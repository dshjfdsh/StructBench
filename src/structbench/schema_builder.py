"""Compile a JSON Schema into a strict Pydantic v2 validator.

Why not just use the ``jsonschema`` package? Because StructBench's whole pitch
is that validation is *deterministic and free*, and Pydantic gives us structured
``ValidationError`` objects that map cleanly onto a five-way error taxonomy. So
we generate a real Pydantic model (or ``TypeAdapter``) from the schema at
runtime.

The critical design decision here is **JSON-exact strictness**. Pydantic's
default ("lax") mode happily coerces ``"123"`` into ``123``, which would silently
bless exactly the failure mode this benchmark exists to catch. Every generated
type therefore declares ``strict=True`` at its own level, with narrowly scoped
``BeforeValidator`` shims only where JSON and Python disagree about what is
valid (e.g. JSON ``1.0`` *is* an integer; ISO-8601 strings *are* datetimes).

Supported keywords: ``$ref`` (local pointers), ``$defs``/``definitions``,
``type`` (including type arrays and ``null``), ``properties``, ``required``,
``additionalProperties`` (boolean form), ``items``, ``enum``, ``const``,
``allOf``, ``anyOf``/``oneOf``, ``pattern``, ``minLength``/``maxLength``,
``minimum``/``maximum``/``exclusiveMinimum``/``exclusiveMaximum``/``multipleOf``,
``minItems``/``maxItems``, and the ``format`` values below.

Deliberately **not** supported (each is recorded as a compile warning rather than
silently ignored): recursive ``$ref`` cycles, ``patternProperties``,
``additionalProperties`` given as a schema, ``if``/``then``/``else``,
``dependentRequired``, ``unevaluatedProperties`` and tuple-form ``items``.
"""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time
from decimal import Decimal
from ipaddress import IPv4Address, IPv6Address
from typing import Annotated, Any, Literal, Union
from uuid import UUID

from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    TypeAdapter,
    create_model,
)
from pydantic_core import PydanticCustomError

__all__ = [
    "CompiledSchema",
    "SchemaCompileError",
    "build_annotation",
    "compile_schema",
    "merge_all_of",
    "resolve_ref",
]


def resolve_ref(ref: str, root: Any) -> Any:
    """Public wrapper for local ``$ref`` JSON Pointer resolution."""
    return _resolve_ref(ref, root)


def merge_all_of(schema: dict[str, Any], root: Any) -> dict[str, Any]:
    """Public wrapper for ``allOf`` flattening."""
    return _merge_all_of(schema, root)


class SchemaCompileError(ValueError):
    """The schema uses a construct StructBench cannot translate."""


# --------------------------------------------------------------------------- #
# JSON-vs-Python strictness shims
# --------------------------------------------------------------------------- #
#
# Each shim raises a *custom* Pydantic error type. That matters: the evaluator
# maps pydantic error types onto the five-way taxonomy, so a bespoke type lets it
# tell "the model put a boolean where a number belongs" (a type mismatch) apart
# from "the model put a malformed RFC 3339 string where a date-time belongs"
# (a format violation) without parsing human-readable messages.


def _json_int(value: Any) -> Any:
    """JSON ``integer`` means "number with no fractional part", so ``1.0`` counts."""
    if isinstance(value, bool):
        raise PydanticCustomError("structbench_bool_as_integer", "a JSON boolean is not an integer")
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _json_number(value: Any) -> Any:
    """Reject booleans; widen ints to float so strict float comparison succeeds."""
    if isinstance(value, bool):
        raise PydanticCustomError("structbench_bool_as_number", "a JSON boolean is not a number")
    if isinstance(value, int):
        return float(value)
    return value


def _json_decimal(value: Any) -> Any:
    """Widen a JSON number to ``Decimal`` for exact ``multipleOf`` arithmetic.

    Strings are deliberately *not* converted: JSON-exact validation must still
    reject ``"123"`` for a numeric field.
    """
    if isinstance(value, bool):
        raise PydanticCustomError("structbench_bool_as_number", "a JSON boolean is not a number")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        # ``str`` yields the shortest round-tripping representation, i.e. exactly
        # the decimal literal the model emitted.
        return Decimal(str(value))
    return value


def _json_datetime(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            raise PydanticCustomError(
                "structbench_bad_datetime",
                "not an RFC 3339 date-time: {value}",
                {"value": value},
            ) from None
    return value


def _json_date(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            raise PydanticCustomError(
                "structbench_bad_date", "not an ISO 8601 date: {value}", {"value": value}
            ) from None
    return value


def _json_time(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return time.fromisoformat(text)
        except ValueError:
            raise PydanticCustomError(
                "structbench_bad_time", "not an ISO 8601 time: {value}", {"value": value}
            ) from None
    return value


def _json_uuid(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return UUID(value)
        except ValueError:
            raise PydanticCustomError(
                "structbench_bad_uuid", "not a UUID: {value}", {"value": value}
            ) from None
    return value


def _json_ipv4(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return IPv4Address(value)
        except ValueError:
            raise PydanticCustomError(
                "structbench_bad_ipv4", "not an IPv4 address: {value}", {"value": value}
            ) from None
    return value


def _json_ipv6(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return IPv6Address(value)
        except ValueError:
            raise PydanticCustomError(
                "structbench_bad_ipv6", "not an IPv6 address: {value}", {"value": value}
            ) from None
    return value


# fmt: off
#: ``format`` -> (base Python type, BeforeValidator shim, regex fallback pattern)
_FORMAT_TYPES: dict[str, tuple[Any, Any]] = {
    "date-time": (datetime, _json_datetime),
    "date": (date, _json_date),
    "time": (time, _json_time),
    "uuid": (UUID, _json_uuid),
    "ipv4": (IPv4Address, _json_ipv4),
    "ipv6": (IPv6Address, _json_ipv6),
}
# fmt: on

#: Formats validated by regex instead of a Python type -- keeps us dependency
#: free (``email-validator`` is not pulled in) while staying deterministic.
#:
#: NOTE: ``pattern`` values are compiled by pydantic-core's Rust regex engine,
#: which supports neither lookarounds nor backreferences. Every pattern here (and
#: in the datasets) must therefore be lookahead-free.
_FORMAT_PATTERNS: dict[str, str] = {
    "email": r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
    "uri": r"^[A-Za-z][A-Za-z0-9+.\-]*:.*$",
    "uri-reference": r"^.*$",
    "hostname": r"^[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?)*$",
    "regex": r"^.*$",
    "duration": r"^P(?:\d+Y)?(?:\d+M)?(?:\d+D)?(?:T(?:\d+H)?(?:\d+M)?(?:\d+(?:\.\d+)?S)?)?$",
    "json-pointer": r"^(?:/(?:[^~/]|~[01])*)*$",
}

#: Extra constraints implied by a ``format`` that a bare regex cannot express.
_FORMAT_EXTRA_CONSTRAINTS: dict[str, dict[str, Any]] = {
    # RFC 1123 caps a hostname at 253 characters.
    "hostname": {"max_length": 253},
}

_UNSUPPORTED_KEYWORDS = (
    "patternProperties",
    "if",
    "then",
    "else",
    "dependentRequired",
    "dependentSchemas",
    "unevaluatedProperties",
    "unevaluatedItems",
    "contains",
    "prefixItems",
)

_JSON_TYPE_NAMES = {
    "string",
    "integer",
    "number",
    "boolean",
    "object",
    "array",
    "null",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _sanitize_identifier(name: str, fallback: str = "Model") -> str:
    """Turn an arbitrary schema title into a legal Python class name."""
    cleaned = re.sub(r"\W", "_", name or "")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"{fallback}_{cleaned}" if cleaned else fallback
    if keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_"
    return cleaned


def _resolve_ref(ref: str, root: Any) -> Any:
    if not isinstance(ref, str) or not ref.startswith("#"):
        raise SchemaCompileError(f"only local JSON Pointer $refs are supported, got {ref!r}")
    node = root
    pointer = ref[1:].lstrip("/")
    if not pointer:
        return node
    for raw in pointer.split("/"):
        part = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list):
            try:
                node = node[int(part)]
            except (ValueError, IndexError):
                raise SchemaCompileError(f"unresolvable $ref {ref!r}") from None
        else:
            raise SchemaCompileError(f"unresolvable $ref {ref!r}")
    return node


def _merge_all_of(schema: dict[str, Any], root: Any) -> dict[str, Any]:
    """Shallow-merge an ``allOf`` group into one schema object."""
    merged: dict[str, Any] = {k: v for k, v in schema.items() if k != "allOf"}
    subschemas = schema.get("allOf") or []
    if not isinstance(subschemas, list):
        raise SchemaCompileError("allOf must be an array of schemas")
    for sub in subschemas:
        if not isinstance(sub, dict):
            continue
        resolved = _resolve_ref(sub["$ref"], root) if "$ref" in sub else sub
        for key, value in resolved.items():
            if key == "properties" and isinstance(value, dict):
                merged.setdefault("properties", {})
                merged["properties"] = {**merged["properties"], **value}
            elif key == "required" and isinstance(value, list):
                existing = list(merged.get("required") or [])
                for item in value:
                    if item not in existing:
                        existing.append(item)
                merged["required"] = existing
            elif key not in merged:
                merged[key] = value
    return merged


def _constraints(schema: dict[str, Any]) -> dict[str, Any]:
    """Translate numeric/string/array JSON Schema constraints into Pydantic ones."""
    out: dict[str, Any] = {"strict": True}

    for json_key, py_key in (
        ("minimum", "ge"),
        ("maximum", "le"),
        ("minLength", "min_length"),
        ("maxLength", "max_length"),
        ("minItems", "min_length"),
        ("maxItems", "max_length"),
    ):
        if isinstance(schema.get(json_key), (int, float)) and not isinstance(
            schema.get(json_key), bool
        ):
            out[py_key] = schema[json_key]

    # Draft 2020-12 uses numbers; draft-04 used booleans paired with min/max.
    for json_key, py_key in (("exclusiveMinimum", "gt"), ("exclusiveMaximum", "lt")):
        value = schema.get(json_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[py_key] = value

    if isinstance(schema.get("multipleOf"), (int, float)) and not isinstance(
        schema.get("multipleOf"), bool
    ):
        out["multiple_of"] = schema["multipleOf"]

    pattern = schema.get("pattern")
    if isinstance(pattern, str):
        out["pattern"] = pattern

    return out


def _is_hashable(value: Any) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


# --------------------------------------------------------------------------- #
# The translator
# --------------------------------------------------------------------------- #


def build_annotation(
    schema: Any,
    root: Any,
    name: str = "StructBenchPayload",
    *,
    stack: tuple[str, ...] = (),
    warnings: list[str] | None = None,
) -> Any:
    """Translate a JSON Schema node into a Python type annotation."""
    if warnings is None:
        warnings = []

    if schema is True or schema is None:
        return Any
    if schema is False:
        raise SchemaCompileError("the boolean schema `false` accepts nothing")
    if not isinstance(schema, dict):
        raise SchemaCompileError(f"schema node must be an object, got {type(schema).__name__}")

    for key in _UNSUPPORTED_KEYWORDS:
        if key in schema:
            warnings.append(f"{name}: keyword {key!r} is ignored by the StructBench validator")

    if "$ref" in schema:
        ref = schema["$ref"]
        if ref in stack:
            warnings.append(f"{name}: recursive $ref {ref!r} is treated as an unconstrained value")
            return Any
        return build_annotation(
            _resolve_ref(ref, root), root, name, stack=(*stack, ref), warnings=warnings
        )

    if "allOf" in schema:
        return build_annotation(
            _merge_all_of(schema, root), root, name, stack=stack, warnings=warnings
        )

    if "const" in schema:
        value = schema["const"]
        if _is_hashable(value):
            return Literal[value]  # type: ignore[valid-type]
        warnings.append(f"{name}: unhashable `const` treated as an unconstrained value")
        return Any

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        if all(_is_hashable(v) for v in enum_values):
            return Literal[tuple(enum_values)]  # type: ignore[valid-type]
        warnings.append(f"{name}: unhashable `enum` values treated as unconstrained")
        return Any

    for combinator in ("anyOf", "oneOf"):
        options = schema.get(combinator)
        if isinstance(options, list) and options:
            built: list[Any] = []
            for index, option in enumerate(options):
                annotation = build_annotation(
                    option, root, f"{name}_{combinator}{index}", stack=stack, warnings=warnings
                )
                if annotation not in built:
                    built.append(annotation)
            if len(built) == 1:
                return built[0]
            return Union[tuple(built)]  # type: ignore[valid-type]

    declared = schema.get("type")

    if isinstance(declared, list):
        variants: list[Any] = []
        for member in declared:
            if member not in _JSON_TYPE_NAMES:
                warnings.append(f"{name}: unknown JSON type {member!r} ignored")
                continue
            variant = build_annotation(
                {**schema, "type": member},
                root,
                f"{name}_{member}",
                stack=stack,
                warnings=warnings,
            )
            if variant not in variants:
                variants.append(variant)
        if not variants:
            return Any
        if len(variants) == 1:
            return variants[0]
        return Union[tuple(variants)]  # type: ignore[valid-type]

    if declared is None:
        if "properties" in schema or "required" in schema:
            declared = "object"
        elif "items" in schema:
            declared = "array"
        else:
            return Any

    if declared not in _JSON_TYPE_NAMES:
        warnings.append(f"{name}: unknown JSON type {declared!r} treated as unconstrained")
        return Any

    if declared == "null":
        return type(None)

    if declared == "boolean":
        return Annotated[bool, Field(**_constraints(schema))]

    if declared == "integer":
        return Annotated[int, BeforeValidator(_json_int), Field(**_constraints(schema))]

    if declared == "number":
        constraints = _constraints(schema)
        if "multiple_of" in constraints:
            # JSON Schema's ``multipleOf`` is exact arithmetic, but binary floating
            # point is not: ``2.5 % 0.01`` is 0.009999999999999787, so a float
            # check would wrongly reject a price like 249.99 paired with a 0.01
            # step. Decimal makes the check correct rather than merely fast.
            for key in ("multiple_of", "ge", "le", "gt", "lt"):
                if key in constraints:
                    constraints[key] = Decimal(str(constraints[key]))
            return Annotated[Decimal, BeforeValidator(_json_decimal), Field(**constraints)]
        return Annotated[float, BeforeValidator(_json_number), Field(**constraints)]

    if declared == "string":
        text = schema.get("format")
        base: Any = str
        shim = None
        if isinstance(text, str) and text in _FORMAT_TYPES:
            base, shim = _FORMAT_TYPES[text]
        elif isinstance(text, str) and text in _FORMAT_PATTERNS:
            text = schema.get("pattern") or _FORMAT_PATTERNS[text]
            schema = {**schema, "pattern": text}
        constraints = _constraints(schema)
        if isinstance(schema.get("format"), str):
            constraints.update(_FORMAT_EXTRA_CONSTRAINTS.get(schema["format"], {}))
        if shim is not None:
            return Annotated[base, BeforeValidator(shim), Field(**constraints)]
        return Annotated[base, Field(**constraints)]

    if declared == "array":
        items = schema.get("items")
        if isinstance(items, list):
            raise SchemaCompileError(
                "tuple-form `items` (an array of schemas) is not supported by the "
                "StructBench validator; use a single schema and `minItems`"
            )
        if items is None:
            item_type: Any = Any
        else:
            item_type = build_annotation(items, root, f"{name}Item", stack=stack, warnings=warnings)
        return Annotated[list[item_type], Field(**_constraints(schema))]  # type: ignore[valid-type]

    if declared == "object":
        return _build_model(schema, root, name, stack=stack, warnings=warnings)

    raise SchemaCompileError(f"unsupported JSON type {declared!r}")  # pragma: no cover


def _build_model(
    schema: dict[str, Any],
    root: Any,
    name: str,
    *,
    stack: tuple[str, ...],
    warnings: list[str],
) -> Any:
    """Build a Pydantic model class for an ``object`` schema node."""
    properties = schema.get("properties") or {}
    additional = schema.get("additionalProperties", True)

    if not properties and additional is not False:
        # A free-form mapping. ``additionalProperties: false`` falls through to
        # the model path below, where a field-less model with ``extra="forbid"``
        # correctly rejects *every* key.
        return Annotated[dict[str, Any], Field(strict=True)]

    required = set(schema.get("required") or ())
    extra_mode = "forbid" if additional is False else "ignore"
    if isinstance(additional, dict):
        warnings.append(
            f"{name}: `additionalProperties` given as a schema is accepted but not validated"
        )
        extra_mode = "allow"

    fields: dict[str, Any] = {}
    used_names: set[str] = set()

    for index, (prop_name, prop_schema) in enumerate(properties.items()):
        annotation = build_annotation(
            prop_schema if isinstance(prop_schema, dict) else {},
            root,
            f"{name}_{_sanitize_identifier(str(prop_name), 'Field')}",
            stack=stack,
            warnings=warnings,
        )

        safe = (
            str(prop_name)
            if str(prop_name).isidentifier() and not keyword.iskeyword(str(prop_name))
            else _sanitize_identifier(str(prop_name), "field")
        )
        while safe in used_names:
            safe = f"{safe}__{index}"
        used_names.add(safe)

        needs_alias = safe != prop_name
        if prop_name in required:
            default: Any = ...
        else:
            default = prop_schema.get("default") if isinstance(prop_schema, dict) else None

        if needs_alias:
            fields[safe] = (annotation, Field(default, alias=str(prop_name)))
        else:
            fields[safe] = (annotation, default)

    return create_model(  # type: ignore[call-overload]
        _sanitize_identifier(name, "Payload"),
        __config__=ConfigDict(
            strict=True,
            extra=extra_mode,  # type: ignore[arg-type]
            populate_by_name=True,
        ),
        **fields,
    )


# --------------------------------------------------------------------------- #
# Compilation entry point
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class CompiledSchema:
    """A validated, reusable Pydantic validator plus any fidelity caveats."""

    adapter: TypeAdapter[Any]
    name: str
    schema: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


_CACHE: dict[tuple[str, str], CompiledSchema] = {}


def compile_schema(
    schema: dict[str, Any],
    name: str = "StructBenchPayload",
    *,
    use_cache: bool = True,
) -> CompiledSchema:
    """Compile ``schema`` into a strict, reusable validator.

    Raises :class:`SchemaCompileError` if the schema cannot be translated.
    """
    import json

    cache_key = (name, json.dumps(schema, sort_keys=True, ensure_ascii=False, default=str))
    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key]

    warnings: list[str] = []
    annotation = build_annotation(schema, schema, _sanitize_identifier(name), warnings=warnings)
    try:
        adapter = TypeAdapter(annotation)
    except SchemaCompileError:
        raise
    except Exception as exc:
        raise SchemaCompileError(
            f"could not build a validator for {name!r}: {type(exc).__name__}: {exc}. "
            "Note that `pattern` values are compiled by pydantic-core's Rust regex "
            "engine, which supports neither lookarounds ((?=, ?!, ?<=, ?<!)) nor "
            "backreferences -- rewrite the pattern without them."
        ) from None
    compiled = CompiledSchema(adapter=adapter, name=name, schema=schema, warnings=warnings)

    if use_cache:
        _CACHE[cache_key] = compiled
    return compiled
