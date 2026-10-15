"""Synthesise schema-conforming values.

Two consumers depend on this:

* the offline :class:`~structbench.providers.mock.MockProvider`, which needs a
  plausible "correct" answer to then deliberately corrupt; and
* :mod:`structbench.selfcheck`, which proves a dataset case is *satisfiable* by
  building the ideal response and asserting that it passes.

Neither needs a general-purpose faker: it needs a deterministic value that a
strict, JSON-exact validator will accept. Where that is impossible -- a regex no
candidate satisfies, or a schema whose constraints contradict each other -- we
raise :class:`SynthError` rather than guess, because that is exactly the kind of
dataset bug worth surfacing.
"""

from __future__ import annotations

import copy
import re
from decimal import ROUND_CEILING, Decimal
from typing import Any

from structbench import jsonpath as jp
from structbench.models import Case, GroundTruth
from structbench.regexgen import sample_for_pattern
from structbench.schema_builder import merge_all_of, resolve_ref

__all__ = [
    "SynthError",
    "apply_ground_truth",
    "find_schema_path",
    "is_nullable",
    "schema_at_path",
    "synth_value",
    "synthesize_case_value",
]


class SynthError(RuntimeError):
    """No value can be constructed for this schema (or the schema is unusable)."""


#: Canonical examples for JSON Schema ``format`` values. All are accepted by the
#: shims in :mod:`structbench.schema_builder`.
_FORMAT_SAMPLES: dict[str, str] = {
    "date-time": "2024-01-02T03:04:05Z",
    "date": "2024-01-02",
    "time": "03:04:05",
    "uuid": "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
    "email": "ada@example.com",
    "uri": "https://example.com/a",
    "uri-reference": "/a",
    "hostname": "example.com",
    "ipv4": "127.0.0.1",
    "ipv6": "::1",
    "regex": "^a$",
    "duration": "P1D",
    "json-pointer": "/a",
}

#: Tried in order when a ``pattern`` has to be satisfied. Covers the shapes that
#: actually appear in StructBench fixtures (codes, ids, timestamps, slugs, money,
#: dotted versions, hostnames).
_PATTERN_CANDIDATES: tuple[str, ...] = (
    "AB",
    "abc",
    "ABC123",
    "a1",
    "A1",
    "x",
    "1",
    "1234",
    "1234567890",
    "8675309",
    "2024-01-02",
    "2024-01-02T03:04:05Z",
    "2024-01-02T03:04:05.123Z",
    "BK-4471",
    "active",
    "SELECT 1",
    "ada@example.com",
    "https://example.com/a",
    "example.com",
    "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
    "127.0.0.1",
    "::1",
    # Dotted shapes: money, decimals, semver, hostnames, IPv4-in-range checks.
    "19.90",
    "0.00",
    "1.0",
    "99.99",
    "12.34",
    "v1.2.3",
    "10.0.0.1",
    "sub.example.com",
)


def _wrap_plain(schema: dict[str, Any], root: Any, depth: int) -> dict[str, Any]:
    """Resolve ``$ref``/``allOf`` until no wrapper remains."""
    for _ in range(32):
        if "$ref" in schema:
            schema = resolve_ref(schema["$ref"], root)
            continue
        if "allOf" in schema:
            schema = merge_all_of(schema, root)
            continue
        return schema
    raise SynthError("$ref/allOf chain is too deep to resolve")


def _declared_types(schema: dict[str, Any]) -> list[str]:
    declared = schema.get("type")
    if isinstance(declared, str):
        return [declared]
    if isinstance(declared, list):
        return [item for item in declared if isinstance(item, str)]
    return []


def synth_value(
    schema: Any,
    root: Any = None,
    *,
    depth: int = 0,
    path: str = "$",
) -> Any:
    """Build one deterministic value that conforms to ``schema``."""
    if depth > 16:
        raise SynthError(f"{path}: schema nests too deeply to synthesise")
    if schema is True or schema is None:
        return None
    if not isinstance(schema, dict):
        raise SynthError(f"{path}: schema node must be an object")

    if root is None:
        root = schema
    schema = _wrap_plain(schema, root, depth)

    if "const" in schema:
        return schema["const"]

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return enum_values[0]

    for combinator in ("anyOf", "oneOf"):
        options = schema.get(combinator)
        if isinstance(options, list) and options:
            errors: list[str] = []
            for index, option in enumerate(options):
                try:
                    return synth_value(
                        option, root, depth=depth + 1, path=f"{path}.{combinator}[{index}]"
                    )
                except SynthError as exc:
                    errors.append(str(exc))
            raise SynthError(
                f"{path}: no branch of `{combinator}` can be synthesised ({'; '.join(errors)})"
            )

    types = _declared_types(schema)
    if types and "null" in types and len(types) > 1:
        types = [item for item in types if item != "null"]
    if types == ["null"]:
        return None

    if not types:
        if "properties" in schema or "required" in schema:
            types = ["object"]
        elif "items" in schema:
            types = ["array"]
        elif "pattern" in schema or "format" in schema or "minLength" in schema:
            types = ["string"]
        else:
            return None

    kind = types[0]
    if kind == "object":
        return _synthesize_object(schema, root, depth, path)
    if kind == "array":
        return _synthesize_array(schema, root, depth, path)
    if kind == "string":
        return _synthesize_string(schema, path)
    if kind == "integer":
        return _synthesize_number(schema, path, integral=True)
    if kind == "number":
        return _synthesize_number(schema, path, integral=False)
    if kind == "boolean":
        return True
    if kind == "null":
        return None
    raise SynthError(f"{path}: unsupported JSON type {kind!r}")


def _synthesize_object(schema: dict[str, Any], root: Any, depth: int, path: str) -> dict[str, Any]:
    properties = schema.get("properties") or {}
    if not isinstance(properties, dict):
        raise SynthError(f"{path}: `properties` must be an object")
    result: dict[str, Any] = {}
    for name, prop_schema in properties.items():
        child_path = f"{path}.{name}"
        if isinstance(prop_schema, dict) and "default" in prop_schema:
            result[name] = prop_schema["default"]
            continue
        result[name] = synth_value(prop_schema, root, depth=depth + 1, path=child_path)
    return result


def _synthesize_array(schema: dict[str, Any], root: Any, depth: int, path: str) -> list[Any]:
    minimum = schema.get("minItems") or 0
    maximum = schema.get("maxItems")
    if isinstance(maximum, int) and minimum > maximum:
        raise SynthError(f"{path}: minItems {minimum} exceeds maxItems {maximum}")

    items = schema.get("items")
    if items is None:
        return [None] * int(minimum)

    if isinstance(items, list):  # tuple form is unsupported by the validator
        raise SynthError(f"{path}: tuple-form `items` is not supported")

    count = max(int(minimum), 1)
    sample = synth_value(items, root, depth=depth + 1, path=f"{path}[]")
    # Deep-copy per element: sharing one object across indices would make a
    # ground-truth rule on index [1] silently rewrite index [0] as well.
    return [copy.deepcopy(sample) for _ in range(count)]


def _synthesize_string(schema: dict[str, Any], path: str) -> str:
    minimum = schema.get("minLength") or 0
    maximum = schema.get("maxLength")

    pattern = schema.get("pattern")
    if isinstance(pattern, str):
        # JSON Schema (and Pydantic) treat `pattern` as a partial match.
        candidate = _string_matching(pattern, minimum, maximum, full=False)
        if candidate is None:
            raise SynthError(
                f"{path}: no synthesised value satisfies pattern {pattern!r} "
                f"(minLength={minimum}, maxLength={maximum})"
            )
        return candidate

    text_format = schema.get("format")
    if isinstance(text_format, str) and text_format in _FORMAT_SAMPLES:
        text = _FORMAT_SAMPLES[text_format]
    else:
        text = "sample"

    if maximum is not None and len(text) > maximum:
        text = text[:maximum]
    if len(text) < minimum:
        text = text + "x" * (minimum - len(text))
    return text


def _string_matching(
    pattern: str, minimum: int, maximum: int | None, *, full: bool = True
) -> str | None:
    """Find a deterministic string that satisfies ``pattern``.

    Human-readable candidates are tried first (they make self-check diagnostics
    easier to read), then the regex sampler in :mod:`structbench.regexgen` handles
    patterns no fixed list could ever cover, such as ``^CUST-[0-9]{4}$``.

    ``full=True`` requires a full match -- StructBench's ``must_match`` rule.
    ``full=False`` accepts a partial match, which is what JSON Schema's
    ``pattern`` keyword means (and what Pydantic enforces).
    """
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise SynthError(f"invalid pattern {pattern!r}: {exc}") from None

    def fits(text: str) -> bool:
        if len(text) < minimum:
            return False
        return maximum is None or len(text) <= maximum

    def matches(text: str) -> bool:
        if full:
            return compiled.fullmatch(text) is not None
        return compiled.search(text) is not None

    candidates = list(_PATTERN_CANDIDATES)
    for size in range(1, minimum + 9):
        candidates.extend(("1" * size, "a" * size, "A" * size, "0" * size))
    for candidate in candidates:
        if fits(candidate) and matches(candidate):
            return candidate

    return sample_for_pattern(pattern, min_length=minimum, max_length=maximum, full=full)


def _synthesize_number(schema: dict[str, Any], path: str, *, integral: bool) -> float:
    exclusive_min = schema.get("exclusiveMinimum")
    inclusive_min = schema.get("minimum")
    maximum = schema.get("maximum")
    exclusive_max = schema.get("exclusiveMaximum")
    multiple = schema.get("multipleOf")

    if isinstance(inclusive_min, (int, float)) and not isinstance(inclusive_min, bool):
        value: float = float(inclusive_min)
    elif isinstance(exclusive_min, (int, float)) and not isinstance(exclusive_min, bool):
        step = 1.0 if integral else 0.5
        value = float(exclusive_min) + step
    else:
        value = 1.0

    if isinstance(multiple, (int, float)) and not isinstance(multiple, bool) and multiple > 0:
        # Decimal arithmetic: ``2.5 % 0.01`` is not 0 in binary floating point, so
        # a float-derived multiple can land on 0.9999999999999999 and be rejected
        # by the exact check the validator performs.
        step = Decimal(str(multiple))
        steps = (Decimal(str(value)) / step).to_integral_value(rounding=ROUND_CEILING)
        value = float(steps * step)
        if value == 0 and inclusive_min is None:
            value = float(step)

    if isinstance(maximum, (int, float)) and not isinstance(maximum, bool) and value > maximum:
        raise SynthError(
            f"{path}: no value satisfies minimum {inclusive_min} and maximum {maximum}"
        )
    if (
        isinstance(exclusive_max, (int, float))
        and not isinstance(exclusive_max, bool)
        and value >= exclusive_max
    ):
        raise SynthError(f"{path}: no value satisfies the exclusive maximum {exclusive_max}")

    return int(value) if integral else float(value)


# --------------------------------------------------------------------------- #
# Schema navigation
# --------------------------------------------------------------------------- #


def schema_at_path(schema: Any, tokens: list[str | int], root: Any = None) -> dict[str, Any] | None:
    """Return the sub-schema at ``tokens``, or ``None`` when it cannot be found."""
    if root is None:
        root = schema
    node = schema
    for token in tokens:
        if not isinstance(node, dict):
            return None
        try:
            node = _wrap_plain(node, root, 0)
        except SynthError:
            return None

        if isinstance(token, int):
            items = node.get("items")
            if isinstance(items, list):
                if token >= len(items):
                    return None
                node = items[token]
            elif isinstance(items, dict):
                node = items
            else:
                return None
            continue

        properties = node.get("properties") or {}
        if isinstance(properties, dict) and token in properties:
            node = properties[token]
            continue
        additional = node.get("additionalProperties")
        if isinstance(additional, dict):
            node = additional
            continue
        return None
    return node if isinstance(node, dict) else None


def find_schema_path(
    schema: dict[str, Any],
    root: dict[str, Any] | None = None,
    *,
    predicate: Any,
    prefix: tuple[str | int, ...] = (),
) -> tuple[str | int, ...] | None:
    """Find the first path (depth-first) whose sub-schema satisfies ``predicate``.

    Used to locate a field worth corrupting: the first ``enum``, the first numeric
    leaf, the first ``must_be_null`` target, and so on.
    """
    if root is None:
        root = schema
    try:
        schema = _wrap_plain(schema, root, 0)
    except SynthError:
        return None

    if predicate(schema, prefix):
        return prefix or None

    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, child in properties.items():
            if not isinstance(child, dict):
                continue
            found = find_schema_path(child, root, predicate=predicate, prefix=(*prefix, name))
            if found:
                return found

    items = schema.get("items")
    if isinstance(items, dict):
        found = find_schema_path(items, root, predicate=predicate, prefix=(*prefix, 0))
        if found:
            return found

    for combinator in ("anyOf", "oneOf"):
        options = schema.get(combinator)
        if isinstance(options, list):
            for option in options:
                if not isinstance(option, dict):
                    continue
                found = find_schema_path(option, root, predicate=predicate, prefix=prefix)
                if found:
                    return found

    return None


def is_nullable(schema: Any, root: Any = None) -> bool:
    """True when ``schema`` admits an explicit JSON ``null``."""
    if root is None:
        root = schema
    if not isinstance(schema, dict):
        return True
    try:
        schema = _wrap_plain(schema, root, 0)
    except SynthError:
        return True

    if "const" in schema:
        return schema["const"] is None
    enum_values = schema.get("enum")
    if isinstance(enum_values, list):
        return None in enum_values

    for combinator in ("anyOf", "oneOf"):
        options = schema.get(combinator)
        if isinstance(options, list) and options:
            return any(is_nullable(option, root) for option in options)

    all_of = schema.get("allOf")
    if isinstance(all_of, list) and all_of:
        return all(is_nullable(option, root) for option in all_of)

    types = _declared_types(schema)
    if not types:
        return True  # unconstrained
    return "null" in types


# --------------------------------------------------------------------------- #
# Case-level synthesis
# --------------------------------------------------------------------------- #


def apply_ground_truth(
    value: Any,
    truth: GroundTruth,
    root_schema: dict[str, Any] | None,
    *,
    case_id: str = "<case>",
) -> Any:
    """Fold the value-level rules into an already schema-conforming ``value``.

    Ordering matters: exact values (``must_equal``) are applied last so they win
    over the softer rules. If two rules genuinely contradict each other the
    resulting value fails validation, and the self-check reports the case as
    unsatisfiable -- which is the correct outcome.
    """
    if root_schema is None:
        return value

    for path in truth.must_be_null:
        tokens = jp.parse_path(path)
        sub = schema_at_path(root_schema, tokens, root_schema)
        if sub is not None and not is_nullable(sub, root_schema):
            raise SynthError(
                f"case {case_id}: `must_be_null` targets {path}, but the schema there does "
                'not permit null -- declare the type as e.g. ["string", "null"]'
            )
        value = jp.set_(value, tokens, None)

    for path, needle in truth.must_contain.items():
        tokens = jp.parse_path(path)
        sub = schema_at_path(root_schema, tokens, root_schema) or {}
        current = jp.get(value, path)
        text = current if isinstance(current, str) else ""
        if needle not in text:
            text = text + needle
        minimum = sub.get("minLength")
        maximum = sub.get("maxLength")
        if isinstance(minimum, int) and len(text) < minimum:
            text = text + "x" * (minimum - len(text))
        if isinstance(maximum, int) and len(text) > maximum:
            raise SynthError(
                f"case {case_id}: `must_contain` on {path} needs {needle!r} but maxLength "
                f"is {maximum}"
            )
        value = jp.set_(value, tokens, text)

    for path, pattern in truth.must_match.items():
        tokens = jp.parse_path(path)
        sub = schema_at_path(root_schema, tokens, root_schema) or {}
        text = _string_matching(pattern, sub.get("minLength") or 0, sub.get("maxLength"))
        if text is None:
            raise SynthError(
                f"case {case_id}: cannot synthesise a value for `must_match` {pattern!r} at {path}"
            )
        value = jp.set_(value, tokens, text)

    for path, expected in truth.must_equal.items():
        value = jp.set_(value, jp.parse_path(path), expected)

    for path in truth.forbidden_paths:
        tokens = jp.parse_path(path)
        # A path that the schema also *requires* is a contradictory fixture.
        parent = schema_at_path(root_schema, tokens[:-1], root_schema) or {}
        if tokens[-1] in (parent.get("required") or []):
            raise SynthError(
                f"case {case_id}: {path} is listed in `forbidden_paths` but the schema "
                "also marks it as required"
            )
        jp.delete(value, tokens)

    return value


def synthesize_case_value(case: Case) -> Any:
    """The ideal payload for ``case``.

    For JSON-mode cases this is the response body; for tool-mode ``call`` cases it
    is the tool-call argument object. Returns ``None`` for ``clarify`` cases,
    which are supposed to have no payload at all.
    """
    if case.is_tool:
        if case.expect.behaviour != "call":
            return None
        schema = case.tool_by_name(case.expect.expected_tool)
        root_schema = schema.parameters if schema else None
    else:
        root_schema = case.target_schema

    if not root_schema:
        raise SynthError(f"case {case.id}: no schema to synthesise from")

    value = synth_value(root_schema, root_schema)
    return apply_ground_truth(value, case.ground_truth, root_schema, case_id=case.id)
