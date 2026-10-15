"""Dataset self-consistency checks.

A benchmark fixture can fail in two silent ways, and both would poison the
leaderboard:

1. **Unsatisfiable** -- no response can ever pass (a ``must_be_null`` rule on a
   field the schema forbids from being null; a ``must_equal`` value that
   contradicts an ``enum``). Every model would score zero on that case.
2. **Non-discriminating** -- *any* response passes (a schema with no required
   fields, no enums and ``additionalProperties: true``), so the case measures
   nothing while inflating scores.

:func:`check_case` builds the ideal response with
:func:`structbench.synth.synthesize_case_value`, asserts it passes, then applies
deterministic mutations that *must* fail. Both directions are verified per case,
which is why the bundled suite can be trusted without a judge model.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from structbench.evaluator import compile_case, evaluate
from structbench.models import Case, Completion, ToolCall
from structbench.schema_builder import merge_all_of, resolve_ref
from structbench.synth import SynthError, find_schema_path, synthesize_case_value

__all__ = [
    "CaseCheck",
    "CheckReport",
    "check_case",
    "check_cases",
    "ideal_completion",
]

#: A response that satisfies a ``clarify`` case.
_CLARIFY_RESPONSE = (
    "I need one more detail before I can do that: which date should I use? "
    "Please provide the missing information."
)

#: Key injected when probing whether the schema rejects undeclared fields.
_PROBE_KEY = "__structbench_probe__"


@dataclass
class CaseCheck:
    """The outcome of checking a single case."""

    case_id: str
    category: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class CheckReport:
    """The outcome of checking a whole suite."""

    checks: list[CaseCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    @property
    def errors(self) -> list[str]:
        return [f"{check.case_id}: {message}" for check in self.checks for message in check.errors]

    @property
    def warnings(self) -> list[str]:
        return [
            f"{check.case_id}: {message}" for check in self.checks for message in check.warnings
        ]

    def summary(self) -> str:
        passed = sum(1 for check in self.checks if check.ok)
        return (
            f"{passed}/{len(self.checks)} cases passed self-check; "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _effective_root(case: Case) -> dict[str, Any] | None:
    """The schema whose *root* constraints the discrimination probes inspect."""
    if case.is_tool:
        tool = case.tool_by_name(case.expect.expected_tool)
        return tool.parameters if tool is not None else None
    return case.target_schema


def _flatten(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve root-level ``$ref``/``allOf`` chains."""
    root = schema
    for _ in range(32):
        if "$ref" in schema:
            schema = resolve_ref(schema["$ref"], root)
            continue
        if "allOf" in schema:
            schema = merge_all_of(schema, root)
            continue
        return schema
    return schema


def _scalar_leaf_path(
    schema: dict[str, Any], root: dict[str, Any], prefix: tuple[str | int, ...] = ()
) -> tuple[str | int, ...] | None:
    """Find a path whose schema declares ``integer``/``number``/``boolean``.

    Corrupting such a leaf to the string ``"x"`` is *guaranteed* to fail under
    JSON-exact strict validation, which makes it a reliable discrimination probe.
    """

    def is_numeric(sub: dict[str, Any], _prefix: tuple[str | int, ...]) -> bool:
        declared = sub.get("type")
        names = declared if isinstance(declared, list) else [declared]
        return any(name in ("integer", "number", "boolean") for name in names if name)

    return find_schema_path(schema, root, predicate=is_numeric, prefix=prefix)


def ideal_completion(case: Case) -> Completion:
    """Build the response that :func:`check_case` asserts must pass."""
    if case.is_tool:
        if case.expect.behaviour == "clarify":
            return Completion(text=_CLARIFY_RESPONSE, latency_ms=1.0)
        value = synthesize_case_value(case)
        return Completion(
            tool_calls=[
                ToolCall(
                    id="call_0",
                    name=str(case.expect.expected_tool),
                    arguments_raw=json.dumps(value),
                    arguments=value,
                )
            ],
            latency_ms=1.0,
        )

    value = synthesize_case_value(case)
    return Completion(text=json.dumps(value), latency_ms=1.0)


def _tool_completion(case: Case, name: str, arguments: Any) -> Completion:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return Completion(
        tool_calls=[ToolCall(id="call_0", name=name, arguments_raw=raw)], latency_ms=1.0
    )


def _describe(attempt: Any) -> str:
    label = attempt.label.value if attempt.label is not None else "no-label"
    issues = "; ".join(f"{issue.path}: {issue.detail}" for issue in attempt.issues[:3])
    return f"{label}/{attempt.detail}" + (f" [{issues}]" if issues else "")


# --------------------------------------------------------------------------- #
# The checks
# --------------------------------------------------------------------------- #


def check_case(case: Case) -> CaseCheck:
    """Verify that ``case`` is satisfiable and discriminating."""
    check = CaseCheck(case_id=case.id, category=case.category)

    try:
        compiled = compile_case(case)
    except Exception as exc:
        check.errors.append(f"case does not compile: {exc}")
        return check

    check.warnings.extend(compiled.warnings)

    # ---- 1. satisfiability ------------------------------------------------ #
    try:
        ideal = ideal_completion(case)
    except SynthError as exc:
        check.errors.append(f"no conforming response could be constructed: {exc}")
        return check
    except Exception as exc:
        check.errors.append(f"synthesis failed: {type(exc).__name__}: {exc}")
        return check

    attempt = evaluate(compiled, ideal)
    if not attempt.passed:
        check.errors.append(
            f"the ideal response does NOT pass ({_describe(attempt)}) -- "
            "the case is unsatisfiable as written"
        )
        return check

    # ---- 2. discrimination ------------------------------------------------ #
    cases_checked = 0

    if case.is_tool and case.expect.behaviour == "clarify":
        # Calling a tool when information is missing must be a failure.
        probe = _tool_completion(case, case.tools[0].name, {})
        cases_checked += 1
        if evaluate(compiled, probe).passed:
            check.errors.append(
                "a tool call on an under-specified request was accepted -- the "
                "anti-hallucination probe cannot fire"
            )
    elif case.is_tool:
        tool = case.tool_by_name(case.expect.expected_tool)
        required = (tool.parameters.get("required") if tool else None) or []
        if required:
            cases_checked += 1
            if evaluate(
                compiled, _tool_completion(case, str(case.expect.expected_tool), {})
            ).passed:
                check.errors.append("an argument object missing every required field was accepted")
        else:
            check.warnings.append(
                "the tool declares no required parameters, so this case cannot detect "
                "omitted arguments"
            )
    else:
        root = _effective_root(case)
        if root is None:
            check.warnings.append("no schema available for discrimination probes")
        else:
            flat = _flatten(copy.deepcopy(root))
            ideal_value = json.loads(ideal.text)

            # 2a. undeclared extra field
            if flat.get("additionalProperties") is False:
                cases_checked += 1
                payload = copy.deepcopy(ideal_value)
                if isinstance(payload, dict):
                    payload[_PROBE_KEY] = 1
                    if evaluate(compiled, Completion(text=json.dumps(payload))).passed:
                        check.errors.append(
                            "adding an undeclared field was accepted even though "
                            "additionalProperties is false"
                        )
            else:
                check.warnings.append(
                    "the root object does not set additionalProperties: false, so "
                    "invented fields will not be detected"
                )

            # 2b. dropped required field
            required = flat.get("required") or []
            if required and isinstance(ideal_value, dict):
                cases_checked += 1
                payload = copy.deepcopy(ideal_value)
                payload.pop(required[0], None)
                if evaluate(compiled, Completion(text=json.dumps(payload))).passed:
                    check.errors.append(f"dropping the required field {required[0]!r} was accepted")

            # 2c. wrong primitive type on a scalar leaf
            leaf = _scalar_leaf_path(copy.deepcopy(root), root)
            if leaf is not None:
                from structbench import jsonpath as jp

                cases_checked += 1
                payload = copy.deepcopy(ideal_value)
                try:
                    payload = jp.set_(payload, list(leaf), "x")
                except jp.PathError:
                    payload = None
                if (
                    payload is not None
                    and evaluate(compiled, Completion(text=json.dumps(payload))).passed
                ):
                    check.errors.append(
                        f"a string substituted for the numeric field {jp.format_path(list(leaf))} "
                        "was accepted"
                    )

    if cases_checked == 0:
        check.warnings.append(
            "no discrimination probe could be applied; this case may not measure anything"
        )

    return check


def check_cases(cases: list[Case]) -> CheckReport:
    """Run :func:`check_case` over a suite."""
    return CheckReport(checks=[check_case(case) for case in cases])
