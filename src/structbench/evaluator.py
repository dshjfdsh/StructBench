"""The deterministic validation sandbox -- StructBench's core.

Pipeline for one model response::

    raw text
      -> extract_payload()      strip a ```json fence, trim surrounding prose
      -> parse_payload()        json.loads, with an ast.literal_eval recovery pass
      -> validate_payload()     strict Pydantic v2 check against the case schema
      -> check_ground_truth()   value-level rules (null-not-invented, no leaks)
      -> Attempt                one of the five ErrorLabels + a fine-grained detail

Two design rules matter:

**Attribution follows root cause.** If the payload cannot be parsed at all, the
verdict is ``Syntax_Truncated`` even when the model *also* wrapped it in a
Markdown fence -- a fence around broken JSON is not the interesting failure.
Only when the payload parses cleanly does a wrapper violation become the verdict.

**Pydantic runs in strict (JSON-exact) mode.** An ``int`` field rejects the string
``"123"``; a non-required field of type ``string`` rejects an explicit ``null``.
Lax coercion would silently bless the exact failures this benchmark exists to
measure. See :mod:`structbench.schema_builder`.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from structbench import jsonpath as jp
from structbench.models import (
    Attempt,
    AttemptStatus,
    Case,
    Completion,
    ErrorLabel,
    FieldIssue,
)
from structbench.schema_builder import CompiledSchema, compile_schema

__all__ = [
    "DETAIL_GLOSS",
    "CompiledCase",
    "Extraction",
    "ParseOutcome",
    "build_feedback",
    "check_ground_truth",
    "classify_json_error",
    "compile_case",
    "evaluate",
    "extract_payload",
    "find_json_span",
    "parse_payload",
    "validate_payload",
]

#: Matches a Markdown fenced block, with or without a language tag and with or
#: without the newline after it (models emit all four combinations).
FENCE_RE = re.compile(
    r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n?(.*?)```",
    re.DOTALL,
)

#: Phrases that count as asking the user for the missing information. Used by
#: ``clarify`` cases; deliberately broad across English and Chinese.
_CLARIFY_RE = re.compile(
    r"\?|？|"
    r"\b(?:missing|required|require|need|needs|need to know|please (?:provide|specify)|"
    r"which|clarif\w*|cannot proceed|unable to proceed|insufficient|ambiguous)\b|"
    r"缺少|缺失|请提供|请问|需要|哪一个|哪一|无法确定|不明确|补充",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Fine-grained detail codes (chart axis + human gloss)
# --------------------------------------------------------------------------- #

DETAIL_GLOSS: dict[str, str] = {
    # -- wrapper ---------------------------------------------------------------
    "fence_leak": "用 ``` 代码块包裹",
    "prose_padding": "前后夹杂寒暄/说明文字",
    "fence_and_prose": "既有代码块又有多余文字",
    # -- syntax ----------------------------------------------------------------
    "empty_output": "空响应",
    "no_json_found": "响应中找不到 JSON",
    "unclosed_bracket": "括号/引号未闭合（输出被截断）",
    "unterminated_string": "字符串未闭合",
    "bad_escape": "非法转义序列",
    "trailing_comma": "多余尾逗号",
    "single_quotes": "使用单引号（非 JSON）",
    "invalid_key": "对象键未加双引号",
    "invalid_delimiter": "缺少分隔符",
    "missing_colon": "缺少冒号",
    "invalid_token": "非法字面量",
    "trailing_data": "JSON 之后还有多余内容",
    "invalid_json": "JSON 语法错误",
    "python_literal": "输出 Python 字面量而非 JSON",
    # -- tool calls ------------------------------------------------------------
    "no_tool_call": "未发起任何工具调用",
    "tool_call_as_text": "把工具参数当普通文本输出",
    "wrong_tool": "调用了错误的工具",
    "undeclared_tool": "调用了未声明的工具（严重幻觉）",
    "unrequested_tool_call": "信息不足却擅自发起调用",
    "no_clarification": "既未追问也未调用",
    "silent_proceed": "未追问就直接编造结果",
    "clarification_requested": "正确追问缺失信息",
    # -- schema ----------------------------------------------------------------
    "missing_field": "缺少必填字段",
    "type_mismatch": "字段类型不匹配",
    "enum_or_const_violation": "枚举/常量取值非法",
    "extra_field": "多出未声明的字段",
    "pattern_violation": "不符合正则约束",
    "format_violation": "不符合格式约束",
    "range_violation": "超出数值范围",
    "string_too_short": "字符串过短",
    "string_too_long": "字符串过长",
    "too_few_items": "数组元素过少",
    "too_many_items": "数组元素过多",
    # -- ground truth ----------------------------------------------------------
    "ground_truth_mismatch": "取值与事实不符",
    "fabricated_missing_value": "为缺失信息编造了取值",
    "missing_required_content": "缺少必须包含的内容",
    "content_leak": "把不该出现的内容泄漏进来",
    "format_mismatch": "值格式不符合要求",
    "forbidden_field": "出现了禁止的字段",
    # -- success / infra -------------------------------------------------------
    "exact_match": "完全合规",
    "provider_error": "请求失败（基础设施问题）",
}


# --------------------------------------------------------------------------- #
# Pydantic error type -> taxonomy
# --------------------------------------------------------------------------- #

_MISMATCH = ErrorLabel.SCHEMA_TYPE_MISMATCH
_HALLUCINATION = ErrorLabel.ENUM_OR_VALUE_HALLUCINATION

#: Exact pydantic error ``type`` -> (label, detail).
_KIND_MAP: dict[str, tuple[ErrorLabel, str]] = {
    "missing": (_MISMATCH, "missing_field"),
    "extra_forbidden": (_HALLUCINATION, "extra_field"),
    # ``enum``/``const`` schemas compile to Literal, which raises literal_error.
    "literal_error": (_HALLUCINATION, "enum_or_const_violation"),
    "enum": (_HALLUCINATION, "enum_or_const_violation"),
    "string_pattern_mismatch": (_MISMATCH, "pattern_violation"),
    "string_too_short": (_MISMATCH, "string_too_short"),
    "string_too_long": (_MISMATCH, "string_too_long"),
    "too_short": (_MISMATCH, "too_few_items"),
    "too_long": (_MISMATCH, "too_many_items"),
    "greater_than": (_MISMATCH, "range_violation"),
    "greater_than_equal": (_MISMATCH, "range_violation"),
    "less_than": (_MISMATCH, "range_violation"),
    "less_than_equal": (_MISMATCH, "range_violation"),
    "multiple_of": (_MISMATCH, "range_violation"),
    "bool_type": (_MISMATCH, "type_mismatch"),
    "bytes_type": (_MISMATCH, "type_mismatch"),
    "date_type": (_MISMATCH, "type_mismatch"),
    "datetime_type": (_MISMATCH, "type_mismatch"),
    "time_type": (_MISMATCH, "type_mismatch"),
    "decimal_type": (_MISMATCH, "type_mismatch"),
    "dict_type": (_MISMATCH, "type_mismatch"),
    "float_type": (_MISMATCH, "type_mismatch"),
    "int_type": (_MISMATCH, "type_mismatch"),
    "list_type": (_MISMATCH, "type_mismatch"),
    "model_attributes_type": (_MISMATCH, "type_mismatch"),
    "model_type": (_MISMATCH, "type_mismatch"),
    "none_required": (_MISMATCH, "type_mismatch"),
    "set_type": (_MISMATCH, "type_mismatch"),
    "string_type": (_MISMATCH, "type_mismatch"),
    "tuple_type": (_MISMATCH, "type_mismatch"),
    "is_instance_of": (_MISMATCH, "type_mismatch"),
    "value_error": (_MISMATCH, "type_mismatch"),
    "json_invalid": (ErrorLabel.SYNTAX_TRUNCATED, "invalid_json"),
    # Raised by our own BeforeValidator shims.
    "structbench_bool_as_integer": (_MISMATCH, "type_mismatch"),
    "structbench_bool_as_number": (_MISMATCH, "type_mismatch"),
    "structbench_bad_datetime": (_MISMATCH, "format_violation"),
    "structbench_bad_date": (_MISMATCH, "format_violation"),
    "structbench_bad_time": (_MISMATCH, "format_violation"),
    "structbench_bad_uuid": (_MISMATCH, "format_violation"),
    "structbench_bad_ipv4": (_MISMATCH, "format_violation"),
    "structbench_bad_ipv6": (_MISMATCH, "format_violation"),
}

_TYPE_SUFFIXES = ("_type", "_parsing")
_MISMATCH_KINDS = frozenset(
    {
        "none_required",
        "is_instance_of",
        "value_error",
        "union_tag_invalid",
        "union_tag_not_found",
        "int_from_float",
    }
)


def classify_pydantic_error(kind: str) -> tuple[ErrorLabel, str]:
    """Map a pydantic error ``type`` onto ``(label, detail)``."""
    mapped = _KIND_MAP.get(kind)
    if mapped is not None:
        return mapped
    if kind in _MISMATCH_KINDS or kind.endswith(_TYPE_SUFFIXES):
        return _MISMATCH, "type_mismatch"
    if kind.endswith("_too_short"):
        return _MISMATCH, "string_too_short"
    if kind.endswith("_too_long"):
        return _MISMATCH, "string_too_long"
    return _MISMATCH, "type_mismatch"


# --------------------------------------------------------------------------- #
# Step 1 -- extraction
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Extraction:
    """The result of pulling a JSON payload out of a raw model response."""

    #: The slice handed to the parser.
    payload_text: str = ""
    #: ``fence``, ``prose``, ``fence+prose`` or ``None`` when the response was
    #: already raw JSON.
    wrapper: str | None = None
    #: False when no ``{``/``[`` was found at all.
    found: bool = False
    #: False when the opening bracket never closed (truncated output).
    complete: bool = True
    #: The original response, for the report's diff view.
    raw_text: str = ""


def find_json_span(text: str) -> tuple[int, int, bool] | None:
    """Locate the first top-level JSON value in ``text``.

    Returns ``(start, end, complete)``. ``complete`` is False when the opening
    bracket is never closed -- the classic truncation failure. String contents and
    backslash escapes are respected so that a brace inside a string literal does
    not confuse the depth counter.
    """
    start = -1
    for index, char in enumerate(text):
        if char in "{[":
            start = index
            break
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0:
                return start, index + 1, True
    return start, len(text), False


def _has_unbalanced_delimiters(text: str) -> bool:
    """True when brackets or a string literal are left open."""
    depth = 0
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
    return depth != 0 or in_string


def extract_payload(text: str | None) -> Extraction:
    """Strip fences/prose and isolate the JSON payload."""
    raw = (text or "").strip()
    result = Extraction(raw_text=text or "")
    if not raw:
        return result

    body = raw
    fenced = False
    fenced_prose = False

    match = FENCE_RE.search(raw)
    if match and match.group(2).strip():
        body = match.group(2).strip()
        fenced = True
        fenced_prose = bool(raw[: match.start()].strip() or raw[match.end() :].strip())

    span = find_json_span(body)
    if span is None:
        # No structural payload: the whole (de-fenced) text is the candidate.
        result.payload_text = body
        result.found = False
        result.complete = False
        wrapper = "fence" if fenced else "prose" if body != raw else None
        result.wrapper = wrapper
        return result

    start, end, complete = span
    result.payload_text = body[start:end]
    result.found = True
    result.complete = complete

    prose = bool(body[:start].strip() or body[end:].strip())
    if fenced and (prose or fenced_prose):
        result.wrapper = "fence+prose"
    elif fenced:
        result.wrapper = "fence"
    elif prose:
        result.wrapper = "prose"
    return result


#: A comma immediately followed by a closing bracket. Used to *infer* a trailing
#: comma on CPython versions whose message for it is generic (<= 3.12 says
#: "Expecting property name..." / "Expecting value").
_TRAILING_COMMA_RE = re.compile(r",\s*[}\]]")


def classify_json_error(text: str, exc: json.JSONDecodeError) -> str:
    """Turn a :class:`json.JSONDecodeError` into a chart-friendly detail code.

    The message catalogue differs between CPython versions -- 3.13 added explicit
    "Illegal trailing comma before end of object/array" messages, while <= 3.12
    reports those positions generically. Both are handled, with the generic case
    falling back to a structural inference.
    """
    if not text.strip():
        return "empty_output"
    message = exc.msg

    # --- messages that are unambiguous on every version ---------------------
    if "trailing comma" in message:
        return "trailing_comma"
    if "Invalid \\escape" in message or "Invalid \\uXXXX" in message:
        return "bad_escape"
    if "Unterminated string" in message:
        return "unterminated_string"

    # --- messages that need a structural second opinion ---------------------
    if "Expecting property name" in message:
        if _TRAILING_COMMA_RE.search(text):
            return "trailing_comma"
        if "'" in text:
            return "single_quotes"
        return "invalid_key"
    if _has_unbalanced_delimiters(text):
        return "unclosed_bracket"
    if "Extra data" in message:
        return "trailing_data"
    if "Expecting ',' delimiter" in message:
        return "invalid_delimiter"
    if "Expecting ':' delimiter" in message:
        return "missing_colon"
    if "Expecting value" in message:
        if _TRAILING_COMMA_RE.search(text):
            return "trailing_comma"
        return "invalid_token"
    return "invalid_json"


@dataclass(slots=True)
class ParseOutcome:
    """Outcome of parsing an extracted payload."""

    value: Any = None
    #: ``None`` means the payload was valid JSON.
    error: str | None = None
    #: True when ``ast.literal_eval`` understood the payload but ``json`` did not
    #: (single quotes, ``True``/``None``, ...). The attempt still fails -- this
    #: only exists so the report can show what the model probably meant.
    recovered_via_ast: bool = False
    message: str = ""


#: Parser details vague enough that "it is a Python literal, not JSON" is a
#: strictly better explanation. A trailing comma, by contrast, is a precise
#: diagnosis that ``ast.literal_eval`` would otherwise mask -- Python dict
#: literals allow trailing commas, so we must not let the recovery pass hide it.
_VAGUE_PARSE_DETAILS = frozenset({"invalid_token", "invalid_json", "invalid_delimiter"})


def parse_payload(payload_text: str) -> ParseOutcome:
    """Parse a payload as strict JSON, with a Python-literal recovery pass."""
    text = (payload_text or "").strip()
    if not text:
        return ParseOutcome(error="empty_output", message="empty payload")

    try:
        return ParseOutcome(value=json.loads(text))
    except json.JSONDecodeError as exc:
        detail = classify_json_error(text, exc)
        message = f"{exc.msg} (line {exc.lineno} column {exc.colno})"

    # Tolerant second pass: models frequently emit Python reprs. We recover the
    # value for diagnostics only -- it never turns a failure into a pass.
    recovered: Any = None
    via_ast = False
    try:
        candidate = ast.literal_eval(text)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        candidate = None
    else:
        if isinstance(candidate, (dict, list)):
            recovered, via_ast = candidate, True

    if via_ast and detail in _VAGUE_PARSE_DETAILS:
        detail = "python_literal"

    return ParseOutcome(
        value=recovered,
        error=detail,
        recovered_via_ast=via_ast,
        message=message,
    )


# --------------------------------------------------------------------------- #
# Step 3 -- schema validation
# --------------------------------------------------------------------------- #


def _issue_from_error(error: dict[str, Any]) -> FieldIssue:
    kind = str(error.get("type", ""))
    label, detail = classify_pydantic_error(kind)
    loc = [token for token in error.get("loc", ()) if token != "__root__"]
    try:
        path = jp.format_path(list(loc))
    except jp.PathError:  # pragma: no cover - defensive
        path = "$" + "".join(f".{token}" for token in loc)
    return FieldIssue(
        path=path,
        message=str(error.get("msg", kind)),
        kind=kind,
        label=label,
        detail=detail,
    )


def validate_payload(compiled: CompiledSchema, payload: Any) -> list[FieldIssue]:
    """Strictly validate ``payload``; return attributed issues (empty == valid)."""
    try:
        compiled.adapter.validate_python(payload)
    except ValidationError as exc:
        return [_issue_from_error(error) for error in exc.errors()]
    return []


# --------------------------------------------------------------------------- #
# Step 4 -- ground truth rules
# --------------------------------------------------------------------------- #


def _rule_issue(path: str, message: str, label: ErrorLabel, detail: str) -> FieldIssue:
    return FieldIssue(path=path, message=message, kind="ground_truth", label=label, detail=detail)


def check_ground_truth(case: Case, value: Any) -> list[FieldIssue]:
    """Apply the case's value-level assertions to a successfully parsed payload."""
    truth = case.ground_truth
    issues: list[FieldIssue] = []

    for path, expected in truth.must_equal.items():
        actual = jp.get(value, path)
        if actual is jp.MISSING:
            issues.append(
                _rule_issue(
                    path,
                    f"expected {expected!r} but the field is absent",
                    _HALLUCINATION,
                    "ground_truth_mismatch",
                )
            )
        elif actual != expected:
            issues.append(
                _rule_issue(
                    path,
                    f"expected {expected!r}, got {actual!r}",
                    _HALLUCINATION,
                    "ground_truth_mismatch",
                )
            )

    for path in truth.must_be_null:
        # Absence is governed by the schema's ``required`` list, so only an
        # explicit non-null value counts as inventing data here.
        if jp.exists(value, path) and not jp.is_null(value, path):
            actual = jp.get(value, path)
            issues.append(
                _rule_issue(
                    path,
                    f"must be null rather than invent a value, got {actual!r}",
                    _HALLUCINATION,
                    "fabricated_missing_value",
                )
            )

    for path, needle in truth.must_contain.items():
        actual = jp.get(value, path)
        if not isinstance(actual, str) or needle not in actual:
            issues.append(
                _rule_issue(
                    path, f"value must contain {needle!r}", _MISMATCH, "missing_required_content"
                )
            )

    for path, needle in truth.must_not_contain.items():
        actual = jp.get(value, path)
        if isinstance(actual, str) and needle in actual:
            issues.append(
                _rule_issue(
                    path, f"value must not contain {needle!r}", _HALLUCINATION, "content_leak"
                )
            )

    for path, pattern in truth.must_match.items():
        actual = jp.get(value, path)
        try:
            matched = isinstance(actual, str) and re.fullmatch(pattern, actual) is not None
        except re.error as exc:  # pragma: no cover - dataset authoring bug
            issues.append(
                _rule_issue(
                    path, f"invalid rule regex {pattern!r}: {exc}", _MISMATCH, "format_mismatch"
                )
            )
            continue
        if not matched:
            issues.append(
                _rule_issue(
                    path,
                    f"value must match /{pattern}/, got {actual!r}",
                    _MISMATCH,
                    "format_mismatch",
                )
            )

    for path in truth.forbidden_paths:
        if jp.exists(value, path):
            issues.append(
                _rule_issue(
                    path, "this field must not be present at all", _HALLUCINATION, "forbidden_field"
                )
            )

    return issues


def _dominant(issues: list[FieldIssue]) -> FieldIssue:
    """Pick the headline issue: an invented value outranks a shape problem."""
    for issue in issues:
        if issue.label is _HALLUCINATION:
            return issue
    return issues[0]


# --------------------------------------------------------------------------- #
# Compilation
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class CompiledCase:
    """A case plus its pre-compiled validators (built once, reused per attempt)."""

    case: Case
    payload: CompiledSchema | None = None
    tools: dict[str, CompiledSchema] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def compile_case(case: Case) -> CompiledCase:
    """Compile a case's schemas into reusable strict validators."""
    compiled = CompiledCase(case=case)

    # Fail fast on malformed rule paths: a typo should break loading, not a run.
    truth = case.ground_truth
    for path in (
        list(truth.must_equal)
        + truth.must_be_null
        + list(truth.must_contain)
        + list(truth.must_not_contain)
        + list(truth.must_match)
        + truth.forbidden_paths
    ):
        try:
            jp.parse_path(path)
        except jp.PathError as exc:
            raise ValueError(f"case {case.id!r}: {exc}") from None

    if case.target_schema is not None:
        schema = compile_schema(case.target_schema, f"{case.id}_payload")
        compiled.payload = schema
        compiled.warnings.extend(schema.warnings)

    for tool in case.tools:
        if not tool.parameters:
            compiled.warnings.append(
                f"case {case.id!r}: tool {tool.name!r} has an empty parameter schema, "
                "so its arguments cannot be validated"
            )
        schema = compile_schema(tool.parameters, f"{case.id}_tool_{tool.name}")
        compiled.tools[tool.name] = schema
        compiled.warnings.extend(schema.warnings)

    return compiled


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def _fail(attempt: Attempt, issues: list[FieldIssue]) -> None:
    dominant = _dominant(issues)
    attempt.passed = False
    attempt.issues = issues
    attempt.label = dominant.label
    attempt.detail = dominant.detail


def _pass(attempt: Attempt, detail: str) -> None:
    attempt.passed = True
    attempt.label = ErrorLabel.VALID_PASS
    attempt.detail = detail


def _looks_like_clarification(text: str) -> bool:
    return bool(text) and _CLARIFY_RE.search(text) is not None


def evaluate(
    compiled: CompiledCase,
    completion: Completion,
    round_index: int = 1,
) -> Attempt:
    """Score one provider response against one case."""
    case = compiled.case
    attempt = Attempt(
        round=round_index,
        raw_text=completion.text or "",
        tool_calls=completion.tool_calls,
        latency_ms=completion.latency_ms,
        ttft_ms=completion.ttft_ms,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
    )

    if not completion.ok:
        attempt.status = AttemptStatus.PROVIDER_ERROR
        attempt.provider_error = completion.provider_error
        attempt.passed = False
        attempt.label = None
        attempt.detail = "provider_error"
        attempt.feedback = ""
        return attempt

    if case.is_tool:
        _evaluate_tool(compiled, completion, attempt)
    else:
        _evaluate_json_payload(compiled, completion, attempt)

    if not attempt.passed:
        attempt.feedback = build_feedback(case, attempt)
    return attempt


def _evaluate_json_payload(
    compiled: CompiledCase, completion: Completion, attempt: Attempt
) -> None:
    case = compiled.case
    extraction = extract_payload(completion.text)
    attempt.payload_text = extraction.payload_text
    attempt.wrapper = extraction.wrapper

    outcome = parse_payload(extraction.payload_text)
    attempt.parsed = outcome.value if outcome.error is None or outcome.recovered_via_ast else None

    # 1. Root-cause priority: an unparseable payload outranks everything else.
    if outcome.error is not None:
        if not extraction.found:
            # No bracket anywhere. A bare scalar payload never lands here because
            # ``json.loads("42")`` succeeds, so this really is "no JSON at all".
            detail = "empty_output" if not (attempt.raw_text or "").strip() else "no_json_found"
            message = "the response contains no JSON object or array"
        else:
            detail = outcome.error
            message = outcome.message or outcome.error
            if outcome.recovered_via_ast:
                message += (
                    " -- the payload parses as Python, but JSON requires double-quoted "
                    "keys/strings and lowercase true/false/null"
                )
        _fail(
            attempt,
            [
                FieldIssue(
                    path="$",
                    message=message,
                    kind="json",
                    label=ErrorLabel.SYNTAX_TRUNCATED,
                    detail=detail,
                )
            ],
        )
        return

    # 2. Wrapper contract, checked only once we know the JSON itself is sound.
    if extraction.wrapper and not case.allow_wrapper:
        _fail(
            attempt,
            [
                FieldIssue(
                    path="$",
                    message=(
                        "the response must be raw JSON with no Markdown fence and no "
                        f"surrounding commentary (detected: {extraction.wrapper})"
                    ),
                    kind="wrapper",
                    label=ErrorLabel.MARKDOWN_WRAPPER_LEAK,
                    detail={
                        "fence": "fence_leak",
                        "prose": "prose_padding",
                        "fence+prose": "fence_and_prose",
                    }[extraction.wrapper],
                )
            ],
        )
        return

    if compiled.payload is None:  # pragma: no cover - guarded by Case validation
        _pass(attempt, "exact_match")
        return

    # 3. Structural + type validation.
    issues = validate_payload(compiled.payload, outcome.value)

    # 4. Value-level rules, only when the structure is sound.
    if not issues:
        issues = check_ground_truth(case, outcome.value)

    if issues:
        _fail(attempt, issues)
    else:
        _pass(attempt, "exact_match")


def _evaluate_tool(compiled: CompiledCase, completion: Completion, attempt: Attempt) -> None:
    case = compiled.case
    calls = completion.tool_calls

    if case.expect.behaviour == "clarify":
        _evaluate_clarify(compiled, completion, attempt)
        return

    # -- behaviour == "call" -------------------------------------------------
    expected = case.expect.expected_tool
    if not calls:
        # Some models put the arguments in `content` instead of emitting a call.
        outcome = parse_payload(extract_payload(completion.text).payload_text)
        as_text = outcome.error is None and isinstance(outcome.value, dict)
        _fail(
            attempt,
            [
                FieldIssue(
                    path="$",
                    message=(
                        "expected a tool call to "
                        f"{expected!r} but the model replied with the arguments as plain text"
                        if as_text
                        else f"expected a tool call to {expected!r} but none was issued"
                    ),
                    kind="tool_call",
                    label=ErrorLabel.SYNTAX_TRUNCATED,
                    detail="tool_call_as_text" if as_text else "no_tool_call",
                )
            ],
        )
        return

    declared = {tool.name for tool in case.tools}
    chosen = next((call for call in calls if call.name == expected), None)
    if chosen is None:
        called = calls[0].name
        undeclared = called not in declared
        _fail(
            attempt,
            [
                FieldIssue(
                    path="$",
                    message=(
                        f"called undeclared tool {called!r}; expected {expected!r}"
                        if undeclared
                        else f"called {called!r}; expected {expected!r}"
                    ),
                    kind="tool_call",
                    label=_HALLUCINATION,
                    detail="undeclared_tool" if undeclared else "wrong_tool",
                )
            ],
        )
        return

    raw_arguments = chosen.arguments_raw or ""
    extraction = extract_payload(raw_arguments)
    attempt.payload_text = extraction.payload_text
    attempt.wrapper = extraction.wrapper
    outcome = parse_payload(extraction.payload_text)
    attempt.parsed = outcome.value if outcome.error is None else None

    if outcome.error is not None:
        _fail(
            attempt,
            [
                FieldIssue(
                    path="$",
                    message=(
                        f"arguments for {expected!r} are not valid JSON: "
                        f"{outcome.message or outcome.error}"
                    ),
                    kind="json",
                    label=ErrorLabel.SYNTAX_TRUNCATED,
                    detail=f"tool_args_{outcome.error}",
                )
            ],
        )
        return

    if extraction.wrapper and not case.allow_wrapper:
        _fail(
            attempt,
            [
                FieldIssue(
                    path="$",
                    message=f"tool arguments must be raw JSON (detected: {extraction.wrapper})",
                    kind="wrapper",
                    label=ErrorLabel.MARKDOWN_WRAPPER_LEAK,
                    detail={
                        "fence": "fence_leak",
                        "prose": "prose_padding",
                        "fence+prose": "fence_and_prose",
                    }[extraction.wrapper],
                )
            ],
        )
        return

    arguments = outcome.value
    schema = compiled.tools.get(expected)
    issues: list[FieldIssue] = []
    if schema is not None:
        issues = validate_payload(schema, arguments)
    if not issues and isinstance(arguments, dict):
        issues = check_ground_truth(case, arguments)
    if not issues and not isinstance(arguments, dict):
        issues = [
            FieldIssue(
                path="$",
                message=f"tool arguments must be a JSON object, got {type(arguments).__name__}",
                kind="tool_args",
                label=_MISMATCH,
                detail="type_mismatch",
            )
        ]

    if issues:
        _fail(attempt, issues)
    else:
        _pass(attempt, "exact_match")


def _evaluate_clarify(compiled: CompiledCase, completion: Completion, attempt: Attempt) -> None:
    """Score an anti-hallucination case where the model should refuse to guess."""
    calls = completion.tool_calls

    if calls:
        invented = ", ".join(sorted({call.name for call in calls}))
        _fail(
            attempt,
            [
                FieldIssue(
                    path="$",
                    message=(
                        f"the request is missing a required argument, so the model should have "
                        f"asked for it instead of calling {invented}"
                    ),
                    kind="tool_call",
                    label=_HALLUCINATION,
                    detail="unrequested_tool_call",
                )
            ],
        )
        return

    text = completion.text or ""
    if _looks_like_clarification(text):
        _pass(attempt, "clarification_requested")
        return

    attempt.payload_text = extract_payload(text).payload_text
    outcome = parse_payload(attempt.payload_text)
    attempt.parsed = outcome.value if outcome.error is None else None
    if outcome.error is None:
        _fail(
            attempt,
            [
                FieldIssue(
                    path="$",
                    message="the model proceeded without asking for the missing argument",
                    kind="tool_call",
                    label=_HALLUCINATION,
                    detail="silent_proceed",
                )
            ],
        )
    else:
        _fail(
            attempt,
            [
                FieldIssue(
                    path="$",
                    message=(
                        "the model neither asked for the missing argument nor produced a "
                        f"usable response ({outcome.error})"
                    ),
                    kind="tool_call",
                    label=ErrorLabel.SYNTAX_TRUNCATED,
                    detail="no_clarification",
                )
            ],
        )


# --------------------------------------------------------------------------- #
# Healing feedback
# --------------------------------------------------------------------------- #

_FEEDBACK_HEADER = "StructBench validation feedback"


def build_feedback(case: Case, attempt: Attempt, *, max_issues: int = 12) -> str:
    """Build the deterministic repair message replayed to the model.

    The exact text matters: it is the *only* new information the model receives
    between attempt 1 and attempt 2, so Heal@1 measures precisely "given a
    precise error message, can the model repair its own output?".
    """
    lines = [
        _FEEDBACK_HEADER,
        f"Your previous response for case `{case.id}` failed deterministic validation.",
    ]

    if attempt.status is AttemptStatus.PROVIDER_ERROR:
        return ""
    if attempt.label is not None:
        lines.append(f"Verdict: {attempt.label.value} ({attempt.detail})")
    if attempt.wrapper:
        lines.append(
            "- Reply with the raw payload only: no Markdown code fence, no preamble, "
            "no trailing commentary."
        )

    for issue in attempt.issues[:max_issues]:
        lines.append(f"- at {issue.path}: {issue.message}")
    if len(attempt.issues) > max_issues:
        lines.append(f"- ...and {len(attempt.issues) - max_issues} more problem(s)")

    if case.is_tool and case.expect.behaviour == "call":
        lines.append(
            f"- Call the tool `{case.expect.expected_tool}` exactly once with valid JSON arguments."
        )
    elif case.is_tool and case.expect.behaviour == "clarify":
        lines.append(
            "- The request is missing information you need. Ask for it; do not call a tool."
        )
    else:
        lines.append("- The payload must match the requested JSON schema exactly.")

    lines.append("Reply with the corrected output only.")
    return "\n".join(lines)
