"""Tests for the JSON Schema compiler, JSONPath helpers and the reporters.

The schema compiler is where correctness either holds or silently rots: a lax
conversion would accept the very outputs the benchmark exists to reject, so the
strictness matrix is pinned here case by case.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from structbench import jsonpath as jp
from structbench.metrics import compute_run_result
from structbench.models import Attempt, CaseResult, ErrorLabel
from structbench.reporter import (
    build_payload,
    render_html_report,
    render_markdown,
    render_text_report,
    write_html_report,
)
from structbench.schema_builder import (
    SchemaCompileError,
    build_annotation,
    compile_schema,
)


def validates(schema: dict, value) -> bool:
    try:
        compile_schema(schema, "probe").adapter.validate_python(value)
    except ValidationError:
        return False
    return True


def error_types(schema: dict, value) -> list[str]:
    try:
        compile_schema(schema, "probe").adapter.validate_python(value)
    except ValidationError as exc:
        return [error["type"] for error in exc.errors()]
    return []


# --------------------------------------------------------------------------- #
# JSON-exact strictness
# --------------------------------------------------------------------------- #


class TestStrictness:
    def test_integer_rejects_a_numeric_string(self) -> None:
        assert validates({"type": "integer"}, 5) is True
        assert validates({"type": "integer"}, "5") is False
        assert error_types({"type": "integer"}, "5") == ["int_type"]

    def test_json_says_an_integer_valued_float_is_an_integer(self) -> None:
        assert validates({"type": "integer"}, 5.0) is True
        assert validates({"type": "integer"}, 5.5) is False

    def test_number_accepts_int_and_float_but_not_strings(self) -> None:
        assert validates({"type": "number"}, 5) is True
        assert validates({"type": "number"}, 5.5) is True
        assert validates({"type": "number"}, "5.5") is False

    def test_boolean_is_never_a_number(self) -> None:
        assert error_types({"type": "integer"}, True) == ["structbench_bool_as_integer"]
        assert error_types({"type": "number"}, True) == ["structbench_bool_as_number"]

    def test_boolean_rejects_truthy_lookalikes(self) -> None:
        assert validates({"type": "boolean"}, True) is True
        for value in ("true", 1, 0, "yes"):
            assert validates({"type": "boolean"}, value) is False

    def test_string_rejects_null_and_numbers(self) -> None:
        assert validates({"type": "string"}, "x") is True
        assert validates({"type": "string"}, None) is False
        assert validates({"type": "string"}, 1) is False

    def test_type_array_permits_only_the_declared_null(self) -> None:
        schema = {"type": ["string", "null"]}
        assert validates(schema, None) is True
        assert validates(schema, "x") is True
        assert validates(schema, 1) is False

    def test_optional_field_still_rejects_an_explicit_null(self) -> None:
        # Absence is allowed, but an explicit null is not "a string".
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": False,
        }
        assert validates(schema, {}) is True
        assert validates(schema, {"a": None}) is False

    def test_containers_must_be_containers(self) -> None:
        assert error_types({"type": "array", "items": {"type": "string"}}, "x") == ["list_type"]
        assert error_types({"type": "array", "items": {"type": "string"}}, [1]) == ["string_type"]
        assert error_types({"type": "object"}, []) == ["dict_type"]


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #


class TestStructure:
    def test_missing_and_extra_fields(self) -> None:
        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
            "additionalProperties": False,
        }
        assert error_types(schema, {}) == ["missing"]
        assert error_types(schema, {"a": 1, "b": 2}) == ["extra_forbidden"]

    def test_extra_keys_are_allowed_when_declared(self) -> None:
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
        assert validates(schema, {"a": 1, "b": "anything"}) is True

    def test_an_empty_object_that_forbids_extras_rejects_everything(self) -> None:
        schema = {"type": "object", "additionalProperties": False}
        assert validates(schema, {}) is True
        assert validates(schema, {"a": 1}) is False

    def test_enum_and_const_compile_to_literals(self) -> None:
        assert error_types({"enum": ["a", "b"]}, "c") == ["literal_error"]
        assert error_types({"const": 7}, 8) == ["literal_error"]

    def test_constraints(self) -> None:
        assert error_types({"type": "integer", "minimum": 1}, 0) == ["greater_than_equal"]
        assert error_types({"type": "integer", "exclusiveMinimum": 1}, 1) == ["greater_than"]
        assert error_types({"type": "integer", "multipleOf": 3}, 4) == ["multiple_of"]
        assert error_types({"type": "string", "maxLength": 2}, "abc") == ["string_too_long"]
        assert error_types({"type": "array", "items": {}, "minItems": 2}, [1]) == ["too_short"]
        assert error_types({"type": "string", "pattern": "^[A-Z]{2}$"}, "abc") == [
            "string_pattern_mismatch"
        ]

    def test_multiple_of_uses_exact_decimal_arithmetic(self) -> None:
        # 2.5 % 0.01 is not exactly 0 in binary floating point, but JSON Schema's
        # multipleOf is exact arithmetic -- a price of 2.50 must be accepted.
        schema = {"type": "number", "multipleOf": 0.01}
        assert validates(schema, 2.5) is True
        assert validates(schema, 249.99) is True
        assert validates(schema, 2.505) is False

    def test_unions_and_refs(self) -> None:
        assert validates({"anyOf": [{"type": "integer"}, {"type": "string"}]}, "x") is True
        assert validates({"anyOf": [{"type": "integer"}, {"type": "string"}]}, True) is False

        schema = {
            "$defs": {
                "P": {
                    "type": "object",
                    "properties": {"n": {"type": "string"}},
                    "required": ["n"],
                    "additionalProperties": False,
                }
            },
            "type": "object",
            "properties": {"p": {"$ref": "#/$defs/P"}},
            "required": ["p"],
            "additionalProperties": False,
        }
        assert validates(schema, {"p": {"n": "v"}}) is True
        assert error_types(schema, {"p": {}}) == ["missing"]

    def test_all_of_is_merged(self) -> None:
        schema = {
            "allOf": [
                {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}},
                    "required": ["a"],
                },
                {"type": "object", "properties": {"b": {"type": "string"}}, "required": ["b"]},
            ],
            "additionalProperties": False,
        }
        assert validates(schema, {"a": 1, "b": "x"}) is True
        assert error_types(schema, {"a": 1}) == ["missing"]

    def test_formats(self) -> None:
        assert validates({"type": "string", "format": "date-time"}, "2024-01-02T03:04:05Z") is True
        assert validates({"type": "string", "format": "date-time"}, "yesterday") is False
        assert validates({"type": "string", "format": "date"}, "2024-01-02") is True
        assert (
            validates({"type": "string", "format": "uuid"}, "3f2504e0-4f89-11d3-9a0c-0305e82c3301")
            is True
        )
        assert validates({"type": "string", "format": "email"}, "a@b.co") is True
        assert validates({"type": "string", "format": "email"}, "nope") is False
        assert validates({"type": "string", "format": "ipv4"}, "127.0.0.1") is True
        assert validates({"type": "string", "format": "ipv6"}, "::1") is True
        # The hostname pattern must stay lookahead-free: pydantic-core's Rust
        # regex engine rejects lookarounds outright.
        assert validates({"type": "string", "format": "hostname"}, "sub.example.com") is True

    def test_alias_handling_for_non_identifier_keys(self) -> None:
        schema = {
            "type": "object",
            "properties": {"user-id": {"type": "integer"}, "2fa": {"type": "boolean"}},
            "required": ["user-id"],
            "additionalProperties": False,
        }
        assert validates(schema, {"user-id": 3, "2fa": True}) is True
        assert error_types(schema, {"user-id": "3"}) == ["int_type"]

    def test_unsupported_keywords_produce_warnings_not_silence(self) -> None:
        compiled = compile_schema(
            {"type": "object", "properties": {"a": {"type": "string"}}, "if": {"type": "object"}},
            "warn",
        )
        assert any("ignored" in warning for warning in compiled.warnings)

    def test_recursive_refs_degrade_to_unconstrained_values(self) -> None:
        schema = {
            "$defs": {
                "N": {
                    "type": "object",
                    "properties": {"next": {"$ref": "#/$defs/N"}},
                    "additionalProperties": False,
                }
            },
            "$ref": "#/$defs/N",
        }
        compiled = compile_schema(schema, "recursive")
        assert any("recursive" in warning for warning in compiled.warnings)
        assert compiled.adapter.validate_python({"next": {"next": {}}}) is not None

    def test_unsupported_constructs_raise_with_guidance(self) -> None:
        with pytest.raises(SchemaCompileError, match="tuple-form|lookaround|unsupported"):
            compile_schema({"type": "array", "items": [{"type": "string"}]}, "tuple")
        with pytest.raises(SchemaCompileError, match="accepts nothing"):
            compile_schema(False, "never")
        with pytest.raises(SchemaCompileError, match="lookaround"):
            compile_schema({"type": "string", "pattern": "^(?=x)y$"}, "lookahead")

    def test_unknown_type_is_reported_not_ignored(self) -> None:
        compiled = compile_schema({"type": "quaternion"}, "weird")
        assert any("unknown JSON type" in warning for warning in compiled.warnings)
        assert compiled.adapter.validate_python("anything") == "anything"

    def test_build_annotation_exposes_the_translation(self) -> None:
        assert build_annotation({"type": "boolean"}, {"type": "boolean"}) is not None


# --------------------------------------------------------------------------- #
# JSONPath helpers
# --------------------------------------------------------------------------- #


class TestJsonPath:
    def test_parse_and_format_round_trip(self) -> None:
        assert jp.parse_path("$.a.b[0].c") == ["a", "b", 0, "c"]
        assert jp.format_path(["a", "b", 0, "c"]) == "$.a.b[0].c"
        assert jp.parse_path("$") == []
        assert jp.format_path([]) == "$"

    def test_quoted_keys(self) -> None:
        assert jp.parse_path("$['weird key'].v") == ["weird key", "v"]
        assert jp.format_path(["weird key"]) == "$['weird key']"

    def test_invalid_paths_raise(self) -> None:
        with pytest.raises(jp.PathError):
            jp.parse_path("$.a..b")
        with pytest.raises(jp.PathError):
            jp.parse_path("$.a[")

    def test_get_distinguishes_absent_from_null(self) -> None:
        value = {"a": None, "b": {"c": 1}, "arr": [10, 20]}
        assert jp.get(value, "$.a") is None
        assert jp.exists(value, "$.a") is True
        assert jp.is_null(value, "$.a") is True
        assert jp.get(value, "$.zzz") is jp.MISSING
        assert jp.exists(value, "$.zzz") is False
        assert jp.get(value, "$.b.c") == 1
        assert jp.get(value, "$.arr[1]") == 20
        assert jp.get(value, "$.arr[9]", default="fallback") == "fallback"

    def test_set_creates_missing_containers(self) -> None:
        # Skipped array slots are filled with the container the *next* token
        # needs, so the structure stays navigable rather than becoming a null run.
        value = jp.set_({}, "$.a.b[1].c", 5)
        assert value == {"a": {"b": [{}, {"c": 5}]}}

    def test_set_on_a_scalar_root_replaces_it(self) -> None:
        assert jp.set_("old", "$", "new") == "new"

    def test_delete_reports_whether_it_removed_anything(self) -> None:
        value = {"a": {"b": 1}, "c": [1, 2]}
        assert jp.delete(value, "$.a.b") is True
        assert jp.delete(value, "$.a.b") is False
        assert jp.delete(value, "$.c[0]") is True
        assert value == {"a": {}, "c": [2]}

    def test_iter_leaf_paths(self) -> None:
        paths = {
            jp.format_path(list(tokens)) for tokens, _ in jp.iter_leaf_paths({"a": {"b": [1]}})
        }
        assert paths == {"$.a.b[0]"}


# --------------------------------------------------------------------------- #
# Reporters
# --------------------------------------------------------------------------- #


def sample_result():
    def record(case_id: str, category: str, passed: bool, detail: str) -> CaseResult:
        attempt = Attempt(
            round=1,
            passed=passed,
            label=ErrorLabel.VALID_PASS if passed else ErrorLabel.SYNTAX_TRUNCATED,
            detail="exact_match" if passed else detail,
            raw_text='{"a": 1}',
            latency_ms=120.0,
            ttft_ms=40.0,
            prompt_tokens=10,
            completion_tokens=20,
        )
        return CaseResult(
            case_id=case_id,
            category=category,
            model="mock/good",
            title=f"case {case_id}",
            difficulty=3,
            attempts=[attempt],
            passed_at_1=passed,
            passed_after_heal=passed,
            healed=False,
            first_label=attempt.label,
            first_detail=attempt.detail,
            final_label=attempt.label,
            final_detail=attempt.detail,
            total_latency_ms=120.0,
            total_tokens=30,
        )

    return compute_run_result(
        version="0.0-test",
        records=[
            record("a", "deep_nesting", True, "exact_match"),
            record("b", "escape_hell", False, "unclosed_bracket"),
        ],
        models=[("mock/good", "mock", "mock/good")],
        datasets=["datasets/tiny.yaml"],
        case_count=2,
        concurrency=4,
        max_rounds=2,
        meta={"json_mode": False},
    )


class TestReporters:
    def test_markdown_leaderboard(self) -> None:
        body = render_markdown(sample_result())
        assert "| Model | Pass@1 |" in body
        assert "`mock/good`" in body
        assert "deep_nesting" in body
        assert "语法残缺" in body  # the Chinese taxonomy headers

    def test_console_text_report_renders_without_a_terminal(self) -> None:
        text = render_text_report(sample_result())
        assert "StructBench leaderboard" in text
        assert "mock/good" in text

    def test_payload_structure(self) -> None:
        payload = build_payload(sample_result())
        assert payload["case_count"] == 2
        assert payload["categories"] == ["deep_nesting", "escape_hell"]
        model = payload["models"][0]
        assert model["pass_at_1_rate"] == 50.0
        assert model["by_category"]["deep_nesting"] == 100.0
        assert model["error_distribution"]["Syntax_Truncated"] == 1
        assert payload["failure_count"] == 1
        assert payload["failures"][0]["case_id"] == "b"

    def test_html_report_is_self_contained(self) -> None:
        html = render_html_report(sample_result())
        assert html.startswith("<!DOCTYPE html>")
        assert "sb-data" in html
        assert "chart-radar" in html
        # The JSON payload must be embedded, and ``<`` escaped so that the script
        # block cannot be terminated early by model output.
        assert "\\u003c" in html or "<" not in html.split("sb-data")[1][:2000]
        assert "echarts" in html

    def test_html_report_escapes_script_terminators(self) -> None:
        result = sample_result()
        result.records[0].attempts[0].raw_text = "</script><script>alert(1)</script>"
        html = render_html_report(result)
        assert "</script><script>alert(1)" not in html

    def test_write_html_report_creates_parent_directories(self, workdir) -> None:
        target = workdir / "nested" / "deep" / "report.html"
        written = write_html_report(sample_result(), target)
        assert written.is_file()
        assert written.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

    def test_html_report_embeds_a_local_echarts_bundle(self, workdir) -> None:
        library = workdir / "echarts.min.js"
        library.write_text(
            "window.echarts = {init: function () { return {setOption: function () {}}; }};",
            encoding="utf-8",
        )
        html = render_html_report(sample_result(), inline_echarts=library)
        assert "window.echarts" in html
        assert "cdn.jsdelivr.net" not in html

    def test_missing_echarts_bundle_is_reported(self, workdir) -> None:
        with pytest.raises(FileNotFoundError):
            render_html_report(sample_result(), inline_echarts=workdir / "absent.js")

    def test_payload_embeds_dataset_references_when_given(self) -> None:
        from tests import _data

        payload = build_payload(sample_result(), cases=[_data.nested_case()])
        failure = payload["failures"][0]
        # case "b" is not in the supplied dataset, so the reference is minimal
        # rather than wrong.
        assert failure["reference"]["id"] == "b"
        assert json.dumps(payload)  # the payload must stay JSON-serialisable
