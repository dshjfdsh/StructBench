"""Unit tests for the deterministic validation sandbox.

The point of this file is that the validator itself must never be the bug: every
pathology class the benchmark claims to detect is pinned to an exact
``(ErrorLabel, detail)`` pair here.
"""

from __future__ import annotations

import copy
import json

import pytest

from structbench.evaluator import (
    check_ground_truth,
    compile_case,
    evaluate,
    extract_payload,
    parse_payload,
    validate_payload,
)
from structbench.models import (
    AttemptStatus,
    Case,
    Completion,
    ErrorLabel,
    ToolCall,
)
from tests._data import GOOD_PAYLOAD, NESTED_SCHEMA

MISMATCH = ErrorLabel.SCHEMA_TYPE_MISMATCH
HALLUCINATION = ErrorLabel.ENUM_OR_VALUE_HALLUCINATION
WRAPPER = ErrorLabel.MARKDOWN_WRAPPER_LEAK
SYNTAX = ErrorLabel.SYNTAX_TRUNCATED
PASS = ErrorLabel.VALID_PASS


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def mutate(**updates: object) -> str:
    """Deep-copy the canonical payload, apply updates, and serialise it."""
    data = copy.deepcopy(GOOD_PAYLOAD)
    for key, value in updates.items():
        if key == "user" and isinstance(value, dict):
            data["user"].update(value)
        else:
            data[key] = value
    return json.dumps(data)


def drop(*path: str) -> str:
    """Serialise the canonical payload with a nested key removed."""
    data = copy.deepcopy(GOOD_PAYLOAD)
    target = data
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]
    return json.dumps(data)


def verdict(case: Case, payload: str) -> tuple[bool, ErrorLabel | None, str, list[str]]:
    attempt = evaluate(compile_case(case), Completion(text=payload, latency_ms=1.0))
    return attempt.passed, attempt.label, attempt.detail, [i.path for i in attempt.issues]


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #


class TestExtractPayload:
    def test_clean_json_has_no_wrapper(self) -> None:
        text = json.dumps(GOOD_PAYLOAD)
        result = extract_payload(text)
        assert result.wrapper is None
        assert result.found is True
        assert result.complete is True
        assert json.loads(result.payload_text) == GOOD_PAYLOAD

    def test_fenced_json(self) -> None:
        result = extract_payload(f"```json\n{json.dumps(GOOD_PAYLOAD)}\n```")
        assert result.wrapper == "fence"
        assert result.found is True

    def test_language_tagless_fence(self) -> None:
        result = extract_payload(f"```\n{json.dumps(GOOD_PAYLOAD)}\n```")
        assert result.wrapper == "fence"

    def test_prose_padding(self) -> None:
        result = extract_payload(
            f"Sure! Here you go:\n{json.dumps(GOOD_PAYLOAD)}\nHope that helps."
        )
        assert result.wrapper == "prose"
        assert json.loads(result.payload_text) == GOOD_PAYLOAD

    def test_fence_and_prose(self) -> None:
        result = extract_payload(f"Here you go:\n```json\n{json.dumps(GOOD_PAYLOAD)}\n```\nDone.")
        assert result.wrapper == "fence+prose"

    def test_braces_inside_strings_do_not_confuse_the_scanner(self) -> None:
        text = '{"code": "if (x) { return \\"}\\" }", "n": 1}'
        result = extract_payload(text)
        assert result.found is True
        assert result.complete is True
        assert json.loads(result.payload_text)["n"] == 1

    def test_trailing_prose_after_json_is_detected(self) -> None:
        result = extract_payload('{"a": 1} and that is the answer.')
        assert result.wrapper == "prose"
        assert result.payload_text == '{"a": 1}'

    def test_no_json_at_all(self) -> None:
        result = extract_payload("I am sorry, I cannot help with that.")
        assert result.found is False
        assert result.wrapper is None

    def test_empty_input(self) -> None:
        result = extract_payload("")
        assert result.found is False
        assert result.payload_text == ""

    def test_truncated_payload_is_incomplete(self) -> None:
        result = extract_payload('{"user": {"id": 1}')
        assert result.found is True
        assert result.complete is False


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


class TestParsePayload:
    @pytest.mark.parametrize(
        ("text", "detail"),
        [
            ("", "empty_output"),
            ("   \n  ", "empty_output"),
            ('{"a": 1', "unclosed_bracket"),
            ('{"a": [1, 2', "unclosed_bracket"),
            ('{"a": "unterminated', "unterminated_string"),
            ('{"a": "\\xda"}', "bad_escape"),
            ('{"a": 1,}', "trailing_comma"),
            ("[1, 2,]", "trailing_comma"),
            ('{"a" 1}', "missing_colon"),
            ('{"a": 1, b: 2}', "invalid_key"),
            ('{"a": 1} {"b": 2}', "trailing_data"),
            ('{"a": True}', "python_literal"),
            ('{"a": None}', "python_literal"),
            ("{'a': 1}", "single_quotes"),
            ("this is not json at all", "invalid_token"),
        ],
    )
    def test_error_details(self, text: str, detail: str) -> None:
        outcome = parse_payload(text)
        assert outcome.error == detail

    def test_valid_json_has_no_error(self) -> None:
        outcome = parse_payload('{"a": 1}')
        assert outcome.error is None
        assert outcome.value == {"a": 1}

    def test_bare_scalar_is_valid_json(self) -> None:
        assert parse_payload("42").error is None

    def test_python_literal_is_recovered_for_diagnostics_only(self) -> None:
        outcome = parse_payload("{'a': [1, 2], 'b': None}")
        assert outcome.recovered_via_ast is True
        assert outcome.value == {"a": [1, 2], "b": None}
        # Recovered or not, it is still a failure.
        assert outcome.error is not None

    def test_trailing_comma_detail_survives_ast_recovery(self) -> None:
        # ``ast.literal_eval`` accepts Python's trailing commas, so the specific
        # JSON diagnosis must not be masked by the recovery pass.
        outcome = parse_payload('{"a": 1,}')
        assert outcome.error == "trailing_comma"
        assert outcome.recovered_via_ast is True


# --------------------------------------------------------------------------- #
# JSON-mode evaluation
# --------------------------------------------------------------------------- #


class TestJsonEvaluation:
    def test_canonical_payload_passes(self, nested_case: Case) -> None:
        passed, label, detail, _ = verdict(nested_case, json.dumps(GOOD_PAYLOAD))
        assert passed is True
        assert label is PASS
        assert detail == "exact_match"

    def test_fence_leak(self, nested_case: Case) -> None:
        payload = f"```json\n{json.dumps(GOOD_PAYLOAD)}\n```"
        passed, label, detail, _ = verdict(nested_case, payload)
        assert (passed, label, detail) == (False, WRAPPER, "fence_leak")

    def test_prose_padding(self, nested_case: Case) -> None:
        payload = f"Certainly! Here is the JSON:\n{json.dumps(GOOD_PAYLOAD)}"
        passed, label, detail, _ = verdict(nested_case, payload)
        assert (passed, label, detail) == (False, WRAPPER, "prose_padding")

    def test_fence_and_prose(self, nested_case: Case) -> None:
        payload = f"Here:\n```json\n{json.dumps(GOOD_PAYLOAD)}\n```\nDone."
        passed, label, detail, _ = verdict(nested_case, payload)
        assert (passed, label, detail) == (False, WRAPPER, "fence_and_prose")

    def test_allow_wrapper_opt_out(self) -> None:
        case = Case(
            id="wrapped_ok",
            category="x",
            prompt="reply with a fenced json block",
            target_schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}},
                "required": ["a"],
                "additionalProperties": False,
            },
            allow_wrapper=True,
        )
        passed, label, _, _ = verdict(case, '```json\n{"a": 1}\n```')
        assert passed is True
        assert label is PASS

    def test_syntax_truncated_outranks_wrapper_leak(self, nested_case: Case) -> None:
        # Root-cause attribution: a fence around broken JSON is not the headline.
        passed, label, detail, _ = verdict(nested_case, '```json\n{"user": {"id": 1')
        assert (passed, label, detail) == (False, SYNTAX, "unclosed_bracket")

    def test_int_emitted_as_string(self, nested_case: Case) -> None:
        passed, label, detail, paths = verdict(nested_case, mutate(user={"id": "8675309"}))
        assert (passed, label, detail) == (False, MISMATCH, "type_mismatch")
        assert "$.user.id" in paths

    def test_missing_required_field(self, nested_case: Case) -> None:
        passed, label, detail, paths = verdict(nested_case, drop("user", "name"))
        assert (passed, label, detail) == (False, MISMATCH, "missing_field")
        assert "$.user.name" in paths

    def test_enum_hallucination(self, nested_case: Case) -> None:
        passed, label, detail, paths = verdict(nested_case, mutate(status="activating"))
        assert (passed, label, detail) == (False, HALLUCINATION, "enum_or_const_violation")
        assert "$.status" in paths

    def test_extra_top_level_field(self, nested_case: Case) -> None:
        passed, label, detail, paths = verdict(nested_case, mutate(api_key="sk-secret"))
        assert (passed, label, detail) == (False, HALLUCINATION, "extra_field")
        assert "$.api_key" in paths

    def test_extra_nested_field(self, nested_case: Case) -> None:
        passed, label, detail, paths = verdict(nested_case, mutate(user={"password": "hunter2"}))
        assert (passed, label, detail) == (False, HALLUCINATION, "extra_field")
        assert "$.user.password" in paths

    def test_fabricated_value_for_unknown_field(self, nested_case: Case) -> None:
        # The schema permits a string here, but the ground truth demands null:
        # this is exactly the "invent rather than admit ignorance" failure.
        passed, label, detail, paths = verdict(nested_case, mutate(note="nothing to report"))
        assert (passed, label, detail) == (False, HALLUCINATION, "fabricated_missing_value")
        assert "$.note" in paths

    def test_ground_truth_value_mismatch(self, nested_case: Case) -> None:
        passed, label, detail, paths = verdict(nested_case, mutate(user={"id": 1234}))
        assert (passed, label, detail) == (False, HALLUCINATION, "ground_truth_mismatch")
        assert "$.user.id" in paths

    def test_string_too_short(self, nested_case: Case) -> None:
        passed, label, detail, _ = verdict(nested_case, mutate(user={"name": "A"}))
        assert (passed, label, detail) == (False, MISMATCH, "string_too_short")

    def test_array_too_short(self, nested_case: Case) -> None:
        passed, label, detail, _ = verdict(nested_case, mutate(user={"tags": []}))
        assert (passed, label, detail) == (False, MISMATCH, "too_few_items")

    def test_bad_datetime_format(self, nested_case: Case) -> None:
        passed, label, _, paths = verdict(nested_case, mutate(user={"deleted_at": "yesterday"}))
        assert passed is False
        assert label in (MISMATCH, HALLUCINATION)
        assert "$.user.deleted_at" in paths

    def test_empty_response(self, nested_case: Case) -> None:
        passed, label, detail, _ = verdict(nested_case, "")
        assert (passed, label, detail) == (False, SYNTAX, "empty_output")

    def test_prose_only_response(self, nested_case: Case) -> None:
        passed, label, detail, _ = verdict(nested_case, "I cannot help with that request.")
        assert (passed, label, detail) == (False, SYNTAX, "no_json_found")

    def test_absent_optional_field_satisfies_must_be_null(self, nested_case: Case) -> None:
        # Presence is governed by the schema's `required` list, so an omitted
        # optional field must not be double-penalised.
        payload = drop("note")
        passed, _, _, _ = verdict(nested_case, payload)
        assert passed is True


# --------------------------------------------------------------------------- #
# tool-call evaluation
# --------------------------------------------------------------------------- #


class TestToolCallEvaluation:
    def test_correct_call_passes(self, tool_call_case: Case) -> None:
        completion = Completion(
            tool_calls=[
                ToolCall(name="get_weather", arguments_raw='{"city": "Beijing", "unit": "celsius"}')
            ]
        )
        attempt = evaluate(compile_case(tool_call_case), completion)
        assert attempt.passed is True
        assert attempt.detail == "exact_match"

    def test_invented_tool_argument(self, tool_call_case: Case) -> None:
        completion = Completion(
            tool_calls=[
                ToolCall(
                    name="get_weather",
                    arguments_raw='{"city": "Beijing", "unit": "celsius", "api_key": "sk-x"}',
                )
            ]
        )
        attempt = evaluate(compile_case(tool_call_case), completion)
        assert (attempt.passed, attempt.label, attempt.detail) == (
            False,
            HALLUCINATION,
            "extra_field",
        )

    def test_enum_violation_in_arguments(self, tool_call_case: Case) -> None:
        completion = Completion(
            tool_calls=[
                ToolCall(name="get_weather", arguments_raw='{"city": "Beijing", "unit": "kelvin"}')
            ]
        )
        attempt = evaluate(compile_case(tool_call_case), completion)
        assert (attempt.passed, attempt.label, attempt.detail) == (
            False,
            HALLUCINATION,
            "enum_or_const_violation",
        )

    def test_declared_but_wrong_tool(self, tool_call_case: Case) -> None:
        completion = Completion(
            tool_calls=[ToolCall(name="get_weather_forecast", arguments_raw='{"city": "Beijing"}')]
        )
        attempt = evaluate(compile_case(tool_call_case), completion)
        assert (attempt.passed, attempt.label, attempt.detail) == (
            False,
            HALLUCINATION,
            "wrong_tool",
        )

    def test_undeclared_tool(self, tool_call_case: Case) -> None:
        completion = Completion(
            tool_calls=[ToolCall(name="delete_all_bookings", arguments_raw="{}")]
        )
        attempt = evaluate(compile_case(tool_call_case), completion)
        assert (attempt.passed, attempt.label, attempt.detail) == (
            False,
            HALLUCINATION,
            "undeclared_tool",
        )

    def test_no_call_at_all(self, tool_call_case: Case) -> None:
        attempt = evaluate(compile_case(tool_call_case), Completion(text="Beijing is sunny."))
        assert (attempt.passed, attempt.label, attempt.detail) == (
            False,
            SYNTAX,
            "no_tool_call",
        )

    def test_arguments_emitted_as_text(self, tool_call_case: Case) -> None:
        completion = Completion(text='{"city": "Beijing", "unit": "celsius"}')
        attempt = evaluate(compile_case(tool_call_case), completion)
        assert (attempt.passed, attempt.label, attempt.detail) == (
            False,
            SYNTAX,
            "tool_call_as_text",
        )

    def test_truncated_arguments(self, tool_call_case: Case) -> None:
        completion = Completion(
            tool_calls=[ToolCall(name="get_weather", arguments_raw='{"city": "Beij')]
        )
        attempt = evaluate(compile_case(tool_call_case), completion)
        assert (attempt.passed, attempt.label, attempt.detail) == (
            False,
            SYNTAX,
            "tool_args_unterminated_string",
        )

    def test_fenced_arguments(self, tool_call_case: Case) -> None:
        completion = Completion(
            tool_calls=[
                ToolCall(
                    name="get_weather",
                    arguments_raw='```json\n{"city": "Beijing", "unit": "celsius"}\n```',
                )
            ]
        )
        attempt = evaluate(compile_case(tool_call_case), completion)
        assert (attempt.passed, attempt.label, attempt.detail) == (
            False,
            WRAPPER,
            "fence_leak",
        )


# --------------------------------------------------------------------------- #
# anti-hallucination (clarify) evaluation
# --------------------------------------------------------------------------- #


class TestClarifyEvaluation:
    def test_asking_a_question_passes(self, clarify_case: Case) -> None:
        completion = Completion(text="Which date would you like to move it to?")
        attempt = evaluate(compile_case(clarify_case), completion)
        assert attempt.passed is True
        assert attempt.detail == "clarification_requested"

    def test_chinese_clarification_passes(self, clarify_case: Case) -> None:
        completion = Completion(text="请问您想改签到哪一天？")
        attempt = evaluate(compile_case(clarify_case), completion)
        assert attempt.passed is True

    def test_calling_anyway_is_a_hallucination(self, clarify_case: Case) -> None:
        completion = Completion(
            tool_calls=[
                ToolCall(
                    name="rebook_flight",
                    arguments_raw='{"booking_id": "BK-4471", "new_date": "2025-06-01"}',
                )
            ]
        )
        attempt = evaluate(compile_case(clarify_case), completion)
        assert (attempt.passed, attempt.label, attempt.detail) == (
            False,
            HALLUCINATION,
            "unrequested_tool_call",
        )

    def test_silently_proceeding_is_a_hallucination(self, clarify_case: Case) -> None:
        completion = Completion(text='{"booking_id": "BK-4471", "new_date": "2025-06-01"}')
        attempt = evaluate(compile_case(clarify_case), completion)
        assert (attempt.passed, attempt.label, attempt.detail) == (
            False,
            HALLUCINATION,
            "silent_proceed",
        )

    def test_rambling_without_a_question(self, clarify_case: Case) -> None:
        completion = Completion(text="Sure, let me take a look at that booking for you.")
        attempt = evaluate(compile_case(clarify_case), completion)
        assert (attempt.passed, attempt.label, attempt.detail) == (
            False,
            SYNTAX,
            "no_clarification",
        )


# --------------------------------------------------------------------------- #
# infrastructure, feedback and compiler guards
# --------------------------------------------------------------------------- #


class TestProviderFailure:
    def test_transport_error_is_not_a_model_pathology(self, nested_case: Case) -> None:
        completion = Completion(provider_error="HTTP 503: upstream unavailable")
        attempt = evaluate(compile_case(nested_case), completion)
        assert attempt.status is AttemptStatus.PROVIDER_ERROR
        assert attempt.label is None
        assert attempt.detail == "provider_error"
        assert attempt.feedback == ""


class TestFeedback:
    def test_feedback_names_the_problem_and_forbids_a_fence(self, nested_case: Case) -> None:
        completion = Completion(text=f"```json\n{json.dumps(GOOD_PAYLOAD)}\n```")
        attempt = evaluate(compile_case(nested_case), completion)
        assert "StructBench validation feedback" in attempt.feedback
        assert "Markdown_Wrapper_Leak" in attempt.feedback
        assert "no Markdown code fence" in attempt.feedback

    def test_tool_feedback_restates_the_expected_tool(self, tool_call_case: Case) -> None:
        attempt = evaluate(
            compile_case(tool_call_case),
            Completion(
                tool_calls=[ToolCall(name="get_weather", arguments_raw='{"city": "Beijing"}')]
            ),
        )
        assert "get_weather" in attempt.feedback
        assert "missing" in attempt.feedback.lower()

    def test_feedback_is_bounded(self, nested_case: Case) -> None:
        payload = mutate(
            api_key="x",
            secret="y",
            token="z",
            user={"id": "nope", "name": 5, "tags": "notalist", "extra1": 1, "extra2": 2},
        )
        attempt = evaluate(compile_case(nested_case), Completion(text=payload))
        assert attempt.feedback.count("\n- at ") <= 12


class TestCompilerGuards:
    def test_invalid_rule_path_fails_at_compile_time(self) -> None:
        case = Case(
            id="bad_path",
            category="x",
            prompt="p",
            target_schema={"type": "object"},
            ground_truth={"must_equal": {"$.a..b": 1}},
        )
        with pytest.raises(ValueError, match="invalid JSONPath"):
            compile_case(case)

    def test_case_requires_a_schema_for_json_mode(self) -> None:
        with pytest.raises(ValueError, match="requires target_schema"):
            Case(id="x", category="c", prompt="p", mode="json")

    def test_call_behaviour_requires_a_known_tool(self) -> None:
        with pytest.raises(ValueError, match="expected_tool"):
            Case(
                id="x",
                category="c",
                prompt="p",
                tools=[{"name": "a", "parameters": {"type": "object"}}],
                expect={"behaviour": "call"},
            )
        with pytest.raises(ValueError, match="not among the declared tools"):
            Case(
                id="x",
                category="c",
                prompt="p",
                tools=[{"name": "a", "parameters": {"type": "object"}}],
                expect={"behaviour": "call", "expected_tool": "b"},
            )

    def test_mode_is_inferred_from_tools(self) -> None:
        assert (
            Case(id="x", category="c", prompt="p", tools=[{"name": "a", "parameters": {}}]).mode
            == "tool"
        )
        assert (
            Case(id="y", category="c", prompt="p", target_schema={"type": "object"}).mode == "json"
        )


class TestGroundTruthRules:
    def test_must_contain_and_forbidden(self) -> None:
        case = Case(
            id="gt",
            category="x",
            prompt="p",
            target_schema=NESTED_SCHEMA,
            ground_truth={
                "must_contain": {"$.note": "SELECT"},
                "forbidden_paths": ["$.api_key"],
                "must_match": {"$.status": "act.*"},
            },
        )

        issues = check_ground_truth(case, {"note": "no sql here", "status": "nope", "api_key": 1})
        details = {issue.detail for issue in issues}
        assert details == {
            "missing_required_content",
            "format_mismatch",
            "forbidden_field",
        }

        assert check_ground_truth(case, {"note": "SELECT 1", "status": "active"}) == []

    def test_duplicate_case_ids_are_rejected(self) -> None:
        from pydantic import ValidationError

        from structbench.models import Dataset

        with pytest.raises(ValidationError, match="duplicate case ids"):
            Dataset(
                cases=[
                    {
                        "id": "a",
                        "category": "c",
                        "prompt": "p",
                        "target_schema": {"type": "object"},
                    },
                    {
                        "id": "a",
                        "category": "c",
                        "prompt": "p",
                        "target_schema": {"type": "object"},
                    },
                ]
            )

    def test_validate_payload_reports_aliased_paths(self) -> None:
        from structbench.schema_builder import compile_schema

        schema = {
            "type": "object",
            "properties": {"user-id": {"type": "integer"}},
            "required": ["user-id"],
            "additionalProperties": False,
        }
        issues = validate_payload(compile_schema(schema, "alias"), {"user-id": "1"})
        assert [issue.path for issue in issues] == ["$.user-id"]
