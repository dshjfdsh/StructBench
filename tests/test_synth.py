"""Tests for the regex sampler, the value synthesiser and the dataset self-check.

These cover the machinery that makes a dataset *trustworthy*: the ability to
construct a conforming response for any case (so the case is provably solvable)
and to prove that obvious mutations fail (so the case actually measures
something).
"""

from __future__ import annotations

import re

import pytest

from structbench.models import Case
from structbench.regexgen import sample_for_pattern
from structbench.selfcheck import check_case, ideal_completion
from structbench.synth import (
    SynthError,
    apply_ground_truth,
    is_nullable,
    schema_at_path,
    synth_value,
    synthesize_case_value,
)
from tests import _data

# --------------------------------------------------------------------------- #
# Regex sampling
# --------------------------------------------------------------------------- #

PATTERNS = [
    r"^CUST-[0-9]{4}$",
    r"^[A-Z]{2}-\d{4}$",
    r"^\d{4}-\d{2}-\d{2}$",
    r"^[0-9]+\.[0-9]{2}$",
    r"^BK-[0-9]{4}$",
    r"^(foo|bar)-\d{3}$",
    r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$",
    r"^#[0-9a-fA-F]{6}$",
    r"^v\d+\.\d+\.\d+$",
    r"^[a-z]+(?:_[a-z]+)*$",
    r"^(?P<code>[A-Z]{3})-(\d{2})$",
    r"^(?:ID|REF)-\d+$",
    r"^\w{8,22}$",
    r"^[^;]{3,10}$",
]


@pytest.mark.parametrize("pattern", PATTERNS)
def test_sampler_produces_a_full_match(pattern: str) -> None:
    sample = sample_for_pattern(pattern)
    assert sample is not None, f"could not sample {pattern!r}"
    assert re.fullmatch(pattern, sample), f"{sample!r} does not match {pattern!r}"


def test_sampler_respects_length_bounds() -> None:
    # An unbounded upper quantifier is the only kind that can be grown to length.
    sample = sample_for_pattern(r"^[0-9]{4,}$", min_length=12, max_length=12)
    assert sample is not None and len(sample) == 12


def test_partial_semantics_can_pad_to_min_length() -> None:
    # Unanchored, so trailing filler keeps the partial match intact.
    sample = sample_for_pattern(r"CUST-[0-9]{4}", min_length=20, full=False)
    assert sample is not None
    assert len(sample) >= 20
    assert re.search(r"CUST-[0-9]{4}", sample)


def test_anchored_pattern_cannot_be_padded_beyond_its_own_length() -> None:
    # Honest failure beats a wrong answer: `$` forbids the filler trick.
    assert sample_for_pattern(r"^CUST-[0-9]{4}$", min_length=20, full=False) is None


def test_sampler_returns_none_for_unsupported_constructs() -> None:
    # Lookarounds cannot be sampled, and pydantic-core's Rust engine rejects them
    # outright -- returning None (not a guess) is the honest answer.
    assert sample_for_pattern(r"^(?=x)y$") is None
    assert sample_for_pattern(r"a{") is None


def test_sampler_handles_negated_classes() -> None:
    sample = sample_for_pattern(r"^[^;]{3,10}$")
    assert sample is not None and ";" not in sample


# --------------------------------------------------------------------------- #
# Schema-driven synthesis
# --------------------------------------------------------------------------- #


def test_synth_value_covers_nested_shapes() -> None:
    schema = _data.NESTED_SCHEMA
    value = synth_value(schema, schema)
    assert isinstance(value, dict)
    assert isinstance(value["user"]["id"], int)
    assert isinstance(value["user"]["tags"], list) and value["user"]["tags"]
    assert value["status"] in ("active", "banned")
    # Nullable fields may be null but must exist.
    assert "deleted_at" in value["user"]


def test_synth_value_honours_enum_and_const() -> None:
    assert synth_value({"enum": ["b", "a"]}) == "b"
    assert synth_value({"const": 42}) == 42


def test_synth_value_reports_unsatisfiable_schemas() -> None:
    with pytest.raises(SynthError):
        synth_value({"type": "number", "minimum": 10, "maximum": 5})


def test_optional_fields_are_still_emitted_but_not_required() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    assert synth_value(schema, schema) == {"a": "sample", "b": 1}


def test_schema_at_path_navigates_refs_and_items() -> None:
    schema = {
        "$defs": {"P": {"type": "object", "properties": {"n": {"type": "string"}}}},
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"$ref": "#/$defs/P"}},
        },
    }
    found = schema_at_path(schema, ["items", 0, "n"], schema)
    assert found is not None and found["type"] == "string"
    assert schema_at_path(schema, ["items", 0, "missing"], schema) is None


def test_is_nullable_understands_type_arrays_and_unions() -> None:
    assert is_nullable({"type": ["string", "null"]})
    assert not is_nullable({"type": "string"})
    assert is_nullable({"anyOf": [{"type": "string"}, {"type": "null"}]})
    assert not is_nullable({"enum": ["a", "b"]})
    assert is_nullable({"const": None})


# --------------------------------------------------------------------------- #
# Ground-truth folding
# --------------------------------------------------------------------------- #


def test_apply_ground_truth_pins_values_and_nulls() -> None:
    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "note": {"type": ["string", "null"]},
        },
        "required": ["id", "note"],
        "additionalProperties": False,
    }
    truth = _data.nested_case().ground_truth.__class__(
        must_equal={"$.id": 99}, must_be_null=["$.note"]
    )
    value = apply_ground_truth({"id": 1, "note": "x"}, truth, schema)
    assert value == {"id": 99, "note": None}


def test_must_be_null_on_a_non_nullable_field_is_an_error() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    truth = _data.nested_case().ground_truth.__class__(must_be_null=["$.a"])
    with pytest.raises(SynthError, match="does not permit null"):
        apply_ground_truth({"a": "x"}, truth, schema)


def test_forbidden_path_that_is_also_required_is_an_error() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    truth = _data.nested_case().ground_truth.__class__(forbidden_paths=["$.a"])
    with pytest.raises(SynthError, match="forbidden_paths"):
        apply_ground_truth({"a": "x"}, truth, schema)


def test_must_contain_pads_to_min_length() -> None:
    schema = {
        "type": "object",
        "properties": {"q": {"type": "string", "minLength": 20}},
        "required": ["q"],
        "additionalProperties": False,
    }
    truth = _data.nested_case().ground_truth.__class__(must_contain={"$.q": "SELECT"})
    value = apply_ground_truth({"q": ""}, truth, schema)
    assert "SELECT" in value["q"] and len(value["q"]) >= 20


# --------------------------------------------------------------------------- #
# Case-level synthesis
# --------------------------------------------------------------------------- #


def test_synthesize_case_value_for_json_case() -> None:
    value = synthesize_case_value(_data.nested_case())
    assert value["user"]["id"] == 8675309  # pinned by must_equal
    assert value["note"] is None  # pinned by must_be_null


def test_synthesize_case_value_for_tool_case() -> None:
    value = synthesize_case_value(_data.tool_call_case())
    assert isinstance(value, dict)
    assert set(value) >= {"city", "unit"}


def test_synthesize_case_value_returns_none_for_clarify() -> None:
    assert synthesize_case_value(_data.clarify_case()) is None


def test_ideal_completion_shapes_match_the_mode() -> None:
    json_completion = ideal_completion(_data.nested_case())
    assert json_completion.tool_calls == [] and json_completion.text

    tool_completion = ideal_completion(_data.tool_call_case())
    assert [call.name for call in tool_completion.tool_calls] == ["get_weather"]

    clarify_completion = ideal_completion(_data.clarify_case())
    assert clarify_completion.tool_calls == []
    assert "?" in clarify_completion.text


# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #


def test_fixture_cases_pass_the_self_check() -> None:
    for factory in (_data.nested_case, _data.tool_call_case, _data.clarify_case):
        check = check_case(factory())
        assert check.ok, f"{check.case_id}: {check.errors}"


def test_self_check_catches_an_unsatisfiable_case() -> None:
    case = Case(
        id="unsat",
        category="x",
        prompt="p",
        target_schema={
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "additionalProperties": False,
        },
        ground_truth={"must_be_null": ["$.a"]},
    )
    check = check_case(case)
    assert not check.ok
    assert "does not permit null" in " ".join(check.errors)


def test_self_check_catches_a_vacuous_case() -> None:
    # No required fields, no enums, extra keys allowed -> nothing can fail.
    case = Case(
        id="vacuous",
        category="x",
        prompt="p",
        target_schema={"type": "object", "properties": {"a": {"type": "string"}}},
    )
    check = check_case(case)
    assert any("additionalProperties" in warning for warning in check.warnings)
    assert any(
        "measures nothing" in warning or "not measure" in warning for warning in check.warnings
    )


def test_self_check_verifies_that_inventing_a_field_fails() -> None:
    case = Case(
        id="extra_ok",
        category="x",
        prompt="p",
        target_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
            "additionalProperties": True,
        },
    )
    check = check_case(case)
    # The schema permits extra keys, so the report must say so rather than claim
    # the case catches invented fields.
    assert any("additionalProperties" in warning for warning in check.warnings)
