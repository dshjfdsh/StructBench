"""Integrity gate for the bundled benchmark suite.

This is the test that keeps the project's headline number trustworthy: every
shipped case must be *satisfiable* (some response passes) and *discriminating*
(an obvious mutation fails). A case that fails either check would silently
distort every leaderboard the README publishes.
"""

from __future__ import annotations

from collections import Counter

import pytest

from structbench.loader import default_dataset_dir, load_cases
from structbench.selfcheck import check_cases

EXPECTED_CATEGORIES = {"deep_nesting", "escape_hell", "tool_hallucination", "edge_types"}


@pytest.fixture(scope="module")
def suite():
    return load_cases()


def test_suite_loads_from_four_files(suite) -> None:
    cases, files = suite
    assert len(files) == 4
    assert len(cases) == 86


def test_every_category_is_well_populated(suite) -> None:
    cases, _ = suite
    counts = Counter(case.category for case in cases)
    assert set(counts) == EXPECTED_CATEGORIES
    # Each dimension must be big enough for its per-dimension rate to mean
    # something: a 3-case dimension has a 33% granularity.
    for category, count in counts.items():
        assert count >= 20, f"{category} only has {count} cases"


def test_case_ids_are_unique_and_well_formed(suite) -> None:
    cases, _ = suite
    ids = [case.id for case in cases]
    assert len(ids) == len(set(ids))
    prefixes = {
        "deep_nesting": "dn_",
        "escape_hell": "esc_",
        "tool_hallucination": "hal_",
        "edge_types": "typ_",
    }
    for case in cases:
        assert case.id.startswith(prefixes[case.category]), case.id


def test_difficulty_and_rounds_are_sane(suite) -> None:
    cases, _ = suite
    assert {case.difficulty for case in cases} >= {2, 3, 4, 5}
    assert sum(1 for case in cases if case.difficulty == 5) >= 8
    assert all(1 <= case.max_rounds <= 5 for case in cases)
    # Healing is the headline feature, so most cases must allow a second round.
    assert sum(1 for case in cases if case.max_rounds >= 2) >= len(cases) * 0.9


def test_all_four_behaviours_are_covered(suite) -> None:
    cases, _ = suite
    behaviours = Counter(case.expect.behaviour for case in cases)
    assert behaviours["json"] >= 50
    assert behaviours["call"] >= 10
    assert behaviours["clarify"] >= 8
    assert behaviours["null_value"] >= 2


def test_tool_cases_declare_decoys_and_expectations(suite) -> None:
    cases, _ = suite
    tool_cases = [case for case in cases if case.is_tool]
    assert len(tool_cases) >= 20
    for case in tool_cases:
        assert case.tools, case.id
        if case.expect.behaviour == "call":
            assert case.tool_by_name(case.expect.expected_tool) is not None, case.id
        # Decoys are the point of the anti-hallucination dimension: a single tool
        # cannot test tool *selection*.
        if case.category == "tool_hallucination":
            assert len(case.tools) >= 2, case.id


def test_every_case_is_documented(suite) -> None:
    cases, _ = suite
    for case in cases:
        assert case.prompt.strip(), case.id
        assert case.title.strip(), case.id
        assert case.notes.strip(), f"{case.id} has no authoring notes"
        # Titles show up in CLI tables and report rows; keep them readable.
        assert len(case.title) <= 80, f"{case.id} title is {len(case.title)} characters"


def test_suite_passes_the_self_check(suite) -> None:
    cases, _ = suite
    report = check_cases(cases)

    failures = {check.case_id: check.errors for check in report.checks if not check.ok}
    assert not failures, f"unsatisfiable or non-discriminating cases: {failures}"

    # Warnings indicate lost fidelity (an unsupported keyword being ignored, or an
    # object that will not catch an invented field). The shipped suite is clean.
    assert report.warnings == [], f"self-check warnings: {report.warnings}"


def test_bundled_directory_is_the_one_that_was_loaded(suite) -> None:
    _, files = suite
    bundled = default_dataset_dir()
    assert all(path.parent == bundled for path in files)
