"""Tests for metric aggregation.

The definitions under test are the ones the README publishes, so they are pinned
here: Pass@1 is over first attempts, Heal@1 is an overall pass rate that includes
healing, and ``repair_rate`` is ``healed / failed_at_1`` (undefined when nothing
failed).
"""

from __future__ import annotations

import pytest

from structbench.metrics import (
    compute_category_scores,
    compute_difficulty_scores,
    compute_model_score,
    compute_run_result,
    label_counts_from_attempts,
    latency_stats,
    percentile,
    safe_ratio,
    top_error_labels,
)
from structbench.models import (
    Attempt,
    AttemptStatus,
    CaseResult,
    ErrorLabel,
)


def make_attempt(
    round_index: int = 1,
    *,
    passed: bool = True,
    label: ErrorLabel | None = ErrorLabel.VALID_PASS,
    detail: str = "exact_match",
    latency_ms: float | None = 100.0,
    ttft_ms: float | None = 20.0,
    prompt_tokens: int | None = 10,
    completion_tokens: int | None = 40,
    status: AttemptStatus = AttemptStatus.OK,
) -> Attempt:
    return Attempt(
        round=round_index,
        passed=passed,
        label=label,
        detail=detail,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        status=status,
    )


def make_record(
    case_id: str,
    *,
    model: str = "mock/x",
    category: str = "deep_nesting",
    difficulty: int = 3,
    passed_at_1: bool = True,
    passed_after_heal: bool = True,
    rounds: int = 1,
    first_label: ErrorLabel | None = ErrorLabel.VALID_PASS,
    first_detail: str = "exact_match",
    attempts: list[Attempt] | None = None,
) -> CaseResult:
    if attempts is None:
        attempts = []
        for index in range(1, rounds + 1):
            passed = passed_at_1 if index == 1 else passed_after_heal
            attempts.append(
                make_attempt(
                    index,
                    passed=passed,
                    label=ErrorLabel.VALID_PASS if passed else ErrorLabel.SCHEMA_TYPE_MISMATCH,
                    detail="exact_match" if passed else "missing_field",
                )
            )
    return CaseResult(
        case_id=case_id,
        category=category,
        model=model,
        title=case_id,
        difficulty=difficulty,
        attempts=attempts,
        passed_at_1=passed_at_1,
        passed_after_heal=passed_after_heal,
        healed=passed_after_heal and not passed_at_1,
        first_label=first_label,
        first_detail=first_detail,
        final_label=attempts[-1].label,
        final_detail=attempts[-1].detail,
    )


# --------------------------------------------------------------------------- #
# Primitive statistics
# --------------------------------------------------------------------------- #


class TestPercentile:
    def test_median_of_even_sample_interpolates(self) -> None:
        assert percentile([1, 2, 3, 4], 0.5) == 2.5

    def test_single_sample(self) -> None:
        assert percentile([7.0], 0.95) == 7.0

    def test_empty_is_none(self) -> None:
        assert percentile([], 0.5) is None

    def test_extremes_are_min_and_max(self) -> None:
        values = [5.0, 1.0, 9.0, 3.0]
        assert percentile(values, 0.0) == 1.0
        assert percentile(values, 1.0) == 9.0

    def test_p95_of_a_hundred_values(self) -> None:
        assert percentile([float(i) for i in range(1, 101)], 0.95) == pytest.approx(95.05)


class TestLatencyStats:
    def test_ignores_unknowns_and_counts_samples(self) -> None:
        stats = latency_stats([100.0, None, 300.0, None])
        assert stats.samples == 2
        assert stats.mean == 200.0
        assert stats.min == 100.0
        assert stats.max == 300.0

    def test_empty_series(self) -> None:
        assert latency_stats([]).samples == 0
        assert latency_stats([None]).mean is None


def test_safe_ratio_never_divides_by_zero() -> None:
    assert safe_ratio(3, 4) == 0.75
    assert safe_ratio(0, 0) == 0.0


# --------------------------------------------------------------------------- #
# Model scorecards
# --------------------------------------------------------------------------- #


class TestModelScore:
    def test_counts_and_rates(self) -> None:
        records = [
            make_record("a", passed_at_1=True),
            make_record("b", passed_at_1=True),
            make_record("c", passed_at_1=False, passed_after_heal=True, rounds=2),
            make_record("d", passed_at_1=False, passed_after_heal=False, rounds=2),
        ]
        score = compute_model_score("mock/x", records)
        assert score.total == 4
        assert score.pass_at_1 == 2
        assert score.pass_at_1_rate == 0.5
        assert score.failed_at_1 == 2
        assert score.healed == 1
        assert score.heal_at_1 == 3
        assert score.heal_at_1_rate == 0.75
        assert score.repair_rate == 0.5

    def test_repair_rate_is_none_without_failures(self) -> None:
        score = compute_model_score("mock/x", [make_record("a")])
        assert score.repair_rate is None

    def test_error_distribution_covers_first_attempts_only(self) -> None:
        records = [
            make_record("ok", passed_at_1=True),
            make_record(
                "bad",
                passed_at_1=False,
                passed_after_heal=False,
                rounds=2,
                first_label=ErrorLabel.ENUM_OR_VALUE_HALLUCINATION,
                first_detail="extra_field",
            ),
        ]
        score = compute_model_score("mock/x", records)
        assert score.error_distribution[ErrorLabel.VALID_PASS.value] == 1
        assert score.error_distribution[ErrorLabel.ENUM_OR_VALUE_HALLUCINATION.value] == 1
        assert score.error_distribution[ErrorLabel.MARKDOWN_WRAPPER_LEAK.value] == 0
        assert score.detail_distribution == {"extra_field": 1}

    def test_provider_failures_are_counted_but_excluded_from_the_taxonomy(self) -> None:
        broken = make_record(
            "broken",
            passed_at_1=False,
            passed_after_heal=False,
            first_label=None,
            first_detail="provider_error",
            attempts=[
                make_attempt(
                    1,
                    passed=False,
                    label=None,
                    detail="provider_error",
                    status=AttemptStatus.PROVIDER_ERROR,
                )
            ],
        )
        score = compute_model_score("mock/x", [broken, make_record("ok")])
        assert score.provider_failures == 1
        assert score.unusable is False
        assert score.error_distribution[ErrorLabel.VALID_PASS.value] == 1

    def test_all_requests_failing_marks_the_model_unusable(self) -> None:
        broken = make_record(
            "broken",
            passed_at_1=False,
            passed_after_heal=False,
            first_label=None,
            first_detail="provider_error",
            attempts=[
                make_attempt(
                    1,
                    passed=False,
                    label=None,
                    detail="provider_error",
                    status=AttemptStatus.PROVIDER_ERROR,
                )
            ],
        )
        assert compute_model_score("mock/x", [broken]).unusable is True

    def test_throughput_excludes_the_time_to_first_token(self) -> None:
        # Two attempts: 100ms total each minus 20ms TTFT = 80ms generating 40
        # tokens each -> 80 tokens over 0.16s = 500 tok/s.
        records = [
            make_record(
                "a",
                attempts=[make_attempt(1, latency_ms=100.0, ttft_ms=20.0, completion_tokens=40)]
                * 1,
            )
        ]
        records.append(
            make_record(
                "b",
                attempts=[make_attempt(1, latency_ms=100.0, ttft_ms=20.0, completion_tokens=40)],
            )
        )
        score = compute_model_score("mock/x", records)
        assert score.tps == pytest.approx(500.0)

    def test_token_totals_include_every_round(self) -> None:
        record = make_record(
            "a",
            passed_at_1=False,
            passed_after_heal=True,
            rounds=2,
        )
        score = compute_model_score("mock/x", [record])
        assert score.prompt_tokens == 20
        assert score.completion_tokens == 80
        assert score.total_tokens == 100

    def test_latency_is_aggregated_over_all_rounds(self) -> None:
        record = make_record("a", passed_at_1=False, passed_after_heal=True, rounds=2)
        score = compute_model_score("mock/x", [record])
        assert score.latency.samples == 2
        assert score.ttft.samples == 2


# --------------------------------------------------------------------------- #
# Grouped scores
# --------------------------------------------------------------------------- #


class TestGrouping:
    def test_category_scores_are_ordered_by_size(self) -> None:
        records = [
            make_record("a", category="big"),
            make_record("b", category="big"),
            make_record("c", category="small"),
        ]
        scores = compute_category_scores(records)
        assert [score.category for score in scores] == ["big", "small"]
        assert scores[0].total == 2

    def test_difficulty_scores_are_keyed_by_level(self) -> None:
        records = [
            make_record("a", difficulty=2),
            make_record("b", difficulty=5, passed_at_1=False, passed_after_heal=False),
        ]
        scores = compute_difficulty_scores(records)
        assert set(scores) == {"2", "5"}
        assert scores["2"].pass_at_1_rate == 1.0
        assert scores["5"].pass_at_1_rate == 0.0

    def test_per_category_blocks_are_attached_to_the_model_score(self) -> None:
        records = [
            make_record("a", category="deep_nesting"),
            make_record("b", category="escape_hell", passed_at_1=False, passed_after_heal=False),
        ]
        score = compute_model_score("mock/x", records)
        by_category = {block.category: block for block in score.by_category}
        assert by_category["deep_nesting"].pass_at_1_rate == 1.0
        assert by_category["escape_hell"].pass_at_1_rate == 0.0


# --------------------------------------------------------------------------- #
# Run assembly
# --------------------------------------------------------------------------- #


def test_compute_run_result_attributes_records_per_model() -> None:
    records = [
        make_record("a", model="mock/one"),
        make_record("a", model="mock/two", passed_at_1=False, passed_after_heal=False),
    ]
    result = compute_run_result(
        version="0.0-test",
        records=records,
        models=[("mock/one", "mock", "mock/one"), ("mock/two", "mock", "mock/two")],
        datasets=["datasets/x.yaml"],
        case_count=1,
        concurrency=4,
        max_rounds=2,
    )
    assert result.structbench_version == "0.0-test"
    assert result.score_for("mock/one").pass_at_1_rate == 1.0
    assert result.score_for("mock/two").pass_at_1_rate == 0.0
    assert len(result.records_for("mock/one")) == 1
    assert result.score_for("nope") is None


def test_label_counts_helper() -> None:
    attempts = [
        make_attempt(1, passed=True),
        make_attempt(2, passed=False, label=ErrorLabel.SYNTAX_TRUNCATED, detail="unclosed_bracket"),
        make_attempt(3, passed=False, label=None, status=AttemptStatus.PROVIDER_ERROR),
    ]
    counts = label_counts_from_attempts(attempts)
    assert counts == {ErrorLabel.VALID_PASS.value: 1, ErrorLabel.SYNTAX_TRUNCATED.value: 1}


def test_top_error_labels_ignores_the_success_label() -> None:
    score = compute_model_score(
        "mock/x",
        [
            make_record("a"),
            make_record(
                "b",
                passed_at_1=False,
                passed_after_heal=False,
                first_label=ErrorLabel.SYNTAX_TRUNCATED,
                first_detail="unclosed_bracket",
            ),
        ],
    )
    top = top_error_labels(score)
    assert top[0][0] == ErrorLabel.SYNTAX_TRUNCATED.value
    assert top[0][1] == 1
    assert all(label != ErrorLabel.VALID_PASS.value for label, _ in top)
