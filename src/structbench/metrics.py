"""Metric aggregation: Pass@1, Heal@1, repair rate, latency and cost.

Definitions used consistently across the CLI, the HTML report and the README:

``Pass@1``
    Fraction of cases whose **first** attempt was fully compliant. This is raw,
    unaided structured-output reliability.

``Heal@1``
    Fraction of cases that passed within the healing budget (attempt 1 plus the
    replayed-error retries). Note the spec's wording -- "首次失败后…二次修正后的
    **总通过率**" -- so Heal@1 is an overall pass rate, not a conditioned one. It
    is always ``>= Pass@1``.

``repair_rate``
    ``healed / failed_at_1``: of the cases the model broke on the first try, how
    many did it repair once told precisely what was wrong? This is the diagnostic
    metric behind Heal@1, and it is ``None`` when the model broke nothing (there
    is then nothing to repair and the ratio is undefined -- reporting 0% or 100%
    would both be misleading).

Latency and throughput are aggregated over *all* attempts, because that is the
real cost of running the suite; Pass@1/Heal@1 are aggregated over cases.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence

from structbench.models import (
    ALL_LABELS,
    Attempt,
    AttemptStatus,
    CaseResult,
    CategoryScore,
    ErrorLabel,
    LatencyStats,
    ModelScore,
    RunResult,
)

__all__ = [
    "compute_category_scores",
    "compute_model_score",
    "compute_run_result",
    "latency_stats",
    "percentile",
    "safe_ratio",
]


def percentile(values: Sequence[float], quantile: float) -> float | None:
    """Linearly interpolated percentile (the same convention as numpy)."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def latency_stats(values: Iterable[float | None]) -> LatencyStats:
    """Summarise a series of millisecond measurements, ignoring unknowns."""
    present = [float(value) for value in values if value is not None]
    if not present:
        return LatencyStats(samples=0)
    return LatencyStats(
        mean=sum(present) / len(present),
        p50=percentile(present, 0.50),
        p95=percentile(present, 0.95),
        min=min(present),
        max=max(present),
        samples=len(present),
    )


def safe_ratio(numerator: int, denominator: int) -> float:
    """A fraction that degrades to 0.0 instead of raising on an empty set."""
    return float(numerator) / float(denominator) if denominator else 0.0


def _score_block(records: Sequence[CaseResult]) -> dict[str, float | int]:
    """The pass/fail core shared by per-category and whole-run scorecards."""
    total = len(records)
    pass_at_1 = sum(1 for record in records if record.passed_at_1)
    heal_at_1 = sum(1 for record in records if record.passed_after_heal)
    healed = sum(1 for record in records if record.healed)
    failed_at_1 = total - pass_at_1
    return {
        "total": total,
        "pass_at_1": pass_at_1,
        "heal_at_1": heal_at_1,
        "failed_at_1": failed_at_1,
        "healed": healed,
        "pass_at_1_rate": safe_ratio(pass_at_1, total),
        "heal_at_1_rate": safe_ratio(heal_at_1, total),
        "repair_rate": safe_ratio(healed, failed_at_1) if failed_at_1 else None,
    }


def compute_category_scores(records: Sequence[CaseResult]) -> list[CategoryScore]:
    """Per-dimension scorecards, ordered by descending case count then name."""
    grouped: dict[str, list[CaseResult]] = defaultdict(list)
    for record in records:
        grouped[record.category].append(record)

    scores = [
        CategoryScore(category=category, **_score_block(group))  # type: ignore[arg-type]
        for category, group in grouped.items()
    ]
    scores.sort(key=lambda score: (-score.total, score.category))
    return scores


def compute_difficulty_scores(records: Sequence[CaseResult]) -> dict[str, CategoryScore]:
    """Per-difficulty scorecards, keyed by the difficulty level as a string."""
    grouped: dict[int, list[CaseResult]] = defaultdict(list)
    for record in records:
        grouped[record.difficulty].append(record)
    return {
        str(level): CategoryScore(category=f"difficulty_{level}", **_score_block(group))  # type: ignore[arg-type]
        for level, group in sorted(grouped.items())
    }


def _all_attempts(records: Sequence[CaseResult]) -> list[Attempt]:
    return [attempt for record in records for attempt in record.attempts]


def compute_model_score(
    model: str,
    records: Sequence[CaseResult],
    *,
    provider: str = "",
    error_distribution: dict[str, int] | None = None,
    detail_distribution: dict[str, int] | None = None,
) -> ModelScore:
    """Build the full scorecard for one model from its case records."""
    attempts = _all_attempts(records)
    real_attempts = [attempt for attempt in attempts if attempt.status is AttemptStatus.OK]

    completion_tokens = sum(attempt.completion_tokens or 0 for attempt in attempts)
    prompt_tokens = sum(attempt.prompt_tokens or 0 for attempt in attempts)

    # Throughput is measured over generation time only (total latency minus the
    # wait for the first token), which is what "tokens per second" means to a user.
    generation_ms = 0.0
    generation_tokens = 0
    for attempt in real_attempts:
        if attempt.completion_tokens and attempt.latency_ms:
            window = attempt.latency_ms - (attempt.ttft_ms or 0.0)
            if window > 0:
                generation_ms += window
                generation_tokens += attempt.completion_tokens
    tps = (generation_tokens / (generation_ms / 1000.0)) if generation_ms > 0 else None

    provider_failures = sum(
        1 for attempt in attempts if attempt.status is AttemptStatus.PROVIDER_ERROR
    )

    distribution = (
        error_distribution
        if error_distribution is not None
        else _first_attempt_label_distribution(records)
    )

    return ModelScore(
        model=model,
        provider=provider,
        provider_failures=provider_failures,
        unusable=bool(records) and provider_failures >= len(records),
        latency=latency_stats(attempt.latency_ms for attempt in attempts if attempt.latency_ms),
        ttft=latency_stats(attempt.ttft_ms for attempt in attempts),
        tps=tps,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        error_distribution=distribution,
        detail_distribution=detail_distribution or _first_attempt_detail_distribution(records),
        by_category=compute_category_scores(records),
        by_difficulty=compute_difficulty_scores(records),
        **_score_block(records),  # type: ignore[arg-type]
    )


def _first_attempt_label_distribution(records: Sequence[CaseResult]) -> dict[str, int]:
    """Label counts over first attempts -- the model's native, unaided behaviour."""
    counts = {label.value: 0 for label in ALL_LABELS}
    for record in records:
        if record.attempts and record.attempts[0].status is AttemptStatus.PROVIDER_ERROR:
            continue
        label = record.first_label
        if label is None:
            continue
        counts[label.value] = counts.get(label.value, 0) + 1
    return counts


def _first_attempt_detail_distribution(records: Sequence[CaseResult]) -> dict[str, int]:
    """Fine-grained cause counts over *failed* first attempts."""
    counts: dict[str, int] = {}
    for record in records:
        if record.passed_at_1:
            continue
        detail = record.first_detail or "unknown"
        counts[detail] = counts.get(detail, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def compute_run_result(
    *,
    version: str,
    records: Sequence[CaseResult],
    models: Sequence[tuple[str, str, str]],
    datasets: Sequence[str],
    case_count: int,
    concurrency: int,
    max_rounds: int,
    meta: dict | None = None,
) -> RunResult:
    """Assemble a serialisable run archive.

    ``models`` is a sequence of ``(display_name, provider, key)`` so that records
    (keyed by the internal model key) can be attributed to the right scorecard.
    """
    scorecards: list[ModelScore] = []
    for display, provider, key in models:
        subset = [record for record in records if record.model == key]
        scorecards.append(compute_model_score(display, subset, provider=provider))

    return RunResult(
        structbench_version=version,
        datasets=list(datasets),
        case_count=case_count,
        concurrency=concurrency,
        max_rounds=max_rounds,
        models=scorecards,
        records=list(records),
        meta=dict(meta or {}),
    )


def label_counts_from_attempts(attempts: Sequence[Attempt]) -> dict[str, int]:
    """Label histogram for an arbitrary set of attempts (used by tests/reports)."""
    counts: dict[str, int] = {}
    for attempt in attempts:
        if attempt.label is None:
            continue
        counts[attempt.label.value] = counts.get(attempt.label.value, 0) + 1
    return counts


def top_error_labels(score: ModelScore, limit: int = 3) -> list[tuple[str, int]]:
    """The most frequent *failure* labels for a model, largest first."""
    failures = [
        (label.value, score.error_distribution.get(label.value, 0))
        for label in ALL_LABELS
        if label is not ErrorLabel.VALID_PASS
    ]
    return sorted(failures, key=lambda item: -item[1])[:limit]
