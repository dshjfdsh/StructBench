"""Turn a :class:`~structbench.models.RunResult` into SVG charts.

This module is the bridge between the benchmark's numbers and the pictures people
actually look at. It exists separately from :mod:`structbench.reporter.svg` so that
the drawing code stays a pure function of numbers, while every decision about
*what* to plot -- which models, which dimensions, what counts as a failure -- lives
in one place.

The charts produced here are the ones committed to the repository and embedded in
both READMEs, because GitHub cannot render the interactive HTML report.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from structbench.models import ALL_LABELS, LABEL_ZH, ErrorLabel, RunResult
from structbench.reporter.svg import (
    PALETTE,
    Series,
    grouped_bar_chart,
    radar_chart,
    stacked_bar_chart,
)

__all__ = [
    "CATEGORY_LABELS",
    "CHART_BUILDERS",
    "build_charts",
    "error_taxonomy_series",
    "model_colours",
    "ordered_models",
    "pass_heal_series",
    "radar_series",
    "write_charts",
]

#: Human-facing names for the four dimensions. The dataset keys stay in English
#: (they are API), but a chart axis is read by a person.
CATEGORY_LABELS: dict[str, str] = {
    "deep_nesting": "Deep nesting",
    "escape_hell": "Escape hell",
    "tool_hallucination": "Tool hallucination",
    "edge_types": "Edge types",
}

#: Distinct colours for the five verdicts. ``Valid_Pass`` is green because it is
#: the good outcome; the four failures get warm/cool contrast so a stacked bar is
#: readable without consulting the legend.
LABEL_COLOURS: dict[ErrorLabel, str] = {
    ErrorLabel.MARKDOWN_WRAPPER_LEAK: "#f59e0b",
    ErrorLabel.SYNTAX_TRUNCATED: "#dc2626",
    ErrorLabel.SCHEMA_TYPE_MISMATCH: "#9333ea",
    ErrorLabel.ENUM_OR_VALUE_HALLUCINATION: "#be123c",
    ErrorLabel.VALID_PASS: "#16a34a",
}


def ordered_models(result: RunResult) -> list:
    """Scorecards, best Pass@1 first, so every chart shares one model order.

    A consistent order matters more than it looks: if the radar, the bars and the
    taxonomy each sorted differently, comparing them across charts would be
    needlessly hard.
    """
    return sorted(result.models, key=lambda score: (-score.pass_at_1_rate, score.model))


def model_colours(result: RunResult) -> dict[str, str]:
    """Stable colour per model, assigned by rank."""
    return {
        score.model: PALETTE[index % len(PALETTE)]
        for index, score in enumerate(ordered_models(result))
    }


def _categories(result: RunResult) -> list[str]:
    """Dimension keys in a stable, meaningful order (dataset order, not alphabetical)."""
    preferred = list(CATEGORY_LABELS)
    present = {block.category for score in result.models for block in score.by_category}
    ordered = [name for name in preferred if name in present]
    ordered += sorted(present - set(preferred))
    return ordered


def radar_series(result: RunResult, *, include_unusable: bool = False) -> list[Series]:
    """Per-dimension Pass@1 as a 0..100 series per model."""
    colours = model_colours(result)
    categories = _categories(result)
    series: list[Series] = []

    for score in ordered_models(result):
        if score.unusable and not include_unusable:
            continue
        by_category = {block.category: block for block in score.by_category}
        series.append(
            Series(
                name=score.model,
                colour=colours[score.model],
                values=[
                    None
                    if category not in by_category
                    else round(by_category[category].pass_at_1_rate * 100.0, 1)
                    for category in categories
                ],
            )
        )
    return series


def pass_heal_series(result: RunResult) -> list[Series]:
    """Pass@1 and Heal@1 side by side -- the headline gap.

    The gap is the project's main claim, so it is deliberately the first chart a
    visitor sees after the leaderboard.
    """
    models = ordered_models(result)
    return [
        Series(
            name="Pass@1",
            colour="#2f6fed",
            values=[round(s.pass_at_1_rate * 100, 1) for s in models],
        ),
        Series(
            name="Heal@1",
            colour="#16a34a",
            values=[round(s.heal_at_1_rate * 100, 1) for s in models],
        ),
    ]


def error_taxonomy_series(result: RunResult, *, include_pass: bool = True) -> list[Series]:
    """The five verdict classes, stacked, over *first* attempts."""
    models = ordered_models(result)
    labels = (
        list(ALL_LABELS)
        if include_pass
        else [x for x in ALL_LABELS if x is not ErrorLabel.VALID_PASS]
    )
    return [
        Series(
            name=LABEL_ZH[label],
            colour=LABEL_COLOURS[label],
            values=[float(score.error_distribution.get(label.value, 0)) for score in models],
        )
        for label in labels
    ]


#: ``name -> builder`` for the committed chart set. The names double as filenames
#: (``<name>.svg``), so they are part of the public surface.
CHART_BUILDERS: dict[str, str] = {
    "radar": "Dimension profile (Pass@1 per dimension)",
    "pass-heal": "Pass@1 versus Heal@1",
    "taxonomy": "Error taxonomy (first attempts)",
}


def build_charts(
    result: RunResult,
    *,
    width: int = 880,
    height: int = 560,
    subtitle: str = "",
) -> dict[str, str]:
    """Render every committed chart, keyed by name.

    ``subtitle`` is appended to each chart title so a committed image records which
    run produced it -- otherwise a screenshot in a README is unverifiable.
    """
    models = ordered_models(result)
    categories = _categories(result)
    suffix = f" — {subtitle}" if subtitle else ""

    charts: dict[str, str] = {}

    axis_labels = [CATEGORY_LABELS.get(name, name) for name in categories]
    charts["radar"] = radar_chart(
        axis_labels,
        radar_series(result),
        title=f"Dimension profile{suffix}",
        width=width,
        height=height,
    )

    charts["pass-heal"] = grouped_bar_chart(
        [score.model for score in models],
        pass_heal_series(result),
        title=f"Pass@1 vs Heal@1{suffix}",
        y_label="pass rate (%)",
        width=width,
        height=int(height * 0.82),
        max_value=100.0,
    )

    charts["taxonomy"] = stacked_bar_chart(
        [score.model for score in models],
        error_taxonomy_series(result),
        title=f"Error taxonomy, first attempts{suffix}",
        y_label="cases",
        width=width,
        height=int(height * 0.82),
    )

    return charts


def write_charts(
    result: RunResult,
    directory: str | Path,
    *,
    names: Sequence[str] | None = None,
    **kwargs,
) -> dict[str, Path]:
    """Render and write the charts, returning ``name -> path``.

    Parent directories are created, so ``--charts docs/charts`` works on a fresh
    clone.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    charts = build_charts(result, **kwargs)
    written: dict[str, Path] = {}
    for name, svg in charts.items():
        if names is not None and name not in names:
            continue
        path = target / f"{name}.svg"
        path.write_text(svg, encoding="utf-8")
        written[name] = path
    return written
