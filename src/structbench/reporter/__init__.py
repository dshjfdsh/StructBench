"""Reporting: terminal, Markdown, interactive HTML and static SVG."""

from __future__ import annotations

from structbench.reporter.charts import (
    CATEGORY_LABELS,
    build_charts,
    error_taxonomy_series,
    model_colours,
    ordered_models,
    pass_heal_series,
    radar_series,
    write_charts,
)
from structbench.reporter.console import (
    case_detail,
    category_table,
    detail_table,
    error_table,
    failure_table,
    leaderboard_table,
    render_markdown,
    render_report,
    render_text_report,
    score_bars,
    visual_summary,
)
from structbench.reporter.html import build_payload, render_html_report, write_html_report
from structbench.reporter.svg import (
    PALETTE,
    ChartError,
    Series,
    grouped_bar_chart,
    radar_chart,
    stacked_bar_chart,
)

__all__ = [
    "CATEGORY_LABELS",
    "PALETTE",
    "ChartError",
    "Series",
    "build_charts",
    "build_payload",
    "case_detail",
    "category_table",
    "detail_table",
    "error_table",
    "error_taxonomy_series",
    "failure_table",
    "grouped_bar_chart",
    "leaderboard_table",
    "model_colours",
    "ordered_models",
    "pass_heal_series",
    "radar_chart",
    "radar_series",
    "render_html_report",
    "render_markdown",
    "render_report",
    "render_text_report",
    "score_bars",
    "stacked_bar_chart",
    "visual_summary",
    "write_charts",
    "write_html_report",
]
