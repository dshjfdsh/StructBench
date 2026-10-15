"""Reporting: the dual output (Rich terminal leaderboard + HTML radar report)."""

from __future__ import annotations

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
)
from structbench.reporter.html import build_payload, render_html_report, write_html_report

__all__ = [
    "build_payload",
    "case_detail",
    "category_table",
    "detail_table",
    "error_table",
    "failure_table",
    "leaderboard_table",
    "render_html_report",
    "render_markdown",
    "render_report",
    "render_text_report",
    "write_html_report",
]
