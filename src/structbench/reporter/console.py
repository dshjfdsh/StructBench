"""Rich terminal reports: leaderboard, per-dimension matrix, error taxonomy.

Everything here is derived from a :class:`~structbench.models.RunResult`, so the
same numbers appear in the console and in the HTML report -- there is exactly one
place where a rate is computed (:mod:`structbench.metrics`).
"""

from __future__ import annotations

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from structbench.evaluator import DETAIL_GLOSS
from structbench.models import (
    ALL_LABELS,
    LABEL_ZH,
    AttemptStatus,
    ErrorLabel,
    ModelScore,
    RunResult,
)

__all__ = [
    "case_detail",
    "category_table",
    "detail_table",
    "error_table",
    "failure_table",
    "leaderboard_table",
    "render_markdown",
    "render_report",
    "render_text_report",
]

_LABEL_STYLE = {
    ErrorLabel.MARKDOWN_WRAPPER_LEAK: "yellow",
    ErrorLabel.SYNTAX_TRUNCATED: "red",
    ErrorLabel.SCHEMA_TYPE_MISMATCH: "magenta",
    ErrorLabel.ENUM_OR_VALUE_HALLUCINATION: "bright_red",
    ErrorLabel.VALID_PASS: "green",
}


def rate_text(rate: float | None, *, width: int = 6) -> Text:
    """Colour a 0..1 rate by how alarming it is."""
    if rate is None:
        return Text("—", style="dim", justify="right")
    percent = rate * 100.0
    if percent >= 90:
        style = "bold green"
    elif percent >= 75:
        style = "green"
    elif percent >= 60:
        style = "yellow"
    elif percent >= 40:
        style = "dark_orange"
    else:
        style = "bold red"
    return Text(f"{percent:.1f}%", style=style, justify="right")


def _ms(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value / 1000:.2f}s"
    return f"{value:.0f}ms"


def _count(value: int | None) -> str:
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    return str(value)


def _model_label(score: ModelScore) -> Text:
    text = Text(score.model, style="bold")
    if score.unusable:
        text.append("  ⚠ all requests failed", style="bold red")
    elif score.provider_failures:
        text.append(f"  ⚠ {score.provider_failures} failed", style="yellow")
    return text


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def leaderboard_table(result: RunResult) -> Table:
    """The headline ranking.

    The column count is deliberately small and every cell is ``no_wrap``: an
    80-column terminal must show real numbers, never ``9…``.
    """
    table = Table(
        title="StructBench leaderboard",
        title_style="bold cyan",
        header_style="bold",
        pad_edge=False,
    )
    table.add_column("Model", no_wrap=True, min_width=16)
    table.add_column("Pass@1", justify="right", no_wrap=True)
    table.add_column("Heal@1", justify="right", no_wrap=True)
    table.add_column("Repair", justify="right", no_wrap=True)
    table.add_column("P50", justify="right", no_wrap=True)
    table.add_column("P95", justify="right", no_wrap=True)
    table.add_column("TTFT", justify="right", no_wrap=True)
    table.add_column("TPS", justify="right", no_wrap=True)
    table.add_column("Tokens", justify="right", no_wrap=True)

    for score in sorted(result.models, key=lambda item: -item.pass_at_1_rate):
        table.add_row(
            _model_label(score),
            rate_text(score.pass_at_1_rate),
            rate_text(score.heal_at_1_rate),
            rate_text(score.repair_rate),
            _ms(score.latency.p50),
            _ms(score.latency.p95),
            _ms(score.ttft.p50),
            f"{score.tps:.0f}" if score.tps else "—",
            _count(score.total_tokens),
        )
    return table


def category_table(result: RunResult) -> Table:
    """Pass@1 per dimension -- where each model is actually weak."""
    categories: list[str] = []
    for score in result.models:
        for block in score.by_category:
            if block.category not in categories:
                categories.append(block.category)

    table = Table(
        title="Pass@1 by dimension",
        title_style="bold cyan",
        header_style="bold",
        pad_edge=False,
    )
    table.add_column("Model", no_wrap=True, min_width=16)
    for category in categories:
        table.add_column(category, justify="right", no_wrap=True)
    table.add_column("Overall", justify="right", no_wrap=True)
    table.add_column("n", justify="right", no_wrap=True, style="dim")

    for score in sorted(result.models, key=lambda item: -item.pass_at_1_rate):
        cells = []
        by_category = {block.category: block for block in score.by_category}
        for category in categories:
            block = by_category.get(category)
            cells.append(rate_text(block.pass_at_1_rate if block else None, width=5))
        table.add_row(
            _model_label(score),
            *cells,
            rate_text(score.pass_at_1_rate, width=5),
            str(score.total),
        )
    return table


def error_table(result: RunResult) -> Table:
    """The five-way taxonomy, per model, over first attempts."""
    table = Table(
        title="Error taxonomy — first attempts",
        title_style="bold cyan",
        header_style="bold",
        pad_edge=False,
    )
    table.add_column("Model", no_wrap=True, min_width=16)
    for label in ALL_LABELS:
        table.add_column(LABEL_ZH[label], justify="right", style=_LABEL_STYLE[label], no_wrap=True)

    for score in sorted(result.models, key=lambda item: -item.pass_at_1_rate):
        table.add_row(
            _model_label(score),
            *[
                Text(str(score.error_distribution.get(label.value, 0)), justify="right")
                for label in ALL_LABELS
            ],
        )
    return table


def detail_table(result: RunResult, *, limit: int = 4) -> Table:
    """The most frequent fine-grained causes per model -- the actionable view."""
    table = Table(
        title="Top failure causes (first attempts)",
        title_style="bold cyan",
        header_style="bold",
        pad_edge=False,
    )
    table.add_column("Model", no_wrap=True, min_width=16)
    table.add_column("Cause", overflow="fold")
    table.add_column("n", justify="right", no_wrap=True)
    table.add_column("What it means", style="dim", overflow="fold")

    for score in sorted(result.models, key=lambda item: -item.pass_at_1_rate):
        top = list(score.detail_distribution.items())[:limit]
        if not top:
            table.add_row(_model_label(score), Text("none", style="green"), "0", "no failures")
            continue
        for index, (detail, count) in enumerate(top):
            table.add_row(
                _model_label(score) if index == 0 else "",
                Text(detail, style="bold"),
                str(count),
                DETAIL_GLOSS.get(detail, ""),
            )
    return table


def failure_table(result: RunResult, model: str, *, limit: int = 12) -> Table:
    """The cases a model could not fix even after being told what was wrong."""
    table = Table(
        title=f"Hardest failures — {model}",
        title_style="bold cyan",
        header_style="bold",
        pad_edge=False,
    )
    table.add_column("Case", overflow="fold")
    table.add_column("Dimension", overflow="fold")
    table.add_column("Round 1 verdict", overflow="fold")
    table.add_column("First failure", overflow="fold")

    records = [record for record in result.records_for(model) if not record.passed_after_heal]
    records.sort(key=lambda record: (-record.difficulty, record.case_id))

    for record in records[:limit]:
        label = record.first_label.value if record.first_label else "(request failed)"
        table.add_row(
            record.case_id,
            record.category,
            Text(
                label,
                style=_LABEL_STYLE.get(record.first_label, "dim") if record.first_label else "dim",
            ),
            f"{record.first_detail} — {DETAIL_GLOSS.get(record.first_detail, '')}",
        )
    if not records:
        table.add_row("—", "—", Text("none", style="green"), "every case passed")
    return table


def case_detail(result: RunResult, model: str, case_id: str, *, max_chars: int = 1200) -> Panel:
    """A single case's story: verdict per round, the errors, and the raw output."""
    record = next(
        (item for item in result.records_for(model) if item.case_id == case_id),
        None,
    )
    if record is None:
        return Panel(Text(f"no record for {case_id} under {model}", style="red"))

    lines: list[Text] = []
    verdict = (
        Text("PASS at round 1", style="bold green")
        if record.passed_at_1
        else Text("healed in round 2", style="bold yellow")
        if record.healed
        else Text("FAILED", style="bold red")
    )
    lines.append(verdict)
    lines.append(Text(f"dimension: {record.category}   difficulty: {'★' * record.difficulty}"))

    for attempt in record.attempts:
        header = Text(f"\n── round {attempt.round} ", style="bold")
        if attempt.status is AttemptStatus.PROVIDER_ERROR:
            header.append(f"provider error: {attempt.provider_error}", style="yellow")
            lines.append(header)
            continue
        label = attempt.label.value if attempt.label else "?"
        header.append(
            label, style=_LABEL_STYLE.get(attempt.label, "dim") if attempt.label else "dim"
        )
        header.append(f" / {attempt.detail}", style="dim")
        lines.append(header)
        for issue in attempt.issues[:6]:
            lines.append(Text(f"   • {issue.path}: {issue.message}", style="dim"))

        raw = attempt.raw_text or "(no text content)"
        if attempt.tool_calls:
            calls = "; ".join(f"{call.name}({call.arguments_raw})" for call in attempt.tool_calls)
            raw = f"{raw}\n[tool calls] {calls}".strip()
        if len(raw) > max_chars:
            raw = raw[:max_chars] + f"\n… (+{len(raw) - max_chars} more characters)"
        lines.append(Text(raw, style="grey62"))

    return Panel(
        Group(*lines),
        title=f"{model} · {case_id}",
        border_style="cyan",
        expand=False,
    )


# --------------------------------------------------------------------------- #
# Composite rendering
# --------------------------------------------------------------------------- #


def render_report(
    result: RunResult,
    console: Console,
    *,
    show_failures: int = 8,
) -> None:
    """Print the whole console report."""
    header = Text()
    header.append("StructBench ", style="bold white")
    header.append(f"v{result.structbench_version}", style="dim")
    header.append(
        f"  ·  {len(result.models)} model(s) × {result.case_count} case(s)"
        f"  ·  concurrency {result.concurrency}"
        f"  ·  up to {result.max_rounds} round(s)/case\n",
        style="dim",
    )
    header.append(f"datasets: {', '.join(result.datasets) or '—'}", style="dim")
    console.print(Panel(header, border_style="cyan", expand=False))

    console.print(leaderboard_table(result))
    console.print()
    console.print(category_table(result))
    console.print()
    console.print(error_table(result))
    console.print()
    console.print(detail_table(result))

    for score in sorted(result.models, key=lambda item: -item.pass_at_1_rate):
        console.print()
        console.print(failure_table(result, score.model, limit=show_failures))


def render_text_report(result: RunResult) -> str:
    """Plain-text rendering, for piping into a file or a CI log."""
    console = Console(
        record=True,
        width=160,
        force_terminal=False,
        no_color=True,
        legacy_windows=False,
    )
    render_report(result, console)
    return console.export_text()


def _markdown_rate(rate: float | None) -> str:
    if rate is None:
        return "n/a"
    return f"{rate * 100:.1f}%"


def render_markdown(result: RunResult, *, title: str = "StructBench leaderboard") -> str:
    """Render the headline tables as Markdown.

    This is the artifact you paste into a README: the whole point of the project
    is a leaderboard people can trust, and a Markdown copy keeps it reviewable in
    a pull request instead of buried in an HTML blob.
    """
    lines: list[str] = [f"### {title}", ""]

    lines.append("| Model | Pass@1 | Heal@1 | Repair | P50 | P95 | TTFT p50 | TPS | Tokens |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for score in sorted(result.models, key=lambda item: -item.pass_at_1_rate):
        lines.append(
            f"| `{score.model}` | **{_markdown_rate(score.pass_at_1_rate)}** "
            f"| {_markdown_rate(score.heal_at_1_rate)} "
            f"| {_markdown_rate(score.repair_rate)} "
            f"| {_ms(score.latency.p50)} | {_ms(score.latency.p95)} | {_ms(score.ttft.p50)} "
            f"| {score.tps:.1f} | {_count(score.total_tokens)} |"
        )

    categories: list[str] = []
    for score in result.models:
        for block in score.by_category:
            if block.category not in categories:
                categories.append(block.category)

    if categories:
        lines += ["", "#### Pass@1 by dimension", ""]
        lines.append("| Model | " + " | ".join(categories) + " |")
        lines.append("| :--- | " + " | ".join("---:" for _ in categories) + " |")
        for score in sorted(result.models, key=lambda item: -item.pass_at_1_rate):
            by_category = {block.category: block for block in score.by_category}
            cells = [
                _markdown_rate(by_category[category].pass_at_1_rate)
                if category in by_category
                else "—"
                for category in categories
            ]
            lines.append(f"| `{score.model}` | " + " | ".join(cells) + " |")

    lines += ["", "#### Error taxonomy (first attempts)", ""]
    lines.append("| Model | " + " | ".join(LABEL_ZH[label] for label in ALL_LABELS) + " |")
    lines.append("| :--- | " + " | ".join("---:" for _ in ALL_LABELS) + " |")
    for score in sorted(result.models, key=lambda item: -item.pass_at_1_rate):
        cells = [str(score.error_distribution.get(label.value, 0)) for label in ALL_LABELS]
        lines.append(f"| `{score.model}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        f"<sub>{len(result.models)} model(s) × {result.case_count} case(s) · "
        f"up to {result.max_rounds} round(s)/case · generated by StructBench "
        f"v{result.structbench_version} on {result.created_at}</sub>",
        "",
    ]
    return "\n".join(lines)
