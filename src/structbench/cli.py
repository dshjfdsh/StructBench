"""The ``structbench`` command-line interface.

Commands
--------
``run``       execute the benchmark against one or more models
``report``    re-render the console/HTML/Markdown/SVG output from a saved archive
``chart``     write static SVG charts from a saved archive
``validate``  self-check the datasets (every case satisfiable and discriminating)
``list``      enumerate the bundled cases
``personas``  describe the offline simulator personas
``providers`` list the known provider presets and their environment variables
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from structbench import __version__
from structbench.loader import DatasetError, default_dataset_dir, load_cases
from structbench.models import (
    PROVIDER_PRESETS,
    ModelSpec,
    RunResult,
    parse_model_spec,
)
from structbench.providers import PERSONA_NAMES, PERSONAS, ProviderConfigError, ProviderOptions
from structbench.reporter import (
    ChartError,
    render_markdown,
    render_report,
    write_charts,
    write_html_report,
)
from structbench.runner import RunConfig, Runner
from structbench.selfcheck import check_cases

app = typer.Typer(
    name="structbench",
    help=(
        "Stress-test LLMs on extreme JSON schemas, nested tool calls, and "
        "self-healing recovery. Every verdict is deterministic -- no judge model."
    ),
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

console = Console()
err_console = Console(stderr=True)

EXIT_OK = 0
EXIT_FAILED_GATE = 1
EXIT_CONFIG_ERROR = 2
EXIT_INTERRUPTED = 130


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _split(values: list[str] | None) -> list[str]:
    """Flatten repeated and comma-separated option values."""
    out: list[str] = []
    for value in values or []:
        out.extend(part.strip() for part in str(value).split(",") if part.strip())
    return out


def _parse_specs(raw: list[str]) -> list[ModelSpec]:
    if not raw:
        raise typer.BadParameter(
            "at least one model is required, e.g. -m mock:good or -m deepseek:deepseek-chat"
        )
    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for item in raw:
        try:
            spec = parse_model_spec(item)
        except ValueError as exc:
            raise typer.BadParameter(f"invalid model spec {item!r}: {exc}") from None
        if spec.display not in seen:
            seen.add(spec.display)
            specs.append(spec)
    return specs


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"structbench {__version__}")
        raise typer.Exit(EXIT_OK)


def _write_markdown(result: RunResult, path: Path, *, title: str | None = None) -> Path | None:
    """Write (or stream) the Markdown leaderboard. ``-`` means stdout."""
    body = render_markdown(result, title=title or "StructBench leaderboard")
    if str(path) == "-":
        console.print(body)
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _write_charts(result: RunResult, directory: Path, *, title: str | None = None) -> dict:
    """Render the static SVG chart set into ``directory``."""
    subtitle = f"{result.case_count} cases · v{result.structbench_version}"
    if title:
        subtitle = f"{title} · {subtitle}"
    return write_charts(result, directory, subtitle=subtitle)


@app.callback()
def main_callback(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """StructBench — structured-output and tool-call stress testing for LLMs."""


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


@app.command("run")
def run_command(
    models: Annotated[
        list[str],
        typer.Option(
            "--models",
            "-m",
            help="Model spec(s), comma-separated. "
            "Examples: mock:good, deepseek:deepseek-chat, ollama:qwen2.5:7b",
        ),
    ],
    datasets: Annotated[
        list[Path] | None,
        typer.Option(
            "--datasets",
            "-d",
            help="Dataset file(s) or directories. Defaults to the bundled suite.",
        ),
    ] = None,
    categories: Annotated[
        list[str] | None, typer.Option("--categories", "-c", help="Filter by category.")
    ] = None,
    tags: Annotated[list[str] | None, typer.Option("--tags", "-t", help="Filter by tag.")] = None,
    ids: Annotated[list[str] | None, typer.Option("--ids", help="Filter by case id.")] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", "-n", help="Keep only the first N cases.")
    ] = None,
    concurrency: Annotated[int, typer.Option("--concurrency", help="Max in-flight requests.")] = 5,
    rps: Annotated[
        float | None,
        typer.Option("--rps", help="Optional requests-per-second cap (free-tier friendly)."),
    ] = None,
    max_rounds: Annotated[
        int | None,
        typer.Option("--max-rounds", help="Override the per-case attempt budget (1 = no healing)."),
    ] = None,
    temperature: Annotated[
        float | None, typer.Option("--temperature", help="Sampling temperature.")
    ] = None,
    max_tokens: Annotated[
        int | None, typer.Option("--max-tokens", help="Cap on generated tokens.")
    ] = None,
    json_mode: Annotated[
        bool,
        typer.Option(
            "--json-mode/--no-json-mode",
            help="Send response_format=json_object. OFF by default so that native "
            "JSON discipline is what gets measured.",
        ),
    ] = False,
    tool_choice: Annotated[
        str | None,
        typer.Option("--tool-choice", help="OpenAI-style tool_choice (default: auto)."),
    ] = None,
    stream: Annotated[
        bool, typer.Option("--stream/--no-stream", help="Stream responses to measure TTFT.")
    ] = True,
    timeout: Annotated[
        float, typer.Option("--timeout", help="Per-request timeout in seconds.")
    ] = 120.0,
    transport_retries: Annotated[
        int, typer.Option("--transport-retries", help="Retries for 429/5xx responses.")
    ] = 2,
    base_url: Annotated[
        str | None, typer.Option("--base-url", help="Override the provider endpoint.")
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="API key. Prefer the provider's environment variable."),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Seed for the offline simulator.")] = 0,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Directory for the result archive and report.")
    ] = Path("results"),
    report: Annotated[Path | None, typer.Option("--report", help="HTML report path.")] = None,
    no_report: Annotated[
        bool, typer.Option("--no-report", help="Skip HTML report generation.")
    ] = False,
    markdown: Annotated[
        Path | None,
        typer.Option("--markdown", help="Also write a Markdown leaderboard ('-' for stdout)."),
    ] = None,
    charts: Annotated[
        Path | None,
        typer.Option(
            "--charts",
            help="Also write static SVG charts (radar, pass-heal, taxonomy) into this directory.",
        ),
    ] = None,
    no_charts: Annotated[
        bool,
        typer.Option("--no-charts", help="Skip the terminal bar-chart summary."),
    ] = False,
    inline_echarts: Annotated[
        Path | None,
        typer.Option(
            "--inline-echarts",
            help="Inline this echarts.min.js instead of the vendored copy.",
        ),
    ] = None,
    echarts_src: Annotated[
        str | None,
        typer.Option(
            "--echarts-src",
            help="Load echarts from this URL instead (smaller file, needs network).",
        ),
    ] = None,
    title: Annotated[str | None, typer.Option("--title", help="Report title.")] = None,
    show_failures: Annotated[
        int, typer.Option("--show-failures", help="Failed cases to list per model in the terminal.")
    ] = 8,
    fail_under: Annotated[
        float | None,
        typer.Option(
            "--fail-under", help="Exit non-zero if any model's Pass@1 (0-100) is below this."
        ),
    ] = None,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Only print the leaderboard.")
    ] = False,
) -> None:
    """Run the benchmark suite against one or more models."""
    try:
        specs = _parse_specs(_split(models))
        cases, files = load_cases(
            [str(path) for path in datasets] if datasets else None,
            categories=_split(categories) or None,
            tags=_split(tags) or None,
            ids=_split(ids) or None,
            limit=limit,
            max_rounds=max_rounds,
        )
    except DatasetError as exc:
        err_console.print(
            Panel(Text(str(exc), style="bold red"), title="dataset error", border_style="red")
        )
        raise typer.Exit(EXIT_CONFIG_ERROR) from None
    except typer.BadParameter as exc:
        err_console.print(
            Panel(Text(str(exc), style="bold red"), title="configuration error", border_style="red")
        )
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    options = ProviderOptions(
        base_url=base_url,
        api_key=api_key,
        timeout_s=timeout,
        stream=stream,
        transport_retries=max(transport_retries, 0),
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )

    if not quiet:
        console.print(
            Panel(
                Text.assemble(
                    (f"{len(specs)} model(s) × {len(cases)} case(s)", "bold"),
                    (f"  ·  concurrency {concurrency}", "dim"),
                    (
                        f"  ·  up to {max(cases, key=lambda c: c.max_rounds).max_rounds} round(s)/case",
                        "dim",
                    ),
                    (f"\ndatasets: {', '.join(path.name for path in files)}", "dim"),
                    (f"\nmodels:   {', '.join(spec.display for spec in specs)}", "dim"),
                ),
                title="StructBench run",
                border_style="cyan",
                expand=False,
            )
        )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TextColumn("[dim]{task.fields[status]}"),
        TimeElapsedColumn(),
        console=console,
        transient=quiet,
        disable=quiet,
    )

    config = RunConfig(
        models=specs,
        cases=cases,
        concurrency=concurrency,
        options=options,
        json_mode=json_mode,
        tool_choice=tool_choice,
        seed=seed,
        rps=rps,
        dataset_files=[str(path) for path in files],
    )

    with progress:
        tasks = {
            spec.display: progress.add_task(spec.display, total=len(cases), status="")
            for spec in specs
        }

        def on_case_done(model: str, case, result) -> None:
            task_id = tasks.get(model)
            if task_id is not None:
                mark = "ok" if result.passed_at_1 else ("healed" if result.healed else "fail")
                progress.update(task_id, advance=1, status=f"[dim]{case.id} {mark}[/dim]")

        config.progress = on_case_done

        try:
            result = asyncio.run(Runner(config).run())
        except ProviderConfigError as exc:
            err_console.print(
                Panel(Text(str(exc), style="bold red"), title="provider error", border_style="red")
            )
            raise typer.Exit(EXIT_CONFIG_ERROR) from None
        except ValueError as exc:
            err_console.print(
                Panel(Text(str(exc), style="bold red"), title="invalid case", border_style="red")
            )
            raise typer.Exit(EXIT_CONFIG_ERROR) from None
        except KeyboardInterrupt:  # pragma: no cover - interactive
            err_console.print(Text("\ninterrupted", style="bold yellow"))
            raise typer.Exit(EXIT_INTERRUPTED) from None

    console.print()
    render_report(result, console, show_failures=show_failures, show_charts=not no_charts)

    artifacts: list[Path] = []
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = output / f"run-{stamp}.json"
    archive.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    artifacts.append(archive)

    if not no_report:
        report_path = report or (output / "report.html")
        try:
            write_html_report(
                result,
                report_path,
                cases=cases,
                title=title,
                inline_echarts=inline_echarts,
                echarts_src=echarts_src,
            )
        except FileNotFoundError as exc:
            err_console.print(Text(str(exc), style="yellow"))
        else:
            artifacts.append(report_path)

    if markdown is not None:
        written = _write_markdown(result, markdown, title=title)
        if written is not None:
            artifacts.append(written)

    if charts is not None:
        artifacts.extend(_write_charts(result, charts, title=title).values())

    console.print()
    for path in artifacts:
        console.print(f"[green]✓[/green] wrote [bold]{path}[/bold]")

    if fail_under is not None:
        offenders = [
            score.model for score in result.models if score.pass_at_1_rate * 100.0 < fail_under
        ]
        if offenders:
            err_console.print(
                Panel(
                    Text(
                        f"Pass@1 below {fail_under:.1f}% for: {', '.join(offenders)}",
                        style="bold red",
                    ),
                    title="quality gate failed",
                    border_style="red",
                )
            )
            raise typer.Exit(EXIT_FAILED_GATE)
        console.print(f"[green]✓[/green] quality gate passed (Pass@1 ≥ {fail_under:.1f}%)")


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


@app.command("report")
def report_command(
    archive: Annotated[
        Path, typer.Argument(help="Run archive JSON produced by `structbench run`.")
    ],
    output: Annotated[Path | None, typer.Option("--output", "-o", help="HTML output path.")] = None,
    datasets: Annotated[
        list[Path] | None,
        typer.Option("--datasets", "-d", help="Datasets to embed prompts/schemas in the report."),
    ] = None,
    title: Annotated[str | None, typer.Option("--title", help="Report title.")] = None,
    inline_echarts: Annotated[
        Path | None,
        typer.Option(
            "--inline-echarts", help="Inline this echarts.min.js instead of the vendored one."
        ),
    ] = None,
    echarts_src: Annotated[
        str | None,
        typer.Option("--echarts-src", help="Load echarts from this URL (smaller, needs network)."),
    ] = None,
    markdown: Annotated[
        Path | None,
        typer.Option("--markdown", help="Also write a Markdown leaderboard ('-' for stdout)."),
    ] = None,
    charts: Annotated[
        Path | None,
        typer.Option("--charts", help="Also write the static SVG charts into this directory."),
    ] = None,
    print_console: Annotated[
        bool, typer.Option("--console/--no-console", help="Also print the terminal report.")
    ] = True,
) -> None:
    """Re-render the report from a saved run archive."""
    if not archive.is_file():
        err_console.print(Text(f"no such archive: {archive}", style="bold red"))
        raise typer.Exit(EXIT_CONFIG_ERROR)

    try:
        result = RunResult.model_validate_json(archive.read_text(encoding="utf-8"))
    except Exception as exc:
        err_console.print(
            Panel(
                Text(f"{archive} is not a valid StructBench archive: {exc}", style="bold red"),
                border_style="red",
            )
        )
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    cases = None
    try:
        if datasets:
            cases, _ = load_cases([str(path) for path in datasets])
        else:
            paths = [Path(item) for item in result.datasets if Path(item).is_file()]
            if paths:
                cases, _ = load_cases([str(path) for path in paths])
    except DatasetError as exc:
        err_console.print(
            Text(
                f"note: could not load datasets ({exc}); the report will omit prompts and schemas.",
                style="yellow",
            )
        )

    if print_console:
        render_report(result, console)

    target = output or archive.with_suffix(".html")
    try:
        write_html_report(
            result,
            target,
            cases=cases,
            title=title,
            inline_echarts=inline_echarts,
            echarts_src=echarts_src,
        )
    except FileNotFoundError as exc:
        err_console.print(Text(str(exc), style="bold red"))
        raise typer.Exit(EXIT_CONFIG_ERROR) from None
    console.print(f"\n[green]✓[/green] wrote [bold]{target}[/bold]")

    if markdown is not None:
        written = _write_markdown(result, markdown, title=title)
        if written is not None:
            console.print(f"[green]✓[/green] wrote [bold]{written}[/bold]")

    if charts is not None:
        for path in _write_charts(result, charts, title=title).values():
            console.print(f"[green]✓[/green] wrote [bold]{path}[/bold]")


# --------------------------------------------------------------------------- #
# chart
# --------------------------------------------------------------------------- #


@app.command("chart")
def chart_command(
    archive: Annotated[
        Path, typer.Argument(help="Run archive JSON produced by `structbench run`.")
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Directory for the SVG files.")
    ] = Path("charts"),
    title: Annotated[str | None, typer.Option("--title", help="Chart title prefix.")] = None,
    width: Annotated[int, typer.Option("--width", help="Canvas width in pixels.")] = 880,
    height: Annotated[int, typer.Option("--height", help="Canvas height in pixels.")] = 560,
) -> None:
    """Write static SVG charts from a saved run archive.

    SVG is the format that works where the interactive HTML report cannot: GitHub
    renders a committed ``.svg`` inline in a README or a Release description, so
    these are the charts a visitor to the repository actually sees.
    """
    if not archive.is_file():
        err_console.print(Text(f"no such archive: {archive}", style="bold red"))
        raise typer.Exit(EXIT_CONFIG_ERROR)

    try:
        result = RunResult.model_validate_json(archive.read_text(encoding="utf-8"))
    except Exception as exc:
        err_console.print(
            Panel(
                Text(f"{archive} is not a valid StructBench archive: {exc}", style="bold red"),
                border_style="red",
            )
        )
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    if not result.models:
        err_console.print(Text("the archive contains no model scorecards", style="bold red"))
        raise typer.Exit(EXIT_CONFIG_ERROR)

    try:
        written = write_charts(
            result, output, width=width, height=height, subtitle=title or _chart_subtitle(result)
        )
    except ChartError as exc:
        err_console.print(
            Panel(Text(str(exc), style="bold red"), title="chart error", border_style="red")
        )
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    for name, path in written.items():
        console.print(
            f"[green]✓[/green] [bold]{name}[/bold] → {path} ({path.stat().st_size:,} bytes)"
        )
    console.print(
        f"\n[dim]Embed in Markdown:[/dim] ![dimension profile]({written['radar'].as_posix()})"
    )


def _chart_subtitle(result: RunResult) -> str:
    return f"{result.case_count} cases · v{result.structbench_version}"


# --------------------------------------------------------------------------- #
# validate / list
# --------------------------------------------------------------------------- #


@app.command("validate")
def validate_command(
    datasets: Annotated[
        list[Path] | None, typer.Option("--datasets", "-d", help="Dataset file(s) or directories.")
    ] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Only report problems.")] = False,
) -> None:
    """Check that every case is satisfiable and discriminating."""
    try:
        cases, files = load_cases([str(path) for path in datasets] if datasets else None)
    except DatasetError as exc:
        err_console.print(
            Panel(Text(str(exc), style="bold red"), title="dataset error", border_style="red")
        )
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    report = check_cases(cases)

    table = Table(title="Dataset self-check", title_style="bold cyan", header_style="bold")
    table.add_column("Case", overflow="fold")
    table.add_column("Dimension", overflow="fold")
    table.add_column("Trap it must catch", style="dim", overflow="fold")

    for check in report.checks:
        if not check.ok:
            table.add_row(
                Text(check.case_id, style="bold red"),
                check.category,
                Text("; ".join(check.errors), style="red"),
            )
        elif not quiet:
            table.add_row(check.case_id, check.category, Text("ok", style="green"))

    if not quiet:
        console.print(f"loaded {len(cases)} case(s) from {len(files)} file(s)")
        console.print(table)

    for warning in report.warnings:
        console.print(Text(f"warning: {warning}", style="yellow"))

    if report.ok:
        console.print(f"[green]✓[/green] {report.summary()}")
        raise typer.Exit(EXIT_OK)

    err_console.print(f"[red]✗[/red] {report.summary()}")
    for error in report.errors:
        err_console.print(Text(f"  error: {error}", style="red"))
    raise typer.Exit(EXIT_FAILED_GATE)


@app.command("list")
def list_command(
    datasets: Annotated[
        list[Path] | None, typer.Option("--datasets", "-d", help="Dataset file(s) or directories.")
    ] = None,
    categories: Annotated[
        list[str] | None, typer.Option("--categories", "-c", help="Filter by category.")
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show tags and the first line of the prompt.")
    ] = False,
) -> None:
    """List the cases in the suite."""
    try:
        cases, files = load_cases(
            [str(path) for path in datasets] if datasets else None,
            categories=_split(categories) or None,
        )
    except DatasetError as exc:
        err_console.print(Panel(Text(str(exc), style="bold red"), border_style="red"))
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    table = Table(
        title=f"{len(cases)} case(s) from {len(files)} file(s)",
        title_style="bold cyan",
        header_style="bold",
    )
    table.add_column("Id", overflow="fold")
    table.add_column("Dimension", overflow="fold")
    table.add_column("Mode", overflow="fold")
    table.add_column("Behaviour", overflow="fold")
    table.add_column("Diff", justify="right")
    table.add_column("Rounds", justify="right")
    table.add_column("Title", overflow="fold")
    if verbose:
        table.add_column("Tags", overflow="fold")

    for case in cases:
        row = [
            case.id,
            case.category,
            case.mode or "json",
            case.expect.behaviour,
            "★" * case.difficulty,
            str(case.max_rounds),
            case.title,
        ]
        if verbose:
            row.append(", ".join(case.tags))
        table.add_row(*row)

    console.print(table)


@app.command("personas")
def personas_command() -> None:
    """Describe the built-in offline simulator personas."""
    table = Table(
        title="Mock personas (use -m mock:<name>)", title_style="bold cyan", header_style="bold"
    )
    table.add_column("Name")
    table.add_column("Pass@1", justify="right")
    table.add_column("Heal@1", justify="right")
    table.add_column("Description", overflow="fold")

    for name in PERSONA_NAMES:
        persona = PERSONAS[name]
        table.add_row(
            f"mock:{name}",
            f"{persona.pass_rate * 100:.0f}%",
            f"{persona.heal_rate * 100:.0f}%",
            persona.description,
        )
    console.print(table)
    console.print(
        Text(
            "\nThe simulator synthesises a schema-conforming answer and then corrupts it "
            "according to the persona; the real evaluator judges the result, so a mock run "
            "exercises the genuine attribution path.",
            style="dim",
        )
    )


@app.command("providers")
def providers_command() -> None:
    """List provider presets and the environment variable each expects."""
    table = Table(title="Provider presets", title_style="bold cyan", header_style="bold")
    table.add_column("Prefix", no_wrap=True)
    table.add_column("API key variable", no_wrap=True)
    table.add_column("Set?", justify="right", no_wrap=True)

    for name, preset in PROVIDER_PRESETS.items():
        env_var = preset["api_key_env"]
        configured = "—" if name == "ollama" else ("yes" if os.environ.get(env_var) else "no")
        table.add_row(
            f"{name}:<model>",
            env_var,
            Text(configured, style="green" if configured == "yes" else "dim"),
        )
    console.print(table)

    # Endpoints are printed separately: a four-column table would wrap them
    # illegibly on an 80-column terminal.
    endpoints = Table.grid(padding=(0, 2))
    endpoints.add_column(style="dim", no_wrap=True)
    endpoints.add_column(style="dim")
    for name, preset in PROVIDER_PRESETS.items():
        endpoints.add_row(name, preset["base_url"])
    console.print(Text("\ndefault endpoints", style="bold"))
    console.print(endpoints)

    console.print(
        Text(
            f"\nbundled datasets: {default_dataset_dir()}\n"
            "offline demo:     structbench run -m mock:good,mock:flaky",
            style="dim",
        )
    )


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
