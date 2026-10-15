"""Command-line interface tests.

The CLI is the project's public surface, so these run the real Typer app through
``CliRunner``: an offline ``mock:`` run, the dataset self-check, archive
re-rendering and the CI quality gate all have to work end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from structbench.cli import app

runner = CliRunner()

TINY_DATASET = """\
version: 1
category: deep_nesting
description: Two-case fixture for CLI tests.
cases:
  - id: cli_001
    title: single integer field
    prompt: Return the object as JSON.
    target_schema:
      type: object
      additionalProperties: false
      properties:
        a:
          type: integer
      required: [a]
  - id: cli_002
    title: nested string field
    prompt: Return the object as JSON.
    tags: [nesting]
    target_schema:
      type: object
      additionalProperties: false
      properties:
        outer:
          type: object
          additionalProperties: false
          properties:
            name:
              type: string
              minLength: 2
          required: [name]
      required: [outer]
"""


@pytest.fixture
def dataset_file(workdir: Path) -> Path:
    path = workdir / "tiny.yaml"
    path.write_text(TINY_DATASET, encoding="utf-8")
    return path


def invoke(*args: str, **kwargs):
    return runner.invoke(app, list(args), **kwargs)


# --------------------------------------------------------------------------- #
# Informational commands
# --------------------------------------------------------------------------- #


def test_version_flag() -> None:
    result = invoke("--version")
    assert result.exit_code == 0
    assert "structbench" in result.output


def test_help_lists_every_command() -> None:
    result = invoke("--help")
    assert result.exit_code == 0
    for command in ("run", "report", "validate", "list", "personas", "providers"):
        assert command in result.output


def test_personas_command_lists_every_persona() -> None:
    from structbench.providers import PERSONA_NAMES

    result = invoke("personas")
    assert result.exit_code == 0
    for name in PERSONA_NAMES:
        assert f"mock:{name}" in result.output


def test_providers_command_shows_env_vars() -> None:
    result = invoke("providers")
    assert result.exit_code == 0
    assert "STRUCTBENCH_DEEPSEEK_API_KEY" in result.output
    assert "STRUCTBENCH_SILICONFLOW_API_KEY" in result.output
    assert "api.deepseek.com" in result.output
    assert "localhost:11434" in result.output


def test_list_command(dataset_file: Path) -> None:
    result = invoke("list", "-d", str(dataset_file))
    assert result.exit_code == 0
    assert "cli_001" in result.output
    assert "cli_002" in result.output


def test_list_command_filters_by_category(dataset_file: Path) -> None:
    ok = invoke("list", "-d", str(dataset_file), "-c", "deep_nesting")
    assert ok.exit_code == 0
    missing = invoke("list", "-d", str(dataset_file), "-c", "escape_hell")
    assert missing.exit_code == 2


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #


def test_validate_accepts_the_bundled_suite() -> None:
    result = invoke("validate", "--quiet")
    assert result.exit_code == 0
    assert "86/86" in result.output


def test_validate_rejects_a_contradictory_case(workdir: Path) -> None:
    path = workdir / "broken.yaml"
    path.write_text(
        """\
version: 1
category: deep_nesting
cases:
  - id: bad_001
    prompt: Return the object.
    target_schema:
      type: object
      additionalProperties: false
      properties:
        a:
          type: string
      required: [a]
    ground_truth:
      must_be_null: ["$.a"]
""",
        encoding="utf-8",
    )
    result = invoke("validate", "-d", str(path))
    assert result.exit_code == 1
    assert "does not permit null" in result.output


def test_validate_rejects_a_missing_file() -> None:
    result = invoke("validate", "-d", "nope/absent.yaml")
    assert result.exit_code == 2


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


def test_offline_mock_run_writes_archive_and_report(dataset_file: Path, workdir: Path) -> None:
    out = workdir / "out"
    result = invoke(
        "run",
        "-m",
        "mock:perfect",
        "-d",
        str(dataset_file),
        "-o",
        str(out),
        "--quiet",
        "--markdown",
        str(out / "board.md"),
    )
    assert result.exit_code == 0, result.output

    archives = list(out.glob("run-*.json"))
    assert len(archives) == 1
    report = out / "report.html"
    assert report.is_file() and report.stat().st_size > 2000
    assert (out / "board.md").is_file()

    payload = json.loads(archives[0].read_text(encoding="utf-8"))
    assert payload["case_count"] == 2
    assert payload["models"][0]["model"] == "mock/perfect"
    assert payload["models"][0]["pass_at_1_rate"] == 1.0

    html = report.read_text(encoding="utf-8")
    assert "sb-data" in html
    assert "chart-radar" in html


def test_run_limit_and_no_report(dataset_file: Path, workdir: Path) -> None:
    out = workdir / "out"
    result = invoke(
        "run",
        "-m",
        "mock:perfect",
        "-d",
        str(dataset_file),
        "-n",
        "1",
        "-o",
        str(out),
        "--no-report",
        "--quiet",
    )
    assert result.exit_code == 0
    assert not (out / "report.html").exists()
    archive = next(out.glob("run-*.json"))
    assert json.loads(archive.read_text(encoding="utf-8"))["case_count"] == 1


def test_run_fails_the_quality_gate(dataset_file: Path, workdir: Path) -> None:
    # A threshold above 100% can never be met, so the gate must trip.
    result = invoke(
        "run",
        "-m",
        "mock:perfect",
        "-d",
        str(dataset_file),
        "-o",
        str(workdir / "out"),
        "--fail-under",
        "101",
        "--quiet",
    )
    assert result.exit_code == 1
    assert "quality gate failed" in result.output


def test_run_passes_the_quality_gate(dataset_file: Path, workdir: Path) -> None:
    result = invoke(
        "run",
        "-m",
        "mock:perfect",
        "-d",
        str(dataset_file),
        "-o",
        str(workdir / "out"),
        "--fail-under",
        "90",
        "--quiet",
    )
    assert result.exit_code == 0
    assert "quality gate passed" in result.output


def test_run_without_a_usable_provider_exits_2(dataset_file: Path, monkeypatch) -> None:
    for name in ("STRUCTBENCH_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    result = invoke("run", "-m", "deepseek:deepseek-chat", "-d", str(dataset_file), "--quiet")
    assert result.exit_code == 2
    assert "no API key" in result.output


def test_run_rejects_an_unknown_persona(dataset_file: Path) -> None:
    result = invoke("run", "-m", "mock:nope", "-d", str(dataset_file), "--quiet")
    assert result.exit_code == 2
    assert "unknown mock persona" in result.output


def test_run_requires_at_least_one_model(dataset_file: Path) -> None:
    result = invoke("run", "-d", str(dataset_file), "--quiet")
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def test_report_command_rerenders_from_an_archive(dataset_file: Path, workdir: Path) -> None:
    out = workdir / "out"
    assert (
        invoke(
            "run",
            "-m",
            "mock:perfect",
            "-d",
            str(dataset_file),
            "-o",
            str(out),
            "--quiet",
            "--no-report",
        ).exit_code
        == 0
    )
    archive = next(out.glob("run-*.json"))

    target = workdir / "rerender.html"
    result = invoke(
        "report",
        str(archive),
        "-o",
        str(target),
        "-d",
        str(dataset_file),
        "--no-console",
        "--markdown",
        "-",
    )
    assert result.exit_code == 0, result.output
    assert target.is_file()
    assert "| Model | Pass@1 |" in result.output


def test_report_rejects_a_non_archive(workdir: Path) -> None:
    path = workdir / "not-an-archive.json"
    path.write_text('{"hello": "world"}', encoding="utf-8")
    result = invoke("report", str(path), "--no-console")
    assert result.exit_code == 2
    assert "not a valid StructBench archive" in result.output


def test_report_rejects_a_missing_archive(workdir: Path) -> None:
    result = invoke("report", str(workdir / "absent.json"))
    assert result.exit_code == 2
    assert "no such archive" in result.output
