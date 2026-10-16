"""Tests for the CI helper scripts.

These scripts are not part of the installed package, so they get no import-time
protection from the normal test run. That matters: `check_deps.py` shipped using
`tomllib`, which only exists on Python 3.11+, and CI runs the matrix down to 3.10
-- so the helper crashed on exactly the interpreter it was meant to protect. The
guard below pins that class of mistake.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))

#: Modules added in 3.11 or later. Importing any of these unconditionally breaks
#: the 3.10 leg of the CI matrix. Each is allowed only inside a `try:` block that
#: has a fallback -- `check_scripts_guard_against_newer_stdlib` enforces that.
NEWER_STDLIB = {
    "tomllib": "3.11",
    "typing.Self": "3.11",
    "enum.StrEnum": "3.11",
    "asyncio.TaskGroup": "3.11",
    "ExceptionGroup": "3.11",
    "datetime.UTC": "3.11",
    "itertools.batched": "3.12",
    "typing.override": "3.12",
}


def load_script(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_scripts_directory_is_not_empty() -> None:
    assert SCRIPTS, "no scripts found; this guard would pass vacuously"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_every_script_parses(path: Path) -> None:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_scripts_guard_against_newer_stdlib(path: Path) -> None:
    """A 3.11+ stdlib import must sit in a ``try`` with a fallback."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    guarded: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                guarded.update(alias.name for alias in child.names)
            elif isinstance(child, ast.ImportFrom) and child.module:
                name = child.module
                if child.names:
                    name = f"{child.module}.{child.names[0].name}"
                guarded.add(name)

    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.names:
                names = [f"{node.module}.{node.names[0].name}", node.module]
            else:
                names = [node.module]

        for name in names:
            if name in NEWER_STDLIB and name not in guarded:
                pytest.fail(
                    f"{path.name}: imports {name!r} (Python {NEWER_STDLIB[name]}+) "
                    "without a try/except fallback; CI still runs Python 3.10"
                )


def test_check_deps_parses_pyproject() -> None:
    module = load_script(ROOT / "scripts" / "check_deps.py")
    data = module.tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "project" in data
    assert data["project"]["name"] == "structbench"


def test_check_deps_declares_the_tomli_backport_for_310() -> None:
    """The fallback import only works if the backport is an actual dependency."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "tomli" in text, "check_deps.py falls back to tomli, so it must be declared"
    assert "python_version < '3.11'" in text, (
        "tomli must be conditional, or it becomes a needless dependency on 3.11+"
    )


def test_check_deps_runs_end_to_end() -> None:
    """Full run against PyPI: every declared requirement must still resolve."""
    module = load_script(ROOT / "scripts" / "check_deps.py")
    assert module.main() == 0


def test_readme_checker_runs_end_to_end() -> None:
    module = load_script(ROOT / "scripts" / "check_readmes.py")
    assert module.main() == 0


def test_ci_gate_rejects_a_missing_archive(workdir: Path) -> None:
    module = load_script(ROOT / "scripts" / "ci_gate.py")
    assert module.main(["ci_gate", str(workdir)]) == 1


def test_every_script_has_a_docstring() -> None:
    """These are user-facing helpers; an undocumented one is a support burden."""
    for path in SCRIPTS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstring = ast.get_docstring(tree)
        assert docstring, f"{path.name} has no module docstring"
        assert "Usage::" in docstring, f"{path.name} does not document its usage"


class TestReportOfflineCheck:
    """The guard that stopped a blank-page report from being released again."""

    checker = load_script(ROOT / "scripts" / "check_report_offline.py")

    def offline_report(self, workdir: Path) -> Path:
        """A report with the library inlined, produced by the real generator."""
        sys.path.insert(0, str(ROOT / "src"))
        try:
            from structbench.metrics import compute_run_result
            from structbench.models import Attempt, AttemptStatus, CaseResult
            from structbench.reporter import render_html_report
        finally:
            sys.path.pop(0)

        attempt = Attempt(
            round=1,
            passed=True,
            raw_text='{"a": 1}',
            status=AttemptStatus.OK,
        )
        record = CaseResult(
            case_id="c1",
            category="deep_nesting",
            model="mock/x",
            title="t",
            attempts=[attempt],
            passed_at_1=True,
            passed_after_heal=True,
        )
        result = compute_run_result(
            version="0.0-test",
            records=[record],
            models=[("mock/x", "mock", "mock/x")],
            datasets=["d.yaml"],
            case_count=1,
            concurrency=1,
            max_rounds=2,
        )
        path = workdir / "report.html"
        path.write_text(render_html_report(result), encoding="utf-8")
        return path

    def test_offline_report_passes(self, workdir: Path) -> None:
        assert self.checker.check(self.offline_report(workdir)) == []

    def test_cdn_report_is_rejected(self, workdir: Path) -> None:
        """The exact shape that shipped as a blank page in v0.1.0."""
        path = workdir / "cdn.html"
        path.write_text(
            '<html><head><script src="https://cdn.example.test/echarts.js"></script>'
            "</head><body>sb-data chart-radar</body></html>",
            encoding="utf-8",
        )
        problems = self.checker.check(path)
        assert problems, "a CDN-dependent report must be rejected"
        assert any("remote script" in p for p in problems)

    def test_report_without_data_is_rejected(self, workdir: Path) -> None:
        path = workdir / "nodata.html"
        path.write_text("<html>" + "x" * 500_000 + "</html>", encoding="utf-8")
        problems = self.checker.check(path)
        assert any("run data is not embedded" in p for p in problems)

    def test_suspiciously_small_report_is_rejected(self, workdir: Path) -> None:
        path = workdir / "tiny.html"
        path.write_text("<html><body>sb-data chart-radar</body></html>", encoding="utf-8")
        problems = self.checker.check(path)
        assert any("not inlined" in p for p in problems)

    def test_missing_file_is_reported(self, workdir: Path) -> None:
        problems = self.checker.check(workdir / "absent.html")
        assert any("no such report" in p for p in problems)

    def test_cli_exit_codes(self, workdir: Path) -> None:
        assert self.checker.main(["check", str(self.offline_report(workdir))]) == 0

        bad = workdir / "bad.html"
        bad.write_text("<html></html>", encoding="utf-8")
        assert self.checker.main(["check", str(bad)]) == 1
        assert self.checker.main(["check"]) == 2

    def test_cli_output_names_the_problem(self, workdir: Path, capsys) -> None:
        bad = workdir / "bad.html"
        bad.write_text("<html></html>", encoding="utf-8")
        self.checker.main(["check", str(bad)])
        assert "FAIL" in capsys.readouterr().out
