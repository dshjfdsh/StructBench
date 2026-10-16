"""Tests for the release tooling: packaging and release-note composition.

Two failure modes here are silent and only surface after a release is published:

1. **The wheel omits the datasets.** `pip install structbench` then works right up
   until `structbench validate` cannot find the bundled suite. The packaging
   config has to be exercised, not eyeballed.
2. **The release body links to charts that were never uploaded**, or loses the
   changelog because the heading format changed. A broken image on a Release page
   looks worse than no image.

Both are covered here, plus the note composer's own edge cases.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


notes = load("release_notes")


# --------------------------------------------------------------------------- #
# Packaging
# --------------------------------------------------------------------------- #


def test_pyproject_declares_the_console_script() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'structbench = "structbench.cli:main"' in text


def test_pyproject_force_includes_the_datasets() -> None:
    """Without this, `pip install` yields a package with no benchmark suite."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "force-include" in text, "datasets must be force-included into the wheel"
    assert '"datasets" = "structbench/datasets"' in text


def test_loader_looks_inside_the_package_first() -> None:
    """The packaged copy must be preferred over a repository checkout."""
    source = (ROOT / "src" / "structbench" / "loader.py").read_text(encoding="utf-8")
    packaged = source.index("_packaged_dataset_dir")
    repository = source.index("_repository_dataset_dir")
    # Both must be referenced by default_dataset_dir, packaged first.
    default = source[source.index("def default_dataset_dir") :]
    assert default.index("_packaged_dataset_dir") < default.index("_repository_dataset_dir")
    assert packaged < repository


def test_bundled_datasets_are_discoverable_from_source() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from structbench.loader import discover_files

    files = discover_files(None)
    assert len(files) == 4
    assert all(path.suffix == ".yaml" for path in files)


# --------------------------------------------------------------------------- #
# Release note composition
# --------------------------------------------------------------------------- #


class TestChangelogExtraction:
    def test_extracts_the_requested_version(self) -> None:
        changelog = (
            "# Changelog\n\n"
            "## [Unreleased]\n\n"
            "## [0.2.0] — 2025-11-01\n\nSecond release.\n\n"
            "## [0.1.0] — 2025-10-16\n\nFirst release.\n"
        )
        assert notes.changelog_section(changelog, "0.1.0") == "First release."
        assert notes.changelog_section(changelog, "0.2.0") == "Second release."

    def test_does_not_bleed_into_the_next_version(self) -> None:
        changelog = "## [0.1.0]\n\nOnly mine.\n\n## [0.0.9]\n\nNot mine.\n"
        section = notes.changelog_section(changelog, "0.1.0")
        assert section == "Only mine."
        assert "Not mine" not in section

    def test_missing_version_returns_none(self) -> None:
        assert notes.changelog_section("## [1.0.0]\n\nx\n", "9.9.9") is None

    def test_empty_section_returns_none(self) -> None:
        assert notes.changelog_section("## [0.1.0]\n\n## [0.0.1]\n\nx\n", "0.1.0") is None

    def test_version_from_tag_strips_the_prefix(self) -> None:
        assert notes.version_from_tag("v0.1.0") == "0.1.0"
        assert notes.version_from_tag("v0.2.0-rc1") == "0.2.0-rc1"


class TestReleaseBody:
    def make_assets(self, directory: Path, *, charts: bool = True) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        if charts:
            for name, _ in notes.CHARTS:
                (directory / name).write_text("<svg/>", encoding="utf-8")
        (directory / "leaderboard.md").write_text("| Model | Pass@1 |\n| m | 1 |", encoding="utf-8")
        return directory

    def test_body_contains_install_and_charts(self, workdir: Path) -> None:
        assets = self.make_assets(workdir / "assets")
        body = notes.build_body(
            tag="v0.1.0",
            repository="owner/repo",
            assets=assets,
            changelog_path=ROOT / "CHANGELOG.md",
        )
        assert "pip install structbench" in body
        assert "owner/repo/releases/download/v0.1.0/pass-heal.svg" in body
        assert "## Leaderboard" in body
        assert "## What changed" in body

    def test_body_includes_the_matching_changelog_section(self, workdir: Path) -> None:
        assets = self.make_assets(workdir / "assets")
        changelog = workdir / "CHANGELOG.md"
        changelog.write_text(
            "# Changelog\n\n## [0.1.0] — today\n\n### Added\n\n- a thing\n",
            encoding="utf-8",
        )
        body = notes.build_body(
            tag="v0.1.0", repository="o/r", assets=assets, changelog_path=changelog
        )
        assert "- a thing" in body

    def test_unknown_version_says_so_instead_of_failing(self, workdir: Path) -> None:
        assets = self.make_assets(workdir / "assets")
        changelog = workdir / "CHANGELOG.md"
        changelog.write_text("## [9.9.9]\n\nother\n", encoding="utf-8")
        body = notes.build_body(
            tag="v1.2.3", repository="o/r", assets=assets, changelog_path=changelog
        )
        assert "No `[1.2.3]` section found" in body

    def test_missing_chart_is_refused_by_default(self, workdir: Path) -> None:
        assets = self.make_assets(workdir / "assets", charts=False)
        with pytest.raises(FileNotFoundError, match="missing chart asset"):
            notes.build_body(tag="v0.1.0", repository="o/r", assets=assets)

    def test_missing_chart_can_be_tolerated_for_previews(self, workdir: Path) -> None:
        assets = self.make_assets(workdir / "assets", charts=False)
        body = notes.build_body(tag="v0.1.0", repository="o/r", assets=assets, require_charts=False)
        assert "## Install" in body
        # The charts *section* must be absent; the install snippet legitimately
        # links to the release assets, so its URL is not evidence of a chart.
        assert "## Charts from this release" not in body
        assert "![Pass@1 vs Heal@1]" not in body

    def test_body_warns_that_the_models_are_simulated(self, workdir: Path) -> None:
        """A Release page must not read as a claim about real models."""
        assets = self.make_assets(workdir / "assets")
        body = notes.build_body(
            tag="v0.1.0", repository="o/r", assets=assets, changelog_path=ROOT / "CHANGELOG.md"
        )
        assert "simulated" in body.lower()


class TestReleaseNotesCli:
    def test_main_writes_the_output_file(self, workdir: Path) -> None:
        assets = workdir / "assets"
        assets.mkdir()
        for name, _ in notes.CHARTS:
            (assets / name).write_text("<svg/>", encoding="utf-8")
        out = workdir / "body.md"

        code = notes.main(
            [
                "--tag",
                "v0.1.0",
                "--repository",
                "o/r",
                "--assets",
                str(assets),
                "--changelog",
                str(ROOT / "CHANGELOG.md"),
                "--output",
                str(out),
            ]
        )
        assert code == 0
        assert out.is_file()
        assert "## Install" in out.read_text(encoding="utf-8")

    def test_main_reports_a_missing_chart(self, workdir: Path, capsys) -> None:
        assets = workdir / "assets"
        assets.mkdir()
        code = notes.main(["--tag", "v0.1.0", "--repository", "o/r", "--assets", str(assets)])
        assert code == 1
        assert "missing chart asset" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Workflow wiring
# --------------------------------------------------------------------------- #


class TestReleaseWorkflow:
    """The workflow is configuration, so its invariants are asserted directly."""

    @pytest.fixture(scope="class")
    def workflow(self) -> str:
        return (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    def test_triggers_on_version_tags(self, workflow: str) -> None:
        assert 'tags: ["v*"]' in workflow

    def test_has_the_expected_jobs(self, workflow: str) -> None:
        for job in ("verify", "charts", "build", "release"):
            assert re.search(rf"^  {job}:", workflow, re.MULTILINE), f"missing job {job}"

    def test_release_waits_for_charts_and_build(self, workflow: str) -> None:
        assert "needs: [charts, build]" in workflow

    def test_grants_write_permission_for_the_release(self, workflow: str) -> None:
        assert "contents: write" in workflow

    def test_verifies_before_publishing(self, workflow: str) -> None:
        # A release must not ship unverified code.
        assert "uv run pytest -q" in workflow
        assert "validate --quiet" in workflow

    def test_no_node20_actions(self, workflow: str) -> None:
        """Every action must be on a Node 24 generation.

        v4/v5 of the first-party actions and v2 of the popular third-party release
        action still target node20, which GitHub force-migrates and reports as
        deprecated on every job.
        """
        banned = {
            "actions/checkout@v4": "use @v5",
            "actions/upload-artifact@v5": "use @v6",
            "actions/download-artifact@v6": "use @v7",
            "astral-sh/setup-uv@v5": "use @v7",
            "astral-sh/ruff-action@v3": "ruff installs from the dev extra instead",
            "softprops/action-gh-release@v2": "use the gh CLI instead",
        }
        for spec, advice in banned.items():
            assert spec not in workflow, f"{spec} targets node20; {advice}"

    def test_publishes_via_the_gh_cli(self, workflow: str) -> None:
        assert "gh release create" in workflow
        assert "GH_TOKEN" in workflow

    def test_prerelease_detection(self, workflow: str) -> None:
        # A hyphen in the tag must produce a pre-release, not a full release.
        assert "--prerelease" in workflow

    def test_wheel_is_install_tested(self, workflow: str) -> None:
        """Guards the classic packaging bug: a wheel that omits the datasets."""
        assert "uv pip install --python /tmp/check dist/*.whl" in workflow
        assert "/tmp/check/bin/structbench validate" in workflow


class TestCiWorkflowStillSane:
    @pytest.fixture(scope="class")
    def workflow(self) -> str:
        return (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    def test_no_node20_actions(self, workflow: str) -> None:
        for spec in (
            "actions/checkout@v4",
            "actions/upload-artifact@v5",
            "astral-sh/setup-uv@v5",
            "astral-sh/ruff-action@v3",
        ):
            assert spec not in workflow, f"{spec} targets node20"

    def test_runs_the_guards(self, workflow: str) -> None:
        for script in ("check_readmes.py", "check_deps.py", "ci_gate.py"):
            assert script in workflow, f"{script} is not run in CI"
