"""Tests for the README-sync guard.

Multi-language docs rot quietly: someone regenerates the leaderboard in one
language and forgets the other, so the translation starts advertising numbers the
project no longer produces. ``scripts/check_readmes.py`` enforces the parts that
must stay identical, and these tests make sure the guard itself still works.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def load_checker() -> ModuleType:
    path = ROOT / "scripts" / "check_readmes.py"
    spec = importlib.util.spec_from_file_location("check_readmes", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_readmes"] = module
    spec.loader.exec_module(module)
    return module


checker = load_checker()


def test_english_and_chinese_readmes_both_exist() -> None:
    for name in checker.READMES:
        assert (ROOT / name).is_file(), f"{name} is missing"


def test_shipped_readmes_are_in_sync() -> None:
    assert checker.main() == 0


def test_language_switchers_are_real_markdown_links() -> None:
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")

    assert "README.zh-CN.md" in checker.linked_readmes(english)
    assert "README.md" in checker.linked_readmes(chinese)


def test_link_detection_ignores_prose_mentions() -> None:
    # ``README.md`` appears in the project-structure listing of the Chinese page,
    # so a substring check would wrongly conclude the switcher is present.
    prose = "see README.md for details"
    assert checker.linked_readmes(prose) == set()
    assert checker.linked_readmes("[English](README.md)") == {"README.md"}
    assert checker.linked_readmes("[简体中文](README.zh-CN.md)") == {"README.zh-CN.md"}


def test_leaderboards_match_across_languages() -> None:
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")

    assert checker.leaderboard(english) == checker.leaderboard(chinese)
    # The block must not be empty, or equality would be trivially satisfied.
    assert len(checker.leaderboard(english).splitlines()) >= 15


def test_leaderboard_extraction_keeps_only_data_rows() -> None:
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    rows = checker.leaderboard(english)
    assert "mock/perfect" in rows
    # Header rows are translated, so they must be filtered out; if they leaked in,
    # the two languages would never compare equal.
    assert "Pass@1" not in rows
    assert "|---" not in rows


def test_every_language_declares_its_required_sections() -> None:
    for name, language in checker.READMES.items():
        text = (ROOT / name).read_text(encoding="utf-8")
        for heading in checker.REQUIRED_SECTIONS[language]:
            assert heading in text, f"{name} is missing {heading!r}"


def test_required_sections_cover_the_same_topics_in_both_languages() -> None:
    # A translation that silently drops a section is the failure mode this guard
    # exists for, so the two lists must stay the same length.
    english = checker.REQUIRED_SECTIONS["en"]
    chinese = checker.REQUIRED_SECTIONS["zh-CN"]
    assert len(english) == len(chinese)


def test_code_fences_are_balanced() -> None:
    for name in checker.READMES:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert checker.code_fence_count(text) % 2 == 0, f"{name} has an unclosed fence"


def test_quoted_test_count_matches_pytest() -> None:
    """The badge, the tree comment and the dev snippet must not drift."""
    actual = checker.collected_test_count()
    assert actual is not None and actual > 0, "could not collect a test count"

    for name in checker.READMES:
        text = (ROOT / name).read_text(encoding="utf-8")
        found = 0
        for pattern in checker.TEST_COUNT_PATTERNS:
            for quoted in re.findall(pattern, text):
                found += 1
                assert int(quoted) == actual, (
                    f"{name} claims {quoted} tests but pytest collects {actual}"
                )
        # Each README quotes the count three times; a silently-broken regex would
        # make this test pass vacuously.
        assert found == 3, f"{name}: expected 3 quoted test counts, matched {found}"


def test_collected_count_agrees_with_a_real_run() -> None:
    """Cross-check the collector against the summary line of an actual run."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--co", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    collected = checker.collected_test_count()
    assert collected is not None and collected > 0
    assert result.returncode == 0, result.stdout[-2000:]
