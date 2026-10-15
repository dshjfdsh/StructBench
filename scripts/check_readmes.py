#!/usr/bin/env python
"""Check that the localised READMEs stay in sync.

A multi-language README rots in a specific, quiet way: someone updates the
leaderboard in `README.md` and forgets `README.zh-CN.md`, so the Chinese page
advertises numbers the project no longer produces. That is worse than not having a
translation, because it looks authoritative.

This script enforces the parts that must be *identical* (the leaderboard data,
which is generated), the language switchers, the per-language section list, and
the quoted test count -- which is derived from pytest rather than trusted, because
a hard-coded number in a badge, a tree comment and a dev snippet, in two
languages, is guaranteed to drift.

Usage::

    python scripts/check_readmes.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Every README must exist and cross-link to all the others.
READMES: dict[str, str] = {
    "README.md": "en",
    "README.zh-CN.md": "zh-CN",
}

LEADERBOARD_START = "<!-- LEADERBOARD:START -->"
LEADERBOARD_END = "<!-- LEADERBOARD:END -->"

#: Headings that must be present in every translation, keyed by language, so a
#: section cannot silently disappear from one page.
REQUIRED_SECTIONS: dict[str, tuple[str, ...]] = {
    "en": (
        "## Quick start",
        "## Reference run",
        "## How it works",
        "## Design decisions",
        "## CLI",
        "## Extending the suite",
        "## Project structure",
        "## Development",
        "## License",
    ),
    "zh-CN": (
        "## 快速上手",
        "## 参考跑分",
        "## 工作原理",
        "## 设计取舍",
        "## 命令行",
        "## 扩展题库",
        "## 项目结构",
        "## 开发",
        "## 许可证",
    ),
}


def read(name: str) -> str:
    path = ROOT / name
    if not path.is_file():
        raise SystemExit(f"missing README: {name}")
    return path.read_text(encoding="utf-8")


def leaderboard(text: str) -> str:
    """Extract the generated leaderboard block, normalised for comparison.

    Only the *numbers and labels* matter, so the comparison strips the table
    header row (which is translated) and keeps the data rows.
    """
    start = text.find(LEADERBOARD_START)
    end = text.find(LEADERBOARD_END)
    if start < 0 or end < 0 or end < start:
        return ""
    block = text[start + len(LEADERBOARD_START) : end]

    rows: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        # Keep only data rows: markdown table rows that are not the separator and
        # not a header (headers contain "Pass@1" and no backticked model name).
        if not stripped.startswith("|"):
            continue
        if set(stripped) <= set("|-: "):
            continue
        if "`mock/" not in stripped:
            continue
        rows.append(re.sub(r"\s+", " ", stripped))
    return "\n".join(rows)


def code_fence_count(text: str) -> int:
    return len(re.findall(r"^```", text, flags=re.MULTILINE))


def collected_test_count() -> int | None:
    """Ask pytest how many tests exist, without running them.

    Hard-coded counts in a README always drift: the number is quoted in a badge, a
    directory-tree comment and a dev snippet, in two languages. Deriving it makes
    the docs verify themselves instead of relying on someone remembering.

    Note that ``--collect-only -q`` prints one ``path: count`` line per file and no
    grand total, so the per-file counts are summed. The grand-total form is tried
    first for robustness across pytest versions.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    total = 0
    found_per_file = False
    for line in result.stdout.splitlines():
        grand = re.search(r"^(\d+)\s+tests?\s+collected", line.strip())
        if grand:
            return int(grand.group(1))
        per_file = re.match(r"^[^\s].*:\s*(\d+)\s*$", line.strip())
        if per_file and ("/" in line or "\\" in line):
            total += int(per_file.group(1))
            found_per_file = True

    return total if found_per_file else None


#: Places a README quotes the test count. Each must agree with pytest.
TEST_COUNT_PATTERNS: tuple[str, ...] = (
    r"badge/tests-(\d+)%20passing",
    r"tests/\s+#\s+(\d+)\s+(?:tests|项测试)",
    r"pytest -q\s+#\s+(\d+)\s+(?:tests|项测试)",
)


def linked_readmes(text: str) -> set[str]:
    """Names of READMEs this text actually links to as markdown links.

    A plain substring check is not enough: ``README.md`` also appears in the
    project-structure listing, so removing the language switcher would still look
    "linked". Only ``[label](target)`` counts.
    """
    found: set[str] = set()
    for match in re.finditer(r"\[[^\]]*\]\(([^)\s#]+)", text):
        target = match.group(1).strip()
        if target in READMES:
            found.add(target)
    return found


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    texts = {name: read(name) for name in READMES}

    # 1. Every README must offer a way to reach all the others.
    for name, text in texts.items():
        available = linked_readmes(text)
        for other in READMES:
            if other == name:
                continue
            if other not in available:
                errors.append(
                    f"{name}: no markdown link to {other} (a language switcher is "
                    "required; the name merely appearing in prose does not count)"
                )

    # 2. Leaderboard data must match exactly across languages.
    reference = leaderboard(texts["README.md"])
    if not reference:
        errors.append("README.md: no leaderboard block found between the markers")
    for name, text in texts.items():
        if name == "README.md":
            continue
        if leaderboard(text) != reference:
            errors.append(
                f"{name}: leaderboard data differs from README.md — regenerate both "
                "with `--markdown` instead of editing by hand"
            )

    # 3. Required sections present per language.
    for name, language in READMES.items():
        for heading in REQUIRED_SECTIONS[language]:
            if heading not in texts[name]:
                errors.append(f"{name}: missing section {heading!r}")

    # 4. Code fences should be balanced, and roughly comparable across languages.
    for name, text in texts.items():
        if code_fence_count(text) % 2 != 0:
            errors.append(f"{name}: unbalanced ``` code fences")

    counts = {name: code_fence_count(text) for name, text in texts.items()}
    if len(set(counts.values())) > 1:
        warnings.append(
            "code-fence counts differ across languages "
            f"({counts}); expected for translated prose, but check nothing was dropped"
        )

    # 5. Every quoted test count must match reality.
    actual = collected_test_count()
    if actual is None:
        warnings.append("could not collect the test count; skipping the accuracy check")
    else:
        for name, text in texts.items():
            for pattern in TEST_COUNT_PATTERNS:
                for quoted in re.findall(pattern, text):
                    if int(quoted) != actual:
                        errors.append(
                            f"{name}: claims {quoted} tests but pytest collects {actual} "
                            f"(pattern: {pattern})"
                        )

    for warning in warnings:
        print(f"warning: {warning}")
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    print(f"OK: {len(READMES)} README(s) in sync ({', '.join(READMES.values())})")
    print(f"    leaderboard rows compared: {len(reference.splitlines())}")
    if actual is not None:
        print(f"    quoted test count verified against pytest: {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
