#!/usr/bin/env python
"""Assert that a generated report.html renders without any network access.

The interactive report is the artefact people download from a Release and
double-click. When it loaded ECharts from a CDN -- as it did in v0.1.0 -- the file
opened as a blank page for anyone offline, or anywhere the CDN is unreachable. That
failure is invisible in CI unless something checks for it, so this script is that
check, and it runs in the release pipeline before the artefact is published.

Usage::

    python scripts/check_report_offline.py path/to/report.html
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

#: A report with the library inlined is roughly this large. Anything much smaller
#: means the library was not embedded, i.e. the report will not draw offline.
MIN_INLINE_SIZE = 400_000


def check(path: Path) -> list[str]:
    """Return a list of problems; empty means the report is offline-safe."""
    if not path.is_file():
        return [f"no such report: {path}"]

    html = path.read_text(encoding="utf-8")
    problems: list[str] = []

    external = re.findall(r'<script[^>]+src="(https?://[^"]+)"', html)
    if external:
        problems.append(
            f"the report references a remote script ({external[0]}), so it will render "
            "as a blank page without a network; the library must be inlined"
        )

    if "sb-data" not in html:
        problems.append("the run data is not embedded in the report")

    if "chart-radar" not in html:
        problems.append("the radar chart is missing from the report")

    if len(html) < MIN_INLINE_SIZE:
        problems.append(
            f"the report is only {len(html):,} chars, which suggests the charting "
            f"library was not inlined (expected at least {MIN_INLINE_SIZE:,})"
        )

    return problems


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip())
        return 2

    targets = [Path(arg) for arg in argv[1:]]
    failed = False
    for path in targets:
        problems = check(path)
        size = path.stat().st_size if path.is_file() else 0
        print(f"{path}")
        print(f"  size: {size:,} chars")
        if problems:
            failed = True
            for problem in problems:
                print(f"  FAIL: {problem}")
        else:
            print("  OK: renders offline, data embedded, charts present")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
