#!/usr/bin/env python
"""Development helper: validate dataset files without building the full CLI.

Usage::

    python scripts/selfcheck.py                       # every bundled dataset
    python scripts/selfcheck.py datasets/01_deep_nesting.yaml

Exits non-zero when any case is unsatisfiable or non-discriminating, so it can
be wired into CI (and into ``structbench validate``, which reuses the same code).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from structbench.loader import load_cases
from structbench.selfcheck import check_cases


def main(argv: list[str]) -> int:
    paths = argv[1:] or None
    cases, files = load_cases(paths)

    print(f"Loaded {len(cases)} case(s) from {len(files)} file(s)")
    for path in files:
        print(f"  - {path}")

    report = check_cases(cases)
    for check in report.checks:
        if not check.ok:
            print(f"\nFAIL {check.case_id} ({check.category})")
            for message in check.errors:
                print(f"  error: {message}")
        for message in check.warnings:
            print(f"WARN {check.case_id}: {message}")

    print(f"\n{report.summary()}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
