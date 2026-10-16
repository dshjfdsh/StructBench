"""Development helper: render the committed chart set from a run archive.

Usage::

    python scripts/make_charts.py results/run-*.json docs/charts

Used to refresh the images embedded in the READMEs, and by the release workflow.
Kept separate from the CLI so that a contributor can regenerate the charts with a
single command without remembering the `structbench chart` flags.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from structbench.models import RunResult
from structbench.reporter.charts import write_charts


def main(argv: list[str]) -> int:
    patterns = [arg for arg in argv[1:] if not arg.startswith("--")] or ["results/run-*.json"]
    archive_pattern = patterns[0]
    out_dir = patterns[1] if len(patterns) > 1 else "docs/charts"

    matches = sorted(glob.glob(archive_pattern))
    if not matches:
        print(f"no run archive matched {archive_pattern!r}")
        return 1

    archive = matches[-1]
    result = RunResult.model_validate_json(Path(archive).read_text(encoding="utf-8"))

    subtitle = f"{result.case_count} cases · v{result.structbench_version}"
    written = write_charts(result, out_dir, subtitle=subtitle)

    print(f"source: {archive}")
    for name, path in written.items():
        print(f"  {name:10} -> {path}  ({path.stat().st_size:,} bytes)")
    print(f"wrote {len(written)} chart(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
