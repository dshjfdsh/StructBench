#!/usr/bin/env python
"""CI acceptance gate: assert the offline reference run is internally consistent.

The `demo-run` CI job executes the whole suite against the offline simulator and
then runs this script. It is a *self-consistency* check on the harness, not a
claim about any real model: the personas are built to produce known behaviour, so
a mismatch means StructBench itself regressed somewhere.

Checks:
  1. `mock:perfect` scores exactly 100% Pass@1 — if the "never fails" persona
     loses even one case, either the evaluator got stricter in a way the
     synthesizer cannot satisfy, or a dataset case became unsatisfiable.
  2. Every model records a scorecard, and the report archive round-trips.
  3. `Heal@1 >= Pass@1` for every model — a structural invariant of the two
     metrics (healing can only add passes, never remove them).
  4. The jagged personas really are jagged: each of `escape_phobic`,
     `typing_weak` and `tool_butcher` must be weakest in its own dimension.
     This is what proves the per-dimension attribution actually discriminates.

Usage::

    python scripts/ci_gate.py ci-results
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: persona -> the dimension it is designed to fail hardest
JAGGED_PERSONAS = {
    "mock/escape_phobic": "escape_hell",
    "mock/typing_weak": "edge_types",
    "mock/tool_butcher": "tool_hallucination",
}

MIN_MODELS = 5


def fail(message: str) -> None:
    print(f"FAIL: {message}")


def main(argv: list[str]) -> int:
    results_dir = Path(argv[1] if len(argv) > 1 else "ci-results")
    archives = sorted(results_dir.glob("run-*.json"))
    if not archives:
        fail(f"no run archive found in {results_dir}")
        return 1

    # Archives are named run-YYYYmmdd-HHMMSS.json, so the lexicographically last
    # name is the newest. Sorting by mtime instead is unreliable: two runs can
    # land in the same filesystem timestamp tick.
    archive = archives[-1]
    payload = json.loads(archive.read_text(encoding="utf-8"))
    models = {entry["model"]: entry for entry in payload.get("models", [])}
    print(f"archive: {archive}")
    print(f"cases:   {payload.get('case_count')}   models: {len(models)}\n")

    errors: list[str] = []

    if len(models) < MIN_MODELS:
        errors.append(f"expected at least {MIN_MODELS} scorecards, got {len(models)}")

    # 1. The upper bound must be exact.
    perfect = models.get("mock/perfect")
    if perfect is None:
        errors.append("mock/perfect is missing from the archive")
    else:
        rate = perfect["pass_at_1_rate"]
        print(f"mock/perfect  Pass@1={rate:.4f}  Heal@1={perfect['heal_at_1_rate']:.4f}")
        # The archive stores rates as fractions (0..1); only the Markdown report
        # renders them as percentages.
        if abs(rate - 1.0) > 1e-9:
            errors.append(
                f"mock/perfect scored {rate:.4f} instead of 1.0 — a case it should "
                "satisfy is failing"
            )

    # 3. Metric invariant.
    print()
    for name, entry in sorted(models.items()):
        p1 = entry["pass_at_1_rate"]
        h1 = entry["heal_at_1_rate"]
        print(f"{name:24} Pass@1={p1 * 100:6.1f}%  Heal@1={h1 * 100:6.1f}%  n={entry['total']}")
        if h1 < p1:
            errors.append(f"{name}: Heal@1 ({h1}) < Pass@1 ({p1})")

    # 4. Per-dimension jaggedness.
    print()
    for name, dimension in JAGGED_PERSONAS.items():
        entry = models.get(name)
        if entry is None:
            errors.append(f"{name} is missing from the archive")
            continue
        # The archive serialises by_category as a list of CategoryScore objects;
        # only the HTML payload (build_payload) reshapes it into a mapping.
        raw = entry.get("by_category") or []
        by_category = {
            item["category"]: item["pass_at_1_rate"]
            for item in raw
            if isinstance(item, dict) and item.get("pass_at_1_rate") is not None
        }
        if dimension not in by_category:
            errors.append(f"{name}: no score for its target dimension {dimension!r}")
            continue
        worst = min(by_category, key=lambda key: by_category[key])
        print(
            f"{name:24} weakest={worst:20} "
            f"({by_category[worst] * 100:5.1f}% vs target {dimension} "
            f"{by_category[dimension] * 100:5.1f}%)"
        )
        if worst != dimension:
            errors.append(f"{name} should be weakest on {dimension!r} but is weakest on {worst!r}")

    if errors:
        print()
        for message in errors:
            fail(message)
        return 1

    print(
        "\nOK: harness is self-consistent (perfect=100%, Heal@1>=Pass@1, jagged personas located)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
