"""Tests for the CI acceptance gate.

``scripts/ci_gate.py`` guards the harness itself: it asserts the offline run is
self-consistent (the never-fails persona is perfect, Heal@1 never falls below
Pass@1, and each jagged persona is weakest in its own dimension). A gate that
silently stopped checking anything would be worse than no gate, so its logic is
pinned here against synthetic archives.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_gate() -> ModuleType:
    """Import scripts/ci_gate.py, which is not part of the installed package."""
    path = ROOT / "scripts" / "ci_gate.py"
    spec = importlib.util.spec_from_file_location("ci_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ci_gate"] = module
    spec.loader.exec_module(module)
    return module


gate = load_gate()

CATEGORIES = ["deep_nesting", "escape_hell", "tool_hallucination", "edge_types"]


def make_model(
    name: str,
    *,
    pass_at_1: float,
    heal_at_1: float | None = None,
    weakest: str | None = None,
) -> dict:
    """A scorecard shaped exactly like the serialised ModelScore."""
    heal_at_1 = pass_at_1 if heal_at_1 is None else heal_at_1
    by_category = []
    for category in CATEGORIES:
        rate = 0.95
        if category == weakest:
            rate = 0.40
        by_category.append(
            {
                "category": category,
                "total": 22,
                "pass_at_1": int(rate * 22),
                "heal_at_1": int(rate * 22),
                "failed_at_1": 22 - int(rate * 22),
                "healed": 0,
                "pass_at_1_rate": rate,
                "heal_at_1_rate": rate,
                "repair_rate": None,
            }
        )
    return {
        "model": name,
        "total": 86,
        "pass_at_1_rate": pass_at_1,
        "heal_at_1_rate": heal_at_1,
        "by_category": by_category,
    }


def write_archive(workdir: Path, models: list[dict], name: str = "run-test.json") -> Path:
    target = workdir / name
    target.write_text(json.dumps({"case_count": 86, "models": models}), encoding="utf-8")
    return target


def healthy_models() -> list[dict]:
    return [
        make_model("mock/perfect", pass_at_1=1.0),
        make_model("mock/good", pass_at_1=0.88, heal_at_1=1.0),
        make_model("mock/flaky", pass_at_1=0.66, heal_at_1=0.95, weakest="escape_hell"),
        make_model("mock/escape_phobic", pass_at_1=0.78, heal_at_1=0.94, weakest="escape_hell"),
        make_model("mock/typing_weak", pass_at_1=0.71, heal_at_1=0.93, weakest="edge_types"),
        make_model(
            "mock/tool_butcher",
            pass_at_1=0.67,
            heal_at_1=0.92,
            weakest="tool_hallucination",
        ),
    ]


def test_gate_passes_on_a_healthy_archive(workdir: Path, capsys) -> None:
    write_archive(workdir, healthy_models())
    assert gate.main(["ci_gate", str(workdir)]) == 0
    output = capsys.readouterr().out
    assert "self-consistent" in output
    # The gate must actually report the rates it inspected, not just exit 0.
    assert "mock/perfect" in output
    assert "escape_hell" in output


def test_gate_fails_when_the_perfect_persona_is_not_perfect(workdir: Path, capsys) -> None:
    models = healthy_models()
    models[0]["pass_at_1_rate"] = 0.99
    write_archive(workdir, models)
    assert gate.main(["ci_gate", str(workdir)]) == 1
    assert "instead of 1.0" in capsys.readouterr().out


def test_gate_fails_when_heal_is_below_pass(workdir: Path, capsys) -> None:
    models = healthy_models()
    models[1]["heal_at_1_rate"] = 0.10  # impossible: healing only adds passes
    write_archive(workdir, models)
    assert gate.main(["ci_gate", str(workdir)]) == 1
    assert "Heal@1" in capsys.readouterr().out


def test_gate_fails_when_a_persona_is_not_weakest_where_expected(workdir: Path, capsys) -> None:
    models = healthy_models()
    # Flatten typing_weak's profile, then make deep_nesting its new worst
    # dimension -- so the gate must complain that edge_types is no longer weakest.
    for item in models[4]["by_category"]:
        item["pass_at_1_rate"] = 0.95
        if item["category"] == "deep_nesting":
            item["pass_at_1_rate"] = 0.30
    write_archive(workdir, models)
    assert gate.main(["ci_gate", str(workdir)]) == 1
    assert "should be weakest on" in capsys.readouterr().out


def test_gate_fails_when_a_persona_is_missing(workdir: Path, capsys) -> None:
    models = [m for m in healthy_models() if m["model"] != "mock/tool_butcher"]
    write_archive(workdir, models)
    assert gate.main(["ci_gate", str(workdir)]) == 1
    out = capsys.readouterr().out
    assert "mock/tool_butcher is missing from the archive" in out


def test_gate_fails_without_any_archive(workdir: Path, capsys) -> None:
    assert gate.main(["ci_gate", str(workdir)]) == 1
    assert "no run archive" in capsys.readouterr().out


def test_gate_picks_the_newest_archive(workdir: Path) -> None:
    write_archive(workdir, [make_model("stale", pass_at_1=0.0)], name="run-001.json")
    write_archive(workdir, healthy_models(), name="run-999.json")
    assert gate.main(["ci_gate", str(workdir)]) == 0


@pytest.mark.parametrize(
    ("persona", "dimension"),
    [
        ("mock/escape_phobic", "escape_hell"),
        ("mock/typing_weak", "edge_types"),
        ("mock/tool_butcher", "tool_hallucination"),
    ],
)
def test_gate_declares_every_jagged_persona(persona: str, dimension: str) -> None:
    assert gate.JAGGED_PERSONAS[persona] == dimension
