#!/usr/bin/env python
"""Pre-flight check: can every declared dependency actually be resolved?

`uv sync` failing in CI because of an unsatisfiable pin is a slow, annoying way to
learn about a typo, and it costs a full CI cycle. This script asks PyPI for each
declared requirement and reports the version that would be selected, so a bad
constraint is caught in one second locally.

Usage::

    python scripts/check_deps.py
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def latest_release(name: str) -> tuple[str, str | None, list[str]]:
    """Return ``(version, requires_python, python_classifiers)`` for a package."""
    url = f"https://pypi.org/pypi/{name}/json"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    info = payload["info"]
    classifiers = [
        c.rsplit("::", 1)[-1].strip()
        for c in info.get("classifiers", [])
        if c.startswith("Programming Language :: Python :: 3.")
    ]
    return info["version"], info.get("requires_python"), classifiers


def satisfied(requirement: str, version: str) -> bool:
    """A deliberately small PEP 440 subset check, enough for this project's pins.

    Handles ``==``, ``>=`` and ``~=``. A *pinned* requirement is reported as
    satisfied only when the latest release matches the pin, because the point of
    the check is "would a fresh resolve pick something we did not expect?".
    """
    text = requirement.strip()
    match = re.match(r"^([A-Za-z0-9_.\-]+)\s*(==|>=|~=|>)\s*([0-9][^\s;]*)", text)
    if not match:
        return True  # unconstrained

    operator, bound_text = match.group(2), match.group(3)

    def as_tuple(value: str) -> tuple[int, int, int]:
        parts = [int(p) for p in re.findall(r"\d+", value)[:3]]
        padded = [*parts, 0, 0, 0][:3]
        return (padded[0], padded[1], padded[2])

    current, bound = as_tuple(version), as_tuple(bound_text)
    if operator == "==":
        return current == bound
    return current >= bound


def main() -> int:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requires_python = data["project"]["requires-python"]
    requirements = list(data["project"]["dependencies"])
    requirements += list(data["project"].get("optional-dependencies", {}).get("dev", []))

    print(f"project requires-python: {requires_python}\n")
    problems: list[str] = []

    for requirement in requirements:
        marker = ""
        if ";" in requirement:
            requirement, _, marker = requirement.partition(";")
            marker = marker.strip()
        name = re.split(r"[<>=!~\[\s]", requirement.strip())[0]
        label = requirement.strip() + (f"  [{marker}]" if marker else "")

        try:
            version, needs, classifiers = latest_release(name)
        except urllib.error.HTTPError as exc:
            problems.append(f"{name}: PyPI lookup failed ({exc.code})")
            print(f"  {label:32} LOOKUP FAILED ({exc.code})")
            continue

        ok = satisfied(requirement, version)
        if not ok and "==" not in requirement:
            # An unmet *lower bound* means a fresh resolve would land somewhere we
            # did not choose. A pin trailing the latest release is the opposite:
            # it is exactly what pinning is for, so it is only reported as a note.
            problems.append(f"{name}: latest is {version}, which does not satisfy {requirement!r}")
        flag = "ok " if ok else "pin"
        note = ""
        if not ok and "==" in requirement:
            note = "  (pinned; trailing latest is expected)"
        print(f"  [{flag}] {label:32} -> {version:12} requires_python={needs}{note}")

        # The dev tooling must run on the newest Python we claim to support.
        if name == "ruff" and classifiers and not any(c.startswith("3.13") for c in classifiers):
            problems.append(f"ruff {version} does not advertise 3.13 support")

    print()
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1
    print(f"OK: all {len(requirements)} requirement(s) resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
