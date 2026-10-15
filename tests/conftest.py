"""Pytest fixtures wrapping the shared case factories in :mod:`tests._data`."""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from structbench.models import Case
from tests import _data

_ROOT = Path(__file__).resolve().parents[1]

#: Scratch directory for tests that need real files on disk.
#:
#: ``tmp_path`` is deliberately avoided: it creates randomly-suffixed directories
#: under the system temp directory, which some confined environments refuse
#: (a locked-down Windows sandbox rejects ``mkdtemp``). A deterministic path
#: inside the repository always works, and is cleaned up afterwards.
WORK_ROOT = _ROOT / ".pytest-work"


@pytest.fixture
def workdir(request: pytest.FixtureRequest) -> Iterator[Path]:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", request.node.name)[:120]
    path = WORK_ROOT / safe
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def nested_case() -> Case:
    return _data.nested_case()


@pytest.fixture
def tool_call_case() -> Case:
    return _data.tool_call_case()


@pytest.fixture
def clarify_case() -> Case:
    return _data.clarify_case()
