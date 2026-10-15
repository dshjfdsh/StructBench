"""Declarative dataset loading.

A dataset file is human-readable YAML (or JSONL) with this shape::

    version: 1
    category: deep_nesting          # inherited by cases that omit it
    description: >
      What this dimension probes and why it breaks models.
    max_rounds: 2                   # optional default healing budget
    cases:
      - id: dn_001
        title: four-level nested profile
        prompt: |
          ...
        target_schema: { ... }       # JSON Schema
        ground_truth:
          must_be_null: ["$.user.deleted_at"]
        notes: |
          Why this case is evil.

Everything is validated by :class:`structbench.models.Case`, so a typo produces a
precise error naming the offending case rather than a mysterious failure mid-run.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from structbench.models import Case, Dataset

__all__ = [
    "DATASET_SUFFIXES",
    "DatasetError",
    "default_dataset_dir",
    "discover_files",
    "load_cases",
    "load_file",
]

DATASET_SUFFIXES = (".yaml", ".yml", ".jsonl")


class DatasetError(RuntimeError):
    """A dataset file is missing, malformed, or internally inconsistent."""


# --------------------------------------------------------------------------- #
# Locating the bundled suite
# --------------------------------------------------------------------------- #


def _packaged_dataset_dir() -> Path | None:
    """``structbench/datasets`` inside an installed wheel."""
    try:
        from importlib.resources import files

        candidate = Path(str(files("structbench") / "datasets"))
    except (ImportError, ModuleNotFoundError, TypeError):  # pragma: no cover
        return None
    return candidate if candidate.is_dir() else None


def _repository_dataset_dir() -> Path | None:
    """The ``datasets/`` directory next to ``pyproject.toml`` in a source tree."""
    candidate = Path(__file__).resolve().parents[2] / "datasets"
    return candidate if candidate.is_dir() else None


def default_dataset_dir() -> Path:
    """Where the bundled benchmark suite lives, packaged copy first."""
    for candidate in (_packaged_dataset_dir(), _repository_dataset_dir()):
        if candidate is not None:
            return candidate
    raise DatasetError(
        "could not locate the bundled datasets directory; pass an explicit path "
        "to `structbench run --datasets <dir>`"
    )


def discover_files(paths: Iterable[str | Path] | None = None) -> list[Path]:
    """Expand directories into a sorted list of dataset files."""
    roots: list[Path] = []
    if paths is None:
        roots.append(default_dataset_dir())
    else:
        for raw in paths:
            roots.append(Path(raw))

    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(
                sorted(
                    child
                    for child in root.iterdir()
                    if child.is_file() and child.suffix.lower() in DATASET_SUFFIXES
                )
            )
        elif root.is_file():
            if root.suffix.lower() not in DATASET_SUFFIXES:
                raise DatasetError(
                    f"{root}: unsupported dataset extension {root.suffix!r} "
                    f"(expected one of {', '.join(DATASET_SUFFIXES)})"
                )
            files.append(root)
        else:
            raise DatasetError(f"{root}: no such dataset file or directory")

    if not files:
        raise DatasetError("no dataset files found")
    return files


# --------------------------------------------------------------------------- #
# Parsing one file
# --------------------------------------------------------------------------- #


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    lines = [f"{path.name}: dataset validation failed"]
    for error in exc.errors()[:12]:
        location = ".".join(str(token) for token in error["loc"])
        lines.append(f"  - {location}: {error['msg']}")
    remaining = len(exc.errors()) - 12
    if remaining > 0:
        lines.append(f"  - ...and {remaining} more error(s)")
    return "\n".join(lines)


def _read_raw_cases(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".jsonl":
        records: list[Any] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path.name}:{lineno}: invalid JSON — {exc.msg}") from None
        return {"version": 1, "cases": records}

    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1} column {mark.column + 1}" if mark else ""
        problem = getattr(exc, "problem", None) or str(exc)
        raise DatasetError(f"{path.name}: invalid YAML{where} — {problem}") from None

    if payload is None:
        raise DatasetError(f"{path.name}: file is empty")
    if isinstance(payload, list):
        return {"version": 1, "cases": payload}
    if not isinstance(payload, dict):
        raise DatasetError(f"{path.name}: top level must be a mapping or a list of cases")
    return payload


def load_file(path: str | Path) -> Dataset:
    """Parse and validate one dataset file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise DatasetError(f"{file_path}: no such file")

    payload = _read_raw_cases(file_path)

    unknown = set(payload) - {"version", "category", "description", "max_rounds", "cases"}
    if unknown:
        raise DatasetError(
            f"{file_path.name}: unknown top-level key(s) {sorted(unknown)}; "
            "allowed keys are version, category, description, max_rounds, cases"
        )

    default_category = payload.get("category")
    default_rounds = payload.get("max_rounds")
    raw_cases = payload.get("cases") or []
    if not isinstance(raw_cases, list):
        raise DatasetError(f"{file_path.name}: `cases` must be a list")

    normalised: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_cases, start=1):
        if not isinstance(raw, dict):
            raise DatasetError(f"{file_path.name}: case #{index} must be a mapping")
        case = dict(raw)
        case.setdefault("category", default_category)
        if default_rounds is not None:
            case.setdefault("max_rounds", default_rounds)
        case["source"] = file_path.name
        if not case.get("category"):
            raise DatasetError(
                f"{file_path.name}: case #{index} ({case.get('id', 'no id')}) has no "
                "`category`, and the file declares no default either"
            )
        normalised.append(case)

    try:
        return Dataset(
            version=int(payload.get("version", 1)),
            category=default_category,
            description=str(payload.get("description") or ""),
            max_rounds=default_rounds,
            cases=normalised,
        )
    except ValidationError as exc:
        raise DatasetError(_format_validation_error(file_path, exc)) from None


# --------------------------------------------------------------------------- #
# Loading and filtering many files
# --------------------------------------------------------------------------- #


def load_cases(
    paths: Sequence[str | Path] | None = None,
    *,
    categories: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
    ids: Sequence[str] | None = None,
    limit: int | None = None,
    max_rounds: int | None = None,
) -> tuple[list[Case], list[Path]]:
    """Load the suite, apply filters, and cross-check case ids.

    ``limit`` keeps a deterministic prefix of the filtered suite, which is what
    makes ``--limit 20`` reproducible across runs.
    """
    files = discover_files(paths)
    cases: list[Case] = []
    seen: dict[str, str] = {}

    for file_path in files:
        dataset = load_file(file_path)
        for case in dataset.cases:
            origin = seen.get(case.id)
            if origin is not None:
                raise DatasetError(
                    f"duplicate case id {case.id!r} in {file_path.name} "
                    f"(already defined in {origin})"
                )
            seen[case.id] = file_path.name
            cases.append(case)

    if not cases:
        raise DatasetError("no cases loaded")

    if categories:
        wanted = {value.strip().lower() for value in categories}
        cases = [case for case in cases if case.category.lower() in wanted]
    if tags:
        wanted_tags = {value.strip().lower() for value in tags}
        cases = [case for case in cases if wanted_tags & {tag.lower() for tag in case.tags}]
    if ids:
        wanted_ids = {value.strip() for value in ids}
        cases = [case for case in cases if case.id in wanted_ids]

    if max_rounds is not None:
        cases = [case.model_copy(update={"max_rounds": max(max_rounds, 1)}) for case in cases]

    if not cases:
        raise DatasetError("the supplied filters matched no cases")

    if limit is not None and limit > 0:
        cases = cases[:limit]

    return cases, files
