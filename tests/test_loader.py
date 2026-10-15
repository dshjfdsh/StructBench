"""Tests for the declarative dataset loader.

A typo in a dataset must produce a precise, actionable error rather than a
mysterious failure halfway through a paid run, so the error paths matter as much
as the happy path here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from structbench.loader import (
    DatasetError,
    default_dataset_dir,
    discover_files,
    load_cases,
    load_file,
)

VALID_CASE = {
    "id": "t_001",
    "title": "tiny",
    "prompt": "Return the object as JSON.",
    "target_schema": {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
        "additionalProperties": False,
    },
}


def write_yaml(workdir: Path, name: str, body: str) -> Path:
    path = workdir / name
    path.write_text(body, encoding="utf-8")
    return path


def write_case_file(workdir: Path, name: str = "ds.yaml", **overrides) -> Path:
    case = {**VALID_CASE, **overrides}
    import yaml

    document = {
        "version": 1,
        "category": "deep_nesting",
        "description": "test fixture",
        "cases": [case],
    }
    path = workdir / name
    path.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Happy paths
# --------------------------------------------------------------------------- #


def test_loads_a_case_and_records_its_source(workdir: Path) -> None:
    path = write_case_file(workdir)
    dataset = load_file(path)
    assert len(dataset.cases) == 1
    case = dataset.cases[0]
    assert case.id == "t_001"
    assert case.source == path.name
    assert case.mode == "json"


def test_category_and_max_rounds_are_inherited(workdir: Path) -> None:
    import yaml

    document = {
        "version": 1,
        "category": "escape_hell",
        "max_rounds": 3,
        "cases": [{"id": "t_002", "prompt": "p", "target_schema": {"type": "object"}}],
    }
    path = workdir / "inherit.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    case = load_file(path).cases[0]
    assert case.category == "escape_hell"
    assert case.max_rounds == 3


def test_tool_mode_is_inferred_from_tools(workdir: Path) -> None:
    import yaml

    document = {
        "version": 1,
        "category": "tool_hallucination",
        "cases": [
            {
                "id": "t_003",
                "prompt": "look it up",
                "tools": [{"name": "lookup", "parameters": {"type": "object"}}],
                "expect": {"behaviour": "call", "expected_tool": "lookup"},
            }
        ],
    }
    path = workdir / "tools.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    assert load_file(path).cases[0].mode == "tool"


def test_jsonl_is_supported(workdir: Path) -> None:
    path = workdir / "ds.jsonl"
    lines = [
        "# a comment line is ignored",
        json.dumps({**VALID_CASE, "id": "j_001", "category": "deep_nesting"}),
        json.dumps({**VALID_CASE, "id": "j_002", "category": "deep_nesting"}),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

    dataset = load_file(path)
    assert [case.id for case in dataset.cases] == ["j_001", "j_002"]


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #


def test_filters_and_limit(workdir: Path) -> None:
    import yaml

    def case(case_id: str, category: str, tags: list[str]) -> dict:
        return {
            "id": case_id,
            "category": category,
            "tags": tags,
            "prompt": "p",
            "target_schema": {"type": "object"},
        }

    path = workdir / "many.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "cases": [
                    case("a", "deep_nesting", ["nesting"]),
                    case("b", "escape_hell", ["escape"]),
                    case("c", "escape_hell", ["escape", "sql"]),
                ],
            }
        ),
        encoding="utf-8",
    )

    cases, files = load_cases([path])
    assert len(cases) == 3 and len(files) == 1

    cases, _ = load_cases([path], categories=["escape_hell"])
    assert [case.id for case in cases] == ["b", "c"]

    cases, _ = load_cases([path], tags=["sql"])
    assert [case.id for case in cases] == ["c"]

    cases, _ = load_cases([path], ids=["a", "c"])
    assert [case.id for case in cases] == ["a", "c"]

    cases, _ = load_cases([path], limit=2)
    assert [case.id for case in cases] == ["a", "b"]

    cases, _ = load_cases([path], max_rounds=1)
    assert all(case.max_rounds == 1 for case in cases)


def test_filters_matching_nothing_is_an_error(workdir: Path) -> None:
    path = write_case_file(workdir)
    with pytest.raises(DatasetError, match="matched no cases"):
        load_cases([path], categories=["nope"])


# --------------------------------------------------------------------------- #
# Error paths
# --------------------------------------------------------------------------- #


def test_unknown_top_level_key_is_rejected(workdir: Path) -> None:
    path = write_yaml(
        workdir,
        "bad.yaml",
        "version: 1\ncategory: x\ndimension: deep\ncases: []\n",
    )
    with pytest.raises(DatasetError, match="unknown top-level key"):
        load_file(path)


def test_duplicate_ids_inside_a_file_are_rejected(workdir: Path) -> None:
    import yaml

    document = {"version": 1, "category": "x", "cases": [VALID_CASE, VALID_CASE]}
    path = workdir / "dup.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(DatasetError, match="duplicate case ids"):
        load_file(path)


def test_duplicate_ids_across_files_are_rejected(workdir: Path) -> None:
    first = write_case_file(workdir, name="a.yaml")
    second = workdir / "b.yaml"
    import yaml

    second.write_text(
        yaml.safe_dump({"version": 1, "category": "x", "cases": [VALID_CASE]}),
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="duplicate case id"):
        load_cases([first, second])


def test_case_without_a_category_is_rejected(workdir: Path) -> None:
    import yaml

    path = workdir / "nocat.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "cases": [VALID_CASE]}), encoding="utf-8")
    with pytest.raises(DatasetError, match="has no `category`"):
        load_file(path)


def test_invalid_yaml_reports_the_line(workdir: Path) -> None:
    path = write_yaml(workdir, "broken.yaml", "version: 1\ncases:\n  - id: x\n   prompt: bad\n")
    with pytest.raises(DatasetError, match="invalid YAML"):
        load_file(path)


def test_empty_file_is_rejected(workdir: Path) -> None:
    path = write_yaml(workdir, "empty.yaml", "\n")
    with pytest.raises(DatasetError, match="file is empty"):
        load_file(path)


def test_missing_file_is_rejected(workdir: Path) -> None:
    with pytest.raises(DatasetError, match="no such file"):
        load_file(workdir / "absent.yaml")


def test_unsupported_extension_is_rejected(workdir: Path) -> None:
    path = workdir / "data.txt"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(DatasetError, match="unsupported dataset extension"):
        discover_files([path])


def test_schema_validation_errors_name_the_case(workdir: Path) -> None:
    import yaml

    path = workdir / "invalid.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "category": "x",
                "cases": [{"id": "bad", "prompt": "p"}],  # no schema, json mode
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="requires target_schema"):
        load_file(path)


# --------------------------------------------------------------------------- #
# Discovery of the bundled suite
# --------------------------------------------------------------------------- #


def test_bundled_suite_is_discoverable() -> None:
    directory = default_dataset_dir()
    assert directory.is_dir()
    files = discover_files(None)
    assert len(files) == 4
    assert all(path.suffix == ".yaml" for path in files)


def test_directory_discovery_is_sorted(workdir: Path) -> None:
    write_case_file(workdir, name="02_b.yaml")
    write_case_file(workdir, name="01_a.yaml")
    files = discover_files([workdir])
    assert [path.name for path in files] == ["01_a.yaml", "02_b.yaml"]
