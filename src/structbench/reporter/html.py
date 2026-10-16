"""Single-file interactive HTML report (Jinja2 + ECharts).

The output is one self-contained ``report.html``: the run archive is embedded as
JSON, the charting library is inlined, and every chart is drawn client-side. That
means the file works when opened straight off disk, with no network at all -- which
matters because this is the artifact people download from a Release and
double-click, and a report that silently needs a CDN renders as a blank page for
anyone offline.

A vendored copy of ECharts ships in ``reporter/vendor/`` for exactly this reason.
``--echarts-src URL`` opts back into the CDN for a much smaller file, and
``--inline-echarts PATH`` inlines a different build (for example a newer release,
or a mirror when the CDN is unreachable).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from structbench.evaluator import DETAIL_GLOSS
from structbench.models import (
    ALL_LABELS,
    FAILURE_LABELS,
    LABEL_ZH,
    Case,
    ErrorLabel,
    RunResult,
)

__all__ = [
    "DEFAULT_ECHARTS_SRC",
    "VENDORED_ECHARTS",
    "build_payload",
    "echarts_library_text",
    "render_html_report",
    "write_html_report",
]

#: CDN location, used only when the caller explicitly asks for a remote library.
DEFAULT_ECHARTS_SRC = "https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"

#: The copy bundled with StructBench, inlined by default so the report is offline.
VENDORED_ECHARTS = Path(__file__).resolve().parent / "vendor" / "echarts.min.js"

_TEMPLATE_DIR = Path(__file__).resolve().parent

#: Raw model output and schemas are truncated in the report so that a run over a
#: large suite does not produce a 50 MB HTML file.
DEFAULT_MAX_RAW_CHARS = 4000
DEFAULT_MAX_FAILURES = 250


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _percent(value: float | None) -> float | None:
    return None if value is None else round(value * 100.0, 2)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… (+{len(text) - limit} characters truncated)"


def _attempt_view(attempt: Any, max_raw_chars: int) -> dict[str, Any]:
    raw = attempt.raw_text or ""
    if attempt.tool_calls:
        calls = "\n".join(
            f"[tool call] {call.name}({call.arguments_raw})" for call in attempt.tool_calls
        )
        raw = f"{raw}\n{calls}".strip() if raw else calls
    return {
        "round": attempt.round,
        "status": attempt.status.value,
        "provider_error": attempt.provider_error,
        "label": attempt.label.value if attempt.label else None,
        "detail": attempt.detail,
        "detail_gloss": DETAIL_GLOSS.get(attempt.detail, ""),
        "passed": attempt.passed,
        "raw": _clip(raw, max_raw_chars),
        "latency_ms": round(attempt.latency_ms, 1),
        "ttft_ms": round(attempt.ttft_ms, 1) if attempt.ttft_ms is not None else None,
        "issues": [
            {
                "path": issue.path,
                "message": issue.message,
                "detail": issue.detail,
                "gloss": DETAIL_GLOSS.get(issue.detail, ""),
                "label": issue.label.value,
            }
            for issue in attempt.issues
        ],
    }


def _case_reference(case: Case, max_raw_chars: int) -> dict[str, Any]:
    if case.is_tool:
        tool = case.tool_by_name(case.expect.expected_tool)
        expected = tool.parameters if tool is not None else {}
        expectation = (
            f"call {case.expect.expected_tool}(...)"
            if case.expect.behaviour == "call"
            else f"behaviour: {case.expect.behaviour}"
        )
        tools = [{"name": item.name, "description": item.description} for item in case.tools]
    else:
        expected = case.target_schema or {}
        expectation = "raw JSON matching the schema"
        tools = []

    return {
        "id": case.id,
        "title": case.title,
        "category": case.category,
        "difficulty": case.difficulty,
        "tags": list(case.tags),
        "prompt": _clip(case.prompt, max_raw_chars),
        "system": _clip(case.system or "", 1000),
        "expected": expected,
        "expectation": expectation,
        "ground_truth": case.ground_truth.model_dump(exclude_defaults=True),
        "tools": tools,
        "notes": case.notes,
    }


def build_payload(
    result: RunResult,
    *,
    cases: Sequence[Case] | None = None,
    max_raw_chars: int = DEFAULT_MAX_RAW_CHARS,
    max_failures: int = DEFAULT_MAX_FAILURES,
    title: str | None = None,
) -> dict[str, Any]:
    """Assemble the JSON that drives the report."""
    case_index = {case.id: case for case in (cases or [])}

    categories: list[str] = []
    for score in result.models:
        for block in score.by_category:
            if block.category not in categories:
                categories.append(block.category)

    difficulties: list[str] = []
    for score in result.models:
        for level in score.by_difficulty:
            if level not in difficulties:
                difficulties.append(level)
    difficulties.sort()

    models: list[dict[str, Any]] = []
    for score in result.models:
        by_category = {block.category: block for block in score.by_category}
        models.append(
            {
                "name": score.model,
                "provider": score.provider,
                "total": score.total,
                "pass_at_1": score.pass_at_1,
                "heal_at_1": score.heal_at_1,
                "pass_at_1_rate": _percent(score.pass_at_1_rate),
                "heal_at_1_rate": _percent(score.heal_at_1_rate),
                "repair_rate": _percent(score.repair_rate),
                "failed_at_1": score.failed_at_1,
                "healed": score.healed,
                "p50_ms": score.latency.p50,
                "p95_ms": score.latency.p95,
                "ttft_p50_ms": score.ttft.p50,
                "tps": score.tps,
                "total_tokens": score.total_tokens,
                "provider_failures": score.provider_failures,
                "unusable": score.unusable,
                "error_distribution": {
                    label.value: score.error_distribution.get(label.value, 0)
                    for label in ALL_LABELS
                },
                "detail_distribution": [
                    {"detail": detail, "count": count, "gloss": DETAIL_GLOSS.get(detail, "")}
                    for detail, count in list(score.detail_distribution.items())[:12]
                ],
                "by_category": {
                    category: _percent(
                        by_category[category].pass_at_1_rate if category in by_category else None
                    )
                    for category in categories
                },
                "by_difficulty": {
                    level: _percent(
                        score.by_difficulty[level].pass_at_1_rate
                        if level in score.by_difficulty
                        else None
                    )
                    for level in difficulties
                },
                "repair_by_category": {
                    category: _percent(
                        by_category[category].repair_rate if category in by_category else None
                    )
                    for category in categories
                },
            }
        )

    failures: list[dict[str, Any]] = []
    for score in result.models:
        for record in result.records_for(score.model):
            if record.passed_at_1:
                continue
            reference = (
                _case_reference(case_index[record.case_id], max_raw_chars)
                if record.case_id in case_index
                else {"id": record.case_id, "title": record.title, "prompt": "", "expected": {}}
            )
            failures.append(
                {
                    "model": score.model,
                    "case_id": record.case_id,
                    "title": record.title,
                    "category": record.category,
                    "difficulty": record.difficulty,
                    "healed": record.healed,
                    "passed_after_heal": record.passed_after_heal,
                    "rounds": record.rounds_used,
                    "first_label": record.first_label.value if record.first_label else None,
                    "first_detail": record.first_detail,
                    "first_detail_gloss": DETAIL_GLOSS.get(record.first_detail, ""),
                    "reference": reference,
                    "attempts": [
                        _attempt_view(attempt, max_raw_chars) for attempt in record.attempts
                    ],
                }
            )
            if len(failures) >= max_failures:
                break
        if len(failures) >= max_failures:
            break

    failures.sort(key=lambda item: (item["category"], -item["difficulty"], item["case_id"]))

    used_details = sorted(
        {detail for model in models for detail, _ in _dist_items(model["detail_distribution"])}
        | {item["first_detail"] for item in failures}
    )

    return {
        "title": title or "StructBench report",
        "version": result.structbench_version,
        "created_at": result.created_at,
        "case_count": result.case_count,
        "concurrency": result.concurrency,
        "max_rounds": result.max_rounds,
        "datasets": list(result.datasets),
        "meta": result.meta,
        "categories": categories,
        "difficulties": difficulties,
        "labels": [
            {
                "key": label.value,
                "zh": LABEL_ZH[label],
                "failure": label is not ErrorLabel.VALID_PASS,
            }
            for label in ALL_LABELS
        ],
        "failure_labels": [label.value for label in FAILURE_LABELS],
        "detail_gloss": {detail: DETAIL_GLOSS.get(detail, "") for detail in used_details},
        "models": models,
        "failures": failures,
        "failure_count": len(failures),
    }


def _dist_items(value: Any) -> list[tuple[str, int]]:
    if isinstance(value, list):
        return [(item["detail"], item["count"]) for item in value]
    return []


def echarts_library_text(path: str | Path | None = None) -> str:
    """Read the JavaScript to inline into the report.

    Defaults to the copy vendored inside the package, so a report is offline-capable
    without the caller doing anything.
    """
    library_path = Path(path) if path is not None else VENDORED_ECHARTS
    if not library_path.is_file():
        raise FileNotFoundError(
            f"echarts bundle not found: {library_path}. Pass --echarts-src URL to load "
            "it from a CDN instead, or --inline-echarts PATH to supply your own build."
        )
    return library_path.read_text(encoding="utf-8")


def render_html_report(
    result: RunResult,
    *,
    cases: Sequence[Case] | None = None,
    title: str | None = None,
    echarts_src: str | None = None,
    inline_echarts: str | Path | None = None,
    offline: bool = True,
    max_raw_chars: int = DEFAULT_MAX_RAW_CHARS,
    max_failures: int = DEFAULT_MAX_FAILURES,
) -> str:
    """Render the report to a string.

    The library is inlined by default (``offline=True``), which is the behaviour a
    downloaded artifact needs. Pass ``echarts_src="https://..."`` for a smaller file
    that requires network access, or ``offline=False`` for the same thing.
    """
    payload = build_payload(
        result,
        cases=cases,
        max_raw_chars=max_raw_chars,
        max_failures=max_failures,
        title=title,
    )

    echarts_library = ""
    remote_src = echarts_src or ""

    if inline_echarts is not None:
        # An explicit local build always wins.
        echarts_library = echarts_library_text(inline_echarts)
        remote_src = ""
    elif offline and not echarts_src:
        echarts_library = echarts_library_text()
        remote_src = ""
    elif not remote_src:
        # offline=False without an explicit URL: fall back to the CDN.
        remote_src = DEFAULT_ECHARTS_SRC

    # ``</script>`` inside the JSON would terminate the host script block early.
    data_json = json.dumps(payload, ensure_ascii=False, default=str).replace("<", "\\u003c")

    template = _environment().get_template("template.html")
    return template.render(
        data_json=data_json,
        title=payload["title"],
        echarts_src=remote_src,
        echarts_inline=echarts_library,
        meta=payload,
    )


def write_html_report(
    result: RunResult,
    path: str | Path,
    **kwargs: Any,
) -> Path:
    """Render and write the report, creating parent directories as needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html_report(result, **kwargs), encoding="utf-8")
    return target
