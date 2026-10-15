"""Tests for the asynchronous runner, the healing loop and archive round-tripping.

The offline simulator makes the whole engine testable without a network or an API
key: a persona whose ``pass_rate`` is 0 and ``heal_rate`` is 1 exercises exactly
the "failed, then repaired after being told what was wrong" path that Heal@1
claims to measure.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from structbench.healer import build_conversation, render_assistant_turn, should_heal
from structbench.models import (
    Attempt,
    AttemptStatus,
    Case,
    Completion,
    ErrorLabel,
    ModelSpec,
    RunResult,
    ToolCall,
    parse_model_spec,
)
from structbench.providers import ProviderOptions
from structbench.providers.base import Provider
from structbench.providers.mock import PERSONAS, MockProvider, Persona
from structbench.runner import RateLimiter, RunConfig, Runner, run_sync
from tests import _data


def _spec(name: str) -> ModelSpec:
    return parse_model_spec(name)


def _simple_case(case_id: str = "c1", *, max_rounds: int = 2) -> Case:
    return Case(
        id=case_id,
        category="deep_nesting",
        title="tiny",
        prompt="Return the object as JSON.",
        target_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
            "additionalProperties": False,
        },
        max_rounds=max_rounds,
    )


def _run(cases, spec, **kwargs) -> RunResult:
    config = RunConfig(
        models=[spec],
        cases=list(cases),
        options=ProviderOptions(),
        concurrency=kwargs.pop("concurrency", 4),
        **kwargs,
    )
    return run_sync(config)


# --------------------------------------------------------------------------- #
# Happy paths and the healing loop
# --------------------------------------------------------------------------- #


def test_perfect_persona_never_fails() -> None:
    result = _run([_simple_case()], _spec("mock:perfect"))
    score = result.models[0]
    assert score.pass_at_1 == 1
    assert score.pass_at_1_rate == 1.0
    assert score.heal_at_1_rate == 1.0
    # Nothing failed, so the repair ratio is undefined rather than 0% or 100%.
    assert score.repair_rate is None
    assert score.error_distribution[ErrorLabel.VALID_PASS.value] == 1


def test_healing_repairs_a_deliberate_first_round_failure() -> None:
    persona = Persona(name="always_fails_then_heals", pass_rate=0.0, heal_rate=1.0)
    provider = MockProvider(_spec("mock:good"), ProviderOptions(), persona=persona)
    case = _simple_case(max_rounds=2)

    async def scenario() -> RunResult:
        runner = Runner(RunConfig(models=[_spec("mock:good")], cases=[case]))
        runner._build_providers = lambda: {_spec("mock:good").display: provider}  # type: ignore[method-assign]
        return await runner.run()

    result = asyncio.run(scenario())
    record = result.records[0]
    assert record.passed_at_1 is False
    assert record.passed_after_heal is True
    assert record.healed is True
    assert record.rounds_used == 2

    score = result.models[0]
    assert score.pass_at_1_rate == 0.0
    assert score.heal_at_1_rate == 1.0
    assert score.repair_rate == 1.0
    assert score.failed_at_1 == 1
    assert score.healed == 1


def test_max_rounds_one_disables_healing() -> None:
    persona = Persona(name="always_fails", pass_rate=0.0, heal_rate=1.0)
    provider = MockProvider(_spec("mock:good"), ProviderOptions(), persona=persona)
    case = _simple_case(max_rounds=1)

    async def scenario() -> RunResult:
        runner = Runner(RunConfig(models=[_spec("mock:good")], cases=[case]))
        runner._build_providers = lambda: {_spec("mock:good").display: provider}  # type: ignore[method-assign]
        return await runner.run()

    record = asyncio.run(scenario()).records[0]
    assert record.rounds_used == 1
    assert record.passed_after_heal is False
    assert record.healed is False


def test_feedback_records_the_verdict_and_the_repair_instruction() -> None:
    persona = Persona(name="always_fails", pass_rate=0.0, heal_rate=0.0)
    provider = MockProvider(_spec("mock:good"), ProviderOptions(), persona=persona)
    case = _simple_case()

    async def scenario() -> RunResult:
        runner = Runner(RunConfig(models=[_spec("mock:good")], cases=[case]))
        runner._build_providers = lambda: {_spec("mock:good").display: provider}  # type: ignore[method-assign]
        return await runner.run()

    record = asyncio.run(scenario()).records[0]
    first = record.attempts[0]
    assert first.feedback.startswith("StructBench validation feedback")
    assert "StructBench validation feedback" in record.attempts[1].feedback


# --------------------------------------------------------------------------- #
# Tool-call and clarify cases
# --------------------------------------------------------------------------- #


def test_tool_case_scores_through_the_runner() -> None:
    result = _run([_data.tool_call_case()], _spec("mock:perfect"))
    record = result.records[0]
    assert record.passed_at_1 is True
    assert record.attempts[0].tool_calls[0].name == "get_weather"


def test_clarify_case_passes_when_the_model_asks() -> None:
    result = _run([_data.clarify_case()], _spec("mock:perfect"))
    assert result.records[0].passed_at_1 is True


def test_multiple_models_produce_one_scorecard_each() -> None:
    result = _run(
        [_simple_case("a"), _simple_case("b")],
        _spec("mock:perfect"),
    )
    assert len(result.models) == 1
    assert result.models[0].total == 2
    assert result.case_count == 2


# --------------------------------------------------------------------------- #
# Infrastructure behaviour
# --------------------------------------------------------------------------- #


class _RecordingProvider(Provider):
    """A provider that reports the peak number of simultaneous calls."""

    def __init__(self, spec: ModelSpec, delay: float = 0.02) -> None:
        super().__init__(spec, ProviderOptions())
        self.delay = delay
        self.in_flight = 0
        self.peak = 0
        self.calls = 0

    async def complete(self, request) -> Completion:
        self.calls += 1
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.in_flight -= 1
        return Completion(text='{"a": 1}', latency_ms=1.0)


def test_concurrency_limit_is_respected(monkeypatch) -> None:
    spec = _spec("mock:good")
    provider = _RecordingProvider(spec)
    monkeypatch.setattr("structbench.runner.build_provider", lambda *a, **k: provider)

    cases = [_simple_case(f"c{i}") for i in range(12)]
    result = run_sync(
        RunConfig(models=[spec], cases=cases, concurrency=3, options=ProviderOptions())
    )

    assert provider.calls == 12
    assert provider.peak <= 3
    assert all(record.passed_at_1 for record in result.records)


def test_provider_errors_are_recorded_and_not_healed(monkeypatch) -> None:
    class _Broken(Provider):
        async def complete(self, request) -> Completion:
            return Completion(provider_error="HTTP 503: upstream unavailable")

    spec = _spec("mock:good")
    monkeypatch.setattr(
        "structbench.runner.build_provider", lambda *a, **k: _Broken(spec, ProviderOptions())
    )
    result = run_sync(RunConfig(models=[spec], cases=[_simple_case()], options=ProviderOptions()))

    record = result.records[0]
    assert record.rounds_used == 1  # a transport failure is never "healed"
    assert record.passed_at_1 is False
    assert record.first_label is None
    assert record.attempts[0].status is AttemptStatus.PROVIDER_ERROR

    score = result.models[0]
    assert score.provider_failures == 1
    assert score.unusable is True


def test_progress_callback_fires_once_per_case() -> None:
    seen: list[tuple[str, str]] = []

    def on_done(model: str, case: Case, result) -> None:
        seen.append((model, case.id))

    _run([_simple_case("a"), _simple_case("b")], _spec("mock:perfect"), progress=on_done)
    assert sorted(case_id for _, case_id in seen) == ["a", "b"]


def test_rate_limiter_spaces_acquisitions() -> None:
    async def scenario() -> float:
        limiter = RateLimiter(20.0)  # 50ms apart
        started = time.monotonic()
        for _ in range(4):
            await limiter.acquire()
        return time.monotonic() - started

    elapsed = asyncio.run(scenario())
    assert elapsed >= 0.14  # three gaps of 50ms, with slack for timer jitter


def test_rate_limiter_rejects_non_positive_rates() -> None:
    with pytest.raises(ValueError, match="positive"):
        RateLimiter(0)


# --------------------------------------------------------------------------- #
# Archive
# --------------------------------------------------------------------------- #


def test_run_archive_round_trips_and_records_metadata() -> None:
    result = _run([_simple_case()], _spec("mock:perfect"), dataset_files=["datasets/x.yaml"])
    payload = result.model_dump_json()
    restored = RunResult.model_validate_json(payload)

    assert restored.case_count == result.case_count
    assert restored.datasets == ["datasets/x.yaml"]
    assert restored.models[0].model == "mock/perfect"
    assert restored.records[0].attempts[0].raw_text
    assert restored.meta["seed"] == 0


def test_compile_failure_is_reported_with_the_case_id() -> None:
    bad = Case(
        id="broken",
        category="x",
        prompt="p",
        target_schema={"type": "object"},
        ground_truth={"must_equal": {"$.a..b": 1}},
    )
    with pytest.raises(ValueError, match="broken"):
        Runner(RunConfig(models=[_spec("mock:perfect")], cases=[bad]))


# --------------------------------------------------------------------------- #
# Healer helpers
# --------------------------------------------------------------------------- #


def test_build_conversation_replays_prompt_and_failure() -> None:
    case = _simple_case()
    attempt = Attempt(
        round=1,
        raw_text='{"b": 1}',
        label=ErrorLabel.SCHEMA_TYPE_MISMATCH,
        detail="missing_field",
        feedback="StructBench validation feedback\n- at $.a: Field required",
    )
    messages = build_conversation(case, [attempt])
    roles = [message.role for message in messages]
    assert roles == ["user", "assistant", "user"]
    assert messages[1].content == '{"b": 1}'
    assert "Field required" in messages[2].content


def test_build_conversation_includes_the_system_prompt() -> None:
    case = _simple_case().model_copy(update={"system": "You are terse."})
    messages = build_conversation(case, [])
    assert messages[0].role == "system"
    assert messages[1].role == "user"


def test_render_assistant_turn_flattens_tool_calls() -> None:
    attempt = Attempt(
        round=1,
        tool_calls=[ToolCall(name="get_weather", arguments_raw='{"city": "Beijing"}')],
    )
    rendered = render_assistant_turn(attempt)
    assert "[tool call] get_weather(" in rendered
    assert "Beijing" in rendered


def test_should_heal_matrix() -> None:
    case = _simple_case(max_rounds=2)
    failed = Attempt(round=1, passed=False, feedback="fix it")
    passed = Attempt(round=1, passed=True, feedback="")
    broken = Attempt(
        round=1,
        passed=False,
        feedback="",
        status=AttemptStatus.PROVIDER_ERROR,
        provider_error="boom",
    )

    assert should_heal(case, failed, 1) is True
    assert should_heal(case, passed, 1) is False
    assert should_heal(case, broken, 1) is False  # never heal a transport failure
    assert should_heal(case, failed, 2) is False  # budget exhausted


def test_personas_are_all_usable() -> None:
    spec = _spec("mock:good")
    for name in PERSONAS:
        provider = MockProvider(spec, ProviderOptions(), persona=name)
        assert provider.persona.name == name


def test_unknown_persona_is_rejected() -> None:
    from structbench.providers import ProviderConfigError

    with pytest.raises(ProviderConfigError, match="unknown mock persona"):
        MockProvider(_spec("mock:good"), ProviderOptions(), persona="nope")
