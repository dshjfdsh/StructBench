"""Deterministic offline simulator.

The MockProvider exists so that StructBench is fully usable with **zero API keys**:
it drives the entire pipeline (prompting, extraction, validation, attribution,
healing, scoring, reporting) against simulated models with known, reproducible
behaviour. That makes the benchmark demoable in the README, testable in CI, and
comparable across machines.

How it works: for each request it synthesises the *ideal* response from the case
schema (see :mod:`structbench.synth`), then deliberately corrupts it according to
a persona's failure profile. Crucially it never labels its own output -- the real
evaluator judges it. So a mock run exercises the genuine attribution path.

Personas are reproducible: every choice is drawn from a ``random.Random`` seeded
by ``(--seed, model, case id, round)``, never from Python's salted ``hash()``.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from typing import Any

from structbench import jsonpath as jp
from structbench.models import Case, Completion, ToolCall
from structbench.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderOptions,
    ProviderRequest,
)
from structbench.schema_builder import merge_all_of, resolve_ref
from structbench.synth import (
    SynthError,
    find_schema_path,
    schema_at_path,
    synth_value,
    synthesize_case_value,
)

__all__ = [
    "DEFAULT_WEIGHTS",
    "PERSONAS",
    "MockProvider",
    "Persona",
]

#: A question that satisfies a ``clarify`` case.
_CLARIFY_TEXT = (
    "I need one more detail before I can proceed: which date should I use? "
    "Please provide the missing information."
)

#: Relative likelihood of each corruption when a simulated model does fail.
DEFAULT_WEIGHTS: dict[str, float] = {
    # wrapper
    "fence_leak": 2.0,
    "prose_padding": 1.6,
    # syntax
    "unclosed_bracket": 2.4,
    "bad_escape": 1.6,
    "trailing_comma": 1.2,
    "python_literal": 1.0,
    "empty_output": 0.4,
    "no_json_found": 0.6,
    # schema
    "type_mismatch": 3.0,
    "missing_field": 2.4,
    "enum_or_const_violation": 1.8,
    "extra_field": 1.4,
    # values
    "fabricated_missing_value": 1.0,
    "ground_truth_mismatch": 0.8,
    # tool calls
    "wrong_tool": 1.6,
    "undeclared_tool": 0.6,
    "tool_call_as_text": 1.0,
    "no_tool_call": 1.2,
    "unrequested_tool_call": 2.0,
    "silent_proceed": 1.0,
    "no_clarification": 0.8,
}


@dataclass(frozen=True)
class Persona:
    """A simulated model's personality: how often it fails, and how."""

    name: str
    description: str = ""
    #: Probability that attempt #1 is fully compliant.
    pass_rate: float = 0.9
    #: Probability that a *healing* round (round >= 2) succeeds.
    heal_rate: float = 0.9
    #: Corruption kind -> relative weight.
    failure_weights: dict[str, float] = field(default_factory=dict)
    #: Category -> pass-rate override, for "good at JSON, terrible at escapes".
    category_pass_rate: dict[str, float] = field(default_factory=dict)
    #: Simulated per-request base latency, milliseconds.
    latency_base_ms: float = 150.0
    #: Simulated per-output-character latency, milliseconds.
    latency_per_char_ms: float = 0.8
    #: Fraction of total latency that elapses before the first token.
    ttft_fraction: float = 0.35

    def weights(self) -> dict[str, float]:
        merged = dict(DEFAULT_WEIGHTS)
        merged.update(self.failure_weights)
        return merged


PERSONAS: dict[str, Persona] = {
    "perfect": Persona(
        name="perfect",
        description="Never fails — the upper bound, useful for smoke tests.",
        pass_rate=1.0,
        heal_rate=1.0,
        latency_base_ms=120.0,
        latency_per_char_ms=0.55,
    ),
    "good": Persona(
        name="good",
        description="Strong model: ~90% first-try, ~90% self-heals.",
        pass_rate=0.90,
        heal_rate=0.90,
        latency_base_ms=150.0,
        latency_per_char_ms=0.75,
    ),
    "flaky": Persona(
        name="flaky",
        description="Mid-tier model: ~62% first-try, weak recovery.",
        pass_rate=0.62,
        heal_rate=0.70,
        failure_weights={"fence_leak": 3.0, "unclosed_bracket": 3.0, "missing_field": 3.0},
        latency_base_ms=240.0,
        latency_per_char_ms=1.1,
    ),
    "weak": Persona(
        name="weak",
        description="Small model: ~35% first-try, poor recovery.",
        pass_rate=0.35,
        heal_rate=0.45,
        failure_weights={
            "unclosed_bracket": 3.0,
            "bad_escape": 2.5,
            "type_mismatch": 3.0,
            "python_literal": 2.0,
            "no_json_found": 1.5,
        },
        latency_base_ms=420.0,
        latency_per_char_ms=1.9,
    ),
    "escape_phobic": Persona(
        name="escape_phobic",
        description="Fine on plain JSON, collapses on multi-language escaping.",
        pass_rate=0.92,
        heal_rate=0.80,
        category_pass_rate={"escape_hell": 0.30},
        failure_weights={
            "bad_escape": 6.0,
            "unclosed_bracket": 4.0,
            "trailing_comma": 2.0,
        },
        latency_base_ms=180.0,
        latency_per_char_ms=0.9,
    ),
    "typing_weak": Persona(
        name="typing_weak",
        description="Nails syntax, muddles the type system and enums.",
        pass_rate=0.88,
        heal_rate=0.72,
        category_pass_rate={"edge_types": 0.50, "deep_nesting": 0.68},
        failure_weights={
            "type_mismatch": 6.0,
            "enum_or_const_violation": 4.0,
            "missing_field": 3.0,
            "extra_field": 2.0,
        },
        latency_base_ms=200.0,
        latency_per_char_ms=0.85,
    ),
    "tool_butcher": Persona(
        name="tool_butcher",
        description="Reliable JSON, unreliable tool selection and arguments.",
        pass_rate=0.86,
        heal_rate=0.65,
        category_pass_rate={"tool_hallucination": 0.45},
        failure_weights={
            "wrong_tool": 4.0,
            "extra_field": 3.0,
            "unrequested_tool_call": 4.0,
            "missing_field": 2.5,
            "type_mismatch": 2.0,
        },
        latency_base_ms=170.0,
        latency_per_char_ms=0.8,
    ),
}

PERSONA_NAMES: tuple[str, ...] = tuple(PERSONAS)


# --------------------------------------------------------------------------- #
# Text-level corruption helpers
# --------------------------------------------------------------------------- #


def _drop_closing(text: str, count: int = 1) -> str:
    """Chop trailing closing brackets -- the classic truncation failure."""
    out = text.rstrip()
    for _ in range(max(count, 1)):
        if out.endswith(("}", "]")):
            out = out[:-1]
        else:
            break
    return out


def _insert_bad_escape(text: str) -> str | None:
    """Inject an illegal ``\\q`` escape inside the first string value."""
    marker = text.find(': "')
    if marker >= 0:
        position = marker + 3
        return f"{text[:position]}\\q{text[position:]}"
    if '"' in text:
        position = text.find('"') + 1
        return f"{text[:position]}\\q{text[position:]}"
    return None


def _insert_trailing_comma(text: str) -> str:
    stripped = text.rstrip()
    if stripped.endswith("}"):
        return stripped[:-1] + ",}"
    if stripped.endswith("]"):
        return stripped[:-1] + ",]"
    return stripped + ","


def _different_value(value: Any) -> Any:
    """A plausible-looking but wrong value of the same JSON type."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return round(value + 1.0, 6)
    if isinstance(value, str):
        return value + "-x"
    if value is None:
        return "unknown"
    return value


def _invent_value(sub: dict[str, Any]) -> Any:
    """A value that violates an ``enum``/``const`` while keeping the type."""
    if "const" in sub:
        return _different_value(sub["const"])
    options = sub.get("enum") or []
    template = options[0] if options else "value"
    if isinstance(template, bool):
        return not template
    if isinstance(template, int):
        candidate = (max(options) if options else 0) + 7
        return candidate
    if isinstance(template, float):
        return round((max(options) if options else 0.0) + 7.0, 6)
    if isinstance(template, str):
        candidate = "invented_value"
        while candidate in options:
            candidate += "_"
        return candidate
    return "invented_value"


def _is_numeric_schema(sub: dict[str, Any], _prefix: tuple[str | int, ...]) -> bool:
    declared = sub.get("type")
    names = declared if isinstance(declared, list) else [declared]
    return any(name in ("integer", "number") for name in names if name)


def _has_enum_schema(sub: dict[str, Any], _prefix: tuple[str | int, ...]) -> bool:
    return "enum" in sub or "const" in sub


# --------------------------------------------------------------------------- #
# Token estimation (only used to simulate metrics; never for scoring)
# --------------------------------------------------------------------------- #


def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 ASCII chars per token, ~1 per CJK character."""
    if not text:
        return 0
    wide = sum(1 for char in text if ord(char) > 0x2E80)
    narrow = len(text) - wide
    return max(1, wide + narrow // 4)


# --------------------------------------------------------------------------- #
# The provider
# --------------------------------------------------------------------------- #


class MockProvider(Provider):
    """A reproducible, offline stand-in for a real chat model."""

    def __init__(
        self,
        spec,
        options: ProviderOptions | None = None,
        persona: str | Persona | None = None,
    ) -> None:
        super().__init__(spec, options)
        chosen = persona if persona is not None else (spec.model or "good")
        if isinstance(chosen, Persona):
            self.persona = chosen
        else:
            key = str(chosen).strip().lower()
            if key not in PERSONAS:
                raise ProviderConfigError(
                    f"unknown mock persona {chosen!r}; available: {', '.join(PERSONA_NAMES)}"
                )
            self.persona = PERSONAS[key]

    # -- entry point ------------------------------------------------------- #

    async def complete(self, request: ProviderRequest) -> Completion:
        persona = self.persona
        case = request.case
        rng = random.Random(
            f"{self.options.seed}:{persona.name}:{getattr(case, 'id', '-')}:{request.round_index}"
        )

        if case is None:  # pragma: no cover - the runner always supplies a case
            return self._finalise(request, '{"ok": true}', [], rng)

        try:
            text, calls = self._simulate(case, request.round_index, rng)
        except SynthError as exc:
            return Completion(
                provider_error=f"mock cannot synthesise this case: {exc}",
                latency_ms=0.0,
            )
        except Exception as exc:
            return Completion(
                provider_error=f"mock simulation failed: {type(exc).__name__}: {exc}",
                latency_ms=0.0,
            )

        return self._finalise(request, text, calls, rng)

    # -- simulation -------------------------------------------------------- #

    def _simulate(
        self, case: Case, round_index: int, rng: random.Random
    ) -> tuple[str, list[ToolCall]]:
        persona = self.persona
        pass_rate = persona.category_pass_rate.get(case.category, persona.pass_rate)
        if round_index > 1:
            # A healing round models a different skill from first-try accuracy.
            pass_rate = persona.heal_rate

        if rng.random() < pass_rate:
            return self._ideal(case)

        kind = self._pick_kind(case, rng)
        return self._corrupt(case, kind, rng)

    def _ideal(self, case: Case) -> tuple[str, list[ToolCall]]:
        if case.is_tool:
            if case.expect.behaviour != "call":
                return _CLARIFY_TEXT, []
            value = synthesize_case_value(case)
            return "", [self._call(case.expect.expected_tool, value)]
        return json.dumps(synthesize_case_value(case)), []

    def _call(self, name: str | None, value: Any) -> ToolCall:
        return ToolCall(
            id="call_0",
            name=str(name),
            arguments_raw=json.dumps(value),
            arguments=value,
        )

    # -- corruption selection --------------------------------------------- #

    def _applicable_kinds(self, case: Case) -> list[str]:
        if case.is_tool and case.expect.behaviour == "clarify":
            return ["unrequested_tool_call", "silent_proceed", "no_clarification"]

        kinds = [
            "fence_leak",
            "prose_padding",
            "unclosed_bracket",
            "trailing_comma",
            "python_literal",
            "type_mismatch",
            "missing_field",
            "extra_field",
            "enum_or_const_violation",
            "empty_output",
            "no_json_found",
        ]
        if case.is_tool:
            kinds = [
                *kinds,
                "wrong_tool",
                "undeclared_tool",
                "tool_call_as_text",
                "no_tool_call",
            ]
        if case.ground_truth.must_be_null:
            kinds.append("fabricated_missing_value")
        if case.ground_truth.must_equal:
            kinds.append("ground_truth_mismatch")
        return kinds

    def _pick_kind(self, case: Case, rng: random.Random) -> str:
        weights = self.persona.weights()
        kinds = [kind for kind in self._applicable_kinds(case) if weights.get(kind, 0) > 0]
        if not kinds:
            return "type_mismatch"
        return rng.choices(kinds, weights=[weights[kind] for kind in kinds], k=1)[0]

    def _corrupt(self, case: Case, kind: str, rng: random.Random) -> tuple[str, list[ToolCall]]:
        if case.is_tool:
            return self._corrupt_tool(case, kind, rng)
        value = synthesize_case_value(case)
        return self._corrupt_payload(kind, value, case.target_schema, case, rng), []

    def _corrupt_tool(
        self, case: Case, kind: str, rng: random.Random
    ) -> tuple[str, list[ToolCall]]:
        if case.expect.behaviour == "clarify":
            tool = case.tools[0]
            args = synth_value(tool.parameters, tool.parameters)
            if kind == "no_clarification":
                return "Sure, let me take a look at that for you.", []
            if kind == "silent_proceed":
                return json.dumps(args), []
            return "", [self._call(tool.name, args)]

        expected = case.expect.expected_tool
        tool = case.tool_by_name(expected)
        root = tool.parameters if tool is not None else {}
        value = synthesize_case_value(case)

        if kind == "wrong_tool":
            others = [item for item in case.tools if item.name != expected]
            if others:
                return "", [self._call(others[0].name, value)]
            return "", []
        if kind == "undeclared_tool":
            return "", [self._call("internal_debug_tool", value)]
        if kind == "tool_call_as_text":
            return json.dumps(value), []
        if kind == "no_tool_call":
            return "I have looked that up and noted it down for you.", []

        body = self._corrupt_payload(kind, value, root, case, rng)
        return "", [ToolCall(id="call_0", name=str(expected), arguments_raw=body, arguments=None)]

    # -- the corruption catalogue ----------------------------------------- #

    def _corrupt_payload(
        self,
        kind: str,
        value: Any,
        root: Any,
        case: Case,
        rng: random.Random,
    ) -> str:
        text = json.dumps(value)

        # ---- text-level corruptions ------------------------------------- #
        if kind == "fence_leak":
            return f"```json\n{text}\n```"
        if kind == "prose_padding":
            return (
                "Sure! Here is the JSON you asked for:\n"
                f"{text}\n"
                "Let me know if you need any changes."
            )
        if kind == "unclosed_bracket":
            return _drop_closing(text, rng.randint(1, 3))
        if kind == "bad_escape":
            broken = _insert_bad_escape(text)
            return broken if broken is not None else _drop_closing(text)
        if kind == "trailing_comma":
            return _insert_trailing_comma(text)
        if kind == "python_literal":
            return repr(value)
        if kind == "empty_output":
            return ""
        if kind == "no_json_found":
            return "I am sorry, but I cannot help with that request."

        # ---- value-level corruptions ------------------------------------ #
        mutated = self._mutate(kind, value, root, case)
        if mutated is not None:
            return json.dumps(mutated)

        # Fall back to a corruption that always applies.
        return _drop_closing(text, 1)

    def _mutate(self, kind: str, value: Any, root: Any, case: Case) -> Any | None:
        if not isinstance(root, dict):
            return None

        if kind == "type_mismatch":
            path = find_schema_path(root, root, predicate=_is_numeric_schema)
            if path:
                return jp.set_(copy.deepcopy(value), list(path), "x")
            if isinstance(value, dict) and value:
                first = next(iter(value))
                return jp.set_(copy.deepcopy(value), [first], "x")
            return None

        flat = _flatten_root(root)
        if kind == "missing_field":
            required = flat.get("required") or []
            if required and isinstance(value, dict) and required[0] in value:
                mutated = copy.deepcopy(value)
                mutated.pop(required[0])
                return mutated
            return None

        if kind == "enum_or_const_violation":
            path = find_schema_path(root, root, predicate=_has_enum_schema)
            if path:
                sub = schema_at_path(root, list(path), root) or {}
                return jp.set_(copy.deepcopy(value), list(path), _invent_value(sub))
            return None

        if kind == "extra_field":
            base = copy.deepcopy(value)
            if not isinstance(base, dict):
                base = {"result": base}
            base["api_key"] = "sk-live-9f2c4e71"
            return base

        if kind == "fabricated_missing_value":
            paths = case.ground_truth.must_be_null
            if paths and jp.exists(value, paths[0]):
                return jp.set_(copy.deepcopy(value), paths[0], "2024-01-02")
            return None

        if kind == "ground_truth_mismatch":
            items = list(case.ground_truth.must_equal.items())
            if items:
                path, expected = items[0]
                return jp.set_(copy.deepcopy(value), path, _different_value(expected))
            return None

        return None

    # -- metrics simulation ---------------------------------------------- #

    def _finalise(
        self,
        request: ProviderRequest,
        text: str,
        calls: list[ToolCall],
        rng: random.Random,
    ) -> Completion:
        persona = self.persona
        payload_chars = len(text) + sum(len(call.arguments_raw) for call in calls)
        jitter = 0.8 + 0.4 * rng.random()
        latency_ms = (
            persona.latency_base_ms + payload_chars * persona.latency_per_char_ms
        ) * jitter
        ttft_ms = latency_ms * persona.ttft_fraction * (0.85 + 0.3 * rng.random())

        prompt_tokens = sum(_estimate_tokens(message.content) for message in request.messages)
        completion_tokens = _estimate_tokens(text) + sum(
            _estimate_tokens(call.arguments_raw) for call in calls
        )

        return Completion(
            text=text,
            tool_calls=calls,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason="tool_calls" if calls else "stop",
        )


def _flatten_root(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve root-level ``$ref``/``allOf`` so probes see real constraints."""
    root = schema
    for _ in range(32):
        if "$ref" in schema:
            schema = resolve_ref(schema["$ref"], root)
            continue
        if "allOf" in schema:
            schema = merge_all_of(schema, root)
            continue
        return schema
    return schema
