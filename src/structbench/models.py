"""Core data models shared by every StructBench subsystem.

Everything that crosses a module boundary -- task cases, provider responses,
per-attempt verdicts, aggregate scores and serialised run archives -- is a
Pydantic model defined here, so dataset files, JSON result archives and report
inputs are all validated by the same source of truth.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------- #
# Error taxonomy
# --------------------------------------------------------------------------- #


class ErrorLabel(str, enum.Enum):
    """The five-way pathology taxonomy that StructBench attributes failures to.

    This is the headline feature of the project: instead of a blunt
    ``pass/fail`` bit, every failed attempt is attributed to exactly one of the
    following root causes, which is what makes the resulting leaderboard
    actionable.
    """

    #: Output *was* valid JSON but the model wrapped it in a ```json fence or
    #: padded it with conversational filler, i.e. it did not follow the
    #: "raw JSON only" contract.
    MARKDOWN_WRAPPER_LEAK = "Markdown_Wrapper_Leak"
    #: The payload could not be parsed at all: unbalanced brackets, an
    #: unterminated string, a bad escape sequence, a trailing comma, or no JSON
    #: at all.
    SYNTAX_TRUNCATED = "Syntax_Truncated"
    #: Syntactically valid JSON that violates the schema's shape: a missing
    #: required field, a wrong primitive type, a pattern/range violation.
    SCHEMA_TYPE_MISMATCH = "Schema_Type_Mismatch"
    #: Valid JSON of the right shape, but the *values* are invented: an enum or
    #: const was violated, an undeclared extra field was added, or the model
    #: fabricated a tool argument the user never supplied.
    ENUM_OR_VALUE_HALLUCINATION = "Enum_Or_Value_Hallucination"
    #: Fully compliant output.
    VALID_PASS = "Valid_Pass"


#: The failure-producing subset of :class:`ErrorLabel`, in a stable order that
#: reports and charts rely on.
FAILURE_LABELS: tuple[ErrorLabel, ...] = (
    ErrorLabel.MARKDOWN_WRAPPER_LEAK,
    ErrorLabel.SYNTAX_TRUNCATED,
    ErrorLabel.SCHEMA_TYPE_MISMATCH,
    ErrorLabel.ENUM_OR_VALUE_HALLUCINATION,
)

#: Every label, including the success label.
ALL_LABELS: tuple[ErrorLabel, ...] = (*FAILURE_LABELS, ErrorLabel.VALID_PASS)

#: Human-readable Chinese gloss used by the console reporter.
LABEL_ZH: dict[ErrorLabel, str] = {
    ErrorLabel.MARKDOWN_WRAPPER_LEAK: "Markdown 包裹泄漏",
    ErrorLabel.SYNTAX_TRUNCATED: "语法残缺/转义错误",
    ErrorLabel.SCHEMA_TYPE_MISMATCH: "Schema 类型不匹配",
    ErrorLabel.ENUM_OR_VALUE_HALLUCINATION: "枚举/参数幻觉",
    ErrorLabel.VALID_PASS: "合规通过",
}


class AttemptStatus(str, enum.Enum):
    """Whether an attempt produced a measurable verdict at all."""

    OK = "ok"
    #: Transport/HTTP/provider-side failure. Recorded separately from the model
    #: pathology taxonomy so infrastructure noise never pollutes the
    #: error-distribution charts.
    PROVIDER_ERROR = "provider_error"


# --------------------------------------------------------------------------- #
# Task definition (the declarative dataset)
# --------------------------------------------------------------------------- #

Behaviour = Literal["json", "call", "clarify", "null_value"]


class ToolDef(BaseModel):
    """An OpenAI-style tool definition handed to the model under test."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class GroundTruth(BaseModel):
    """Value-level assertions applied on top of structural schema validation.

    Structural validation alone cannot catch a model that fills every field with
    a plausible-looking lie, so cases may pin concrete values or forbid entire
    subtrees.
    """

    model_config = ConfigDict(extra="forbid")

    #: JSONPath-lite -> exact expected value.
    must_equal: dict[str, Any] = Field(default_factory=dict)
    #: JSONPath-lite locations that must be present and explicitly ``null``
    #: (the "refuse to invent" contract).
    must_be_null: list[str] = Field(default_factory=list)
    #: JSONPath-lite -> substring that must appear in the string value.
    must_contain: dict[str, str] = Field(default_factory=dict)
    #: JSONPath-lite -> substring that must *not* appear (leak detection).
    must_not_contain: dict[str, str] = Field(default_factory=dict)
    #: JSONPath-lite -> regular expression the string value must fully match.
    must_match: dict[str, str] = Field(default_factory=dict)
    #: JSONPath-lite locations that must not exist at all -- used to catch a
    #: model inventing a whole field (e.g. ``$.api_key``) that was never asked
    #: for.
    forbidden_paths: list[str] = Field(default_factory=list)


class Expectation(BaseModel):
    """The behavioural contract a case checks beyond schema conformance."""

    model_config = ConfigDict(extra="forbid")

    #: ``json``      -- emit JSON matching ``target_schema``.
    #: ``call``      -- emit a tool call to ``expected_tool`` with valid args.
    #: ``clarify``   -- must NOT call a tool; it must ask for the missing input.
    #: ``null_value``-- emit JSON, leaving the ambiguous fields explicitly null.
    behaviour: Behaviour = "json"
    #: For ``call``: the tool that should be invoked.
    expected_tool: str | None = None


class Case(BaseModel):
    """A single declarative boundary test case."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    title: str = ""
    category: str
    tags: list[str] = Field(default_factory=list)
    #: 1 = easy, 5 = adversarial. Used for difficulty-weighted reporting.
    difficulty: int = Field(default=3, ge=1, le=5)
    #: ``json`` evaluates the assistant text; ``tool`` evaluates tool calls.
    mode: Literal["json", "tool"] | None = None
    prompt: str
    system: str | None = None
    target_schema: dict[str, Any] | None = Field(default=None, alias="target_schema")
    tools: list[ToolDef] = Field(default_factory=list)
    expect: Expectation = Field(default_factory=Expectation)
    ground_truth: GroundTruth = Field(default_factory=GroundTruth)
    #: Total attempts allowed, i.e. 1 initial + (max_rounds - 1) healing rounds.
    max_rounds: int = Field(default=2, ge=1, le=5)
    #: By default StructBench enforces the "raw JSON only" contract and flags a
    #: ```json fence or any surrounding prose as ``Markdown_Wrapper_Leak``. Set
    #: this for cases whose prompt explicitly tolerates a fenced block.
    allow_wrapper: bool = False
    #: Why this case is evil -- shown in failure diffs, kept for maintainers.
    notes: str = ""
    #: Populated by the loader with the originating dataset file name.
    source: str | None = None

    @model_validator(mode="after")
    def _infer_mode(self) -> Case:
        if self.mode is None:
            object.__setattr__(self, "mode", "tool" if self.tools else "json")
        return self

    @model_validator(mode="after")
    def _check_contract(self) -> Case:
        if self.mode == "json" and self.target_schema is None:
            raise ValueError(f"case {self.id!r}: mode 'json' requires target_schema")
        if self.mode == "tool" and not self.tools:
            raise ValueError(f"case {self.id!r}: mode 'tool' requires at least one tool")
        if self.expect.behaviour == "call":
            if self.mode != "tool":
                raise ValueError(f"case {self.id!r}: behaviour 'call' requires mode 'tool'")
            if not self.expect.expected_tool:
                raise ValueError(f"case {self.id!r}: behaviour 'call' requires expected_tool")
            names = {t.name for t in self.tools}
            if self.expect.expected_tool not in names:
                raise ValueError(
                    f"case {self.id!r}: expected_tool {self.expect.expected_tool!r} "
                    f"is not among the declared tools {sorted(names)}"
                )
        if self.expect.behaviour == "clarify" and self.mode != "tool":
            raise ValueError(f"case {self.id!r}: behaviour 'clarify' requires mode 'tool'")
        return self

    @property
    def is_tool(self) -> bool:
        return self.mode == "tool"

    def tool_by_name(self, name: str | None) -> ToolDef | None:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None


class Dataset(BaseModel):
    """A parsed dataset file (or the merge of several)."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    category: str | None = None
    description: str = ""
    #: Optional per-file override of the default attempt budget.
    max_rounds: int | None = None
    cases: list[Case] = Field(default_factory=list)

    @field_validator("cases")
    @classmethod
    def _unique_ids(cls, cases: list[Case]) -> list[Case]:
        seen: set[str] = set()
        dupes: list[str] = []
        for case in cases:
            if case.id in seen:
                dupes.append(case.id)
            seen.add(case.id)
        if dupes:
            raise ValueError(f"duplicate case ids: {sorted(set(dupes))}")
        return cases


# --------------------------------------------------------------------------- #
# Provider I/O
# --------------------------------------------------------------------------- #


class ToolCall(BaseModel):
    """A single tool call requested by the model."""

    id: str | None = None
    name: str
    #: Raw string exactly as returned by the provider (may well be broken JSON
    #: -- that is precisely what we are measuring).
    arguments_raw: str = ""
    #: Parsed arguments when they happen to be valid JSON, else ``None``.
    arguments: Any | None = None


class Completion(BaseModel):
    """One provider round-trip, with the system-level metrics we collect."""

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    #: Wall-clock latency of the whole request, milliseconds.
    latency_ms: float = 0.0
    #: Time to first streamed token, milliseconds (``None`` for non-streaming).
    ttft_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    #: Set when the request itself failed; the taxonomy labels are then unused.
    provider_error: str | None = None
    #: Optional raw provider payload, dropped before archiving to keep result
    #: files small.
    raw: dict[str, Any] | None = Field(default=None, exclude=True)

    @property
    def ok(self) -> bool:
        return self.provider_error is None

    @property
    def total_tokens(self) -> int:
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)

    @property
    def tps(self) -> float | None:
        """Completion tokens per second, measured from first token."""
        if not self.completion_tokens:
            return None
        window_ms = self.latency_ms - (self.ttft_ms or 0.0)
        if window_ms <= 0:
            return None
        return self.completion_tokens / (window_ms / 1000.0)


class Message(BaseModel):
    """A chat message as sent to the provider."""

    role: Literal["system", "user", "assistant"]
    content: str


# --------------------------------------------------------------------------- #
# Verdicts and results
# --------------------------------------------------------------------------- #


class FieldIssue(BaseModel):
    """One attributed problem inside a single attempt."""

    path: str = "$"
    message: str = ""
    #: Native pydantic error type (``missing``, ``enum``, ``extra_forbidden``...).
    kind: str = ""
    label: ErrorLabel = ErrorLabel.SCHEMA_TYPE_MISMATCH
    #: Fine-grained, chart-friendly sub-code, e.g. ``unclosed_bracket``.
    detail: str = ""


class Attempt(BaseModel):
    """A single model round-trip plus its deterministic verdict."""

    round: int = 1
    status: AttemptStatus = AttemptStatus.OK
    raw_text: str = ""
    #: The slice of ``raw_text`` actually submitted to the JSON parser, after
    #: fence stripping and prose trimming.
    payload_text: str = ""
    #: Detected output-wrapper violation: ``fence``, ``prose`` or ``fence+prose``.
    wrapper: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    #: Parsed payload, when parsing succeeded *or* was recovered via
    #: ``ast.literal_eval`` (Python-literal output). Diagnostics only.
    parsed: Any | None = None
    #: ``None`` when the attempt never reached the taxonomy (transport failure).
    label: ErrorLabel | None = None
    detail: str = ""
    passed: bool = False
    issues: list[FieldIssue] = Field(default_factory=list)
    #: Human-readable feedback that was (or would be) replayed to the model.
    feedback: str = ""
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    provider_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is AttemptStatus.OK


class CaseResult(BaseModel):
    """The full record for one case under one model."""

    case_id: str
    category: str
    model: str
    title: str = ""
    difficulty: int = 3
    attempts: list[Attempt] = Field(default_factory=list)
    #: Did attempt #1 pass?
    passed_at_1: bool = False
    #: Did any attempt pass (the healing-inclusive score)?
    passed_after_heal: bool = False
    #: Failed attempt #1 but a later attempt passed -- the Heal@1 numerator.
    healed: bool = False
    #: Label of the first attempt -- the model's raw, unaided behaviour.
    first_label: ErrorLabel | None = None
    first_detail: str = ""
    #: Label of the final attempt.
    final_label: ErrorLabel | None = None
    final_detail: str = ""
    total_latency_ms: float = 0.0
    total_tokens: int = 0

    @property
    def rounds_used(self) -> int:
        return len(self.attempts)

    @property
    def needs_heal(self) -> bool:
        return not self.passed_at_1


# --------------------------------------------------------------------------- #
# Aggregates
# --------------------------------------------------------------------------- #


class CategoryScore(BaseModel):
    """Per-dimension scorecard."""

    category: str
    total: int = 0
    pass_at_1: int = 0
    heal_at_1: int = 0
    failed_at_1: int = 0
    healed: int = 0
    pass_at_1_rate: float = 0.0
    heal_at_1_rate: float = 0.0
    #: ``healed / failed_at_1`` -- "of the cases it broke, how many did it fix?".
    #: ``None`` when nothing failed, because the ratio is then undefined.
    repair_rate: float | None = None


class LatencyStats(BaseModel):
    mean: float | None = None
    p50: float | None = None
    p95: float | None = None
    min: float | None = None
    max: float | None = None
    samples: int = 0


class ModelScore(BaseModel):
    """The complete scorecard for one model under test."""

    model: str
    provider: str = ""
    total: int = 0
    pass_at_1: int = 0
    heal_at_1: int = 0
    failed_at_1: int = 0
    healed: int = 0
    pass_at_1_rate: float = 0.0
    heal_at_1_rate: float = 0.0
    #: ``healed / failed_at_1``; ``None`` when nothing failed.
    repair_rate: float | None = None
    provider_failures: int = 0
    #: True when every single request failed -- the run is not interpretable.
    unusable: bool = False
    latency: LatencyStats = Field(default_factory=LatencyStats)
    ttft: LatencyStats = Field(default_factory=LatencyStats)
    tps: float | None = None
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: ``ErrorLabel`` value -> count, over *first* attempts only.
    error_distribution: dict[str, int] = Field(default_factory=dict)
    #: Fine-grained detail code -> count, over first attempts only.
    detail_distribution: dict[str, int] = Field(default_factory=dict)
    by_category: list[CategoryScore] = Field(default_factory=list)
    by_difficulty: dict[str, CategoryScore] = Field(default_factory=dict)


class RunResult(BaseModel):
    """A complete, serialisable benchmark run archive."""

    structbench_version: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    datasets: list[str] = Field(default_factory=list)
    case_count: int = 0
    concurrency: int = 1
    max_rounds: int = 2
    models: list[ModelScore] = Field(default_factory=list)
    records: list[CaseResult] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    def score_for(self, model: str) -> ModelScore | None:
        for score in self.models:
            if score.model == model:
                return score
        return None

    def records_for(self, model: str) -> list[CaseResult]:
        return [r for r in self.records if r.model == model]


# --------------------------------------------------------------------------- #
# Model specs
# --------------------------------------------------------------------------- #

#: Preset providers. ``env`` is the environment variable holding the API key and
#: ``base_url_env`` the optional override for the endpoint.
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "base_url_env": "STRUCTBENCH_DEEPSEEK_BASE_URL",
        "api_key_env": "STRUCTBENCH_DEEPSEEK_API_KEY",
        "label": "DeepSeek",
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "base_url_env": "STRUCTBENCH_DASHSCOPE_BASE_URL",
        "api_key_env": "STRUCTBENCH_DASHSCOPE_API_KEY",
        "label": "Alibaba Qwen",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "base_url_env": "STRUCTBENCH_ZHIPU_BASE_URL",
        "api_key_env": "STRUCTBENCH_ZHIPU_API_KEY",
        "label": "Zhipu GLM",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "base_url_env": "STRUCTBENCH_MOONSHOT_BASE_URL",
        "api_key_env": "STRUCTBENCH_MOONSHOT_API_KEY",
        "label": "Moonshot",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "base_url_env": "STRUCTBENCH_SILICONFLOW_BASE_URL",
        "api_key_env": "STRUCTBENCH_SILICONFLOW_API_KEY",
        "label": "SiliconFlow",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "base_url_env": "STRUCTBENCH_OPENAI_BASE_URL",
        "api_key_env": "STRUCTBENCH_OPENAI_API_KEY",
        "label": "OpenAI",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "base_url_env": "STRUCTBENCH_OLLAMA_BASE_URL",
        "api_key_env": "STRUCTBENCH_OLLAMA_API_KEY",
        "label": "Ollama (local)",
    },
    "custom": {
        "base_url": "http://localhost:8000/v1",
        "base_url_env": "STRUCTBENCH_CUSTOM_BASE_URL",
        "api_key_env": "STRUCTBENCH_CUSTOM_API_KEY",
        "label": "Custom endpoint",
    },
}

#: Heuristics used when a bare model name is given without a ``provider:`` prefix.
_MODEL_PREFIX_HINTS: tuple[tuple[str, str], ...] = (
    ("deepseek", "deepseek"),
    ("qwen", "dashscope"),
    ("qwq", "dashscope"),
    ("glm", "zhipu"),
    ("moonshot", "moonshot"),
    ("kimi", "moonshot"),
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("claude", "custom"),
    ("llama", "ollama"),
    ("mistral", "ollama"),
    ("gemma", "ollama"),
    ("phi", "ollama"),
)


class ModelSpec(BaseModel):
    """A fully resolved ``provider:model`` target."""

    model_config = ConfigDict(frozen=True)

    raw: str
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    label: str = ""

    @property
    def display(self) -> str:
        """``deepseek/deepseek-chat`` -- the name used in reports."""
        if self.label:
            return self.label
        return f"{self.provider}/{self.model}"

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock"


def _matches_family_hint(name: str) -> bool:
    lowered = name.lower()
    return any(lowered.startswith(prefix) for prefix, _ in _MODEL_PREFIX_HINTS)


def parse_model_spec(raw: str) -> ModelSpec:
    """Resolve a CLI model string into a :class:`ModelSpec`.

    Accepted forms::

        mock                       # offline simulator, default persona
        mock:good                  # offline simulator, named persona
        deepseek:deepseek-chat     # explicit provider
        ollama:qwen2.5:7b          # known provider, tag colon preserved
        llama3.1:8b                # bare Ollama tag
        mybox:local-model          # unknown prefix -> custom OpenAI-compatible endpoint
        gpt-4o-mini                # inferred from the model-name prefix

    A colon is ambiguous -- it separates a provider from a model, but it also
    appears inside Ollama tags such as ``llama3.1:8b``. The rule is: a known
    provider (or ``mock``) always wins; otherwise, if the part before the colon
    starts with a recognised model family, the whole string is a model name;
    otherwise the part before the colon names a custom endpoint.
    """
    text = raw.strip()
    if not text:
        raise ValueError("empty model spec")

    provider = ""
    model = text
    head, separator, tail = text.partition(":")
    head_key = head.strip().lower()

    if separator:
        if head_key == "mock" or head_key in PROVIDER_PRESETS:
            provider, model = head_key, tail.strip()
        elif _matches_family_hint(head_key):
            provider, model = "", text  # e.g. "llama3.1:8b" is an Ollama tag
        else:
            provider, model = head_key, tail.strip()

    if provider == "mock" or (not provider and model.lower() == "mock"):
        persona = (model or "good") if provider == "mock" else "good"
        return ModelSpec(raw=raw, provider="mock", model=persona, label=f"mock/{persona}")

    if provider in PROVIDER_PRESETS:
        return ModelSpec(
            raw=raw,
            provider=provider,
            model=model,
            base_url=PROVIDER_PRESETS[provider]["base_url"],
            api_key_env=PROVIDER_PRESETS[provider]["api_key_env"],
        )

    if provider:
        # Unknown provider prefix: treat it as a self-hosted OpenAI-compatible
        # endpoint so users are never blocked by our preset table.
        return ModelSpec(
            raw=raw,
            provider=provider,
            model=model,
            base_url=PROVIDER_PRESETS["custom"]["base_url"],
            api_key_env=PROVIDER_PRESETS["custom"]["api_key_env"],
        )

    # No prefix: infer from the model family.
    lowered = model.lower()
    for prefix, inferred in _MODEL_PREFIX_HINTS:
        if lowered.startswith(prefix):
            return ModelSpec(
                raw=raw,
                provider=inferred,
                model=model,
                base_url=PROVIDER_PRESETS[inferred]["base_url"],
                api_key_env=PROVIDER_PRESETS[inferred]["api_key_env"],
            )

    return ModelSpec(
        raw=raw,
        provider="openai",
        model=model,
        base_url=PROVIDER_PRESETS["openai"]["base_url"],
        api_key_env=PROVIDER_PRESETS["openai"]["api_key_env"],
    )


__all__ = [
    "ALL_LABELS",
    "FAILURE_LABELS",
    "LABEL_ZH",
    "PROVIDER_PRESETS",
    "Attempt",
    "AttemptStatus",
    "Behaviour",
    "Case",
    "CaseResult",
    "CategoryScore",
    "Completion",
    "Dataset",
    "ErrorLabel",
    "Expectation",
    "FieldIssue",
    "GroundTruth",
    "LatencyStats",
    "Message",
    "ModelScore",
    "ModelSpec",
    "RunResult",
    "ToolCall",
    "ToolDef",
    "parse_model_spec",
]
