"""Provider abstraction shared by the real HTTP client and the offline simulator."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from structbench.models import Case, Completion, Message, ModelSpec, ToolDef

__all__ = [
    "Provider",
    "ProviderConfigError",
    "ProviderError",
    "ProviderOptions",
    "ProviderRequest",
    "requires_api_key",
    "resolve_api_key",
]


class ProviderError(RuntimeError):
    """Base class for provider failures."""


class ProviderConfigError(ProviderError):
    """The provider is not usable as configured (usually a missing API key)."""


@dataclass(slots=True)
class ProviderRequest:
    """Everything a provider needs in order to answer one round of a case."""

    messages: list[Message]
    tools: list[ToolDef] = field(default_factory=list)
    #: Ask the server to constrain output to valid JSON. OFF by default: the whole
    #: point is to measure the model's *native* JSON discipline. Turn it on with
    #: ``--json-mode`` when you deliberately want the constrained comparison.
    json_mode: bool = False
    #: OpenAI-style ``tool_choice``; ``None`` means "auto".
    tool_choice: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None

    # --- StructBench-only context (used by the offline simulator) -----------
    case: Case | None = None
    round_index: int = 1
    seed: int = 0


@dataclass(slots=True)
class ProviderOptions:
    """Runtime knobs shared by all providers."""

    base_url: str | None = None
    api_key: str | None = None
    timeout_s: float = 120.0
    #: Stream the response so that TTFT can be measured. Turn off with
    #: ``--no-stream`` for endpoints that mishandle SSE.
    stream: bool = True
    #: Transport-level retries for 429/5xx. Independent of the healing loop.
    transport_retries: int = 2
    request_usage_in_stream: bool = True
    extra_headers: dict[str, str] = field(default_factory=dict)
    temperature: float | None = None
    max_tokens: int | None = None
    #: Seed for providers that simulate their answers (the offline MockProvider).
    #: Real models are not seedable through this interface.
    seed: int = 0
    #: Optional httpx transport. Production uses the default; tests inject
    #: ``httpx.MockTransport`` and deployments can inject proxies or custom TLS.
    transport: Any = None

    @property
    def resolved_temperature(self) -> float | None:
        return self.temperature


def requires_api_key(spec: ModelSpec, base_url: str | None = None) -> bool:
    """False for local endpoints (Ollama and friends), which accept any token."""
    if spec.provider == "ollama":
        return False
    candidate = (base_url or spec.base_url or "").strip()
    if not candidate:
        return True
    host = (urlparse(candidate).hostname or "").lower()
    return host not in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def resolve_api_key(spec: ModelSpec, explicit: str | None = None) -> str | None:
    """Find an API key: explicit flag first, then the provider's env var."""
    if explicit:
        return explicit
    if spec.api_key_env:
        value = os.environ.get(spec.api_key_env)
        if value:
            return value.strip()
        # Accept a couple of conventional aliases so users are not forced to
        # rename what they already have exported.
        for alias in (f"{spec.provider.upper()}_API_KEY",):
            value = os.environ.get(alias)
            if value:
                return value.strip()
    return None


class Provider(ABC):
    """A chat-completion backend that StructBench can measure."""

    def __init__(self, spec: ModelSpec, options: ProviderOptions | None = None) -> None:
        self.spec = spec
        self.options = options or ProviderOptions()

    @property
    def name(self) -> str:
        """The label used in reports."""
        return self.spec.display

    @abstractmethod
    async def complete(self, request: ProviderRequest) -> Completion:
        """Perform one round-trip. Transport problems come back as
        :attr:`Completion.provider_error` rather than as exceptions, so a single
        broken call cannot abort a whole benchmark run."""

    async def aclose(self) -> None:  # pragma: no cover - trivial
        """Release network resources."""
        return None

    async def __aenter__(self) -> Provider:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
