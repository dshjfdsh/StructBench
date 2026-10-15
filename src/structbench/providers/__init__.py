"""Provider registry.

One factory decides which concrete client a ``--models`` entry maps to. Anything
that is not the offline simulator is assumed to speak the OpenAI-compatible
chat-completions protocol, which covers every endpoint StructBench targets.
"""

from __future__ import annotations

from structbench.models import ModelSpec
from structbench.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderError,
    ProviderOptions,
    ProviderRequest,
    requires_api_key,
    resolve_api_key,
)
from structbench.providers.mock import PERSONA_NAMES, PERSONAS, MockProvider, Persona
from structbench.providers.openai_compat import OpenAICompatProvider

__all__ = [
    "PERSONAS",
    "PERSONA_NAMES",
    "MockProvider",
    "OpenAICompatProvider",
    "Persona",
    "Provider",
    "ProviderConfigError",
    "ProviderError",
    "ProviderOptions",
    "ProviderRequest",
    "build_provider",
    "requires_api_key",
    "resolve_api_key",
]


def build_provider(spec: ModelSpec, options: ProviderOptions | None = None) -> Provider:
    """Instantiate the provider for ``spec``.

    Raises :class:`ProviderConfigError` when the target cannot be used as
    configured (an unknown persona, a missing API key, no base URL).
    """
    options = options or ProviderOptions()
    if spec.provider == "mock":
        return MockProvider(spec, options)
    return OpenAICompatProvider(spec, options)
