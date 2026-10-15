"""OpenAI-compatible chat-completions client.

Every mainstream endpoint StructBench targets (DeepSeek, DashScope/Qwen, Zhipu
GLM, Moonshot, SiliconFlow, OpenAI, Ollama, vLLM, LM Studio, ...) speaks this
protocol, so one client covers them all.

Two details are benchmark-critical:

* **Streaming is the default**, because TTFT (time to first token) can only be
  measured while streaming, and TTFT is half of the "is this model usable behind
  an interactive agent?" question.
* **Transport failures are returned, not raised.** A 503 from the provider is not
  a model pathology, so it must not be attributed to the five-way taxonomy and
  must not abort the run; it comes back as ``Completion.provider_error``.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx

from structbench.models import Completion, ToolCall
from structbench.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderOptions,
    ProviderRequest,
    requires_api_key,
    resolve_api_key,
)

__all__ = ["OpenAICompatProvider"]

_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


class _HttpFailure(Exception):
    """A non-2xx provider response."""

    def __init__(self, status: int, body: str, retry_after: float | None = None) -> None:
        super().__init__(f"HTTP {status}: {_shorten(body)}")
        self.status = status
        self.body = body
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.status in _RETRYABLE_STATUS


def _shorten(text: str, limit: int = 400) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[:limit] + "..."


def _parse_retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return min(float(raw), 30.0)
    except ValueError:
        return None


class OpenAICompatProvider(Provider):
    """POST ``{base_url}/chat/completions`` and normalise the response."""

    def __init__(self, spec, options: ProviderOptions | None = None) -> None:
        super().__init__(spec, options)
        resolved_base = self.options.base_url or self.spec.base_url or ""
        if not resolved_base:
            raise ProviderConfigError(
                f"no base URL for provider {self.spec.provider!r}; pass --base-url"
            )
        self._base_url = resolved_base.rstrip("/")
        self._api_key = resolve_api_key(self.spec, self.options.api_key)
        if requires_api_key(self.spec, self._base_url) and not self._api_key:
            env_name = self.spec.api_key_env or "STRUCTBENCH_API_KEY"
            raise ProviderConfigError(
                f"no API key for {self.spec.display}. Export {env_name} "
                f"(see .env.example) or pass --api-key."
            )
        self._client: httpx.AsyncClient | None = None

    # -- plumbing ---------------------------------------------------------- #

    @property
    def endpoint(self) -> str:
        return f"{self._base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.options.extra_headers}
        if self._api_key:
            headers.setdefault("Authorization", f"Bearer {self._api_key}")
        return headers

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.options.timeout_s, connect=15.0),
                follow_redirects=True,
                transport=self.options.transport,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- request building -------------------------------------------------- #

    def _payload(self, request: ProviderRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "messages": [message.model_dump() for message in request.messages],
        }

        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.parameters or {"type": "object", "properties": {}},
                    },
                }
                for tool in request.tools
            ]
            payload["tool_choice"] = request.tool_choice or "auto"

        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}

        temperature = request.temperature
        if temperature is None:
            temperature = self.options.temperature
        if temperature is not None:
            payload["temperature"] = temperature

        max_tokens = request.max_tokens
        if max_tokens is None:
            max_tokens = self.options.max_tokens
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if self.options.stream:
            payload["stream"] = True
            if self.options.request_usage_in_stream:
                payload["stream_options"] = {"include_usage": True}

        return payload

    # -- entry point ------------------------------------------------------- #

    async def complete(self, request: ProviderRequest) -> Completion:
        payload = self._payload(request)
        deadline_retries = max(self.options.transport_retries, 0)
        last_error: str | None = None

        for attempt in range(deadline_retries + 1):
            try:
                if self.options.stream:
                    return await self._send_streaming(payload)
                return await self._send_blocking(payload)
            except _HttpFailure as failure:
                if "stream_options" in failure.body and "stream_options" in payload:
                    # Older/partial OpenAI-compatible servers reject the usage
                    # flag; drop it and retry immediately, once.
                    payload = {k: v for k, v in payload.items() if k != "stream_options"}
                    continue
                last_error = str(failure)
                if not failure.retryable or attempt >= deadline_retries:
                    break
                # An explicit ``Retry-After: 0`` means "retry immediately", so it
                # must not be replaced by the exponential fallback.
                delay = (
                    failure.retry_after
                    if failure.retry_after is not None
                    else min(2.0**attempt, 8.0)
                )
                await asyncio.sleep(delay)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= deadline_retries:
                    break
                await asyncio.sleep(min(2.0**attempt, 8.0))

        return Completion(provider_error=last_error or "request failed")

    # -- transport --------------------------------------------------------- #

    async def _send_blocking(self, payload: dict[str, Any]) -> Completion:
        client = self._http()
        started = time.perf_counter()
        response = await client.post(self.endpoint, json=payload, headers=self._headers())
        latency_ms = (time.perf_counter() - started) * 1000.0

        if response.status_code >= 400:
            raise _HttpFailure(response.status_code, response.text, _parse_retry_after(response))

        try:
            body = response.json()
        except json.JSONDecodeError:
            return Completion(
                provider_error=f"provider returned non-JSON body: {_shorten(response.text)}",
                latency_ms=latency_ms,
            )

        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = body.get("usage") or {}

        return Completion(
            text=message.get("content") or "",
            tool_calls=_parse_tool_calls(message.get("tool_calls")),
            latency_ms=latency_ms,
            # Without streaming we cannot observe the first token, so TTFT is
            # deliberately left unknown rather than guessed.
            ttft_ms=None,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
            raw=body,
        )

    async def _send_streaming(self, payload: dict[str, Any]) -> Completion:
        client = self._http()
        started = time.perf_counter()
        first_token_at: float | None = None
        text_parts: list[str] = []
        tool_slots: dict[int, dict[str, str]] = {}
        usage: dict[str, Any] = {}
        finish_reason: str | None = None

        async with client.stream(
            "POST", self.endpoint, json=payload, headers=self._headers()
        ) as response:
            if response.status_code >= 400:
                raw = (await response.aread()).decode("utf-8", "replace")
                raise _HttpFailure(response.status_code, raw, _parse_retry_after(response))

            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if chunk.get("usage"):
                    usage = chunk["usage"]

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

                delta = choice.get("delta") or choice.get("message") or {}
                content = delta.get("content")
                if content:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    text_parts.append(content)

                for call in delta.get("tool_calls") or []:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    index = call.get("index", 0)
                    slot = tool_slots.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    if call.get("id"):
                        slot["id"] = call["id"]
                    function = call.get("function") or {}
                    if function.get("name"):
                        slot["name"] += function["name"]
                    if function.get("arguments"):
                        slot["arguments"] += function["arguments"]

        latency_ms = (time.perf_counter() - started) * 1000.0
        ttft_ms = (first_token_at - started) * 1000.0 if first_token_at else None

        return Completion(
            text="".join(text_parts),
            tool_calls=[
                ToolCall(
                    id=slot["id"] or None,
                    name=slot["name"],
                    arguments_raw=slot["arguments"],
                )
                for _, slot in sorted(tool_slots.items())
                if slot["name"] or slot["arguments"]
            ],
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=finish_reason,
        )


def _parse_tool_calls(raw: Any) -> list[ToolCall]:
    """Normalise the non-streaming ``tool_calls`` array."""
    calls: list[ToolCall] = []
    if not isinstance(raw, list):
        return calls
    for item in raw:
        if not isinstance(item, dict):
            continue
        function = item.get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, (dict, list)):
            # A few gateways pre-parse the arguments for us.
            arguments_raw = json.dumps(arguments)
            parsed: Any = arguments
        else:
            arguments_raw = arguments or ""
            parsed = None
        calls.append(
            ToolCall(
                id=item.get("id"),
                name=function.get("name") or "",
                arguments_raw=arguments_raw,
                arguments=parsed,
            )
        )
    return calls
