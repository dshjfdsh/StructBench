"""Tests for model-spec parsing and the OpenAI-compatible transport.

The transport is exercised through ``httpx.MockTransport`` (no network, no API
key), which covers the parts of the client that can actually be wrong: SSE
assembly, TTFT measurement, tool-call delta merging, retry policy and the way
transport failures are reported.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from structbench.evaluator import compile_case, evaluate
from structbench.models import ErrorLabel, Message, ToolDef, parse_model_spec
from structbench.providers import ProviderConfigError, ProviderOptions, ProviderRequest
from structbench.providers.base import requires_api_key, resolve_api_key
from structbench.providers.mock import DEFAULT_WEIGHTS, MockProvider, Persona
from structbench.providers.openai_compat import OpenAICompatProvider
from tests import _data


def sse(*chunks: dict[str, Any]) -> bytes:
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return (body + "data: [DONE]\n\n").encode("utf-8")


def provider_with(
    handler,
    *,
    model: str = "deepseek:deepseek-chat",
    options: ProviderOptions | None = None,
) -> OpenAICompatProvider:
    spec = parse_model_spec(model)
    opts = options or ProviderOptions(stream=True, api_key="test-key")
    opts.transport = httpx.MockTransport(handler)
    opts.api_key = opts.api_key or "test-key"
    return OpenAICompatProvider(spec, opts)


def run(provider: OpenAICompatProvider, request: ProviderRequest | None = None):
    async def scenario():
        try:
            return await provider.complete(
                request or ProviderRequest(messages=[Message(role="user", content="hi")])
            )
        finally:
            await provider.aclose()

    return asyncio.run(scenario())


# --------------------------------------------------------------------------- #
# Model specs
# --------------------------------------------------------------------------- #


class TestModelSpec:
    def test_explicit_provider_prefix(self) -> None:
        spec = parse_model_spec("deepseek:deepseek-chat")
        assert (spec.provider, spec.model) == ("deepseek", "deepseek-chat")
        assert spec.display == "deepseek/deepseek-chat"
        assert spec.api_key_env == "STRUCTBENCH_DEEPSEEK_API_KEY"

    def test_ollama_tag_keeps_its_colon(self) -> None:
        spec = parse_model_spec("ollama:qwen2.5:7b")
        assert spec.provider == "ollama"
        assert spec.model == "qwen2.5:7b"
        assert spec.display == "ollama/qwen2.5:7b"
        assert spec.is_mock is False

    def test_mock_persona_is_carried_as_the_model_name(self) -> None:
        spec = parse_model_spec("mock:flaky")
        assert spec.provider == "mock" and spec.model == "flaky"
        assert spec.is_mock is True
        assert parse_model_spec("mock").model == "good"

    @pytest.mark.parametrize(
        ("raw", "provider"),
        [
            ("deepseek-chat", "deepseek"),
            ("qwen-plus", "dashscope"),
            ("glm-4-flash", "zhipu"),
            ("kimi-k2", "moonshot"),
            ("gpt-4o-mini", "openai"),
            ("llama3.1:8b", "ollama"),
        ],
    )
    def test_bare_names_are_inferred(self, raw: str, provider: str) -> None:
        assert parse_model_spec(raw).provider == provider

    def test_unknown_prefix_falls_back_to_a_custom_endpoint(self) -> None:
        spec = parse_model_spec("mybox:local-model")
        assert spec.provider == "mybox"
        assert spec.model == "local-model"
        assert spec.base_url is not None

    def test_empty_spec_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty model spec"):
            parse_model_spec("   ")


class TestCredentials:
    def test_explicit_key_wins(self) -> None:
        spec = parse_model_spec("deepseek:deepseek-chat")
        assert resolve_api_key(spec, "explicit") == "explicit"

    def test_env_var_is_used(self, monkeypatch) -> None:
        spec = parse_model_spec("deepseek:deepseek-chat")
        monkeypatch.setenv("STRUCTBENCH_DEEPSEEK_API_KEY", " from-env ")
        assert resolve_api_key(spec, None) == "from-env"

    def test_provider_alias_is_accepted(self, monkeypatch) -> None:
        spec = parse_model_spec("deepseek:deepseek-chat")
        monkeypatch.delenv("STRUCTBENCH_DEEPSEEK_API_KEY", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "alias")
        assert resolve_api_key(spec, None) == "alias"

    def test_missing_key_returns_none(self, monkeypatch) -> None:
        spec = parse_model_spec("deepseek:deepseek-chat")
        monkeypatch.delenv("STRUCTBENCH_DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        assert resolve_api_key(spec, None) is None

    def test_local_endpoints_need_no_key(self) -> None:
        assert requires_api_key(parse_model_spec("ollama:qwen2.5:7b")) is False
        assert requires_api_key(parse_model_spec("custom:m"), "http://localhost:8000/v1") is False
        assert requires_api_key(parse_model_spec("custom:m"), "http://127.0.0.1:1234/v1") is False
        assert requires_api_key(parse_model_spec("deepseek:deepseek-chat")) is True


# --------------------------------------------------------------------------- #
# Request construction
# --------------------------------------------------------------------------- #


class TestRequestShape:
    def test_missing_key_is_a_config_error(self, monkeypatch) -> None:
        monkeypatch.delenv("STRUCTBENCH_DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(ProviderConfigError, match="no API key"):
            OpenAICompatProvider(parse_model_spec("deepseek:deepseek-chat"), ProviderOptions())

    def test_tools_and_json_mode_are_encoded(self) -> None:
        captured: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]},
            )

        options = ProviderOptions(stream=False, api_key="k")
        provider = provider_with(handler, options=options)
        request = ProviderRequest(
            messages=[Message(role="user", content="weather?")],
            tools=[ToolDef(name="get_weather", description="d", parameters={"type": "object"})],
            json_mode=True,
            temperature=0.0,
            max_tokens=64,
        )
        run(provider, request)

        payload = captured[0]
        assert payload["model"] == "deepseek-chat"
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["tool_choice"] == "auto"
        assert payload["tools"][0]["function"]["name"] == "get_weather"
        assert payload["temperature"] == 0.0
        assert payload["max_tokens"] == 64
        assert "stream" not in payload

    def test_bearer_header_is_sent(self) -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(request.headers)
            return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

        provider_with(handler, options=ProviderOptions(stream=False, api_key="secret"))
        run(provider_with(handler, options=ProviderOptions(stream=False, api_key="secret")))
        assert seen["authorization"] == "Bearer secret"


# --------------------------------------------------------------------------- #
# Blocking responses
# --------------------------------------------------------------------------- #


class TestBlocking:
    def test_content_and_usage_are_parsed(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": '{"a": 1}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7},
                },
            )

        completion = run(provider_with(handler, options=ProviderOptions(stream=False, api_key="k")))
        assert completion.text == '{"a": 1}'
        assert completion.prompt_tokens == 11
        assert completion.completion_tokens == 7
        assert completion.total_tokens == 18
        # TTFT is unknowable without streaming, so it must be reported as unknown.
        assert completion.ttft_ms is None

    def test_tool_calls_are_parsed(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"city": "Beijing"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            )

        completion = run(provider_with(handler, options=ProviderOptions(stream=False, api_key="k")))
        assert len(completion.tool_calls) == 1
        call = completion.tool_calls[0]
        assert call.name == "get_weather"
        assert json.loads(call.arguments_raw) == {"city": "Beijing"}

    def test_pre_parsed_arguments_are_re_serialised(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "c",
                                        "function": {"name": "f", "arguments": {"a": 1}},
                                    }
                                ]
                            }
                        }
                    ]
                },
            )

        call = run(
            provider_with(handler, options=ProviderOptions(stream=False, api_key="k"))
        ).tool_calls[0]
        assert json.loads(call.arguments_raw) == {"a": 1}

    def test_non_json_body_is_a_provider_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>gateway</html>")

        completion = run(provider_with(handler, options=ProviderOptions(stream=False, api_key="k")))
        assert completion.ok is False
        assert "non-JSON body" in completion.provider_error


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


class TestStreaming:
    def test_text_and_ttft_are_captured(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse(
                    {"choices": [{"index": 0, "delta": {"content": "Hel"}}]},
                    {"choices": [{"index": 0, "delta": {"content": "lo"}}]},
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
                ),
            )

        completion = run(provider_with(handler))
        assert completion.text == "Hello"
        assert completion.ttft_ms is not None
        assert completion.ttft_ms <= completion.latency_ms
        assert completion.completion_tokens == 2

    def test_streamed_tool_call_deltas_are_merged(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "function": {
                                                "name": "get_weather",
                                                "arguments": '{"ci',
                                            },
                                        }
                                    ]
                                },
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {"index": 0, "function": {"arguments": 'ty": "Beijing"}'}}
                                    ]
                                },
                            }
                        ]
                    },
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
                ),
            )

        completion = run(provider_with(handler))
        assert completion.text == ""
        call = completion.tool_calls[0]
        assert call.name == "get_weather"
        assert json.loads(call.arguments_raw) == {"city": "Beijing"}

    def test_usage_flag_is_requested_by_default(self) -> None:
        captured: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse())

        run(provider_with(handler))
        assert captured[0]["stream"] is True
        assert captured[0]["stream_options"] == {"include_usage": True}


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #


class TestFailures:
    def test_client_error_is_not_retried(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(400, text="bad request: unknown model")

        completion = run(provider_with(handler))
        assert completion.ok is False
        assert "HTTP 400" in completion.provider_error
        assert calls["n"] == 1

    def test_stream_options_rejection_is_retried_without_the_flag(self) -> None:
        payloads: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            payloads.append(payload)
            if "stream_options" in payload:
                return httpx.Response(400, text="Unrecognized request argument: stream_options")
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse({"choices": [{"index": 0, "delta": {"content": "ok"}}]}),
            )

        completion = run(provider_with(handler))
        assert completion.ok is True
        assert completion.text == "ok"
        assert len(payloads) == 2
        assert "stream_options" not in payloads[1]

    def test_rate_limit_is_retried_honouring_retry_after(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, headers={"retry-after": "0"}, text="slow down")
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse({"choices": [{"index": 0, "delta": {"content": "ok"}}]}),
            )

        completion = run(
            provider_with(handler, options=ProviderOptions(api_key="k", transport_retries=2))
        )
        assert completion.ok is True
        assert calls["n"] == 2

    def test_server_errors_are_retried_then_reported(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503, headers={"retry-after": "0"}, text="unavailable")

        completion = run(
            provider_with(handler, options=ProviderOptions(api_key="k", transport_retries=1))
        )
        assert completion.ok is False
        assert "HTTP 503" in completion.provider_error
        assert calls["n"] == 2  # initial attempt + one retry


# --------------------------------------------------------------------------- #
# Offline simulator
# --------------------------------------------------------------------------- #


def forced_persona(kind: str, **kwargs) -> Persona:
    """A persona that only ever produces one corruption."""
    weights = dict.fromkeys(DEFAULT_WEIGHTS, 0.0)
    weights[kind] = 1.0
    return Persona(
        name=f"only_{kind}",
        pass_rate=kwargs.pop("pass_rate", 0.0),
        heal_rate=kwargs.pop("heal_rate", 0.0),
        failure_weights=weights,
        **kwargs,
    )


def mock_attempt(case, kind: str, spec: str = "mock:good"):
    provider = MockProvider(
        parse_model_spec(spec), ProviderOptions(seed=7), persona=forced_persona(kind)
    )
    request = ProviderRequest(
        messages=[Message(role="user", content=case.prompt)],
        tools=case.tools,
        case=case,
        round_index=1,
    )
    completion = asyncio.run(provider.complete(request))
    return evaluate(compile_case(case), completion)


class TestMockContract:
    """The simulator must produce output the real evaluator attributes as intended."""

    @pytest.mark.parametrize(
        ("kind", "label", "detail"),
        [
            ("fence_leak", ErrorLabel.MARKDOWN_WRAPPER_LEAK, "fence_leak"),
            ("prose_padding", ErrorLabel.MARKDOWN_WRAPPER_LEAK, "prose_padding"),
            ("unclosed_bracket", ErrorLabel.SYNTAX_TRUNCATED, "unclosed_bracket"),
            ("type_mismatch", ErrorLabel.SCHEMA_TYPE_MISMATCH, "type_mismatch"),
            ("missing_field", ErrorLabel.SCHEMA_TYPE_MISMATCH, "missing_field"),
            ("extra_field", ErrorLabel.ENUM_OR_VALUE_HALLUCINATION, "extra_field"),
        ],
    )
    def test_json_corruptions_map_to_the_intended_verdict(
        self, kind: str, label: ErrorLabel, detail: str
    ) -> None:
        case = _data.nested_case()
        attempt = mock_attempt(case, kind)
        assert attempt.passed is False
        assert attempt.label is label, f"{kind} produced {attempt.label}/{attempt.detail}"
        assert attempt.detail == detail

    def test_tool_corruptions_map_to_the_intended_verdict(self) -> None:
        case = _data.tool_call_case()
        assert mock_attempt(case, "wrong_tool").detail in {"wrong_tool", "undeclared_tool"}
        assert mock_attempt(case, "no_tool_call").detail == "no_tool_call"
        assert mock_attempt(case, "tool_call_as_text").detail == "tool_call_as_text"
        assert mock_attempt(case, "extra_field").detail == "extra_field"

    def test_clarify_corruption_calls_the_tool_anyway(self) -> None:
        case = _data.clarify_case()
        attempt = mock_attempt(case, "unrequested_tool_call")
        assert attempt.label is ErrorLabel.ENUM_OR_VALUE_HALLUCINATION
        assert attempt.detail == "unrequested_tool_call"

    def test_perfect_persona_always_produces_a_passing_response(self) -> None:
        for case in (_data.nested_case(), _data.tool_call_case(), _data.clarify_case()):
            provider = MockProvider(
                parse_model_spec("mock:perfect"), ProviderOptions(seed=1), persona="perfect"
            )
            request = ProviderRequest(
                messages=[Message(role="user", content=case.prompt)],
                tools=case.tools,
                case=case,
                round_index=1,
            )
            completion = asyncio.run(provider.complete(request))
            assert evaluate(compile_case(case), completion).passed is True, case.id

    def test_the_simulator_is_deterministic_for_a_given_seed(self) -> None:
        case = _data.nested_case()

        def sample() -> str:
            provider = MockProvider(
                parse_model_spec("mock:good"), ProviderOptions(seed=99), persona="flaky"
            )
            request = ProviderRequest(
                messages=[Message(role="user", content=case.prompt)], case=case, round_index=1
            )
            return asyncio.run(provider.complete(request)).text

        assert sample() == sample()

    def test_metrics_are_simulated_not_measured(self) -> None:
        case = _data.nested_case()
        provider = MockProvider(
            parse_model_spec("mock:perfect"), ProviderOptions(seed=3), persona="perfect"
        )
        completion = asyncio.run(
            provider.complete(
                ProviderRequest(messages=[Message(role="user", content="x")], case=case)
            )
        )
        assert completion.latency_ms > 0
        assert completion.ttft_ms is not None
        assert completion.completion_tokens
