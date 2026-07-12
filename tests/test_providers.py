"""Tests for AI providers."""

import asyncio
import traceback

import httpx
import pytest

from vaultmind.ai.providers import FallbackProvider, get_provider
from vaultmind.ai.providers.anthropic import (
    classify_anthropic_failure,
    is_retryable_anthropic_failure,
)
from vaultmind.ai.providers.base import (
    MAX_EXHAUSTED_ERROR_MESSAGE_LENGTH,
    MAX_IDENTITY_LENGTH,
    MAX_RETAINED_ATTEMPTS,
    FailureKind,
    ProviderExhaustedError,
)
from vaultmind.ai.providers.ollama import (
    OllamaProvider,
    classify_ollama_failure,
    is_retryable_ollama_failure,
)
from vaultmind.ai.providers.openai import classify_openai_failure, is_retryable_openai_failure
from vaultmind.config import AIConfig, AppConfig, EnvSettings, ProviderConfig, ProviderModels


def test_get_anthropic_provider(test_config):
    provider = get_provider(test_config, tier="fast")
    assert provider.model == "claude-sonnet-4-20250514"


def test_get_deep_provider(test_config):
    provider = get_provider(test_config, tier="deep")
    assert provider.model == "claude-opus-4-5"


def test_get_ollama_provider_without_api_key(tmp_vault):
    config = AppConfig(
        vault_path=tmp_vault,
        ai=AIConfig(
            default_provider="ollama",
            fallback_chain=["ollama"],
            providers={
                "ollama": ProviderConfig(
                    base_url="http://localhost:11434",
                    models=ProviderModels(fast="llama3", deep="llama3:70b"),
                )
            },
        ),
        env=EnvSettings(anthropic_api_key="", openai_api_key=""),
    )

    provider = get_provider(config, tier="deep")

    assert isinstance(provider, FallbackProvider)
    assert isinstance(provider.providers[0], OllamaProvider)
    assert provider.model == "llama3:70b"


def test_ollama_provider_complete_uses_native_chat_endpoint():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        payload = httpx.Response(200, json={"message": {"content": "local answer"}})
        return payload

    transport = httpx.MockTransport(handler)
    provider = OllamaProvider(
        base_url="http://ollama.test",
        model="llama3",
        max_tokens=123,
        transport=transport,
    )

    result = asyncio.run(provider.complete("hello", system="system"))

    assert result == "local answer"


class _StubProvider:
    def __init__(
        self,
        name: str,
        model: str,
        outcomes: list[str | BaseException],
        kind: FailureKind = FailureKind.REQUEST,
    ) -> None:
        self.name = name
        self.model = model
        self.outcomes = outcomes
        self.kind = kind
        self.calls = 0

    def classify_failure(self, exc: Exception) -> FailureKind:
        del exc
        return self.kind

    def is_retryable_failure(self, exc: Exception) -> bool:
        del exc
        return self.kind in {
            FailureKind.CONNECTION,
            FailureKind.TIMEOUT,
            FailureKind.RATE_LIMIT,
            FailureKind.SERVER,
        }

    async def complete(self, prompt: str, system: str = "") -> str:
        del prompt, system
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_runtime_fallback_is_ordered_and_returns_success_unchanged():
    first = _StubProvider("first", "fast-a", [RuntimeError("permanent")])
    second = _StubProvider("second", "fast-b", ["  exact response  "])
    third = _StubProvider("third", "fast-c", ["unused"])
    provider = FallbackProvider([first, second, third])

    result = asyncio.run(provider.complete("secret prompt"))

    assert result == "  exact response  "
    assert [first.calls, second.calls, third.calls] == [1, 1, 0]
    assert provider.selected_provider == "second"
    assert provider.selected_model == "fast-b"


def test_empty_completion_advances_to_next_provider():
    first = _StubProvider("first", "a", [" \n\t"])
    second = _StubProvider("second", "b", ["answer"])
    provider = FallbackProvider([first, second])

    assert asyncio.run(provider.complete("prompt")) == "answer"
    assert first.calls == second.calls == 1


@pytest.mark.parametrize(
    "terminal",
    [asyncio.CancelledError(), KeyboardInterrupt(), SystemExit(9)],
)
def test_terminal_exceptions_are_never_swallowed(terminal):
    first = _StubProvider("first", "a", [terminal])
    second = _StubProvider("second", "b", ["must not run"])

    with pytest.raises(type(terminal)):
        asyncio.run(FallbackProvider([first, second]).complete("prompt"))

    assert second.calls == 0


def test_exhaustion_is_typed_ordered_and_sanitized():
    secret = "sk-secret-value"
    prompt = "private prompt contents"
    first = _StubProvider("first", "model-a", [RuntimeError(secret)])
    second = _StubProvider("second", "model-b", [ValueError(prompt)])

    with pytest.raises(ProviderExhaustedError) as exc_info:
        asyncio.run(FallbackProvider([first, second]).complete(prompt))

    error = exc_info.value
    assert [(a.provider, a.model) for a in error.attempts] == [
        ("first", "model-a"),
        ("second", "model-b"),
    ]
    assert error.omitted_attempts == 0
    assert secret not in str(error)
    assert prompt not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert error.__suppress_context__
    assert secret not in "".join(traceback.format_exception(error))
    assert len(str(error)) <= MAX_EXHAUSTED_ERROR_MESSAGE_LENGTH


@pytest.mark.parametrize("omitted_attempts", [-1, True, False, "12", 1.5, None])
def test_exhaustion_defensively_rejects_invalid_omitted_counts(omitted_attempts):
    error = ProviderExhaustedError((), omitted_attempts=omitted_attempts)  # type: ignore[arg-type]

    assert error.omitted_attempts == 0
    assert "attempts omitted" not in str(error)
    assert len(str(error)) <= MAX_EXHAUSTED_ERROR_MESSAGE_LENGTH


def test_exhaustion_bounds_pathologically_large_omitted_count():
    error = ProviderExhaustedError((), omitted_attempts=10**3000)
    message = str(error)

    assert 0 < error.omitted_attempts < 10**3000
    assert f"at least {error.omitted_attempts} attempts omitted" in message
    assert len(message) <= MAX_EXHAUSTED_ERROR_MESSAGE_LENGTH


def test_large_exhaustion_retains_bounded_ordered_attempts_and_reports_omissions():
    provider_count = 1_000
    providers = [
        _StubProvider(
            f"provider-{index}-" + "p" * 100,
            f"model-{index}-" + "m" * 100,
            [RuntimeError(f"secret-{index}")],
        )
        for index in range(provider_count)
    ]

    with pytest.raises(ProviderExhaustedError) as exc_info:
        asyncio.run(FallbackProvider(providers).complete("private prompt"))

    error = exc_info.value
    message = str(error)
    assert len(error.attempts) == MAX_RETAINED_ATTEMPTS
    assert all(len(attempt.provider) == MAX_IDENTITY_LENGTH for attempt in error.attempts)
    assert all(len(attempt.model) == MAX_IDENTITY_LENGTH for attempt in error.attempts)
    assert [attempt.provider.split("-")[1] for attempt in error.attempts] == [
        str(index) for index in range(MAX_RETAINED_ATTEMPTS)
    ]
    assert error.omitted_attempts == provider_count - MAX_RETAINED_ATTEMPTS
    assert message.endswith(f"{error.omitted_attempts} attempts omitted")
    assert len(message) <= MAX_EXHAUSTED_ERROR_MESSAGE_LENGTH
    assert "secret-999" not in message
    assert "private prompt" not in message
    assert error.__cause__ is None
    assert error.__context__ is None
    assert error.__suppress_context__


@pytest.mark.parametrize(
    ("status", "kind"),
    [(401, FailureKind.AUTHENTICATION), (403, FailureKind.PERMISSION)],
)
def test_ollama_auth_http_failures_do_not_retry_and_use_safe_aggregate_labels(status, kind):
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, text="secret response body", request=request)

    provider = OllamaProvider(
        base_url="http://ollama.test",
        model="local-model",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ProviderExhaustedError) as exc_info:
        asyncio.run(FallbackProvider([provider]).complete("private prompt"))

    assert calls == 1
    assert exc_info.value.attempts[0].reason is kind
    assert kind.value in str(exc_info.value)
    assert "secret response body" not in str(exc_info.value)
    assert "private prompt" not in str(exc_info.value)


def test_sdk_failure_classification_retries_only_transient_errors():
    request = httpx.Request("POST", "https://provider.test")
    server_response = httpx.Response(503, request=request)
    client_response = httpx.Response(400, request=request)
    server = httpx.HTTPStatusError("contains secret", request=request, response=server_response)
    client = httpx.HTTPStatusError("contains secret", request=request, response=client_response)

    assert classify_ollama_failure(httpx.ReadTimeout("timeout", request=request)) is FailureKind.TIMEOUT
    assert classify_ollama_failure(server) is FailureKind.SERVER
    assert classify_ollama_failure(client) is FailureKind.REQUEST
    assert is_retryable_ollama_failure(server)
    assert not is_retryable_ollama_failure(client)

    import anthropic
    import openai

    anthropic_auth = anthropic.AuthenticationError(
        "secret", response=httpx.Response(401, request=request), body=None
    )
    anthropic_server = anthropic.InternalServerError(
        "secret", response=httpx.Response(503, request=request), body=None
    )
    openai_auth = openai.AuthenticationError(
        "secret", response=httpx.Response(401, request=request), body=None
    )
    openai_server = openai.InternalServerError(
        "secret", response=httpx.Response(503, request=request), body=None
    )

    assert classify_anthropic_failure(anthropic_auth) is FailureKind.AUTHENTICATION
    assert classify_anthropic_failure(anthropic_server) is FailureKind.SERVER
    assert not is_retryable_anthropic_failure(anthropic_auth)
    assert is_retryable_anthropic_failure(anthropic_server)
    assert classify_openai_failure(openai_auth) is FailureKind.AUTHENTICATION
    assert classify_openai_failure(openai_server) is FailureKind.SERVER
    assert not is_retryable_openai_failure(openai_auth)
    assert is_retryable_openai_failure(openai_server)

    # SDK-independent unexpected exceptions must never be retried.
    assert classify_anthropic_failure(ValueError("secret")) is FailureKind.UNEXPECTED
    assert classify_openai_failure(ValueError("secret")) is FailureKind.UNEXPECTED


def test_registry_builds_available_providers_in_fallback_order(tmp_vault):
    config = AppConfig(
        vault_path=tmp_vault,
        ai=AIConfig(
            fallback_chain=["openai", "anthropic", "ollama"],
            providers={
                "openai": ProviderConfig(models=ProviderModels(fast="gpt-fast", deep="gpt-deep")),
                "anthropic": ProviderConfig(models=ProviderModels(fast="claude-fast", deep="claude-deep")),
                "ollama": ProviderConfig(
                    base_url="http://ollama.test",
                    models=ProviderModels(fast="local-fast", deep="local-deep"),
                ),
            },
        ),
        # OpenAI is deliberately excluded because its credential is absent.
        env=EnvSettings(anthropic_api_key="configured", openai_api_key=""),
    )

    provider = get_provider(config, tier="deep")

    assert isinstance(provider, FallbackProvider)
    assert [(item.name, item.model) for item in provider.providers] == [
        ("anthropic", "claude-deep"),
        ("ollama", "local-deep"),
    ]
