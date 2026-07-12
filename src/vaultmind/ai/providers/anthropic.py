"""Anthropic/Claude provider."""

from __future__ import annotations

import anthropic
import structlog
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from vaultmind.ai.providers.base import TRANSIENT_FAILURES, FailureKind

log = structlog.get_logger()


def classify_anthropic_failure(exc: Exception) -> FailureKind:
    """Map Anthropic SDK failures to safe retry/failover categories."""
    if isinstance(exc, anthropic.APITimeoutError):
        return FailureKind.TIMEOUT
    if isinstance(exc, anthropic.APIConnectionError):
        return FailureKind.CONNECTION
    if isinstance(exc, anthropic.RateLimitError):
        return FailureKind.RATE_LIMIT
    if isinstance(exc, anthropic.AuthenticationError):
        return FailureKind.AUTHENTICATION
    if isinstance(exc, anthropic.PermissionDeniedError):
        return FailureKind.PERMISSION
    if isinstance(exc, anthropic.APIStatusError):
        if exc.status_code == 408:
            return FailureKind.TIMEOUT
        return FailureKind.SERVER if exc.status_code >= 500 else FailureKind.REQUEST
    return FailureKind.UNEXPECTED


def is_retryable_anthropic_failure(exc: BaseException) -> bool:
    """Retry only transient Anthropic SDK failures."""
    return isinstance(exc, Exception) and classify_anthropic_failure(exc) in TRANSIENT_FAILURES


class AnthropicProvider:
    """Anthropic Claude provider using the official SDK."""

    name = "anthropic"

    def __init__(self, api_key: str, model: str, max_tokens: int = 2000) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=0)

    classify_failure = staticmethod(classify_anthropic_failure)

    @staticmethod
    def is_retryable_failure(exc: Exception) -> bool:
        return is_retryable_anthropic_failure(exc)

    @retry(
        retry=retry_if_exception(is_retryable_anthropic_failure),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def complete(self, prompt: str, system: str = "") -> str:
        """Send a prompt to Claude and return the completion text."""
        log.info("anthropic_request", provider=self.name, model=self.model)

        message = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system if system else "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )

        text = message.content[0].text  # type: ignore[union-attr]
        log.info(
            "anthropic_response",
            provider=self.name,
            model=self.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )
        return text
