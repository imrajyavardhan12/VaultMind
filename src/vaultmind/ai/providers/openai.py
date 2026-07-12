"""OpenAI/GPT provider."""

from __future__ import annotations

import openai
import structlog
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from vaultmind.ai.providers.base import TRANSIENT_FAILURES, FailureKind

log = structlog.get_logger()


def classify_openai_failure(exc: Exception) -> FailureKind:
    """Map OpenAI SDK failures to safe retry/failover categories."""
    if isinstance(exc, openai.APITimeoutError):
        return FailureKind.TIMEOUT
    if isinstance(exc, openai.APIConnectionError):
        return FailureKind.CONNECTION
    if isinstance(exc, openai.RateLimitError):
        return FailureKind.RATE_LIMIT
    if isinstance(exc, openai.AuthenticationError):
        return FailureKind.AUTHENTICATION
    if isinstance(exc, openai.PermissionDeniedError):
        return FailureKind.PERMISSION
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code == 408:
            return FailureKind.TIMEOUT
        return FailureKind.SERVER if exc.status_code >= 500 else FailureKind.REQUEST
    return FailureKind.UNEXPECTED


def is_retryable_openai_failure(exc: BaseException) -> bool:
    """Retry only transient OpenAI SDK failures."""
    return isinstance(exc, Exception) and classify_openai_failure(exc) in TRANSIENT_FAILURES


class OpenAIProvider:
    """OpenAI GPT provider using the official SDK."""

    name = "openai"

    def __init__(self, api_key: str, model: str, max_tokens: int = 2000) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = AsyncOpenAI(api_key=api_key, max_retries=0)

    classify_failure = staticmethod(classify_openai_failure)

    @staticmethod
    def is_retryable_failure(exc: Exception) -> bool:
        return is_retryable_openai_failure(exc)

    @retry(
        retry=retry_if_exception(is_retryable_openai_failure),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def complete(self, prompt: str, system: str = "") -> str:
        """Send a prompt to OpenAI and return the completion text."""
        log.info("openai_request", provider=self.name, model=self.model)

        response = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system or "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )

        text = response.choices[0].message.content or ""
        log.info(
            "openai_response",
            provider=self.name,
            model=self.model,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )
        return text
