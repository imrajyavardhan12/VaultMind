"""Ollama provider using the local HTTP API."""

from __future__ import annotations

from typing import Any

import httpx
import structlog
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from vaultmind.ai.providers.base import TRANSIENT_FAILURES, FailureKind

log = structlog.get_logger()


def classify_ollama_failure(exc: Exception) -> FailureKind:
    """Distinguish retryable transport/server failures from permanent HTTP errors."""
    if isinstance(exc, httpx.TimeoutException):
        return FailureKind.TIMEOUT
    if isinstance(exc, httpx.TransportError):
        return FailureKind.CONNECTION
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 401:
            return FailureKind.AUTHENTICATION
        if status == 403:
            return FailureKind.PERMISSION
        if status == 408:
            return FailureKind.TIMEOUT
        if status == 429:
            return FailureKind.RATE_LIMIT
        return FailureKind.SERVER if status >= 500 else FailureKind.REQUEST
    return FailureKind.UNEXPECTED


def is_retryable_ollama_failure(exc: BaseException) -> bool:
    """Retry only transient Ollama transport, rate-limit, and server failures."""
    return isinstance(exc, Exception) and classify_ollama_failure(exc) in TRANSIENT_FAILURES


class OllamaProvider:
    """Local Ollama provider using its native ``/api/chat`` endpoint."""

    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        max_tokens: int = 2000,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self._transport = transport

    classify_failure = staticmethod(classify_ollama_failure)

    @staticmethod
    def is_retryable_failure(exc: Exception) -> bool:
        return is_retryable_ollama_failure(exc)

    @retry(
        retry=retry_if_exception(is_retryable_ollama_failure),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def complete(self, prompt: str, system: str = "") -> str:
        """Send a prompt to Ollama and return the completion text."""
        log.info("ollama_request", provider=self.name, model=self.model)

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": self.max_tokens},
        }

        async with httpx.AsyncClient(transport=self._transport, timeout=120.0) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)

        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            return ""

        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                log.info("ollama_response", provider=self.name, model=self.model)
                return content

        fallback = data.get("response")
        return fallback if isinstance(fallback, str) else ""
