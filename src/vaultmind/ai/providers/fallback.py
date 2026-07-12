"""Ordered runtime failover across configured AI providers."""

from __future__ import annotations

import structlog

from vaultmind.ai.providers.base import (
    MAX_RETAINED_ATTEMPTS,
    TRANSIENT_FAILURES,
    EmptyProviderResponseError,
    FailureKind,
    Provider,
    ProviderAttempt,
    ProviderExhaustedError,
    safe_identity,
)

log = structlog.get_logger()


class FallbackProvider:
    """Try providers in configuration order for every completion request."""

    name = "fallback"

    def __init__(self, providers: list[Provider]) -> None:
        if not providers:
            raise ValueError("FallbackProvider requires at least one provider")
        self.providers = tuple(providers)
        # Retain the old registry's useful model attribute for callers and tests.
        self.model = self.providers[0].model
        self.selected_provider: str | None = None
        self.selected_model: str | None = None

    @staticmethod
    def classify_failure(exc: Exception) -> FailureKind:
        if isinstance(exc, EmptyProviderResponseError):
            return FailureKind.EMPTY_RESPONSE
        return FailureKind.UNEXPECTED

    @staticmethod
    def is_retryable_failure(exc: Exception) -> bool:
        return FallbackProvider.classify_failure(exc) in TRANSIENT_FAILURES

    async def complete(self, prompt: str, system: str = "") -> str:
        """Return the first usable response, or raise a sanitized aggregate."""
        attempts: list[ProviderAttempt] = []
        omitted_attempts = 0
        self.selected_provider = None
        self.selected_model = None

        for provider in self.providers:
            provider_name = safe_identity(getattr(provider, "name", None), fallback="unknown")
            model = safe_identity(provider.model, fallback="unknown")
            log.info("provider_attempt", provider=provider_name, model=model)
            try:
                response = await provider.complete(prompt, system=system)
                if not response.strip():
                    raise EmptyProviderResponseError()
            except Exception as exc:
                # CancelledError, KeyboardInterrupt, and SystemExit inherit from
                # BaseException and intentionally bypass this handler.
                classifier = getattr(provider, "classify_failure", None)
                classified = (
                    FailureKind.EMPTY_RESPONSE
                    if isinstance(exc, EmptyProviderResponseError)
                    else classifier(exc)
                    if callable(classifier)
                    else self.classify_failure(exc)
                )
                reason = (
                    classified if isinstance(classified, FailureKind) else FailureKind.UNEXPECTED
                )
                attempt = ProviderAttempt(provider_name, model, reason)
                if len(attempts) < MAX_RETAINED_ATTEMPTS:
                    attempts.append(attempt)
                else:
                    omitted_attempts += 1
                log.warning(
                    "provider_failed",
                    provider=attempt.provider,
                    model=attempt.model,
                    failure_kind=attempt.reason.value,
                    retryable=attempt.reason in TRANSIENT_FAILURES,
                )
                continue

            self.selected_provider = provider_name
            self.selected_model = model
            log.info("provider_selected", provider=provider_name, model=model)
            return response

        exhausted = ProviderExhaustedError(
            tuple(attempts), omitted_attempts=omitted_attempts
        )
        # Raw SDK exceptions can contain credentials, response bodies, or prompt
        # content. Do not expose the final one through normal exception chaining.
        raise exhausted from None
