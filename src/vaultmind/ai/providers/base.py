"""Typed contracts shared by VaultMind AI providers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class FailureKind(StrEnum):
    """Safe, provider-independent failure classifications."""

    CONNECTION = "connection failure"
    TIMEOUT = "request timed out"
    RATE_LIMIT = "rate limited"
    SERVER = "provider server failure"
    AUTHENTICATION = "authentication failed"
    PERMISSION = "permission denied"
    REQUEST = "invalid request"
    EMPTY_RESPONSE = "empty response"
    UNEXPECTED = "unexpected provider error"


TRANSIENT_FAILURES = frozenset(
    {FailureKind.CONNECTION, FailureKind.TIMEOUT, FailureKind.RATE_LIMIT, FailureKind.SERVER}
)

MAX_IDENTITY_LENGTH = 80
MAX_ATTEMPT_REASON_LENGTH = 80
MAX_RETAINED_ATTEMPTS = 8
MAX_OMITTED_ATTEMPTS = 999_999_999
MAX_EXHAUSTED_ERROR_MESSAGE_LENGTH = 2_048


class ProviderFailureClassifier(Protocol):
    """Interface used by retry policies and the runtime fallback chain."""

    def classify_failure(self, exc: Exception) -> FailureKind:
        """Classify an exception without exposing its potentially sensitive text."""
        ...

    def is_retryable_failure(self, exc: Exception) -> bool:
        """Return whether an individual provider may retry this failure."""
        ...


class Provider(ProviderFailureClassifier, Protocol):
    """Abstract interface implemented by every AI provider."""

    name: str
    model: str

    async def complete(self, prompt: str, system: str = "") -> str:
        """Send a prompt and return the completion text."""
        ...


class ProviderError(Exception):
    """Base class for typed provider-layer failures."""


class EmptyProviderResponseError(ProviderError):
    """A provider returned no usable completion text."""


def _safe_text(value: object, *, fallback: str, limit: int) -> str:
    """Return bounded single-line text without retaining the original value."""
    text = "" if value is None else " ".join(str(value).split())[:limit]
    return text or fallback


def safe_identity(value: object, *, fallback: str) -> str:
    """Return a bounded single-line identity suitable for logs and errors."""
    return _safe_text(value, fallback=fallback, limit=MAX_IDENTITY_LENGTH)


@dataclass(frozen=True)
class ProviderAttempt:
    """A sanitized record of one provider/model attempted by a chain."""

    provider: str
    model: str
    reason: FailureKind

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", safe_identity(self.provider, fallback="unknown"))
        object.__setattr__(self, "model", safe_identity(self.model, fallback="unknown"))
        if not isinstance(self.reason, FailureKind):
            object.__setattr__(self, "reason", FailureKind.UNEXPECTED)

    def render(self) -> str:
        reason = _safe_text(
            self.reason.value,
            fallback=FailureKind.UNEXPECTED.value,
            limit=MAX_ATTEMPT_REASON_LENGTH,
        )
        return f"{self.provider}/{self.model}: {reason}"


def _bounded_omission_count(value: object, additional: int) -> tuple[int, bool]:
    """Return a safe count and whether it is only a lower bound."""
    supplied = int(value) if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0
    if additional >= MAX_OMITTED_ATTEMPTS:
        return MAX_OMITTED_ATTEMPTS, additional > MAX_OMITTED_ATTEMPTS or supplied > 0
    if supplied > MAX_OMITTED_ATTEMPTS - additional:
        return MAX_OMITTED_ATTEMPTS, True
    return supplied + additional, False


def _render_exhausted_message(details: str, omission: str) -> str:
    """Render an exhaustion message with a hard final bound."""
    prefix = "AI provider chain exhausted"
    if omission:
        omission_only = f"{prefix}: {omission}"
        detail_limit = MAX_EXHAUSTED_ERROR_MESSAGE_LENGTH - len(omission_only) - 2
        bounded_details = details[: max(0, detail_limit)].rstrip()
        message = (
            f"{prefix}: {bounded_details}; {omission}" if bounded_details else omission_only
        )
    elif details:
        message = f"{prefix}: {details}"
    else:
        message = prefix

    # Do not rely on the bounds of individual fields: this constructor is public
    # and the final rendered exception must remain bounded under all inputs.
    return message[:MAX_EXHAUSTED_ERROR_MESSAGE_LENGTH]


class ProviderExhaustedError(ProviderError):
    """Every configured provider failed for one completion request."""

    def __init__(
        self,
        attempts: tuple[ProviderAttempt, ...],
        *,
        omitted_attempts: int = 0,
    ) -> None:
        retained = attempts[:MAX_RETAINED_ATTEMPTS]
        self.attempts = retained
        additional = len(attempts) - len(retained)
        self.omitted_attempts, is_lower_bound = _bounded_omission_count(
            omitted_attempts, additional
        )

        details = "; ".join(attempt.render() for attempt in retained)
        qualifier = "at least " if is_lower_bound else ""
        omission = (
            f"... {qualifier}{self.omitted_attempts} attempts omitted"
            if self.omitted_attempts
            else ""
        )
        super().__init__(_render_exhausted_message(details, omission))
