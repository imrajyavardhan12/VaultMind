"""Concurrency-safe prompt-matched replay provider for offline evaluation."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

from vaultmind.ai.providers.base import FailureKind

PROMPT_FINGERPRINT_HEX_CHARS = 16


class ReplayPromptError(Exception):
    """Base class for safe deterministic replay failures."""


class UnmatchedReplayPromptError(ReplayPromptError):
    """No replay rule matched a provider request."""


class ExhaustedReplayResponseError(ReplayPromptError):
    """A matching replay rule has no response left."""


class AmbiguousReplayPromptError(ReplayPromptError):
    """More than one replay rule matched a provider request."""


class ReplayMatch(BaseModel):
    """Conjunctive, order-independent match criteria for a provider request."""

    prompt_contains: list[str] = Field(default_factory=list)
    system_contains: list[str] = Field(default_factory=list)


class ReplayRule(BaseModel):
    """A named match rule and its finite ordered response queue."""

    id: str
    match: ReplayMatch
    responses: list[str]


class ReplayFixture(BaseModel):
    """Serialized replay-provider configuration."""

    rules: list[ReplayRule]


def prompt_fingerprint(prompt: str, system: str = "") -> str:
    """Return a bounded fingerprint without retaining prompt content."""
    prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    system_digest = hashlib.sha256(system.encode("utf-8")).hexdigest()
    return (
        f"prompt_sha256={prompt_digest[:PROMPT_FINGERPRINT_HEX_CHARS]} "
        f"system_sha256={system_digest[:PROMPT_FINGERPRINT_HEX_CHARS]} "
        f"prompt_chars={len(prompt)} system_chars={len(system)}"
    )


class ReplayProvider:
    """Provider whose responses are selected by prompt, never global call order."""

    name = "offline-replay"
    model = "fixture-v1"

    def __init__(self, fixture: ReplayFixture) -> None:
        ids = [rule.id for rule in fixture.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("replay rule ids must be unique")
        self._rules = tuple(fixture.rules)
        self._consumed = {rule.id: 0 for rule in fixture.rules}
        self._call_count = 0
        self._lock = asyncio.Lock()

    @classmethod
    def from_path(cls, path: Path) -> ReplayProvider:
        """Load replay rules from JSON without consulting environment state."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(ReplayFixture.model_validate(data))

    @property
    def call_count(self) -> int:
        """Return every attempted completion request, including replay failures."""
        return self._call_count

    @property
    def consumed_by_rule(self) -> dict[str, int]:
        """Return a stable diagnostic snapshot without prompt or response content."""
        return {rule.id: self._consumed[rule.id] for rule in self._rules}

    async def complete(self, prompt: str, system: str = "") -> str:
        """Atomically match and consume one rule-specific response."""
        async with self._lock:
            self._call_count += 1
            matches = [rule for rule in self._rules if self._matches(rule, prompt, system)]
            fingerprint = prompt_fingerprint(prompt, system)
            if not matches:
                raise UnmatchedReplayPromptError(f"unmatched replay request ({fingerprint})")
            if len(matches) > 1:
                ids = ",".join(sorted(rule.id for rule in matches))[:160]
                raise AmbiguousReplayPromptError(
                    f"ambiguous replay request rules={ids} ({fingerprint})"
                )

            rule = matches[0]
            consumed = self._consumed[rule.id]
            if consumed >= len(rule.responses):
                raise ExhaustedReplayResponseError(
                    f"exhausted replay rule={rule.id[:80]} ({fingerprint})"
                )
            response = rule.responses[consumed]
            self._consumed[rule.id] = consumed + 1
            return response

    @staticmethod
    def _matches(rule: ReplayRule, prompt: str, system: str) -> bool:
        return all(part in prompt for part in rule.match.prompt_contains) and all(
            part in system for part in rule.match.system_contains
        )

    def classify_failure(self, exc: Exception) -> FailureKind:
        """Replay fixture errors are deterministic invalid-request failures."""
        return FailureKind.REQUEST

    def is_retryable_failure(self, exc: Exception) -> bool:
        """Replay mismatches cannot become valid through retrying."""
        return False
