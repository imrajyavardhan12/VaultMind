"""OpenRouter provider using its OpenAI-compatible API."""

from __future__ import annotations

from vaultmind.ai.providers.openai import OpenAICompatibleProvider

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter provider backed by the installed OpenAI SDK."""

    name = "openrouter"

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 2000,
        *,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            base_url=base_url,
        )
