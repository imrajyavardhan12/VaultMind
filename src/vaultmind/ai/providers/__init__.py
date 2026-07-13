"""AI provider registry and runtime fallback construction."""

from __future__ import annotations

__all__ = ["FallbackProvider", "Provider", "ProviderExhaustedError", "get_provider"]

from vaultmind.ai.providers.base import Provider, ProviderExhaustedError
from vaultmind.ai.providers.fallback import FallbackProvider
from vaultmind.config import AppConfig


def get_provider(config: AppConfig, tier: str = "fast") -> Provider:
    """Build every available provider in configured runtime fallback order."""
    from vaultmind.ai.providers.anthropic import AnthropicProvider
    from vaultmind.ai.providers.ollama import OllamaProvider
    from vaultmind.ai.providers.openai import OpenAIProvider
    from vaultmind.ai.providers.openrouter import (
        DEFAULT_OPENROUTER_BASE_URL,
        OpenRouterProvider,
    )

    providers: list[Provider] = []
    for provider_name in config.ai.fallback_chain:
        provider_config = config.ai.providers.get(provider_name)
        if provider_config is None:
            continue

        model = getattr(provider_config.models, tier, provider_config.models.fast)
        if provider_name == "anthropic":
            if not config.env.anthropic_api_key:
                continue
            providers.append(
                AnthropicProvider(
                    api_key=config.env.anthropic_api_key,
                    model=model,
                    max_tokens=config.ai.max_tokens,
                )
            )
        elif provider_name == "openai":
            if not config.env.openai_api_key:
                continue
            providers.append(
                OpenAIProvider(
                    api_key=config.env.openai_api_key,
                    model=model,
                    max_tokens=config.ai.max_tokens,
                )
            )
        elif provider_name == "openrouter":
            if not config.env.openrouter_api_key.strip():
                continue
            providers.append(
                OpenRouterProvider(
                    api_key=config.env.openrouter_api_key,
                    model=model,
                    max_tokens=config.ai.max_tokens,
                    base_url=provider_config.base_url or DEFAULT_OPENROUTER_BASE_URL,
                )
            )
        elif provider_name == "ollama":
            base_url = provider_config.base_url or config.env.ollama_base_url
            if not base_url.strip():
                continue
            providers.append(
                OllamaProvider(
                    base_url=base_url,
                    model=model,
                    max_tokens=config.ai.max_tokens,
                )
            )

    if not providers:
        from vaultmind.utils.display import print_error

        print_error(
            "No AI provider available.\n"
            "Configure a provider and its required credentials in config.yaml and .env."
        )
        raise SystemExit(1)

    return FallbackProvider(providers)
