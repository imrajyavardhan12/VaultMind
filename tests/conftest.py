"""Shared test fixtures."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from vaultmind.config import (
    AIConfig,
    AppConfig,
    EnvSettings,
    FolderConfig,
    ProviderConfig,
    ProviderModels,
)


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    """Create a temporary vault directory structure."""
    folders = [
        "📥 Inbox",
        "📚 Sources/AI",
        "📚 Sources/Tech",
        "📚 Sources/Misc",
        "🛠️ Tools",
        "💬 Discussions",
        "🐦 Threads",
    ]
    for folder in folders:
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def test_config(tmp_vault: Path) -> AppConfig:
    """Create a test AppConfig pointing at a temp vault."""
    return AppConfig(
        vault_path=tmp_vault,
        folders=FolderConfig(),
        ai=AIConfig(
            default_provider="anthropic",
            fallback_chain=["anthropic"],
            providers={
                "anthropic": ProviderConfig(
                    models=ProviderModels(fast="claude-sonnet-4-20250514", deep="claude-opus-4-5")
                )
            },
        ),
        env=EnvSettings(anthropic_api_key="test-key"),
    )


# ---- Fixture Vault (Story #9) ----

FIXTURE_VAULT_PATH = Path(__file__).parent / "fixtures" / "vault"


@pytest.fixture
def fixture_vault(tmp_path: Path) -> AppConfig:
    """Provide a realistic fixture vault for compile integration tests.

    The fixture vault contains:
    - 3 raw source markdown files in 📥 Raw/ (with frontmatter, headings, links, images)
    - 1 existing wiki concept in 🗺️ Wiki/🧠 Concepts/ (attention-mechanisms.md)
    - A realistic vault.manifest.json

    The vault is copied to a temp directory so tests can mutate it safely
    without affecting the fixture source files.
    """
    vault_dest = tmp_path / "fixture_vault"
    shutil.copytree(FIXTURE_VAULT_PATH, vault_dest, dirs_exist_ok=True)

    return AppConfig(
        vault_path=vault_dest,
        folders=FolderConfig(),
        ai=AIConfig(
            default_provider="anthropic",
            fallback_chain=["anthropic"],
            providers={
                "anthropic": ProviderConfig(
                    models=ProviderModels(fast="claude-sonnet-4-20250514", deep="claude-opus-4-5")
                )
            },
        ),
        env=EnvSettings(anthropic_api_key="test-key"),
    )
