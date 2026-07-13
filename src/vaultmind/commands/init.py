"""vm init — interactive setup wizard for new users."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import typer
import yaml

from vaultmind.utils.display import console, print_success, print_warning

CONFIG_DIR = Path.home() / ".config" / "vaultmind"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
ENV_PATH = CONFIG_DIR / ".env"

VAULT_FOLDERS = [
    "📥 Raw",
    "📥 Raw/assets",
    "🗺️ Wiki",
    "🗺️ Wiki/🧠 Concepts",
    "🗺️ Wiki/📊 Queries",
    "🗺️ Wiki/📋 Inbox",
    "🗺️ Wiki/📅 Weekly",
]

VAULT_SCHEMA = """# VaultMind Schema

This vault uses VaultMind's Raw -> Wiki architecture.

## Ownership

- `📥 Raw/` is human or Obsidian Web Clipper owned. VaultMind reads it but does not rewrite it.
- `🗺️ Wiki/` is VaultMind owned. Humans review it.
- `vault.manifest.json` is VaultMind owned.
- `VAULTMIND.md` is human owned and may be edited to tune wiki conventions.

## Concept Pages

Concept pages live in `🗺️ Wiki/🧠 Concepts/`.

Use:

- `[[slug|Display Title]]` wikilinks
- a Sources section with source URLs or raw paths
- concise, encyclopedic prose

Prefer updating existing concept pages over creating near-duplicates.

## Query Pages

Query answers live in `🗺️ Wiki/📊 Queries/`.

Use preview mode when you want terminal output without writing a page.
"""


def init(verbose: bool = False) -> None:
    """Set up VaultMind — creates config and connects your vault + API key."""
    console.print("\n[bold cyan]🧠 VaultMind Setup[/bold cyan]\n")

    # Check for existing config
    if CONFIG_PATH.exists():
        overwrite = typer.confirm(
            f"Config already exists at {CONFIG_PATH}. Overwrite?", default=False
        )
        if not overwrite:
            console.print("[dim]Setup cancelled.[/dim]")
            return

    # 1. Vault path
    vault_path = _ask_vault_path()

    # 2. AI provider
    provider, api_key = _ask_provider()

    # 3. Create vault folders
    _create_vault_folders(vault_path)
    _create_vault_schema(vault_path)

    # 4. Write config.yaml
    _write_config(vault_path, provider)

    # 5. Write .env
    _write_env(provider, api_key)

    console.print()
    print_success(
        "VaultMind is ready!",
        f"Config: {CONFIG_PATH}\n"
        f"Secrets: {ENV_PATH}\n"
        f"Vault: {vault_path}\n\n"
        f"Next steps:\n"
        f"  1. Clip sources into 📥 Raw/ (e.g. with Obsidian Web Clipper)\n"
        f"  2. vm compile\n"
        f'  3. vm ask "your question"',
    )


def _ask_vault_path() -> Path:
    """Prompt for Obsidian vault path and validate it."""
    while True:
        raw = typer.prompt(
            "📁 Obsidian vault path",
            default=str(Path.home() / "Obsidian Vault"),
        )
        vault_path = Path(raw).expanduser().resolve()

        if vault_path.exists() and vault_path.is_dir():
            console.print(f"  [green]✓[/green] Found vault at {vault_path}")
            return vault_path

        create = typer.confirm(
            f"  Directory doesn't exist. Create {vault_path}?", default=True
        )
        if create:
            vault_path.mkdir(parents=True, exist_ok=True)
            console.print(f"  [green]✓[/green] Created {vault_path}")
            return vault_path

        console.print("  [yellow]Try again.[/yellow]")


def _ask_provider() -> tuple[str, str]:
    """Prompt for AI provider choice and API key."""
    console.print("\n[bold]🤖 AI Provider[/bold]")
    console.print("  1. OpenAI  (gpt-4.1)")
    console.print("  2. Anthropic (claude-sonnet)")
    console.print("  3. OpenRouter (OpenAI-compatible, many models)")
    console.print("  4. Ollama  (local, no key needed)\n")

    choice = typer.prompt("Choose provider [1/2/3/4]", default="1")
    provider_map = {
        "1": "openai",
        "2": "anthropic",
        "3": "openrouter",
        "4": "ollama",
    }
    provider = provider_map.get(choice, "openai")

    if provider == "ollama":
        console.print("  [green]✓[/green] Selected Ollama (no API key needed)")
        return provider, ""

    key_names = {
        "anthropic": "Anthropic",
        "openai": "OpenAI",
        "openrouter": "OpenRouter",
    }
    key_urls = {
        "anthropic": "https://console.anthropic.com/",
        "openai": "https://platform.openai.com/api-keys",
        "openrouter": "https://openrouter.ai/keys",
    }
    key_name = key_names[provider]
    console.print("\n  Get your key from:")
    console.print(f"  [dim]{key_urls[provider]}[/dim]")

    api_key = typer.prompt(f"\n🔑 {key_name} API key", hide_input=True)

    if not api_key.strip():
        print_warning("No API key provided. You can add it later in ~/.config/vaultmind/.env")
        return provider, ""

    console.print("  [green]✓[/green] API key saved")
    return provider, api_key.strip()


def _create_vault_folders(vault_path: Path) -> None:
    """Create the standard VaultMind folder structure in the vault."""
    created = 0
    for folder_name in VAULT_FOLDERS:
        folder = vault_path / folder_name
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            created += 1

    if created > 0:
        console.print(f"  [green]✓[/green] Created {created} vault folders")
    else:
        console.print("  [dim]Vault folders already exist[/dim]")


def _create_vault_schema(vault_path: Path) -> None:
    """Create the vault-level schema file if it does not exist."""
    schema_path = vault_path / "VAULTMIND.md"
    if schema_path.exists():
        console.print("  [dim]VAULTMIND.md already exists[/dim]")
        return
    schema_path.write_text(VAULT_SCHEMA, encoding="utf-8")
    console.print("  [green]✓[/green] Created VAULTMIND.md")


def _write_config(vault_path: Path, provider: str) -> None:
    """Write config.yaml to ~/.config/vaultmind/."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    fallback = [provider]
    for p in ["openai", "anthropic", "openrouter", "ollama"]:
        if p != provider:
            fallback.append(p)

    config_data = {
        "vault_path": str(vault_path),
        "folders": {
            "raw": "📥 Raw",
            "wiki": "🗺️ Wiki",
            "wiki_concepts": "🧠 Concepts",
            "wiki_queries": "📊 Queries",
            "wiki_inbox": "📋 Inbox",
            "wiki_weekly": "📅 Weekly",
            "wiki_index": "📇 Index",
        },
        "ai": {
            "default_provider": provider,
            "fallback_chain": fallback,
            "max_tokens": 2000,
            "providers": {
                "anthropic": {
                    "models": {"fast": "claude-sonnet-4-20250514", "deep": "claude-opus-4-5"},
                },
                "openai": {
                    "models": {"fast": "gpt-4.1-mini", "deep": "gpt-4.1"},
                },
                "openrouter": {
                    "models": {
                        "fast": "openai/gpt-4.1-mini",
                        "deep": "openai/gpt-4.1",
                    },
                },
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "models": {"fast": "llama3", "deep": "llama3"},
                },
            },
        },
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    console.print(f"  [green]✓[/green] Config written to {CONFIG_PATH}")


def _write_env(provider: str, api_key: str) -> None:
    """Write .env with API key to ~/.config/vaultmind/ with restricted permissions."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    lines = ["# VaultMind secrets — do not share or commit this file"]
    provider_keys = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    selected_key = provider_keys.get(provider)
    for provider_key in provider_keys.values():
        if provider_key == selected_key:
            lines.append(f"{provider_key}={api_key}")
        else:
            lines.append(f"# {provider_key}=")

    lines.append("")

    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Restrict permissions — only owner can read/write
    os.chmod(ENV_PATH, stat.S_IRUSR | stat.S_IWUSR)

    console.print(f"  [green]✓[/green] Secrets written to {ENV_PATH} (permissions: 600)")
