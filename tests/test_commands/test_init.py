"""Focused tests for OpenRouter setup generation."""

from __future__ import annotations

import importlib
import stat
from pathlib import Path

import yaml

init_cmd = importlib.import_module("vaultmind.commands.init")


def test_ask_provider_offers_openrouter_key_url_without_echoing_secret(monkeypatch):
    answers = iter(["3", "  sk-or-v1-secret  "])
    output: list[str] = []
    monkeypatch.setattr(init_cmd.typer, "prompt", lambda *args, **kwargs: next(answers))
    monkeypatch.setattr(init_cmd.console, "print", lambda value="": output.append(str(value)))

    provider, api_key = init_cmd._ask_provider()

    rendered = "\n".join(output)
    assert provider == "openrouter"
    assert api_key == "sk-or-v1-secret"
    assert "OpenRouter" in rendered
    assert "https://openrouter.ai/keys" in rendered
    assert api_key not in rendered


def test_openrouter_config_contains_all_providers_and_model_slugs(
    monkeypatch, tmp_path: Path
):
    config_dir = tmp_path / "config"
    config_path = config_dir / "config.yaml"
    monkeypatch.setattr(init_cmd, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(init_cmd, "CONFIG_PATH", config_path)
    monkeypatch.setattr(init_cmd.console, "print", lambda *args, **kwargs: None)

    init_cmd._write_config(tmp_path / "vault", "openrouter")

    generated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ai = generated["ai"]
    assert ai["default_provider"] == "openrouter"
    assert ai["fallback_chain"] == ["openrouter", "openai", "anthropic", "ollama"]
    assert list(ai["providers"]) == ["anthropic", "openai", "openrouter", "ollama"]
    assert ai["providers"]["openrouter"]["models"] == {
        "fast": "openai/gpt-4.1-mini",
        "deep": "openai/gpt-4.1",
    }


def test_openrouter_env_activates_only_selected_key_and_keeps_mode_600(
    monkeypatch, tmp_path: Path
):
    config_dir = tmp_path / "config"
    env_path = config_dir / ".env"
    monkeypatch.setattr(init_cmd, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(init_cmd, "ENV_PATH", env_path)
    monkeypatch.setattr(init_cmd.console, "print", lambda *args, **kwargs: None)

    init_cmd._write_env("openrouter", "sk-or-v1-secret")

    lines = env_path.read_text(encoding="utf-8").splitlines()
    assert "# ANTHROPIC_API_KEY=" in lines
    assert "# OPENAI_API_KEY=" in lines
    assert "OPENROUTER_API_KEY=sk-or-v1-secret" in lines
    assert not any(
        line.startswith(("ANTHROPIC_API_KEY=", "OPENAI_API_KEY=")) for line in lines
    )
    assert stat.S_IMODE(env_path.stat().st_mode) == stat.S_IRUSR | stat.S_IWUSR
