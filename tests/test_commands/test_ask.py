"""Tests for vm ask command behavior."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from vaultmind.ai.asker import AskResult
from vaultmind.ai.providers.base import FailureKind, ProviderAttempt, ProviderExhaustedError
from vaultmind.commands import ask as ask_cmd
from vaultmind.main import app


def test_ask_preview_is_strictly_no_write(monkeypatch, test_config):
    """CLI preview skips file logging, query filing, and wiki-log appends."""
    query_dir = test_config.vault_path / test_config.folders.wiki / test_config.folders.wiki_queries
    wiki_log = test_config.vault_path / test_config.folders.wiki / "📋 Log.md"
    wiki_log.parent.mkdir(parents=True, exist_ok=True)
    wiki_log.write_text("# Wiki Log\n\nExisting entry.\n", encoding="utf-8")
    original_log = wiki_log.read_text(encoding="utf-8")

    def fail_setup_logging(*, verbose: bool = False) -> None:
        raise AssertionError(f"filesystem-writing logging setup called (verbose={verbose})")

    async def fake_ask_question(**kwargs):
        assert kwargs["file_answer"] is False
        path = query_dir / "preview-question.md"
        return AskResult(
            answer="Preview answer.",
            slug="preview-question",
            path=path,
            iterations=1,
            gaps=[],
        )

    monkeypatch.setattr(ask_cmd, "setup_logging", fail_setup_logging)
    monkeypatch.setattr(ask_cmd, "load_config", lambda: test_config)
    monkeypatch.setattr(ask_cmd, "get_provider", lambda config, tier: object())
    monkeypatch.setattr(ask_cmd, "ask_question", fake_ask_question)

    result = CliRunner().invoke(app, ["ask", "Preview question?", "--preview"])

    assert result.exit_code == 0, result.output
    assert "Preview answer." in result.output
    assert not query_dir.exists()
    assert not (query_dir / "preview-question.md").exists()
    assert wiki_log.read_text(encoding="utf-8") == original_log


def _exhausted_error() -> ProviderExhaustedError:
    return ProviderExhaustedError(
        (ProviderAttempt("anthropic", "claude", FailureKind.AUTHENTICATION),)
    )


@pytest.mark.parametrize("preview", [False, True])
def test_ask_provider_exhaustion_is_concise_and_nonzero(
    monkeypatch, test_config, preview
):
    async def fail_ask(**kwargs):
        del kwargs
        raise _exhausted_error() from RuntimeError("diagnostic secret")

    monkeypatch.setattr(ask_cmd, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(ask_cmd, "load_config", lambda: test_config)
    monkeypatch.setattr(ask_cmd, "get_provider", lambda config, tier: object())
    monkeypatch.setattr(ask_cmd, "ask_question", fail_ask)

    args = ["ask", "private question"]
    if preview:
        args.append("--preview")
    result = CliRunner().invoke(app, args)

    assert result.exit_code == 1
    assert "AI provider chain exhausted" in result.output
    assert "anthropic/claude: authentication failed" in result.output
    assert "diagnostic secret" not in result.output
    assert "private question" not in result.output
    assert "Traceback" not in result.output


def test_ask_verbose_preserves_provider_exception_chain(monkeypatch, test_config):
    cause = RuntimeError("diagnostic detail")

    async def fail_ask(**kwargs):
        del kwargs
        raise _exhausted_error() from cause

    monkeypatch.setattr(ask_cmd, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(ask_cmd, "load_config", lambda: test_config)
    monkeypatch.setattr(ask_cmd, "get_provider", lambda config, tier: object())
    monkeypatch.setattr(ask_cmd, "ask_question", fail_ask)

    result = CliRunner().invoke(app, ["ask", "question", "--verbose"])

    assert result.exit_code == 1
    assert isinstance(result.exception, ProviderExhaustedError)
    assert result.exception.__cause__ is cause
