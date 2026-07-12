"""Tests for vm ask command behavior."""

from __future__ import annotations

from typer.testing import CliRunner

from vaultmind.ai.asker import AskResult
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
