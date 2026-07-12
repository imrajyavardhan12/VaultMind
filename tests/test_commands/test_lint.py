"""Tests for vm lint command behavior."""

from __future__ import annotations

import pytest
import typer

from vaultmind.commands import lint as lint_cmd


def test_lint_refuses_corrupted_manifest_without_report(fixture_vault, monkeypatch):
    manifest_path = fixture_vault.vault_path / "vault.manifest.json"
    manifest_path.write_text("not-json", encoding="utf-8")
    inbox_dir = (
        fixture_vault.vault_path
        / fixture_vault.folders.wiki
        / fixture_vault.folders.wiki_inbox
    )
    monkeypatch.setattr(lint_cmd, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(lint_cmd, "load_config", lambda: fixture_vault)
    warnings: list[str] = []
    monkeypatch.setattr(lint_cmd, "print_warning", warnings.append)

    with pytest.raises(typer.Exit) as exc_info:
        lint_cmd.lint(preview=False, strict=False, verbose=False)

    assert exc_info.value.exit_code == 1
    assert "refusing to lint" in warnings[0]
    assert not list(inbox_dir.glob("lint-*.md"))
    assert manifest_path.read_text(encoding="utf-8") == "not-json"


def test_lint_preview_no_write(fixture_vault, monkeypatch):
    """Test that --preview mode does not write any file."""
    inbox_dir = fixture_vault.vault_path / fixture_vault.folders.wiki / fixture_vault.folders.wiki_inbox

    # Ensure inbox dir exists but is empty
    inbox_dir.mkdir(parents=True, exist_ok=True)
    for f in inbox_dir.glob("lint-*.md"):
        f.unlink()

    # Monkeypatch config and logging
    monkeypatch.setattr(lint_cmd, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(lint_cmd, "load_config", lambda: fixture_vault)

    # Run with preview
    lint_cmd.lint(preview=True, strict=False, verbose=False)

    # Assert no lint file was written
    lint_files = list(inbox_dir.glob("lint-*.md"))
    assert len(lint_files) == 0


def test_lint_non_preview_writes_file(fixture_vault, monkeypatch):
    """Test that non-preview mode writes a lint report file."""
    inbox_dir = fixture_vault.vault_path / fixture_vault.folders.wiki / fixture_vault.folders.wiki_inbox

    # Ensure inbox dir exists but is empty
    inbox_dir.mkdir(parents=True, exist_ok=True)
    for f in inbox_dir.glob("lint-*.md"):
        f.unlink()

    # Monkeypatch config and logging
    monkeypatch.setattr(lint_cmd, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(lint_cmd, "load_config", lambda: fixture_vault)

    # Run without preview
    lint_cmd.lint(preview=False, strict=False, verbose=False)

    # Assert lint file was written with correct name pattern
    lint_files = list(inbox_dir.glob("lint-*.md"))
    assert len(lint_files) == 1

    # Check file contains expected frontmatter
    lint_file = lint_files[0]
    content = lint_file.read_text(encoding="utf-8")
    assert "kind: lint" in content
    assert "vaultmind: true" in content
    assert "Lint Report" in content


def test_lint_report_has_correct_frontmatter(fixture_vault, monkeypatch):
    """Test that lint report has correct frontmatter structure."""
    inbox_dir = fixture_vault.vault_path / fixture_vault.folders.wiki / fixture_vault.folders.wiki_inbox
    inbox_dir.mkdir(parents=True, exist_ok=True)
    for f in inbox_dir.glob("lint-*.md"):
        f.unlink()

    monkeypatch.setattr(lint_cmd, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(lint_cmd, "load_config", lambda: fixture_vault)

    lint_cmd.lint(preview=False, strict=False, verbose=False)

    lint_files = list(inbox_dir.glob("lint-*.md"))
    assert len(lint_files) == 1

    content = lint_files[0].read_text(encoding="utf-8")

    # Check frontmatter structure
    assert content.startswith("---\n")
    assert "kind: lint" in content
    assert "created:" in content


def test_lint_strict_mode_passes_with_no_errors(fixture_vault, monkeypatch):
    """Test that --strict mode exits 0 when no ERROR findings."""
    # Create a clean vault with no errors
    concepts_dir = (
        fixture_vault.vault_path
        / fixture_vault.folders.wiki
        / fixture_vault.folders.wiki_concepts
    )
    concepts_dir.mkdir(parents=True, exist_ok=True)

    # Remove the broken concept and add a good one
    for f in concepts_dir.glob("*.md"):
        f.unlink()

    good_concept = concepts_dir / "valid-concept.md"
    good_concept.write_text(
        "---\ntitle: Valid Concept\nsources:\n  - https://example.com\n---\n\n"
        "This is a valid concept with no broken links.",
        encoding="utf-8",
    )

    monkeypatch.setattr(lint_cmd, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(lint_cmd, "load_config", lambda: fixture_vault)

    # Should not raise
    lint_cmd.lint(preview=True, strict=True, verbose=False)


def test_lint_strict_mode_fails_with_errors(fixture_vault, monkeypatch):
    """Test that --strict mode exits 1 when ERROR findings exist."""
    # Create a broken wikilink in a concept
    concepts_dir = (
        fixture_vault.vault_path
        / fixture_vault.folders.wiki
        / fixture_vault.folders.wiki_concepts
    )
    concepts_dir.mkdir(parents=True, exist_ok=True)

    bad_concept = concepts_dir / "bad-concept.md"
    bad_concept.write_text(
        "---\ntitle: Bad Concept\n---\n\n[[nonexistent-target]]",
        encoding="utf-8",
    )

    monkeypatch.setattr(lint_cmd, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(lint_cmd, "load_config", lambda: fixture_vault)

    # Should raise Exit(1)
    with pytest.raises(typer.Exit) as exc_info:
        lint_cmd.lint(preview=True, strict=True, verbose=False)

    assert exc_info.value.exit_code == 1


def _report_body(content: str) -> str:
    """Strip frontmatter (incl. the per-run `created` timestamp) from a report."""
    if not content.startswith("---"):
        return content.strip()
    end = content.find("---", 3)
    return content[end + 3 :].strip() if end != -1 else content.strip()


def test_lint_is_idempotent_across_runs(fixture_vault, monkeypatch):
    """Two consecutive non-preview runs must produce identical report bodies.

    Regression guard: the linter must not scan its own report output. A finding
    whose detail renders `[[...]]` would otherwise be re-parsed on the next run
    and produce a spurious broken_wikilink against the report file itself.
    """
    concepts_dir = (
        fixture_vault.vault_path
        / fixture_vault.folders.wiki
        / fixture_vault.folders.wiki_concepts
    )
    concepts_dir.mkdir(parents=True, exist_ok=True)
    # Guarantee a finding whose detail contains a [[...]] wikilink.
    (concepts_dir / "bad-concept.md").write_text(
        "---\ntitle: Bad Concept\nsources:\n  - https://example.com\n---\n\n[[nonexistent-target]]",
        encoding="utf-8",
    )

    inbox_dir = (
        fixture_vault.vault_path / fixture_vault.folders.wiki / fixture_vault.folders.wiki_inbox
    )
    inbox_dir.mkdir(parents=True, exist_ok=True)
    for f in inbox_dir.glob("lint-*.md"):
        f.unlink()

    monkeypatch.setattr(lint_cmd, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(lint_cmd, "load_config", lambda: fixture_vault)

    lint_cmd.lint(preview=False, strict=False, verbose=False)
    first = [p.read_text(encoding="utf-8") for p in inbox_dir.glob("lint-*.md")]
    assert len(first) == 1
    body_one = _report_body(first[0])

    lint_cmd.lint(preview=False, strict=False, verbose=False)
    second = list(inbox_dir.glob("lint-*.md"))
    assert len(second) == 1  # same-day overwrite, not a second file
    body_two = _report_body(second[0].read_text(encoding="utf-8"))

    assert body_one == body_two
    # The report's own path must never appear as a finding subject.
    assert "Inbox/lint-" not in body_two


def test_lint_strict_writes_report_before_exit(fixture_vault, monkeypatch):
    """In non-preview --strict mode, the report is written even when it exits 1."""
    concepts_dir = (
        fixture_vault.vault_path
        / fixture_vault.folders.wiki
        / fixture_vault.folders.wiki_concepts
    )
    concepts_dir.mkdir(parents=True, exist_ok=True)
    (concepts_dir / "bad-concept.md").write_text(
        "---\ntitle: Bad Concept\n---\n\n[[nonexistent-target]]",
        encoding="utf-8",
    )

    inbox_dir = (
        fixture_vault.vault_path / fixture_vault.folders.wiki / fixture_vault.folders.wiki_inbox
    )
    inbox_dir.mkdir(parents=True, exist_ok=True)
    for f in inbox_dir.glob("lint-*.md"):
        f.unlink()

    monkeypatch.setattr(lint_cmd, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(lint_cmd, "load_config", lambda: fixture_vault)

    with pytest.raises(typer.Exit) as exc_info:
        lint_cmd.lint(preview=False, strict=True, verbose=False)

    assert exc_info.value.exit_code == 1
    # The report must exist despite the non-zero exit.
    assert len(list(inbox_dir.glob("lint-*.md"))) == 1


def test_lint_command_creates_correct_directory_structure(fixture_vault, monkeypatch):
    """Test that lint creates inbox directory if it doesn't exist."""
    # Remove inbox dir
    inbox_dir = fixture_vault.vault_path / fixture_vault.folders.wiki / fixture_vault.folders.wiki_inbox
    if inbox_dir.exists():
        import shutil
        shutil.rmtree(inbox_dir)

    monkeypatch.setattr(lint_cmd, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(lint_cmd, "load_config", lambda: fixture_vault)

    lint_cmd.lint(preview=False, strict=False, verbose=False)

    # Check directory was created
    assert inbox_dir.exists()
    assert list(inbox_dir.glob("lint-*.md"))
