"""Tests for the fixture vault (Story #9).

These tests verify that the fixture vault is loadable, scannable,
and that using it does not mutate the source fixture files.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from vaultmind.core.manifest import ManifestReadError, read_manifest, reconcile_manifest
from vaultmind.core.raw_scanner import scan_raw_sources
from vaultmind.schemas import Manifest, ManifestSource


def test_missing_manifest_loads_empty_version_one(tmp_path):
    manifest = read_manifest(tmp_path)
    assert manifest == Manifest(version=1)


@pytest.mark.parametrize("body", ["{bad", '{"last_compiled": "never"}'])
def test_existing_invalid_manifest_raises_typed_error(tmp_path, body):
    (tmp_path / "vault.manifest.json").write_text(body, encoding="utf-8")
    with pytest.raises(ManifestReadError):
        read_manifest(tmp_path)


@pytest.mark.parametrize(
    "body",
    ['{"version": "1"}', '{"version": true}', '{"version": 1.0}', "{}", '{"version": 2}'],
    ids=["string", "boolean", "float", "missing", "unsupported"],
)
def test_persisted_manifest_requires_exact_integer_version_one(tmp_path, body):
    (tmp_path / "vault.manifest.json").write_text(body, encoding="utf-8")
    with pytest.raises(ManifestReadError):
        read_manifest(tmp_path)


def test_dangling_manifest_symlink_raises_typed_error(tmp_path):
    path = tmp_path / "vault.manifest.json"
    try:
        path.symlink_to(tmp_path / "missing-manifest.json")
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ManifestReadError):
        read_manifest(tmp_path)


def test_readable_manifest_symlink_loads_target(tmp_path):
    target = tmp_path / "manifest-target.json"
    target.write_text('{"version": 1}', encoding="utf-8")
    path = tmp_path / "vault.manifest.json"
    try:
        path.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    assert read_manifest(tmp_path) == Manifest(version=1)


def test_unreadable_manifest_symlink_raises_typed_error_where_enforced(tmp_path):
    target = tmp_path / "manifest-target.json"
    target.write_text('{"version": 1}', encoding="utf-8")
    path = tmp_path / "vault.manifest.json"
    try:
        path.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    target.chmod(0)
    try:
        try:
            target.read_text(encoding="utf-8")
        except PermissionError:
            pass
        else:
            pytest.skip("filesystem does not enforce unreadable file permissions")

        with pytest.raises(ManifestReadError):
            read_manifest(tmp_path)
    finally:
        target.chmod(0o600)


def test_reconciliation_preserves_only_cited_missing_history_and_does_not_add_raw(tmp_path):
    cited = "https://example.com/cited"
    stale = "https://example.com/stale"
    current_uncompiled = "https://example.com/current"
    now = datetime.now(UTC)
    manifest = Manifest(
        sources={
            cited: ManifestSource(content_hash="cited", saved_at=now),
            stale: ManifestSource(content_hash="stale", saved_at=now),
        }
    )
    concepts_dir = tmp_path / "concepts"
    concepts_dir.mkdir()
    (concepts_dir / "history.md").write_text(
        f"---\nsources:\n  - {cited}\n---\n\n# History\n",
        encoding="utf-8",
    )

    result = reconcile_manifest(
        manifest,
        concepts_dir=concepts_dir,
        current_raw_keys={current_uncompiled},
    )

    assert set(result.manifest.sources) == {cited}
    assert result.manifest.sources[cited].wiki_articles == ["history"]
    assert current_uncompiled not in result.manifest.sources
    assert set(result.manifest.wiki_articles) == {"history"}


class TestFixtureVaultStructure:
    """Verify the fixture vault has the expected structure."""

    def test_fixture_vault_has_three_raw_sources(self, fixture_vault):
        """Scan returns exactly 3 raw source files."""
        records = scan_raw_sources(fixture_vault)
        assert len(records) == 3
        titles = {r.title for r in records}
        assert "Attention Mechanisms in Transformers" in titles
        assert "RLHF vs DPO: A Comparison of Alignment Methods" in titles
        assert "Sparse Attention Trade-offs in Transformer Models" in titles

    def test_raw_sources_have_frontmatter(self, fixture_vault):
        """Raw sources have source URLs in frontmatter."""
        records = scan_raw_sources(fixture_vault)
        for record in records:
            assert record.source_url is not None, f"Missing source_url for {record.title}"
            assert record.source_url.startswith("https://"), f"Invalid source_url for {record.title}"

    def test_raw_sources_have_body_content(self, fixture_vault):
        """Raw sources have meaningful body content."""
        records = scan_raw_sources(fixture_vault)
        for record in records:
            assert len(record.body) > 100, f"Body too short for {record.title}"
            assert record.content_hash, f"Missing content_hash for {record.title}"

    def test_fixture_vault_has_existing_wiki_concept(self, fixture_vault):
        """Wiki/Concepts contains the existing attention-mechanisms article."""
        concepts_dir = (
            fixture_vault.vault_path
            / fixture_vault.folders.wiki
            / fixture_vault.folders.wiki_concepts
        )
        assert concepts_dir.exists(), "Concepts directory does not exist"
        concept_files = list(concepts_dir.glob("*.md"))
        assert len(concept_files) == 1, f"Expected 1 concept, found {len(concept_files)}"
        assert concept_files[0].stem == "attention-mechanisms"

    def test_existing_concept_has_vaultmind_frontmatter(self, fixture_vault):
        """Existing concept has correct frontmatter."""
        concepts_dir = (
            fixture_vault.vault_path
            / fixture_vault.folders.wiki
            / fixture_vault.folders.wiki_concepts
        )
        concept_path = concepts_dir / "attention-mechanisms.md"
        text = concept_path.read_text(encoding="utf-8")
        assert text.startswith("---")
        assert "vaultmind: true" in text
        assert "kind: concept" in text
        assert "[[transformers]]" in text or "[[" in text


class TestFixtureVaultManifest:
    """Verify the fixture manifest has realistic state."""

    def test_manifest_loads_successfully(self, fixture_vault):
        """Manifest file is valid and loadable."""
        manifest = read_manifest(fixture_vault.vault_path)
        assert isinstance(manifest, Manifest)
        assert manifest.version == 1

    def test_manifest_has_one_compiled_source(self, fixture_vault):
        """Manifest tracks one source as already compiled."""
        manifest = read_manifest(fixture_vault.vault_path)
        assert len(manifest.sources) == 1
        source_url = "https://lena-voita.github.io/posts/annotated_transformers/attention.html"
        assert source_url in manifest.sources
        assert manifest.sources[source_url].wiki_articles == ["attention-mechanisms"]

    def test_manifest_content_hash_matches_real_hash(self, fixture_vault):
        """Manifest source hash matches actual content hash."""
        manifest = read_manifest(fixture_vault.vault_path)
        source_url = "https://lena-voita.github.io/posts/annotated_transformers/attention.html"
        record = next(r for r in scan_raw_sources(fixture_vault) if r.source_url == source_url)
        assert manifest.sources[source_url].content_hash == record.content_hash

    def test_manifest_has_one_wiki_article(self, fixture_vault):
        """Manifest tracks one wiki article."""
        manifest = read_manifest(fixture_vault.vault_path)
        assert len(manifest.wiki_articles) == 1
        assert "attention-mechanisms" in manifest.wiki_articles
        entry = manifest.wiki_articles["attention-mechanisms"]
        assert len(entry.source_urls) == 1
        assert entry.source_urls[0] == "https://lena-voita.github.io/posts/annotated_transformers/attention.html"

    def test_manifest_reconciles_against_representative_vault(self, fixture_vault):
        manifest = read_manifest(fixture_vault.vault_path)
        raw_keys = {
            record.source_url or record.relative_path
            for record in scan_raw_sources(fixture_vault)
        }
        concepts_dir = (
            fixture_vault.vault_path
            / fixture_vault.folders.wiki
            / fixture_vault.folders.wiki_concepts
        )

        result = reconcile_manifest(
            manifest,
            concepts_dir=concepts_dir,
            current_raw_keys=raw_keys,
        )

        source_url = "https://lena-voita.github.io/posts/annotated_transformers/attention.html"
        assert result.manifest.version == 1
        assert set(result.manifest.wiki_articles) == {"attention-mechanisms"}
        assert result.manifest.sources[source_url].wiki_articles == ["attention-mechanisms"]
        assert result.manifest.wiki_articles["attention-mechanisms"].content_hash != "1546d41d"
        assert result.repairs == ("repaired concept hash: attention-mechanisms",)


class TestFixtureVaultIsolation:
    """Verify that mutating the fixture vault does not affect the source."""

    def test_copy_is_independent(self, fixture_vault):
        """Mutating the copied vault does not modify source fixtures."""
        fixture_source = Path(__file__).parent / "fixtures" / "vault"

        # Get original file size
        original_file = fixture_source / "📥 Raw" / "attention-mechanisms-transformers.md"
        original_size = original_file.stat().st_size

        # Modify a file in the copied vault
        concepts_dir = (
            fixture_vault.vault_path
            / fixture_vault.folders.wiki
            / fixture_vault.folders.wiki_concepts
        )
        (concepts_dir / "attention-mechanisms.md").write_text(
            "# Modified\n\nThis should not affect source.",
            encoding="utf-8",
        )

        # Verify source fixture is unchanged
        source_size = original_file.stat().st_size
        assert source_size == original_size, "Source fixture was modified"
        original_text = original_file.read_text(encoding="utf-8")
        assert "This should not affect source." not in original_text
