"""Tests for the compile pipeline internals."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from vaultmind.ai.compiler import _deduplicate_concepts, compile_sources
from vaultmind.commands.compile import _run_compile_async
from vaultmind.core.manifest import read_manifest
from vaultmind.core.raw_scanner import RawSourceRecord, scan_raw_sources
from vaultmind.schemas import ConceptStatus, Manifest, WikiConceptEntry


class StubProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []
        self.model = "stub-model"

    async def complete(self, prompt: str, system: str = "") -> str:
        del system
        self.prompts.append(prompt)
        return self.responses.pop(0)


def _raw_source(*, slug: str, source_url: str | None = None) -> RawSourceRecord:
    return RawSourceRecord(
        path=Path(f"/tmp/{slug}.md"),
        relative_path=f"Clippings/{slug}",
        title=slug,
        source_url=source_url,
        body=f"# {slug}\n\nBody text from {slug}",
        content_hash=f"hash-{slug}",
        raw_tags=[],
    )


def test_compile_sources_updates_existing_article_with_relative_path_raw_body(test_config):
    source = _raw_source(slug="raw-a", source_url=None)
    article_dir = test_config.vault_path / test_config.folders.wiki / test_config.folders.wiki_concepts
    article_dir.mkdir(parents=True, exist_ok=True)
    (article_dir / "concept-a.md").write_text("# Concept A\n\nOld content", encoding="utf-8")

    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": "Concept A",
                    "status": "existing:concept-a",
                    "description": "Existing concept",
                    "source_urls": [source.relative_path],
                    "merge_target": "concept-a",
                }
            ]
        }
    )
    provider = StubProvider([triage, "# Concept A\n\nUpdated content"])

    result, slug_to_urls = asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
        )
    )

    assert result.articles_updated == 1
    assert result.articles_created == 0
    assert slug_to_urls == {"concept-a": [source.relative_path]}
    assert "Body text from raw-a" in provider.prompts[1]


def test_compile_sources_preserves_existing_frontmatter_sources_on_update(test_config):
    source = _raw_source(slug="raw-b", source_url="https://example.com/new-source")
    old_source_url = "https://example.com/old-source"
    article_dir = test_config.vault_path / test_config.folders.wiki / test_config.folders.wiki_concepts
    article_dir.mkdir(parents=True, exist_ok=True)
    article_path = article_dir / "concept-a.md"
    article_path.write_text(
        "\n".join(
            [
                "---",
                "title: Human Concept A",
                "vaultmind: true",
                "kind: concept",
                "sources:",
                f"  - {old_source_url}",
                "---",
                "",
                "# Concept A",
                "",
                "Old content",
            ]
        ),
        encoding="utf-8",
    )

    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": "Concept A",
                    "status": "existing:concept-a",
                    "description": "Existing concept with new source",
                    "source_urls": [source.source_url],
                    "merge_target": "concept-a",
                }
            ]
        }
    )
    provider = StubProvider([triage, "# Concept A\n\nUpdated content"])

    result, slug_to_urls = asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
        )
    )

    updated = article_path.read_text(encoding="utf-8")
    assert result.articles_updated == 1
    assert slug_to_urls == {"concept-a": [source.source_url]}
    assert "title: Human Concept A" in updated
    assert f"  - {old_source_url}" in updated
    assert f"  - {source.source_url}" in updated


def test_fixture_vault_compile_updates_existing_concept_and_preserves_sources(fixture_vault):
    records = scan_raw_sources(fixture_vault)
    source = next(
        record
        for record in records
        if record.source_url == "https://blog.research.google/2023/03/encouraging-sparse-attention-in.html"
    )
    old_source_url = "https://lena-voita.github.io/posts/annotated_transformers/attention.html"

    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": "Attention Mechanisms",
                    "status": "existing:attention-mechanisms",
                    "description": "Existing attention concept expanded with sparse attention trade-offs",
                    "source_urls": [source.source_url],
                    "merge_target": "attention-mechanisms",
                }
            ]
        }
    )
    provider = StubProvider([triage, "# Attention Mechanisms\n\nUpdated with sparse attention."])
    manifest = read_manifest(fixture_vault.vault_path)

    result, slug_to_urls = asyncio.run(
        _run_compile_async(
            [source],
            manifest,
            fixture_vault,
            provider,
            dry_run=False,
        )
    )

    article_path = (
        fixture_vault.vault_path
        / fixture_vault.folders.wiki
        / fixture_vault.folders.wiki_concepts
        / "attention-mechanisms.md"
    )
    updated = article_path.read_text(encoding="utf-8")

    assert result.articles_updated == 1
    assert slug_to_urls == {"attention-mechanisms": [source.source_url]}
    assert source.source_url in manifest.sources
    assert manifest.sources[source.source_url].wiki_articles == ["attention-mechanisms"]
    assert old_source_url in manifest.wiki_articles["attention-mechanisms"].source_urls
    assert source.source_url in manifest.wiki_articles["attention-mechanisms"].source_urls
    assert f"  - {old_source_url}" in updated
    assert f"  - {source.source_url}" in updated


def test_deduplicate_concepts_preserves_existing_status_when_response_omits_status():
    concepts = [
        WikiConceptEntry(
            name="Attention",
            status=ConceptStatus.EXISTING,
            description="Existing concept",
            source_urls=["https://example.com/a"],
            merge_target="attention",
        ),
        WikiConceptEntry(
            name="Self Attention",
            status=ConceptStatus.NEW,
            description="New overlap",
            source_urls=["https://example.com/b"],
        ),
    ]
    provider = StubProvider(
        [
            json.dumps(
                {
                    "concepts": [
                        {
                            "name": "Attention Mechanisms",
                            "description": "Merged concept",
                            "source_urls": ["https://example.com/a", "https://example.com/b"],
                        }
                    ]
                }
            )
        ]
    )

    deduped = asyncio.run(_deduplicate_concepts(concepts, provider))

    assert len(deduped) == 1
    assert deduped[0].status == ConceptStatus.EXISTING
    assert deduped[0].merge_target == "attention"


def test_compile_sources_continues_when_one_concept_create_fails_and_names_it_in_errors(
    test_config, tmp_path
):
    """When one concept's article create fails, other concepts still succeed and error is named."""
    success_source = _raw_source(slug="success-source", source_url="https://example.com/success")
    failing_source = _raw_source(slug="fail-source", source_url="https://example.com/fail")

    # Setup wiki concepts directory for article writes
    article_dir = test_config.vault_path / test_config.folders.wiki / test_config.folders.wiki_concepts
    article_dir.mkdir(parents=True, exist_ok=True)

    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": "Success Concept",
                    "status": "new",
                    "description": "This one will succeed",
                    "source_urls": [success_source.source_url],
                },
                {
                    "name": "Failing Concept",
                    "status": "new",
                    "description": "This one will fail",
                    "source_urls": [failing_source.source_url],
                },
            ]
        }
    )

    # Dedup response — keep the two concepts as-is
    dedup = json.dumps(
        {
            "concepts": [
                {
                    "name": "Success Concept",
                    "status": "new",
                    "description": "This one will succeed",
                    "source_urls": [success_source.source_url],
                },
                {
                    "name": "Failing Concept",
                    "status": "new",
                    "description": "This one will fail",
                    "source_urls": [failing_source.source_url],
                },
            ]
        }
    )

    class FailingProvider(StubProvider):
        """Provider that fails on specific concept names."""

        def __init__(self):
            # Responses for triage, dedup, and one article creation (second will fail)
            super().__init__([
                triage,
                dedup,
                "# Success Concept\n\nSuccess content",
            ])

        async def complete(self, prompt: str, system: str = "") -> str:
            del system
            self.prompts.append(prompt)
            # Fail only on article creation prompts for "Failing Concept"
            # Article creation prompts contain "Write a new wiki article for the concept"
            if "Write a new wiki article" in prompt and "Failing Concept" in prompt:
                raise ValueError("Simulated article creation failure")
            response = self.responses.pop(0)
            return response

    provider = FailingProvider()

    result, slug_to_urls = asyncio.run(
        compile_sources(
            [success_source, failing_source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
        )
    )

    # Success concept should have been created
    assert result.articles_created == 1
    # One error should be recorded with the concept name
    assert len(result.errors) == 1
    assert "Concept 'Failing Concept' failed:" in result.errors[0]
    # Success concept should be in slug_to_urls
    assert "success-concept" in slug_to_urls
    # Article file should exist for success concept
    success_article = article_dir / "success-concept.md"
    assert success_article.exists()


def test_compile_run_skips_manifest_update_for_failed_source(test_config):
    """When one concept fails, only successful sources update the manifest."""
    success_source = _raw_source(slug="success-source", source_url="https://example.com/success")
    failing_source = _raw_source(slug="fail-source", source_url="https://example.com/fail")

    # Setup wiki concepts directory for article writes
    article_dir = test_config.vault_path / test_config.folders.wiki / test_config.folders.wiki_concepts
    article_dir.mkdir(parents=True, exist_ok=True)

    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": "Success Concept",
                    "status": "new",
                    "description": "This one will succeed",
                    "source_urls": [success_source.source_url],
                },
                {
                    "name": "Failing Concept",
                    "status": "new",
                    "description": "This one will fail",
                    "source_urls": [failing_source.source_url],
                },
            ]
        }
    )

    # Dedup response — keep the two concepts as-is
    dedup = json.dumps(
        {
            "concepts": [
                {
                    "name": "Success Concept",
                    "status": "new",
                    "description": "This one will succeed",
                    "source_urls": [success_source.source_url],
                },
                {
                    "name": "Failing Concept",
                    "status": "new",
                    "description": "This one will fail",
                    "source_urls": [failing_source.source_url],
                },
            ]
        }
    )

    class FailingProvider(StubProvider):
        """Provider that fails on specific concept names."""

        def __init__(self):
            # Responses for triage, dedup, and one article creation (second will fail)
            super().__init__([
                triage,
                dedup,
                "# Success Concept\n\nSuccess content",
            ])

        async def complete(self, prompt: str, system: str = "") -> str:
            del system
            self.prompts.append(prompt)
            # Fail only on article creation prompts for "Failing Concept"
            if "Write a new wiki article" in prompt and "Failing Concept" in prompt:
                raise ValueError("Simulated article creation failure")
            response = self.responses.pop(0)
            return response

    provider = FailingProvider()
    manifest = Manifest()

    # Call _run_compile_async directly to test manifest update behavior
    _result, _slug_to_urls = asyncio.run(
        _run_compile_async(
            [success_source, failing_source],
            manifest,
            test_config,
            provider,
            dry_run=False,
        )
    )

    # Verify the manifest was updated only for the successful source
    assert success_source.source_url in manifest.sources
    assert failing_source.source_url not in manifest.sources
    assert "success-concept" in manifest.wiki_articles
