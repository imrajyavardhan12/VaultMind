"""Tests for vm compile command behavior."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
import typer

from vaultmind.ai.providers.base import FailureKind, ProviderAttempt, ProviderExhaustedError
from vaultmind.ai.providers.fallback import FallbackProvider
from vaultmind.commands import compile as compile_cmd
from vaultmind.core.manifest import read_manifest, write_manifest
from vaultmind.core.raw_scanner import RawSourceRecord
from vaultmind.core.wiki_log import wiki_log_path
from vaultmind.schemas import Manifest, ManifestSource, ManifestWikiEntry
from vaultmind.utils.hashing import content_hash


def _provider_exhausted() -> ProviderExhaustedError:
    return ProviderExhaustedError(
        (ProviderAttempt("openai", "gpt", FailureKind.SERVER),)
    )


def _raw_source(
    *,
    slug: str,
    source_url: str | None,
) -> RawSourceRecord:
    return RawSourceRecord(
        path=Path(f"/tmp/{slug}.md"),
        relative_path=f"Clippings/{slug}",
        title=slug,
        source_url=source_url,
        body=f"# {slug}\n\nBody text",
        content_hash=f"hash-{slug}",
        raw_tags=[],
    )


def test_compile_refuses_corrupted_manifest_without_writes(monkeypatch, test_config):
    manifest_path = test_config.vault_path / "vault.manifest.json"
    manifest_path.write_text("{broken", encoding="utf-8")
    original = manifest_path.read_bytes()
    monkeypatch.setattr(compile_cmd, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(compile_cmd, "load_config", lambda: test_config)
    warnings: list[str] = []
    monkeypatch.setattr(compile_cmd, "print_warning", warnings.append)

    with pytest.raises(typer.Exit) as exc_info:
        compile_cmd.compile(full=False, dry_run=False, verbose=False)

    assert exc_info.value.exit_code == 1
    assert "refusing to compile" in warnings[0]
    assert manifest_path.read_bytes() == original
    assert not wiki_log_path(test_config).exists()


def test_compile_full_preserves_manifest_history(monkeypatch, test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    historical = "https://example.com/history"
    concepts_dir = (
        test_config.vault_path
        / test_config.folders.wiki
        / test_config.folders.wiki_concepts
    )
    concepts_dir.mkdir(parents=True)
    (concepts_dir / "concept-a.md").write_text(
        f"---\nsources:\n  - {historical}\n---\n\n# Concept A\n",
        encoding="utf-8",
    )
    now = datetime.now(UTC)
    manifest = Manifest(
        sources={
            historical: ManifestSource(
                content_hash="historical-hash",
                saved_at=now,
                wiki_articles=["concept-a"],
            ),
            source.source_url: ManifestSource(
                content_hash=source.content_hash,
                saved_at=now,
            ),
        },
        wiki_articles={
            "concept-a": ManifestWikiEntry(
                last_updated=now,
                source_urls=[historical],
                content_hash="old-hash",
            )
        },
    )
    captured: dict[str, Manifest] = {}
    monkeypatch.setattr(compile_cmd, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(compile_cmd, "load_config", lambda: test_config)
    monkeypatch.setattr(compile_cmd, "read_manifest", lambda vault_path: manifest)
    monkeypatch.setattr(compile_cmd, "scan_raw_sources", lambda config: [source])
    monkeypatch.setattr(compile_cmd, "get_provider", lambda config, tier="deep": object())
    monkeypatch.setattr(compile_cmd, "print_info", lambda message: None)
    monkeypatch.setattr(compile_cmd, "print_success", lambda title, message: None)
    monkeypatch.setattr(compile_cmd, "print_warning", lambda message: None)

    async def fake_run(sources, manifest_arg, config, provider, dry_run, **kwargs):
        captured["manifest"] = manifest_arg.model_copy(deep=True)
        assert sources == [source]
        return compile_cmd.CompileResult(0, 0, 1, []), {}

    monkeypatch.setattr(compile_cmd, "_run_compile_async", fake_run)
    compile_cmd.compile(full=True, dry_run=True, verbose=False)

    assert historical in captured["manifest"].sources
    assert "concept-a" in captured["manifest"].wiki_articles


def test_compile_repair_only_persists_without_provider_for_hash_repairs(
    monkeypatch, test_config
):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    concepts_dir = (
        test_config.vault_path
        / test_config.folders.wiki
        / test_config.folders.wiki_concepts
    )
    concepts_dir.mkdir(parents=True)
    concept_path = concepts_dir / "concept-a.md"
    concept_path.write_text(
        f"---\nsources:\n  - {source.source_url}\n---\n\n# Concept A\n",
        encoding="utf-8",
    )
    now = datetime.now(UTC)
    write_manifest(
        test_config.vault_path,
        Manifest(
            sources={
                source.source_url: ManifestSource(
                    content_hash=source.content_hash,
                    saved_at=now,
                    wiki_articles=["concept-a"],
                )
            },
            wiki_articles={
                "concept-a": ManifestWikiEntry(
                    last_updated=now,
                    source_urls=[source.source_url],
                    content_hash="stale",
                )
            },
        ),
    )
    monkeypatch.setattr(compile_cmd, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(compile_cmd, "load_config", lambda: test_config)
    monkeypatch.setattr(compile_cmd, "scan_raw_sources", lambda config: [source])
    monkeypatch.setattr(
        compile_cmd,
        "get_provider",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("provider not needed")),
    )
    monkeypatch.setattr(compile_cmd, "print_info", lambda message: None)
    monkeypatch.setattr(compile_cmd, "print_success", lambda title, message: None)

    compile_cmd.compile(full=False, dry_run=False, verbose=False)

    repaired = read_manifest(test_config.vault_path)
    assert repaired.wiki_articles["concept-a"].content_hash == content_hash(
        concept_path.read_text(encoding="utf-8")
    )
    assert "manifest repair" in wiki_log_path(test_config).read_text(encoding="utf-8")


def test_compile_membership_repair_rebuilds_index(monkeypatch, test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    concepts_dir = (
        test_config.vault_path
        / test_config.folders.wiki
        / test_config.folders.wiki_concepts
    )
    concepts_dir.mkdir(parents=True)
    (concepts_dir / "concept-a.md").write_text("# Concept A\n", encoding="utf-8")
    now = datetime.now(UTC)
    manifest = Manifest(
        sources={
            source.source_url: ManifestSource(
                content_hash=source.content_hash,
                saved_at=now,
            )
        }
    )
    rebuilt: list[set[str]] = []
    monkeypatch.setattr(compile_cmd, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(compile_cmd, "load_config", lambda: test_config)
    monkeypatch.setattr(compile_cmd, "read_manifest", lambda vault_path: manifest)
    monkeypatch.setattr(compile_cmd, "scan_raw_sources", lambda config: [source])
    monkeypatch.setattr(compile_cmd, "get_provider", lambda config, tier="deep": object())
    monkeypatch.setattr(compile_cmd, "print_info", lambda message: None)
    monkeypatch.setattr(compile_cmd, "print_success", lambda title, message: None)
    monkeypatch.setattr(
        compile_cmd,
        "_rebuild_wiki_index",
        lambda config, repaired, provider: rebuilt.append(set(repaired.wiki_articles)),
    )

    compile_cmd.compile(full=False, dry_run=False, verbose=False)

    assert rebuilt == [{"concept-a"}]
    assert "concept-a" in read_manifest(test_config.vault_path).wiki_articles


def test_compile_dry_run_does_not_persist_reconciliation(monkeypatch, test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    now = datetime.now(UTC)
    manifest = Manifest(
        sources={
            source.source_url: ManifestSource(
                content_hash=source.content_hash,
                saved_at=now,
            )
        },
        wiki_articles={
            "missing-concept": ManifestWikiEntry(last_updated=now, content_hash="old")
        },
    )
    write_manifest(test_config.vault_path, manifest)
    original = (test_config.vault_path / "vault.manifest.json").read_bytes()
    monkeypatch.setattr(compile_cmd, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(compile_cmd, "load_config", lambda: test_config)
    monkeypatch.setattr(compile_cmd, "scan_raw_sources", lambda config: [source])
    monkeypatch.setattr(compile_cmd, "print_info", lambda message: None)
    monkeypatch.setattr(compile_cmd, "print_success", lambda title, message: None)

    compile_cmd.compile(full=False, dry_run=True, verbose=False)

    assert (test_config.vault_path / "vault.manifest.json").read_bytes() == original
    assert not wiki_log_path(test_config).exists()


def test_compile_incremental_includes_relative_path_only_sources(monkeypatch, test_config):
    from vaultmind.commands import compile as compile_cmd

    source = _raw_source(slug="raw-a", source_url=None)
    called: dict[str, object] = {"run_called": False}

    monkeypatch.setattr(compile_cmd, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(compile_cmd, "load_config", lambda: test_config)
    monkeypatch.setattr(compile_cmd, "read_manifest", lambda vault_path: Manifest())
    monkeypatch.setattr(compile_cmd, "scan_raw_sources", lambda config: [source])
    monkeypatch.setattr(compile_cmd, "get_provider", lambda config, tier="deep": object())
    monkeypatch.setattr(compile_cmd, "print_info", lambda message: None)
    monkeypatch.setattr(compile_cmd, "print_success", lambda title, message: None)
    monkeypatch.setattr(compile_cmd, "print_warning", lambda message: None)

    async def fake_run(sources, manifest, config, provider, dry_run, **kwargs):
        called["run_called"] = True
        called["sources"] = sources
        return (
            compile_cmd.CompileResult(
                articles_created=0,
                articles_updated=0,
                sources_compiled=len(sources),
                errors=[],
            ),
            {},
        )

    monkeypatch.setattr(compile_cmd, "_run_compile_async", fake_run)

    compile_cmd.compile(full=False, dry_run=True, verbose=False)

    assert called["run_called"] is True
    sources = called["sources"]
    assert isinstance(sources, list)
    assert len(sources) == 1
    assert sources[0].relative_path == "Clippings/raw-a"


def test_run_compile_async_upserts_manifest_with_relative_path_key(monkeypatch, test_config):
    from vaultmind.commands import compile as compile_cmd

    source = _raw_source(slug="raw-a", source_url=None)
    manifest = Manifest()

    async def fake_compile_sources(
        sources,
        manifest_arg,
        provider,
        vault_path,
        folders,
        *,
        dry_run=False,
        existing_concepts=None,
        max_touches=5,
    ):
        del manifest_arg, provider, vault_path, folders, dry_run, existing_concepts, max_touches
        return (
            compile_cmd.CompileResult(
                articles_created=1,
                articles_updated=0,
                sources_compiled=len(sources),
                errors=[],
            ),
            {"concept-a": [source.relative_path]},
        )

    monkeypatch.setattr(compile_cmd, "compile_sources", fake_compile_sources)

    result, slug_to_urls = asyncio.run(
        compile_cmd._run_compile_async(
            [source],
            manifest,
            test_config,
            provider=object(),
            dry_run=False,
        )
    )

    assert result.articles_created == 1
    assert slug_to_urls == {"concept-a": ["Clippings/raw-a"]}
    assert source.relative_path in manifest.sources
    assert manifest.sources[source.relative_path].wiki_articles == ["concept-a"]


def test_run_compile_async_merges_one_source_into_multiple_concepts(monkeypatch, test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    manifest = Manifest()
    concepts_dir = test_config.vault_path / test_config.folders.wiki / test_config.folders.wiki_concepts
    concepts_dir.mkdir(parents=True, exist_ok=True)
    (concepts_dir / "concept-a.md").write_text("# Concept A\n", encoding="utf-8")
    (concepts_dir / "concept-b.md").write_text("# Concept B\n", encoding="utf-8")

    async def fake_compile_sources(
        sources,
        manifest_arg,
        provider,
        vault_path,
        folders,
        *,
        dry_run=False,
        existing_concepts=None,
        max_touches=5,
    ):
        del manifest_arg, provider, vault_path, folders, dry_run, existing_concepts, max_touches
        return (
            compile_cmd.CompileResult(
                articles_created=2,
                articles_updated=0,
                sources_compiled=len(sources),
                errors=[],
            ),
            {
                "concept-a": [source.source_url],
                "concept-b": [source.source_url],
            },
        )

    monkeypatch.setattr(compile_cmd, "compile_sources", fake_compile_sources)

    result, _slug_to_urls = asyncio.run(
        compile_cmd._run_compile_async(
            [source],
            manifest,
            test_config,
            provider=object(),
            dry_run=False,
        )
    )

    assert result.articles_created == 2
    assert manifest.sources[source.source_url].wiki_articles == ["concept-a", "concept-b"]
    assert manifest.wiki_articles["concept-a"].source_urls == [source.source_url]
    assert manifest.wiki_articles["concept-b"].source_urls == [source.source_url]


def test_run_compile_async_writes_manifest_and_compile_log(monkeypatch, test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    concepts_dir = test_config.vault_path / test_config.folders.wiki / test_config.folders.wiki_concepts
    concepts_dir.mkdir(parents=True, exist_ok=True)
    (concepts_dir / "concept-a.md").write_text("# Concept A\n", encoding="utf-8")
    manifest = Manifest()

    async def fake_compile_sources(
        sources,
        manifest_arg,
        provider,
        vault_path,
        folders,
        *,
        dry_run=False,
        existing_concepts=None,
        max_touches=5,
    ):
        del manifest_arg, provider, vault_path, folders, dry_run, existing_concepts, max_touches
        return (
            compile_cmd.CompileResult(
                articles_created=1,
                articles_updated=0,
                sources_compiled=len(sources),
                errors=[],
            ),
            {"concept-a": [source.source_url]},
        )

    monkeypatch.setattr(compile_cmd, "compile_sources", fake_compile_sources)

    result, _slug_to_urls = asyncio.run(
        compile_cmd._run_compile_async(
            [source],
            manifest,
            test_config,
            provider=object(),
            dry_run=False,
        )
    )

    written_manifest = read_manifest(test_config.vault_path)
    log_text = wiki_log_path(test_config).read_text(encoding="utf-8")

    assert result.articles_created == 1
    assert source.source_url in written_manifest.sources
    assert written_manifest.sources[source.source_url].wiki_articles == ["concept-a"]
    assert "## [" in log_text
    assert "compile | 1 created, 0 updated, 1 source(s)" in log_text


def test_run_compile_async_preserves_article_frontmatter_sources_in_manifest(
    monkeypatch,
    test_config,
):
    old_source_url = "https://example.com/old-source"
    new_source = _raw_source(slug="raw-b", source_url="https://example.com/new-source")
    concepts_dir = test_config.vault_path / test_config.folders.wiki / test_config.folders.wiki_concepts
    concepts_dir.mkdir(parents=True, exist_ok=True)
    (concepts_dir / "concept-a.md").write_text(
        "\n".join(
            [
                "---",
                "title: Concept A",
                "vaultmind: true",
                "kind: concept",
                "sources:",
                f"  - {old_source_url}",
                "---",
                "",
                "# Concept A",
                "",
                "Updated content",
            ]
        ),
        encoding="utf-8",
    )
    manifest = Manifest()

    async def fake_compile_sources(
        sources,
        manifest_arg,
        provider,
        vault_path,
        folders,
        *,
        dry_run=False,
        existing_concepts=None,
        max_touches=5,
    ):
        del manifest_arg, provider, vault_path, folders, dry_run, existing_concepts, max_touches
        return (
            compile_cmd.CompileResult(
                articles_created=0,
                articles_updated=1,
                sources_compiled=len(sources),
                errors=[],
            ),
            {"concept-a": [new_source.source_url]},
        )

    monkeypatch.setattr(compile_cmd, "compile_sources", fake_compile_sources)

    asyncio.run(
        compile_cmd._run_compile_async(
            [new_source],
            manifest,
            test_config,
            provider=object(),
            dry_run=False,
        )
    )

    assert manifest.sources[new_source.source_url].wiki_articles == ["concept-a"]
    assert manifest.wiki_articles["concept-a"].source_urls == [
        old_source_url,
        new_source.source_url,
    ]


def test_render_dry_run_summary_includes_sources_and_targets():
    from vaultmind.commands import compile as compile_cmd

    source = _raw_source(slug="raw-a", source_url=None)

    summary = compile_cmd._render_dry_run_summary(
        [source],
        {"concept-a": [source.relative_path]},
    )

    assert "Would process 1 raw source(s)." in summary
    assert "raw-a [Clippings/raw-a]" in summary
    assert "→ concept-a" in summary
    assert "Clippings/raw-a" in summary


def test_compile_dry_run_writes_no_state_files(monkeypatch, test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    index_path = test_config.vault_path / test_config.folders.wiki / f"{test_config.folders.wiki_index}.md"
    concept_path = (
        test_config.vault_path
        / test_config.folders.wiki
        / test_config.folders.wiki_concepts
        / "concept-a.md"
    )

    monkeypatch.setattr(compile_cmd, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(compile_cmd, "load_config", lambda: test_config)
    monkeypatch.setattr(compile_cmd, "read_manifest", lambda vault_path: Manifest())
    monkeypatch.setattr(compile_cmd, "scan_raw_sources", lambda config: [source])
    monkeypatch.setattr(compile_cmd, "get_provider", lambda config, tier="deep": object())
    monkeypatch.setattr(compile_cmd, "print_info", lambda message: None)
    monkeypatch.setattr(compile_cmd, "print_warning", lambda message: None)
    monkeypatch.setattr(compile_cmd, "print_success", lambda title, message: None)

    async def fake_run(sources, manifest, config, provider, dry_run, **kwargs):
        assert dry_run is True
        return (
            compile_cmd.CompileResult(
                articles_created=1,
                articles_updated=0,
                sources_compiled=len(sources),
                errors=[],
            ),
            {"concept-a": [source.source_url]},
        )

    monkeypatch.setattr(compile_cmd, "_run_compile_async", fake_run)

    compile_cmd.compile(full=False, dry_run=True, verbose=False)

    assert not (test_config.vault_path / "vault.manifest.json").exists()
    assert not wiki_log_path(test_config).exists()
    assert not index_path.exists()
    assert not concept_path.exists()


def test_compile_incremental_skips_unchanged_sources(monkeypatch, test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    now = compile_cmd.datetime.now(compile_cmd.UTC)
    manifest = Manifest(
        sources={
            source.source_url: ManifestSource(
                content_hash=source.content_hash,
                saved_at=now,
                compiled_at=now,
                wiki_articles=["concept-a"],
            )
        }
    )

    monkeypatch.setattr(compile_cmd, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(compile_cmd, "load_config", lambda: test_config)
    monkeypatch.setattr(compile_cmd, "read_manifest", lambda vault_path: manifest)
    monkeypatch.setattr(compile_cmd, "scan_raw_sources", lambda config: [source])
    monkeypatch.setattr(compile_cmd, "print_success", lambda title, message: None)
    monkeypatch.setattr(compile_cmd, "print_warning", lambda message: None)

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("unchanged sources should not compile")

    monkeypatch.setattr(compile_cmd, "_run_compile_async", fail_if_called)

    compile_cmd.compile(full=False, dry_run=False, verbose=False)


def test_rebuild_wiki_index_writes_wikilinks(test_config):
    class IndexProvider:
        model = "stub-index"

        async def complete(self, prompt: str, system: str = "") -> str:
            del prompt, system
            return "# Wiki Index\n\n- **Concept A**"

    concepts_dir = test_config.vault_path / test_config.folders.wiki / test_config.folders.wiki_concepts
    concepts_dir.mkdir(parents=True, exist_ok=True)
    article_path = concepts_dir / "concept-a.md"
    article_path.write_text("---\ntitle: Concept A\n---\n\n# Concept A\n", encoding="utf-8")
    manifest = Manifest(
        wiki_articles={
            "concept-a": ManifestWikiEntry(
                last_updated=compile_cmd.datetime.now(compile_cmd.UTC),
                source_urls=["https://example.com/raw-a"],
                content_hash=content_hash(article_path.read_text(encoding="utf-8")),
            )
        }
    )

    compile_cmd._rebuild_wiki_index(test_config, manifest, IndexProvider())

    index_path = test_config.vault_path / test_config.folders.wiki / f"{test_config.folders.wiki_index}.md"
    assert "[[concept-a|Concept A]]" in index_path.read_text(encoding="utf-8")


def test_compile_exits_nonzero_when_errors_present(monkeypatch, test_config):
    """When compile result has errors, exit with code 1 and print warnings."""
    source = _raw_source(slug="raw-a", source_url=None)
    warnings_printed = []
    captured_exit_code = None

    monkeypatch.setattr(compile_cmd, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(compile_cmd, "load_config", lambda: test_config)
    monkeypatch.setattr(compile_cmd, "read_manifest", lambda vault_path: Manifest())
    monkeypatch.setattr(compile_cmd, "scan_raw_sources", lambda config: [source])
    monkeypatch.setattr(compile_cmd, "get_provider", lambda config, tier="deep": object())
    monkeypatch.setattr(compile_cmd, "print_info", lambda message: None)
    monkeypatch.setattr(compile_cmd, "print_success", lambda title, message: None)

    def capture_warning(message: str):
        warnings_printed.append(message)

    monkeypatch.setattr(compile_cmd, "print_warning", capture_warning)

    async def fake_run(sources, manifest, config, provider, dry_run, **kwargs):
        return (
            compile_cmd.CompileResult(
                articles_created=0,
                articles_updated=0,
                sources_compiled=len(sources),
                errors=["Concept 'Test Concept' failed: TestError"],
            ),
            {},
        )

    monkeypatch.setattr(compile_cmd, "_run_compile_async", fake_run)

    def mock_exit(code: int):
        nonlocal captured_exit_code
        captured_exit_code = code
        raise StopIteration(code)

    monkeypatch.setattr(typer, "Exit", mock_exit)

    with contextlib.suppress(StopIteration):
        compile_cmd.compile(full=False, dry_run=False, verbose=False)

    assert captured_exit_code == 1
    assert len(warnings_printed) == 1
    assert "Concept 'Test Concept' failed: TestError" in warnings_printed[0]


def test_compile_prints_each_error_with_concept_name(monkeypatch, test_config):
    """Each error in result.errors is printed via print_warning."""
    source = _raw_source(slug="raw-a", source_url=None)
    warnings_printed = []
    captured_exit_code = None

    monkeypatch.setattr(compile_cmd, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(compile_cmd, "load_config", lambda: test_config)
    monkeypatch.setattr(compile_cmd, "read_manifest", lambda vault_path: Manifest())
    monkeypatch.setattr(compile_cmd, "scan_raw_sources", lambda config: [source])
    monkeypatch.setattr(compile_cmd, "get_provider", lambda config, tier="deep": object())
    monkeypatch.setattr(compile_cmd, "print_info", lambda message: None)
    monkeypatch.setattr(compile_cmd, "print_success", lambda title, message: None)

    def capture_warning(message: str):
        warnings_printed.append(message)

    monkeypatch.setattr(compile_cmd, "print_warning", capture_warning)

    async def fake_run(sources, manifest, config, provider, dry_run, **kwargs):
        return (
            compile_cmd.CompileResult(
                articles_created=0,
                articles_updated=0,
                sources_compiled=len(sources),
                errors=[
                    "Concept 'Concept A' failed: error1",
                    "Concept 'Concept B' failed: error2",
                ],
            ),
            {},
        )

    monkeypatch.setattr(compile_cmd, "_run_compile_async", fake_run)

    def mock_exit(code: int):
        nonlocal captured_exit_code
        captured_exit_code = code
        raise StopIteration(code)

    monkeypatch.setattr(typer, "Exit", mock_exit)

    with contextlib.suppress(StopIteration):
        compile_cmd.compile(full=False, dry_run=False, verbose=False)

    assert captured_exit_code == 1
    assert len(warnings_printed) == 2
    assert "Concept 'Concept A' failed: error1" in warnings_printed
    assert "Concept 'Concept B' failed: error2" in warnings_printed


def test_compile_no_extra_warning_when_no_errors(monkeypatch, test_config):
    """When no errors occur, print_success is called and no warnings/exit."""
    source = _raw_source(slug="raw-a", source_url=None)
    warnings_printed = []
    success_printed = []
    exit_raised = False

    monkeypatch.setattr(compile_cmd, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(compile_cmd, "load_config", lambda: test_config)
    monkeypatch.setattr(compile_cmd, "read_manifest", lambda vault_path: Manifest())
    monkeypatch.setattr(compile_cmd, "scan_raw_sources", lambda config: [source])
    monkeypatch.setattr(compile_cmd, "get_provider", lambda config, tier="deep": object())
    monkeypatch.setattr(compile_cmd, "print_info", lambda message: None)

    def capture_warning(message: str):
        warnings_printed.append(message)

    def capture_success(title: str, message: str):
        success_printed.append((title, message))

    monkeypatch.setattr(compile_cmd, "print_warning", capture_warning)
    monkeypatch.setattr(compile_cmd, "print_success", capture_success)

    async def fake_run(sources, manifest, config, provider, dry_run, **kwargs):
        return (
            compile_cmd.CompileResult(
                articles_created=1,
                articles_updated=0,
                sources_compiled=len(sources),
                errors=[],
            ),
            {"concept-a": [source.relative_path]},
        )

    monkeypatch.setattr(compile_cmd, "_run_compile_async", fake_run)
    monkeypatch.setattr(compile_cmd, "_rebuild_wiki_index", lambda config, manifest, provider: None)

    def mock_exit(code: int):
        nonlocal exit_raised
        exit_raised = True
        raise StopIteration(code)

    monkeypatch.setattr(typer, "Exit", mock_exit)

    compile_cmd.compile(full=False, dry_run=False, verbose=False)

    assert exit_raised is False
    assert len(warnings_printed) == 0
    assert len(success_printed) == 1
    assert success_printed[0][0] == "Compile complete"


# ---- --max-touches plumbing ----


def test_run_compile_async_threads_max_touches_into_compile_sources(monkeypatch, test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    captured: dict[str, object] = {}

    async def fake_compile_sources(
        sources,
        manifest_arg,
        provider,
        vault_path,
        folders,
        *,
        dry_run=False,
        existing_concepts=None,
        max_touches=5,
    ):
        del manifest_arg, provider, vault_path, folders, dry_run, existing_concepts
        captured["max_touches"] = max_touches
        return (
            compile_cmd.CompileResult(
                articles_created=0,
                articles_updated=0,
                sources_compiled=len(sources),
                errors=[],
            ),
            {},
        )

    monkeypatch.setattr(compile_cmd, "compile_sources", fake_compile_sources)

    asyncio.run(
        compile_cmd._run_compile_async(
            [source],
            Manifest(),
            test_config,
            provider=object(),
            dry_run=True,
            max_touches=7,
        )
    )

    assert captured["max_touches"] == 7


def test_run_compile_async_propagation_touches_land_in_manifest(monkeypatch, test_config):
    """Touched concept pages should appear in manifest.wiki_articles[target].source_urls
    via the existing disk-scanning manifest-rebuild loop — no extra code path."""
    import json

    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    concepts_dir = (
        test_config.vault_path / test_config.folders.wiki / test_config.folders.wiki_concepts
    )
    concepts_dir.mkdir(parents=True, exist_ok=True)
    target_path = concepts_dir / "existing-target.md"
    target_path.write_text(
        "\n".join(
            [
                "---",
                'title: "Existing Target"',
                "vaultmind: true",
                "kind: concept",
                "sources:",
                "  - https://example.com/legacy",
                "---",
                "",
                "# Existing Target",
                "",
                "## Overview",
                "",
                "Some body.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Real StubProvider with three responses: triage, article create, touch.
    class StubProvider:
        def __init__(self, responses):
            self.responses = responses
            self.model = "stub"

        async def complete(self, prompt: str, system: str = "") -> str:
            del system, prompt
            return self.responses.pop(0)

    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": "New Concept",
                    "status": "new",
                    "description": "A new concept",
                    "source_urls": [source.source_url],
                    "merge_target": None,
                }
            ]
        }
    )
    touch = json.dumps(
        {
            "touches": [
                {
                    "target_slug": "existing-target",
                    "connection_line": "[[new-concept|New Concept]] — related.",
                    "relevance": 8,
                }
            ]
        }
    )
    provider = StubProvider([triage, "# New Concept\n\nBody", touch])

    manifest = Manifest()
    asyncio.run(
        compile_cmd._run_compile_async(
            [source],
            manifest,
            test_config,
            provider,
            dry_run=False,
            max_touches=5,
        )
    )

    # The touched concept must now have the new source in its manifest entry,
    # alongside the pre-existing legacy URL parsed from frontmatter.
    assert "existing-target" in manifest.wiki_articles
    target_urls = manifest.wiki_articles["existing-target"].source_urls
    assert "https://example.com/legacy" in target_urls
    assert source.source_url in target_urls
    assert "existing-target" in manifest.sources[source.source_url].wiki_articles


def test_compile_cli_max_touches_flag_default_is_five(monkeypatch, test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    captured: dict[str, object] = {}

    monkeypatch.setattr(compile_cmd, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(compile_cmd, "load_config", lambda: test_config)
    monkeypatch.setattr(compile_cmd, "read_manifest", lambda vault_path: Manifest())
    monkeypatch.setattr(compile_cmd, "scan_raw_sources", lambda config: [source])
    monkeypatch.setattr(compile_cmd, "get_provider", lambda config, tier="deep": object())
    monkeypatch.setattr(compile_cmd, "print_info", lambda message: None)
    monkeypatch.setattr(compile_cmd, "print_success", lambda title, message: None)
    monkeypatch.setattr(compile_cmd, "print_warning", lambda message: None)

    async def fake_run(sources, manifest, config, provider, dry_run, *, max_touches=5):
        captured["max_touches"] = max_touches
        return (
            compile_cmd.CompileResult(
                articles_created=0,
                articles_updated=0,
                sources_compiled=len(sources),
                errors=[],
            ),
            {},
        )

    monkeypatch.setattr(compile_cmd, "_run_compile_async", fake_run)

    compile_cmd.compile(full=False, dry_run=True, verbose=False, max_touches=5)

    assert captured["max_touches"] == 5

def test_compile_prints_success_on_clean_noop(monkeypatch, test_config):
    """When compile processes zero sources (no creations, updates, or errors), still print success."""
    source = _raw_source(slug="raw-a", source_url=None)
    warnings_printed = []
    success_printed = []

    monkeypatch.setattr(compile_cmd, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(compile_cmd, "load_config", lambda: test_config)
    monkeypatch.setattr(compile_cmd, "read_manifest", lambda vault_path: Manifest())
    monkeypatch.setattr(compile_cmd, "scan_raw_sources", lambda config: [source])
    monkeypatch.setattr(compile_cmd, "get_provider", lambda config, tier="deep": object())
    monkeypatch.setattr(compile_cmd, "print_info", lambda message: None)

    def capture_warning(message: str):
        warnings_printed.append(message)

    def capture_success(title: str, message: str):
        success_printed.append((title, message))

    monkeypatch.setattr(compile_cmd, "print_warning", capture_warning)
    monkeypatch.setattr(compile_cmd, "print_success", capture_success)

    async def fake_run(sources, manifest, config, provider, dry_run, **kwargs):
        return (
            compile_cmd.CompileResult(
                articles_created=0,
                articles_updated=0,
                sources_compiled=len(sources),
                errors=[],
            ),
            {},
        )

    monkeypatch.setattr(compile_cmd, "_run_compile_async", fake_run)
    monkeypatch.setattr(compile_cmd, "_rebuild_wiki_index", lambda config, manifest, provider: None)

    def mock_exit(code: int):
        raise StopIteration(code)

    monkeypatch.setattr(typer, "Exit", mock_exit)

    compile_cmd.compile(full=False, dry_run=False, verbose=False)

    assert len(warnings_printed) == 0
    assert len(success_printed) == 1
    assert success_printed[0][0] == "Compile complete"


def test_run_compile_async_propagation_failure_preserves_retry_hash_and_backlinks(
    monkeypatch, test_config
):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    concepts_dir = (
        test_config.vault_path
        / test_config.folders.wiki
        / test_config.folders.wiki_concepts
    )
    concepts_dir.mkdir(parents=True, exist_ok=True)
    (concepts_dir / "concept-a.md").write_text(
        "---\ntitle: Concept A\nsources:\n"
        f"  - {source.source_url}\n---\n\n# Concept A\n",
        encoding="utf-8",
    )
    manifest = Manifest(
        sources={
            source.source_url: ManifestSource(
                content_hash="previous-hash",
                saved_at=datetime.now(UTC),
                wiki_articles=["older-concept"],
            )
        }
    )

    async def fake_compile_sources(*args, **kwargs):
        del args, kwargs
        return (
            compile_cmd.CompileResult(
                articles_created=1,
                articles_updated=0,
                sources_compiled=1,
                errors=["Propagation call failed"],
                propagation_touches_by_source={source.source_url: ["touched-concept"]},
                propagation_failed_sources={source.source_url},
            ),
            {"concept-a": [source.source_url]},
        )

    monkeypatch.setattr(compile_cmd, "compile_sources", fake_compile_sources)

    asyncio.run(
        compile_cmd._run_compile_async(
            [source], manifest, test_config, provider=object(), dry_run=False
        )
    )

    entry = manifest.sources[source.source_url]
    assert entry.content_hash == "previous-hash"
    assert entry.wiki_articles == ["older-concept", "concept-a", "touched-concept"]
    assert compile_cmd.get_changed_sources(
        manifest, {source.source_url: source.content_hash}
    ) == [source.source_url]


def test_compile_provider_exhaustion_is_concise_normally(monkeypatch):
    errors: list[str] = []

    def fail_compile(**kwargs):
        del kwargs
        raise _provider_exhausted() from RuntimeError("diagnostic secret")

    monkeypatch.setattr(compile_cmd, "_compile", fail_compile)
    monkeypatch.setattr(compile_cmd, "print_error", errors.append)

    with pytest.raises(typer.Exit) as exc_info:
        compile_cmd.compile(full=False, dry_run=False, verbose=False, max_touches=5)

    assert exc_info.value.exit_code == 1
    assert errors == ["AI provider chain exhausted: openai/gpt: provider server failure"]
    assert "diagnostic secret" not in errors[0]


def test_compile_verbose_preserves_provider_exception_chain(monkeypatch):
    cause = RuntimeError("diagnostic detail")

    def fail_compile(**kwargs):
        del kwargs
        raise _provider_exhausted() from cause

    monkeypatch.setattr(compile_cmd, "_compile", fail_compile)
    monkeypatch.setattr(compile_cmd, "print_error", lambda message: None)

    with pytest.raises(ProviderExhaustedError) as exc_info:
        compile_cmd.compile(full=False, dry_run=False, verbose=True, max_touches=5)

    assert exc_info.value.__cause__ is cause


def test_index_rebuild_propagates_provider_exhaustion_without_writing(test_config):
    class FailingProvider:
        name = "ollama"
        model = "local-model"

        @staticmethod
        def classify_failure(exc: Exception) -> FailureKind:
            del exc
            return FailureKind.CONNECTION

        @staticmethod
        def is_retryable_failure(exc: Exception) -> bool:
            del exc
            return True

        async def complete(self, prompt: str, system: str = "") -> str:
            del prompt, system
            raise ConnectionError("private endpoint")

    concepts_dir = (
        test_config.vault_path
        / test_config.folders.wiki
        / test_config.folders.wiki_concepts
    )
    concepts_dir.mkdir(parents=True)
    article = concepts_dir / "concept-a.md"
    article.write_text("# Concept A\n", encoding="utf-8")
    manifest = Manifest(
        wiki_articles={
            "concept-a": ManifestWikiEntry(
                last_updated=datetime.now(UTC),
                content_hash=content_hash(article.read_text(encoding="utf-8")),
            )
        }
    )

    with pytest.raises(ProviderExhaustedError):
        compile_cmd._rebuild_wiki_index(
            test_config, manifest, FallbackProvider([FailingProvider()])
        )

    index_path = (
        test_config.vault_path
        / test_config.folders.wiki
        / f"{test_config.folders.wiki_index}.md"
    )
    assert not index_path.exists()
