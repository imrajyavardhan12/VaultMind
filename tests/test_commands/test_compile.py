"""Tests for vm compile command behavior."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import typer

from vaultmind.commands import compile as compile_cmd
from vaultmind.core.manifest import read_manifest
from vaultmind.core.raw_scanner import RawSourceRecord
from vaultmind.core.wiki_log import wiki_log_path
from vaultmind.schemas import Manifest, ManifestSource, ManifestWikiEntry
from vaultmind.utils.hashing import content_hash


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

    async def fake_run(sources, manifest, config, provider, dry_run):
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
    ):
        del manifest_arg, provider, vault_path, folders, dry_run, existing_concepts
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
    ):
        del manifest_arg, provider, vault_path, folders, dry_run, existing_concepts
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
    ):
        del manifest_arg, provider, vault_path, folders, dry_run, existing_concepts
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

    async def fake_run(sources, manifest, config, provider, dry_run):
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

    async def fake_run(sources, manifest, config, provider, dry_run):
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

    async def fake_run(sources, manifest, config, provider, dry_run):
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

    async def fake_run(sources, manifest, config, provider, dry_run):
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

    async def fake_run(sources, manifest, config, provider, dry_run):
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
