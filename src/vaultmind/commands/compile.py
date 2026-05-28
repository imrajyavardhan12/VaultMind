"""vm compile — compile source notes into wiki concept articles.

Usage:
    vm compile              # incremental: only new/changed sources
    vm compile --full      # full rebuild: all sources
    vm compile --dry-run   # show what would be compiled
    vm compile --verbose   # debug logging
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from pathlib import Path

import structlog
import typer
import yaml

from vaultmind.ai.compiler import CompileResult, compile_sources, rebuild_index
from vaultmind.ai.providers import Provider, get_provider
from vaultmind.config import AppConfig, load_config
from vaultmind.core.manifest import (
    get_changed_sources,
    read_manifest,
    update_compiled_at,
    upsert_source,
    upsert_wiki_article,
    write_manifest,
)
from vaultmind.core.raw_scanner import RawSourceRecord, scan_raw_sources
from vaultmind.core.wiki_log import append_wiki_log
from vaultmind.core.writer import write_markdown_page
from vaultmind.schemas import Manifest
from vaultmind.utils.display import print_info, print_success, print_warning
from vaultmind.utils.hashing import content_hash
from vaultmind.utils.logging import setup_logging

log = structlog.get_logger()


def compile(
    full: bool = typer.Option(False, "--full", help="Full rebuild — skip manifest diff"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would change without writing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """Compile source notes into wiki concept articles.

    Run without flags for incremental compilation (only new/changed sources).
    Use --full to recompile everything.
    """
    setup_logging(verbose=verbose)
    config = load_config()

    manifest = read_manifest(config.vault_path)

    # Scan raw sources from the Raw/ folder (populated by Obsidian Web Clipper)
    all_sources = scan_raw_sources(config)
    if not all_sources:
        print_warning(
            f"No raw sources found in {config.folders.raw}/. "
            "Add articles via Obsidian Web Clipper first."
        )
        return

    # Build source key -> record map (key is source_url, fallback to relative_path)
    source_key_to_source = {s.source_url or s.relative_path: s for s in all_sources}

    # Determine which sources to compile
    if full:
        sources_to_compile = all_sources
        print_info(f"Full rebuild: {len(sources_to_compile)} raw sources")
    else:
        source_hashes = {s.source_url or s.relative_path: s.content_hash for s in all_sources}
        changed_source_keys = get_changed_sources(
            manifest,
            {k: v for k, v in source_hashes.items() if k is not None},
        )
        sources_to_compile = [
            source_key_to_source[key]
            for key in changed_source_keys
            if key in source_key_to_source
        ]
        if not sources_to_compile:
            print_success("All sources are up to date", "Nothing to compile.")
            return
        print_info(f"Incremental compile: {len(sources_to_compile)} new/changed sources")

    if not sources_to_compile:
        print_warning("No sources to compile.")
        return

    # On full rebuild, reset manifest to start fresh
    if full:
        manifest = Manifest()

    # Run compile pipeline
    provider = get_provider(config, tier="deep")

    result, slug_to_urls = asyncio.run(
        _run_compile_async(sources_to_compile, manifest, config, provider, dry_run)
    )

    if dry_run:
        print_success(
            "Dry run",
            _render_dry_run_summary(sources_to_compile, slug_to_urls),
        )
        return

    # Write wiki index
    if result.articles_created > 0 or result.articles_updated > 0:
        _rebuild_wiki_index(config, manifest, provider)

    # Print success unless we're in full-failure case (no creations, no updates, but errors exist)
    if not (result.articles_created == 0 and result.articles_updated == 0 and result.errors):
        print_success(
            "Compile complete",
            f"{result.articles_created} created, "
            f"{result.articles_updated} updated, "
            f"{result.sources_compiled} sources processed."
        )

    # Print errors and exit non-zero if any occurred
    if result.errors:
        for error in result.errors:
            print_warning(error)
        raise typer.Exit(1)


async def _run_compile_async(
    sources: list[RawSourceRecord],
    manifest: Manifest,
    config: AppConfig,
    provider: Provider,
    dry_run: bool,
) -> tuple[CompileResult, dict[str, list[str]]]:
    result, slug_to_urls = await compile_sources(
        sources,
        manifest,
        provider,
        config.vault_path,
        config.folders,
        dry_run=dry_run,
        existing_concepts=_existing_concept_summaries(config),
    )

    if not dry_run and (result.articles_created > 0 or result.articles_updated > 0):
        # slug_to_urls maps slug → source_urls
        # For raw sources, manifest keys are source_url (preferred) or relative_path
        source_key_to_source = {s.source_url or s.relative_path: s for s in sources}

        for slug, urls in slug_to_urls.items():
            for url in urls:
                source = source_key_to_source.get(url)
                if source is None:
                    continue
                upsert_source(
                    manifest,
                    url=url,
                    content_hash=source.content_hash,
                    saved_at=datetime.now(UTC),
                    wiki_articles=[slug],
                )

        # Rebuild manifest wiki_articles from disk (scan wiki concepts directory)
        wiki_concepts_dir = config.vault_path / config.folders.wiki / config.folders.wiki_concepts
        if wiki_concepts_dir.exists():
            for article_path in wiki_concepts_dir.glob("*.md"):
                slug = article_path.stem
                body = article_path.read_text(encoding="utf-8")
                article_hash = content_hash(body)
                # Find source URLs that fed this article, preserving any
                # existing frontmatter provenance on the concept page.
                manifest_source_urls = [
                    url for url, entry in manifest.sources.items()
                    if slug in entry.wiki_articles
                ]
                source_urls = _merge_source_urls(
                    _extract_article_sources(article_path),
                    manifest_source_urls,
                )
                upsert_wiki_article(
                    manifest,
                    slug=slug,
                    content_hash=article_hash,
                    source_urls=source_urls,
                )

        update_compiled_at(manifest)
        write_manifest(config.vault_path, manifest)
        append_wiki_log(
            config,
            event="compile",
            detail=(
                f"{result.articles_created} created, "
                f"{result.articles_updated} updated, "
                f"{result.sources_compiled} source(s)"
            ),
        )

    return result, slug_to_urls


def _extract_article_sources(article_path: Path) -> list[str]:
    """Extract source URLs/keys from concept article frontmatter."""
    try:
        text = article_path.read_text(encoding="utf-8")
    except OSError:
        return []

    if not text.startswith("---"):
        return []

    end = text.find("---", 3)
    if end == -1:
        return []

    try:
        data = yaml.safe_load(text[3:end])
    except yaml.YAMLError:
        return []

    if not isinstance(data, dict):
        return []

    sources = data.get("sources")
    if isinstance(sources, list):
        parsed_sources: list[str] = []
        for source in sources:
            if source is None:
                continue
            source_text = str(source).strip()
            if source_text:
                parsed_sources.append(source_text)
        return parsed_sources
    if isinstance(sources, str) and sources.strip():
        return [sources.strip()]
    return []


def _merge_source_urls(*source_groups: list[str]) -> list[str]:
    """Merge source URL/key lists while preserving first-seen order."""
    merged: list[str] = []
    seen: set[str] = set()
    for group in source_groups:
        for source in group:
            if source in seen:
                continue
            seen.add(source)
            merged.append(source)
    return merged


def _extract_article_title(article_path: Path) -> str:
    """Extract title from frontmatter or first H1 heading of an article."""
    try:
        text = article_path.read_text(encoding="utf-8")
    except Exception:
        return article_path.stem.replace("-", " ").title()

    # Try frontmatter title first
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            fm_text = text[3:end]
            for line in fm_text.splitlines():
                if line.startswith("title:"):
                    return line.split(":", 1)[1].strip().strip('"').strip("'")

    # Fall back to first H1 heading
    heading = re.match(r"^#\s+(.+?)\s*$", text.strip(), re.MULTILINE)
    if heading:
        return heading.group(1).strip()

    return article_path.stem.replace("-", " ").title()


def _existing_concept_summaries(config: AppConfig) -> list[tuple[str, str]]:
    """Return existing concept slugs and titles for compile triage."""
    wiki_concepts_dir = config.vault_path / config.folders.wiki / config.folders.wiki_concepts
    if not wiki_concepts_dir.exists():
        return []

    summaries: list[tuple[str, str]] = []
    for article_path in wiki_concepts_dir.glob("*.md"):
        summaries.append((article_path.stem, _extract_article_title(article_path)))
    return summaries


def _render_dry_run_summary(
    sources: list[RawSourceRecord],
    slug_to_urls: dict[str, list[str]],
) -> str:
    """Render a useful no-write compile plan for humans to review."""
    lines = [
        f"Would process {len(sources)} raw source(s).",
        "",
        "Sources:",
    ]

    for source in sources:
        key = source.source_url or source.relative_path
        lines.append(f"  - {source.title} [{key}]")

    lines.extend(["", f"Concept targets: {len(slug_to_urls)}"])

    if not slug_to_urls:
        lines.append("  - No concept targets were returned by triage.")
        return "\n".join(lines)

    for slug, urls in sorted(slug_to_urls.items()):
        lines.append(f"  → {slug}")
        for url in urls:
            lines.append(f"     - {url}")

    return "\n".join(lines)


def _rebuild_wiki_index(config: AppConfig, manifest: Manifest, provider: Provider) -> None:
    """Rebuild the Wiki/📇 Index.md."""
    wiki_dir = config.vault_path / config.folders.wiki
    wiki_dir.mkdir(parents=True, exist_ok=True)

    index_path = wiki_dir / f"{config.folders.wiki_index}.md"
    existing_index = ""
    if index_path.exists():
        existing_index = index_path.read_text(encoding="utf-8")

    wiki_concepts_dir = config.vault_path / config.folders.wiki / config.folders.wiki_concepts

    # Build article summaries — extract title from frontmatter or heading
    summaries: list[tuple[str, str]] = []
    for slug, _entry in manifest.wiki_articles.items():
        if wiki_concepts_dir.exists():
            article_path = wiki_concepts_dir / f"{slug}.md"
            title = (
                _extract_article_title(article_path)
                if article_path.exists()
                else slug.replace("-", " ").title()
            )
        else:
            title = slug.replace("-", " ").title()
        summaries.append((slug, title))

    if summaries:
        rebuilt = asyncio.run(rebuild_index(existing_index, summaries, provider))
        write_markdown_page(index_path, body=rebuilt)
    elif existing_index:
        # Clear the index if no articles
        write_markdown_page(index_path, body="# Wiki Index\n\nNo articles yet.")


if __name__ == "__main__":
    typer.run(compile)
