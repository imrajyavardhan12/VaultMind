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
from vaultmind.ai.providers import Provider, ProviderExhaustedError, get_provider
from vaultmind.config import AppConfig, load_config
from vaultmind.core.manifest import (
    ManifestReadError,
    ManifestReconciliation,
    get_changed_sources,
    read_manifest,
    reconcile_manifest,
    update_compiled_at,
    upsert_source,
    upsert_wiki_article,
    write_manifest,
)
from vaultmind.core.raw_scanner import RawSourceRecord, scan_raw_sources
from vaultmind.core.wiki_log import append_wiki_log
from vaultmind.core.writer import write_markdown_page
from vaultmind.schemas import Manifest, ManifestSource
from vaultmind.utils.display import print_error, print_info, print_success, print_warning
from vaultmind.utils.hashing import content_hash
from vaultmind.utils.logging import setup_logging

__all__ = ["reconcile_manifest"]

log = structlog.get_logger()


def _validate_max_touches(value: int) -> int:
    """Reject negative --max-touches values at the Typer parse layer."""
    if value < 0:
        raise typer.BadParameter("--max-touches must be >= 0")
    return value


def _concepts_dir(config: AppConfig) -> Path:
    return config.vault_path / config.folders.wiki / config.folders.wiki_concepts


def _report_reconciliation(
    reconciliation: ManifestReconciliation,
    *,
    dry_run: bool,
) -> None:
    if not reconciliation.changed:
        return
    prefix = "Planned manifest reconciliation" if dry_run else "Manifest reconciliation"
    print_info(f"{prefix}: {len(reconciliation.repairs)} repair(s)")
    for repair in reconciliation.repairs:
        print_info(f"  - {repair}")


def _persist_repairs(
    config: AppConfig,
    reconciliation: ManifestReconciliation,
    *,
    provider: Provider | None,
) -> None:
    if not reconciliation.changed:
        return
    write_manifest(config.vault_path, reconciliation.manifest)
    append_wiki_log(
        config,
        event="manifest repair",
        detail=f"{len(reconciliation.repairs)} repair(s)",
    )
    if reconciliation.concept_membership_changed and provider is not None:
        _rebuild_wiki_index(config, reconciliation.manifest, provider)


def compile(
    full: bool = typer.Option(
        False,
        "--full",
        help="Force recompilation of all Raw sources without resetting manifest state",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would change without writing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
    max_touches: int = typer.Option(
        5,
        "--max-touches",
        help=(
            "Max existing concept pages each source may touch in its "
            "Connections section. 0 disables propagation."
        ),
        callback=_validate_max_touches,
    ),
) -> None:
    """Compile source notes into wiki concept articles."""
    try:
        _compile(full=full, dry_run=dry_run, verbose=verbose, max_touches=max_touches)
    except ProviderExhaustedError as exc:
        print_error(str(exc))
        if verbose:
            raise
        raise typer.Exit(1) from None


def _compile(*, full: bool, dry_run: bool, verbose: bool, max_touches: int) -> None:
    """Implement compile while the CLI boundary handles provider exhaustion."""
    setup_logging(verbose=verbose)
    config = load_config()

    try:
        manifest = read_manifest(config.vault_path)
    except ManifestReadError as exc:
        print_warning(
            f"Manifest is unreadable; refusing to compile or write to the vault. {exc}"
        )
        raise typer.Exit(1) from exc

    # Scan Raw and reconcile only against canonical state already on disk. Raw
    # entries are deliberately not inserted until their compilation succeeds.
    all_sources = scan_raw_sources(config)
    raw_keys = {source.source_url or source.relative_path for source in all_sources}
    pre_reconciliation = reconcile_manifest(
        manifest,
        concepts_dir=_concepts_dir(config),
        current_raw_keys=raw_keys,
    )
    manifest = pre_reconciliation.manifest
    _report_reconciliation(pre_reconciliation, dry_run=dry_run)

    if not all_sources:
        if not dry_run:
            _persist_repairs(config, pre_reconciliation, provider=None)
            if pre_reconciliation.concept_membership_changed:
                provider = get_provider(config, tier="deep")
                _rebuild_wiki_index(config, manifest, provider)
        print_warning(
            f"No raw sources found in {config.folders.raw}/. "
            "Add articles via Obsidian Web Clipper first."
        )
        return

    source_key_to_source = {source.source_url or source.relative_path: source for source in all_sources}
    if full:
        sources_to_compile = all_sources
        print_info(f"Forced full recompile: {len(sources_to_compile)} raw sources")
    else:
        source_hashes = {
            source.source_url or source.relative_path: source.content_hash
            for source in all_sources
        }
        changed_source_keys = get_changed_sources(manifest, source_hashes)
        sources_to_compile = [
            source_key_to_source[key]
            for key in changed_source_keys
            if key in source_key_to_source
        ]
        if sources_to_compile:
            print_info(f"Incremental compile: {len(sources_to_compile)} new/changed sources")

    if not sources_to_compile:
        if dry_run:
            print_success("Dry run", "No Raw sources would be compiled.")
        elif pre_reconciliation.changed:
            repair_provider = (
                get_provider(config, tier="deep")
                if pre_reconciliation.concept_membership_changed
                else None
            )
            _persist_repairs(config, pre_reconciliation, provider=repair_provider)
            print_success("Manifest repaired", "No Raw sources needed compilation.")
        else:
            print_success("All sources are up to date", "Nothing to compile.")
        return

    provider = get_provider(config, tier="deep")
    result, slug_to_urls = asyncio.run(
        _run_compile_async(
            sources_to_compile,
            manifest,
            config,
            provider,
            dry_run,
            max_touches=max_touches,
        )
    )

    if dry_run:
        print_success("Dry run", _render_dry_run_summary(sources_to_compile, slug_to_urls))
        return

    post_reconciliation = reconcile_manifest(
        manifest,
        concepts_dir=_concepts_dir(config),
        current_raw_keys=raw_keys,
    )
    manifest = post_reconciliation.manifest
    if post_reconciliation.changed:
        write_manifest(config.vault_path, manifest)

    if pre_reconciliation.changed:
        # _run_compile_async may already have persisted compile state. Record the
        # independent disk repair so repair activity remains visible.
        append_wiki_log(
            config,
            event="manifest repair",
            detail=f"{len(pre_reconciliation.repairs)} repair(s)",
        )
        if not (
            result.articles_created
            or result.articles_updated
            or result.articles_touched
            or post_reconciliation.changed
        ):
            write_manifest(config.vault_path, manifest)

    membership_changed = (
        pre_reconciliation.concept_membership_changed
        or post_reconciliation.concept_membership_changed
    )
    if result.articles_created > 0 or result.articles_updated > 0 or membership_changed:
        _rebuild_wiki_index(config, manifest, provider)

    # Print success unless we're in the full-failure case
    # (no creations, no updates, no touches, but errors exist).
    if not (
        result.articles_created == 0
        and result.articles_updated == 0
        and result.articles_touched == 0
        and result.errors
    ):
        print_success(
            "Compile complete",
            f"{result.articles_created} created, "
            f"{result.articles_updated} updated, "
            f"{result.articles_touched} touched, "
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
    *,
    max_touches: int = 5,
) -> tuple[CompileResult, dict[str, list[str]]]:
    result, slug_to_urls = await compile_sources(
        sources,
        manifest,
        provider,
        config.vault_path,
        config.folders,
        dry_run=dry_run,
        existing_concepts=_existing_concept_summaries(config),
        max_touches=max_touches,
    )

    if not dry_run and (
        result.articles_created > 0
        or result.articles_updated > 0
        or result.articles_touched > 0
    ):
        # Reconcile article provenance in both directions. A propagation failure
        # may happen after article writes, so retain its previous hash (or a
        # non-current placeholder for a new source) to make incremental retry
        # deterministic while still recording any durable article references.
        source_key_to_source = {s.source_url or s.relative_path: s for s in sources}
        source_to_articles: dict[str, list[str]] = {}
        for slug, urls in slug_to_urls.items():
            for source_key in urls:
                articles = source_to_articles.setdefault(source_key, [])
                if slug not in articles:
                    articles.append(slug)
        for source_key, touched_slugs in result.propagation_touches_by_source.items():
            articles = source_to_articles.setdefault(source_key, [])
            articles.extend(slug for slug in touched_slugs if slug not in articles)

        for source_key, articles in source_to_articles.items():
            source = source_key_to_source.get(source_key)
            if source is None:
                continue
            if source_key in result.propagation_failed_sources:
                _merge_failed_source_provenance(
                    manifest,
                    source_key=source_key,
                    wiki_articles=articles,
                )
                continue
            upsert_source(
                manifest,
                url=source_key,
                content_hash=source.content_hash,
                saved_at=datetime.now(UTC),
                wiki_articles=articles,
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


def _merge_failed_source_provenance(
    manifest: Manifest,
    *,
    source_key: str,
    wiki_articles: list[str],
) -> None:
    """Record durable backlinks without marking a failed source current."""
    existing = manifest.sources.get(source_key)
    if existing is not None:
        merged_articles = list(dict.fromkeys([*existing.wiki_articles, *wiki_articles]))
        manifest.sources[source_key] = existing.model_copy(
            update={"wiki_articles": merged_articles}
        )
        return

    manifest.sources[source_key] = ManifestSource(
        content_hash="",
        saved_at=datetime.now(UTC),
        compiled_at=None,
        wiki_articles=list(dict.fromkeys(wiki_articles)),
    )


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

    lines.extend(["", "Propagation: deferred in dry-run."])

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
