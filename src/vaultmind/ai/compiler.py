"""Compile pipeline — sources to wiki concept articles.

Three stages:
1. Concept triage — extract concepts from new/changed sources
2. Article create/update — LLM writes or updates concept articles
3. Index rebuild — update Wiki/📇 Index.md
"""

from __future__ import annotations

__all__ = ["CompileResult", "compile_sources", "rebuild_index"]

import asyncio
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import structlog

from vaultmind.ai.json_utils import clean_json_response
from vaultmind.ai.prompts import (
    COMPILE_ARTICLE_CREATE_PROMPT,
    COMPILE_ARTICLE_UPDATE_PROMPT,
    COMPILE_CONCEPT_DEDUP_PROMPT,
    COMPILE_CONCEPT_TRIAGE_PROMPT,
    COMPILE_INDEX_REBUILD_PROMPT,
)
from vaultmind.ai.providers.base import Provider
from vaultmind.config import FolderConfig
from vaultmind.core.raw_scanner import (
    RawSourceRecord,
    _strip_broken_images,
    format_raw_source_packet,
)
from vaultmind.schemas import ConceptStatus, Manifest, WikiConceptEntry

log = structlog.get_logger()


@dataclass
class CompileResult:
    """Result of a compile run."""

    articles_created: int
    articles_updated: int
    sources_compiled: int
    errors: list[str]


def _extract_h1_title(body: str) -> str | None:
    """Extract title from the first H1 heading in a markdown body."""
    match = re.search(r"^#\s+(.+?)\s*$", body.strip(), re.MULTILINE)
    return match.group(1).strip() if match else None


def _strip_h1(body: str) -> str:
    """Remove the first H1 heading from a markdown body."""
    lines = body.splitlines()
    if lines and re.match(r"^#\s+", lines[0]):
        return "\n".join(lines[1:]).lstrip("\n")
    return body


def _extract_frontmatter(markdown: str) -> dict[str, object]:
    """Extract a markdown frontmatter block as a mapping."""
    if not markdown.startswith("---"):
        return {}

    end = markdown.find("---", 3)
    if end == -1:
        return {}

    try:
        import yaml

        data = yaml.safe_load(markdown[3:end])
    except yaml.YAMLError:
        return {}

    if not isinstance(data, dict):
        return {}
    return data


def _extract_frontmatter_title(markdown: str) -> str | None:
    """Extract a title from wiki article frontmatter."""
    title = _extract_frontmatter(markdown).get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return None


def _extract_frontmatter_sources(markdown: str) -> list[str]:
    """Extract source URLs/keys from a wiki article frontmatter block."""
    data = _extract_frontmatter(markdown)
    if not data:
        return []

    sources = data.get("sources")
    if isinstance(sources, list):
        return [str(source) for source in sources if source]
    if isinstance(sources, str):
        return [sources]
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


def slugify(text: str) -> str:
    """Convert a concept name to a filesystem-safe slug."""
    text = unicodedata.normalize("NFC", text.strip().lower())
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text.strip("-")[:80] or "untitled-concept"


def _source_key(source: RawSourceRecord) -> str:
    """Return the stable manifest/triage key for a raw source."""
    return source.source_url or source.relative_path


async def compile_sources(
    sources: list[RawSourceRecord],
    manifest: Manifest,
    provider: Provider,
    vault_path: Path,
    folders: FolderConfig,
    *,
    dry_run: bool = False,
    existing_concepts: list[tuple[str, str]] | None = None,
) -> tuple[CompileResult, dict[str, list[str]]]:
    """Run the full compile pipeline on raw source documents.

    Returns (CompileResult, slug_to_urls_map) where the map tracks which
    source URLs informed each wiki article slug.
    """
    result = CompileResult(
        articles_created=0,
        articles_updated=0,
        sources_compiled=len(sources),
        errors=[],
    )

    # slug → list of source URLs that informed this article
    slug_to_urls: dict[str, list[str]] = {}

    if not sources:
        return result, slug_to_urls

    # Build source key → source lookup once, reused across all stages. Raw files
    # clipped without a source URL are keyed by their vault-relative path.
    url_to_source = {_source_key(source): source for source in sources}

    # Stage 1: concept triage — uses RAW source content, not AI summaries
    log.info("compile_triage_start", count=len(sources))
    sources_payload = "\n\n---\n\n".join(format_raw_source_packet(s) for s in sources)
    concepts_payload = _format_existing_concepts(existing_concepts or [])

    try:
        triage_response = await provider.complete(
            COMPILE_CONCEPT_TRIAGE_PROMPT.format(
                existing_concepts=concepts_payload,
                new_sources=sources_payload,
            ),
            system="You are a precise librarian. Return only valid JSON.",
        )
    except Exception as exc:
        log.error("compile_triage_failed", error=str(exc))
        result.errors.append(f"Triage failed: {exc}")
        return result, slug_to_urls

    concepts = _parse_concept_triage(triage_response)
    log.info("compile_triage_done", concepts=len(concepts))

    # Stage 1b: deduplicate overlapping concepts
    if len(concepts) > 1:
        concepts = await _deduplicate_concepts(concepts, provider)
        log.info("compile_dedup_done", concepts=len(concepts))

    # Stage 2: create or update each article — all run concurrently
    async def _process_concept(concept: WikiConceptEntry) -> tuple[str, str, bool, bool]:
        """Process a single concept, return (slug, target_slug, was_created, was_updated)."""
        slug = slugify(concept.name)

        if concept.status == ConceptStatus.NEW:
            if dry_run:
                log.info("compile_dry_run_create", concept=concept.name)
                return (slug, slug, False, False)
            await _create_and_write_article(slug, concept, provider, vault_path, folders)
            log.info("compile_article_done", slug=slug, status="created")
            return (slug, slug, True, False)

        # EXISTING or MERGE
        target_slug = concept.merge_target or slug
        if dry_run:
            log.info("compile_dry_run_update", concept=concept.name, target=target_slug)
            return (slug, target_slug, False, False)

        existing_content = _read_wiki_article_content(manifest, target_slug, vault_path, folders)
        if existing_content:
            await _update_and_write_article(
                target_slug, existing_content, concept, url_to_source, provider, vault_path, folders
            )
            log.info("compile_article_done", slug=target_slug, status="updated")
            return (slug, target_slug, False, True)
        else:
            await _create_and_write_article(target_slug, concept, provider, vault_path, folders)
            log.info("compile_article_done", slug=target_slug, status="created_new")
            return (slug, target_slug, True, False)

    async def _process_concept_with_error_handling(
        concept: WikiConceptEntry,
    ) -> tuple[str, str, bool, bool] | None:
        """Wrap _process_concept to attach concept name to any exceptions."""
        try:
            return await _process_concept(concept)
        except Exception as exc:
            result.errors.append(f"Concept '{concept.name}' failed: {exc}")
            return None

    processed = await asyncio.gather(
        *[_process_concept_with_error_handling(c) for c in concepts], return_exceptions=False
    )

    for item in processed:
        if item is None:
            continue
        slug, target_slug, was_created, was_updated = item
        result.articles_created += was_created
        result.articles_updated += was_updated
        # Accumulate source URLs — deduplicate to avoid repeated manifest entries
        concept = next((c for c in concepts if slugify(c.name) == slug), None)
        if concept:
            if target_slug not in slug_to_urls:
                slug_to_urls[target_slug] = []
            for url in concept.source_urls:
                if url not in slug_to_urls[target_slug]:
                    slug_to_urls[target_slug].append(url)

    return result, slug_to_urls


def _format_existing_concepts(existing_concepts: list[tuple[str, str]]) -> str:
    """Format existing concept slugs for the triage prompt."""
    if not existing_concepts:
        return "No existing concept pages yet."
    return "\n".join(f"- {slug}: {title}" for slug, title in sorted(existing_concepts))


async def _create_article(
    concept: WikiConceptEntry,
    provider: Provider,
) -> str:
    """LLM call to create a new wiki article from a concept."""
    source_urls_str = "\n".join(f"- {url}" for url in concept.source_urls)
    prompt = COMPILE_ARTICLE_CREATE_PROMPT.format(
        concept_name=concept.name,
        description=concept.description,
        source_urls=source_urls_str,
    )

    wiki_author = "You are a research wiki author. Write markdown only."
    response = await provider.complete(prompt, system=wiki_author)
    return response.strip()


async def _update_article(
    existing_content: str,
    concept: WikiConceptEntry,
    url_to_source: dict[str, RawSourceRecord],
    provider: Provider,
) -> str:
    """LLM call to update an existing wiki article with new source info."""
    source_blocks: list[str] = []
    for url in concept.source_urls:
        source = url_to_source.get(url)
        if source:
            # Strip broken CDN images from source body
            clean_body = _strip_broken_images(source.body[:800])
            source_blocks.append(f"Source: {url}\n\n{clean_body}")
        else:
            source_blocks.append(f"Source: {url}")

    if source_blocks:
        source_urls_str = "\n\n".join(source_blocks)
    else:
        source_urls_str = "\n".join(f"- {url}" for url in concept.source_urls)

    prompt = COMPILE_ARTICLE_UPDATE_PROMPT.format(
        existing_content=existing_content,
        new_sources=source_urls_str,
    )

    wiki_author = "You are a research wiki author. Write markdown only."
    response = await provider.complete(prompt, system=wiki_author)
    return response.strip()


async def _create_and_write_article(
    slug: str,
    concept: WikiConceptEntry,
    provider: Provider,
    vault_path: Path,
    folders: FolderConfig,
) -> None:
    """Create a wiki article via LLM and write it to disk."""
    body = await _create_article(concept, provider)
    _write_wiki_article(slug, body, concept.name, concept.source_urls, vault_path, folders)


async def _update_and_write_article(
    slug: str,
    existing_content: str,
    concept: WikiConceptEntry,
    url_to_source: dict[str, RawSourceRecord],
    provider: Provider,
    vault_path: Path,
    folders: FolderConfig,
) -> None:
    """Update a wiki article via LLM and write it to disk."""
    body = await _update_article(existing_content, concept, url_to_source, provider)
    title = (
        _extract_frontmatter_title(existing_content)
        or _extract_h1_title(existing_content)
        or slug.replace("-", " ").title()
    )
    source_urls = _merge_source_urls(
        _extract_frontmatter_sources(existing_content),
        concept.source_urls,
    )
    _write_wiki_article(slug, body, title, source_urls, vault_path, folders)


def _write_wiki_article(
    slug: str,
    body: str,
    title: str,
    source_urls: list[str],
    vault_path: Path,
    folders: FolderConfig,
) -> None:
    """Write a wiki article to disk with frontmatter and atomic write."""
    wiki_concepts_dir = vault_path / folders.wiki / folders.wiki_concepts
    wiki_concepts_dir.mkdir(parents=True, exist_ok=True)

    article_path = wiki_concepts_dir / f"{slug}.md"

    content_body = _strip_h1(body)

    frontmatter = {
        "title": title,
        "vaultmind": True,
        "kind": "concept",
        "sources": source_urls,
    }
    fm_lines = ["---"]
    for k, v in frontmatter.items():
        if isinstance(v, list):
            fm_lines.append(f"{k}:")
            for item in v:
                fm_lines.append(f"  - {item}")
        else:
            fm_lines.append(f"{k}: {v}")
    fm_lines.append("---")
    frontmatter_str = "\n".join(fm_lines)

    content = f"{frontmatter_str}\n\n{content_body.strip()}\n"

    import tempfile
    fd, tmp = tempfile.mkstemp(dir=wiki_concepts_dir, suffix=".md.tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(content)
        Path(tmp).replace(article_path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise

    log.info("wiki_article_written", slug=slug, path=str(article_path))


async def rebuild_index(
    existing_index: str,
    article_summaries: list[tuple[str, str]],
    provider: Provider,
) -> str:
    """Rebuild the Wiki/📇 Index.md based on current concept articles.

    article_summaries: list of (slug, title)
    """
    summaries_str = "\n".join(
        f"- [[{slug}|{title}]]"
        for slug, title in article_summaries
    )

    # Build title -> slug mapping for post-processing
    title_to_slug = {title: slug for slug, title in article_summaries}

    prompt = COMPILE_INDEX_REBUILD_PROMPT.format(
        existing_index=existing_index,
        article_summaries=summaries_str,
    )

    index_maintainer = "You are a wiki index maintainer. Write markdown only."
    response = await provider.complete(prompt, system=index_maintainer)
    result = response.strip()

    # Post-process: convert **Title** back to [[slug|Title]] using known slugs
    result = _convert_bold_to_wikilink(result, title_to_slug)

    return result


def _convert_bold_to_wikilink(text: str, title_to_slug: dict[str, str]) -> str:
    """Convert markdown bold text **Title** back to Obsidian [[slug|Title]] links."""
    def replace_bold(match: re.Match[str]) -> str:
        title = match.group(1).strip()
        slug = title_to_slug.get(title)
        if slug:
            return f"[[{slug}|{title}]]"
        return match.group(0)  # Keep original if no slug match
    return re.sub(r"\*\*(.+?)\*\*", replace_bold, text)


def _parse_concept_triage(response: str) -> list[WikiConceptEntry]:
    """Parse the LLM's concept triage JSON response."""
    cleaned = clean_json_response(response)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        log.warning("concept_triage_parse_failed", response=response[:200])
        return []

    concepts: list[WikiConceptEntry] = []
    raw_concepts = data.get("concepts", [])

    if not isinstance(raw_concepts, list):
        return []

    for item in raw_concepts:
        concept = _parse_concept_item(item)
        if concept is not None:
            concepts.append(concept)

    return concepts


def _parse_concept_item(item: object) -> WikiConceptEntry | None:
    """Parse a raw concept dict from a model response."""
    if not isinstance(item, dict):
        return None

    name = item.get("name", "")
    status_raw = item.get("status", "new")
    description = item.get("description", "")
    source_urls = item.get("source_urls", [])
    merge_target_raw = item.get("merge_target")

    if not isinstance(name, str) or not name.strip():
        return None

    status, merge_target = _parse_concept_status(status_raw, merge_target_raw)

    return WikiConceptEntry(
        name=name.strip(),
        status=status,
        description=description if isinstance(description, str) else "",
        source_urls=[u for u in source_urls if isinstance(u, str)] if isinstance(source_urls, list) else [],
        merge_target=merge_target,
    )


def _parse_concept_status(
    status_raw: object,
    merge_target_raw: object = None,
) -> tuple[ConceptStatus, str | None]:
    """Parse `new`, `existing:slug`, or `merge:slug` status strings."""
    merge_target = merge_target_raw.strip() if isinstance(merge_target_raw, str) else None

    if not isinstance(status_raw, str):
        return ConceptStatus.NEW, None

    status_text = status_raw.strip().lower()
    if status_text.startswith("existing:"):
        return ConceptStatus.EXISTING, status_text.split(":", 1)[1].strip() or merge_target
    if status_text.startswith("merge:"):
        return ConceptStatus.MERGE, status_text.split(":", 1)[1].strip() or merge_target
    if status_text == "existing" and merge_target:
        return ConceptStatus.EXISTING, merge_target
    if status_text == "merge" and merge_target:
        return ConceptStatus.MERGE, merge_target
    return ConceptStatus.NEW, None


def _format_concept_status(concept: WikiConceptEntry) -> str:
    """Format concept status for round-tripping through the dedupe prompt."""
    if concept.status == ConceptStatus.NEW:
        return "new"
    if concept.merge_target:
        return f"{concept.status.value}:{concept.merge_target}"
    return concept.status.value


async def _deduplicate_concepts(
    concepts: list[WikiConceptEntry],
    provider: Provider,
) -> list[WikiConceptEntry]:
    """Merge overlapping/near-duplicate concepts via LLM."""
    import yaml

    concepts_data = [
        {
            "name": c.name,
            "status": _format_concept_status(c),
            "description": c.description,
            "source_urls": c.source_urls,
            "merge_target": c.merge_target,
        }
        for c in concepts
    ]

    concepts_yaml = yaml.dump(
        {"concepts": concepts_data},
        default_flow_style=False,
        allow_unicode=True,
    )

    prompt = COMPILE_CONCEPT_DEDUP_PROMPT.format(concepts=concepts_yaml)
    librarian = "You are a precise librarian. Return only valid JSON."
    response = await provider.complete(prompt, system=librarian)

    cleaned = clean_json_response(response)
    try:
        data = json.loads(cleaned)
        deduped = data.get("concepts", [])
        if not isinstance(deduped, list):
            return concepts

        result: list[WikiConceptEntry] = []
        for item in deduped:
            concept = _parse_concept_item(item)
            if concept is None:
                continue
            if isinstance(item, dict) and not item.get("status"):
                concept = _infer_dedup_status(concept, concepts)
            result.append(concept)
        return result
    except json.JSONDecodeError:
        log.warning("concept_dedup_parse_failed", response=response[:200])
        return concepts


def _infer_dedup_status(
    deduped: WikiConceptEntry,
    original_concepts: list[WikiConceptEntry],
) -> WikiConceptEntry:
    """Preserve existing/merge status when a dedupe response omits status.

    Older prompt responses may only return name/description/source_urls. If the
    deduped concept clearly came from an existing concept, keep updating the
    existing wiki article instead of accidentally creating a new one.
    """
    deduped_urls = set(deduped.source_urls)
    deduped_slug = slugify(deduped.name)

    for original in original_concepts:
        if original.status == ConceptStatus.NEW:
            continue
        same_name = slugify(original.name) == deduped_slug
        shares_source = bool(deduped_urls & set(original.source_urls))
        if same_name or shares_source:
            return deduped.model_copy(
                update={
                    "status": original.status,
                    "merge_target": original.merge_target,
                }
            )

    return deduped


def _read_wiki_article_content(
    manifest: Manifest,
    slug: str,
    vault_path: Path,
    folders: FolderConfig,
) -> str:
    """Read the current content of an existing wiki article from disk.

    Returns empty string if the file doesn't exist.
    """
    wiki_path = vault_path / folders.wiki / folders.wiki_concepts / f"{slug}.md"
    if wiki_path.exists():
        return wiki_path.read_text(encoding="utf-8").strip()
    return ""
