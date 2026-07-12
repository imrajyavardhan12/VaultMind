"""Compile pipeline — sources to wiki concept articles.

Three stages:
1. Concept triage — extract concepts from new/changed sources.
2. Article create/update — LLM writes or updates concept articles.
3. Cross-page propagation — patch existing concept pages so they cross-link
   to the new/updated concepts (Connections + Sources only).

Index rebuild lives in vaultmind.commands.compile (it depends on the manifest
state assembled at the command layer, not on this module).
"""

from __future__ import annotations

__all__ = ["CompileResult", "compile_sources", "rebuild_index"]

import asyncio
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from vaultmind.ai.json_utils import clean_json_response
from vaultmind.ai.prompts import (
    COMPILE_ARTICLE_CREATE_PROMPT,
    COMPILE_ARTICLE_UPDATE_PROMPT,
    COMPILE_CONCEPT_DEDUP_PROMPT,
    COMPILE_CONCEPT_TRIAGE_PROMPT,
    COMPILE_GRAPH_TOUCH_PROMPT,
    COMPILE_INDEX_REBUILD_PROMPT,
)
from vaultmind.ai.providers.base import Provider
from vaultmind.config import FolderConfig
from vaultmind.core.linter import extract_wikilinks
from vaultmind.core.raw_scanner import RawSourceRecord, format_raw_source_packet
from vaultmind.core.writer import write_markdown_page
from vaultmind.schemas import ConceptStatus, Manifest, WikiConceptEntry

log = structlog.get_logger()


@dataclass
class CompileResult:
    """Result of a compile run."""

    articles_created: int
    articles_updated: int
    sources_compiled: int
    errors: list[str]
    articles_touched: int = 0
    propagation_touches_by_source: dict[str, list[str]] = field(default_factory=dict)
    propagation_failed_sources: set[str] = field(default_factory=set)


def _extract_h1_title(body: str) -> str | None:
    """Extract title from the first H1 heading in a markdown body."""
    match = re.search(r"^#\s+(.+?)\s*$", body.strip(), re.MULTILINE)
    return match.group(1).strip() if match else None


def _markdown_line_text(line: str) -> str:
    """Remove only a line ending, retaining all Markdown-significant whitespace."""
    return line[:-2] if line.endswith("\r\n") else line[:-1] if line.endswith("\n") else line


def _strip_model_frontmatter(body: str) -> str:
    """Remove leading model-authored YAML frontmatter, including CRLF form."""
    lines = body.splitlines(keepends=True)
    if not lines:
        return body

    opening_index = 0
    while opening_index < len(lines) and not _markdown_line_text(lines[opening_index]).strip():
        opening_index += 1
    if opening_index == len(lines) or not re.fullmatch(
        r"---[ \t]*", _markdown_line_text(lines[opening_index])
    ):
        return body

    for closing_index in range(opening_index + 1, len(lines)):
        if re.fullmatch(r"---[ \t]*", _markdown_line_text(lines[closing_index])):
            return "".join(lines[:opening_index] + lines[closing_index + 1 :])
    return body


def _strip_model_page_wrappers(body: str) -> str:
    """Remove model metadata and real H1s with Markdown fence awareness.

    Original line endings, indentation, and all non-wrapper content are kept.
    Fences follow CommonMark's relevant rules: 0--3 leading spaces, matching
    backtick/tilde marker, and a closing run at least as long as the opener
    followed only by whitespace.
    """
    cleaned = _strip_model_frontmatter(body)
    lines = cleaned.splitlines(keepends=True)
    remove: set[int] = set()
    outside: list[bool] = [False] * len(lines)
    fence_char: str | None = None
    fence_length = 0

    for index, line in enumerate(lines):
        text = _markdown_line_text(line)
        outside[index] = fence_char is None
        if fence_char is not None:
            closer = re.fullmatch(rf" {{0,3}}({re.escape(fence_char)}+)[ \t]*", text)
            if closer and len(closer.group(1)) >= fence_length:
                fence_char = None
                fence_length = 0
            continue

        opener = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", text)
        if opener:
            marker, remainder = opener.groups()
            # A backtick fence's info string cannot itself contain a backtick.
            if marker[0] == "~" or "`" not in remainder:
                fence_char = marker[0]
                fence_length = len(marker)
                continue

        if re.match(r"^ {0,3}#(?:[ \t]+|$)", text):
            remove.add(index)
            continue

        if not re.fullmatch(r" {0,3}=+[ \t]*", text) or index == 0:
            continue
        previous = _markdown_line_text(lines[index - 1])
        previous_indent = len(previous) - len(previous.lstrip(" "))
        if (
            outside[index - 1]
            and index - 1 not in remove
            and previous.strip()
            and previous_indent <= 3
            and not re.match(r"^ {0,3}(?:#{1,6}(?:[ \t]+|$)|>|[-+*][ \t]+)", previous)
        ):
            remove.update((index - 1, index))

    return "".join(line for index, line in enumerate(lines) if index not in remove)


def _frontmatter_span(markdown: str) -> tuple[int, int, int] | None:
    """Return YAML content and body offsets for line-delimited frontmatter."""
    lines = markdown.splitlines(keepends=True)
    if not lines or not re.fullmatch(r"---[ \t]*", _markdown_line_text(lines[0])):
        return None

    offset = len(lines[0])
    for line in lines[1:]:
        line_start = offset
        offset += len(line)
        if re.fullmatch(r"---[ \t]*", _markdown_line_text(line)):
            return len(lines[0]), line_start, offset
    return None


def _extract_frontmatter(markdown: str) -> dict[str, object]:
    """Extract a line-delimited markdown frontmatter block as a mapping."""
    span = _frontmatter_span(markdown)
    if span is None:
        return {}
    content_start, content_end, _body_start = span

    try:
        import yaml

        data = yaml.safe_load(markdown[content_start:content_end])
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
            normalized = source.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
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
    max_touches: int = 5,
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

    # Accept attribution once, before any article prompt or page is generated.
    # All later artifacts consume these same ordered references.
    concepts = [
        concept.model_copy(
            update={
                "source_urls": _bounded_source_references(
                    concept.source_urls,
                    context=f"concept:{concept.name}",
                    preferred_keys=set(url_to_source),
                )
            }
        )
        for concept in concepts
    ]

    # Stage 2: create or update each article — all run concurrently
    async def _process_concept(
        concept: WikiConceptEntry,
    ) -> tuple[str, str, bool, bool, list[str]]:
        """Process a concept and return its exact accepted attribution list."""
        slug = slugify(concept.name)

        if concept.status == ConceptStatus.NEW:
            if dry_run:
                log.info("compile_dry_run_create", concept=concept.name)
                return (slug, slug, False, False, concept.source_urls)
            await _create_and_write_article(
                slug, concept, url_to_source, provider, vault_path, folders
            )
            log.info("compile_article_done", slug=slug, status="created")
            return (slug, slug, True, False, concept.source_urls)

        # EXISTING or MERGE
        target_slug = concept.merge_target or slug
        if dry_run:
            log.info("compile_dry_run_update", concept=concept.name, target=target_slug)
            return (slug, target_slug, False, False, concept.source_urls)

        existing_content = _read_wiki_article_content(manifest, target_slug, vault_path, folders)
        if existing_content:
            accepted_sources = _merge_accepted_source_references(
                _extract_frontmatter_sources(existing_content),
                concept.source_urls,
                context=f"article:{target_slug}",
            )
            accepted_concept = concept.model_copy(update={"source_urls": accepted_sources})
            await _update_and_write_article(
                target_slug,
                existing_content,
                accepted_concept,
                url_to_source,
                provider,
                vault_path,
                folders,
            )
            log.info("compile_article_done", slug=target_slug, status="updated")
            return (slug, target_slug, False, True, accepted_sources)
        else:
            await _create_and_write_article(
                target_slug, concept, url_to_source, provider, vault_path, folders
            )
            log.info("compile_article_done", slug=target_slug, status="created_new")
            return (slug, target_slug, True, False, concept.source_urls)

    async def _process_concept_with_error_handling(
        concept: WikiConceptEntry,
    ) -> tuple[str, str, bool, bool, list[str]] | None:
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
        _slug, target_slug, was_created, was_updated, accepted_sources = item
        result.articles_created += was_created
        result.articles_updated += was_updated
        slug_to_urls[target_slug] = _bounded_source_references(
            _merge_source_urls(slug_to_urls.get(target_slug, []), accepted_sources),
            context=f"slug:{target_slug}",
        )

    # Stage 3: cross-page propagation — patch existing concept pages so they
    # cross-link to the new/updated concepts.
    if not dry_run and slug_to_urls and max_touches > 0:
        await _propagate_to_existing_concepts(
            sources=sources,
            slug_to_urls=slug_to_urls,
            provider=provider,
            vault_path=vault_path,
            folders=folders,
            max_touches=max_touches,
            result=result,
        )

    return result, slug_to_urls


def _format_existing_concepts(existing_concepts: list[tuple[str, str]]) -> str:
    """Format existing concept slugs for the triage prompt."""
    if not existing_concepts:
        return "No existing concept pages yet."
    return "\n".join(f"- {slug}: {title}" for slug, title in sorted(existing_concepts))


# Bounds cover both the complete article prompt and its independently useful
# components. The formatter never enforces these by slicing a completed packet:
# doing so can leave packet metadata but remove all original body text.
ARTICLE_SOURCE_PER_SOURCE_CHARS = 6000
ARTICLE_SOURCE_TOTAL_CHARS = 24000
ARTICLE_SOURCE_REFERENCE_LIMIT = 64
ARTICLE_EXISTING_CONTENT_CHARS = 24000
ARTICLE_PROMPT_TOTAL_CHARS = 50000
ARTICLE_DESCRIPTION_CHARS = 4000
ARTICLE_CONCEPT_NAME_CHARS = 500
_SOURCE_LABEL_CHARS = 256
_SOURCE_PACKET_MIN_CHARS = 96


def _bounded_source_references(
    source_keys: list[str],
    *,
    context: str,
    preferred_keys: set[str] | None = None,
) -> list[str]:
    """Deduplicate and cap attribution, retaining resolvable keys first."""
    deduplicated = _merge_source_urls(source_keys)
    if preferred_keys:
        preferred = [key for key in deduplicated if key in preferred_keys]
        remaining = [key for key in deduplicated if key not in preferred_keys]
        deduplicated = [*preferred, *remaining]
    retained = deduplicated[:ARTICLE_SOURCE_REFERENCE_LIMIT]
    if len(retained) < len(deduplicated):
        log.warning(
            "article_source_references_truncated",
            context=context,
            accepted=len(retained),
            dropped=len(deduplicated) - len(retained),
            limit=ARTICLE_SOURCE_REFERENCE_LIMIT,
        )
    return retained


def _merge_accepted_source_references(
    existing: list[str],
    incoming: list[str],
    *,
    context: str,
) -> list[str]:
    """Bound attribution while reserving room for every accepted incoming key."""
    accepted_incoming = _bounded_source_references(incoming, context=f"{context}:incoming")
    incoming_set = set(accepted_incoming)
    prior = [key for key in _merge_source_urls(existing) if key not in incoming_set]
    prior_capacity = ARTICLE_SOURCE_REFERENCE_LIMIT - len(accepted_incoming)
    combined = [*prior[:prior_capacity], *accepted_incoming]
    dropped = len(prior) - prior_capacity
    if dropped > 0:
        log.warning(
            "article_source_references_truncated",
            context=context,
            accepted=len(combined),
            dropped=dropped,
            limit=ARTICLE_SOURCE_REFERENCE_LIMIT,
        )
    return combined


def _truncate_with_marker(value: str, max_chars: int, marker: str = "…") -> str:
    """Bound text to exactly ``max_chars`` at most, including its marker."""
    if len(value) <= max_chars:
        return value
    if max_chars <= len(marker):
        return marker[:max_chars]
    return value[: max_chars - len(marker)].rstrip() + marker


def _source_label(key: str, max_chars: int = _SOURCE_LABEL_CHARS) -> str:
    """Return a bounded, stable representation of an attributed source key."""
    if len(key) <= max_chars:
        return key
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    suffix = f"… [sha256:{digest}]"
    if max_chars <= len(suffix):
        return ("~" + digest)[:max_chars]
    return _truncate_with_marker(key, max_chars - len(suffix), "") + suffix


def _bounded_raw_source_packet(source: RawSourceRecord, packet_chars: int) -> str:
    """Format a packet that always contains a non-empty Raw body field."""
    key = _source_key(source)
    metadata = (
        f"Title: {_truncate_with_marker(source.title, 160)}\n"
        f"Source: {_source_label(key)}\n"
        f"Tags: {_truncate_with_marker(', '.join(source.raw_tags) or 'none', 160)}\n\n"
        "Raw body:\n"
    )
    body_budget = max(1, packet_chars - len(metadata))
    body = _truncate_with_marker(source.body or "[Empty Raw body]", body_budget)
    packet = metadata + body
    if len(packet) <= packet_chars:
        return packet

    # Tiny budgets still retain the semantic packet invariant instead of
    # returning an arbitrary metadata prefix.
    minimal_prefix = "Raw body:\n"
    return minimal_prefix + _truncate_with_marker(
        source.body or "[Empty Raw body]", max(1, packet_chars - len(minimal_prefix))
    )


def _format_article_source_context(
    source_keys: list[str],
    url_to_source: dict[str, RawSourceRecord],
    *,
    max_chars: int = ARTICLE_SOURCE_TOTAL_CHARS,
) -> str:
    """Format attributed citations and as many complete Raw packets as fit.

    Every key has a stable citation representation. Long keys use a readable
    prefix plus digest. Resolved packets are retained only when enough room
    remains for their ``Raw body`` field; unresolved citations remain visible
    even when their optional unavailable packet does not fit.
    """
    max_chars = min(max_chars, ARTICLE_SOURCE_TOTAL_CHARS)
    keys = _bounded_source_references(
        source_keys,
        context="source_context",
        preferred_keys=set(url_to_source),
    )
    if not keys:
        return _truncate_with_marker("No attributed sources.", max_chars)

    header = "Attributed source citations:\n"
    packet_header = "\n\nRaw source packets:\n"
    per_label = min(
        _SOURCE_LABEL_CHARS,
        max(18, (max_chars - len(header) - len(packet_header) - 3 * len(keys)) // len(keys)),
    )
    citation_lines = [f"- {_source_label(key, per_label)}" for key in keys]
    citations = header + "\n".join(citation_lines)

    # ARTICLE_SOURCE_REFERENCE_LIMIT guarantees this fits the normal hard
    # context bound while keeping every retained key individually identifiable.
    if len(citations) + len(packet_header) > max_chars:
        raise ValueError("source context budget is too small for accepted attribution labels")

    prefix = citations + packet_header
    remaining = max_chars - len(prefix)
    separator = "\n\n---\n\n"
    packets: list[str] = []

    resolved = [url_to_source[key] for key in keys if key in url_to_source]
    separator_chars = len(separator)
    capacity = max(0, (remaining + separator_chars) // (_SOURCE_PACKET_MIN_CHARS + separator_chars))
    retained = resolved[:capacity]
    resolved_reserve = len(retained) * (_SOURCE_PACKET_MIN_CHARS + separator_chars)

    # Preserve explicit unavailable markers only after reserving body-bearing
    # packets. Their citation lines above already carry the attribution.
    for key in keys:
        if key in url_to_source:
            continue
        candidate = f"Source: {_source_label(key)}\n\n[Raw source unavailable]"
        needed = len(candidate) + (len(separator) if packets else 0)
        if needed + resolved_reserve <= remaining:
            packets.append(candidate)
            remaining -= needed

    for index, source in enumerate(retained):
        sources_left = len(retained) - index
        separator_cost = separator_chars if packets else 0
        future_separators = separator_chars * max(0, sources_left - 1)
        available = min(
            ARTICLE_SOURCE_PER_SOURCE_CHARS,
            (remaining - separator_cost - future_separators) // sources_left,
        )
        packet = _bounded_raw_source_packet(source, available)
        packets.append(packet)
        remaining -= len(packet) + separator_cost

    return prefix + separator.join(packets)


async def _create_article(
    concept: WikiConceptEntry,
    url_to_source: dict[str, RawSourceRecord],
    provider: Provider,
) -> str:
    """LLM call to create a new wiki article from grounded Raw sources."""
    concept_name = _truncate_with_marker(concept.name, ARTICLE_CONCEPT_NAME_CHARS)
    description = _truncate_with_marker(concept.description, ARTICLE_DESCRIPTION_CHARS)
    prompt_without_sources = COMPILE_ARTICLE_CREATE_PROMPT.format(
        concept_name=concept_name,
        description=description,
        source_context="",
    )
    source_context = _format_article_source_context(
        concept.source_urls,
        url_to_source,
        max_chars=ARTICLE_PROMPT_TOTAL_CHARS - len(prompt_without_sources),
    )
    prompt = COMPILE_ARTICLE_CREATE_PROMPT.format(
        concept_name=concept_name,
        description=description,
        source_context=source_context,
    )
    assert len(prompt) <= ARTICLE_PROMPT_TOTAL_CHARS

    wiki_author = "You are a research wiki author. Write markdown only."
    response = await provider.complete(prompt, system=wiki_author)
    return response


async def _update_article(
    existing_content: str,
    concept: WikiConceptEntry,
    url_to_source: dict[str, RawSourceRecord],
    provider: Provider,
) -> str:
    """LLM call to update an existing wiki article with new source info."""
    bounded_existing = _truncate_with_marker(
        existing_content,
        ARTICLE_EXISTING_CONTENT_CHARS,
        "\n\n[Existing article truncated]",
    )
    prompt_without_sources = COMPILE_ARTICLE_UPDATE_PROMPT.format(
        existing_content=bounded_existing,
        new_sources="",
    )
    source_context = _format_article_source_context(
        concept.source_urls,
        url_to_source,
        max_chars=ARTICLE_PROMPT_TOTAL_CHARS - len(prompt_without_sources),
    )
    prompt = COMPILE_ARTICLE_UPDATE_PROMPT.format(
        existing_content=bounded_existing,
        new_sources=source_context,
    )
    assert len(prompt) <= ARTICLE_PROMPT_TOTAL_CHARS

    wiki_author = "You are a research wiki author. Write markdown only."
    response = await provider.complete(prompt, system=wiki_author)
    return response


async def _create_and_write_article(
    slug: str,
    concept: WikiConceptEntry,
    url_to_source: dict[str, RawSourceRecord],
    provider: Provider,
    vault_path: Path,
    folders: FolderConfig,
) -> None:
    """Create a wiki article via LLM and write it to disk."""
    body = await _create_article(concept, url_to_source, provider)
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
    _write_wiki_article(slug, body, title, concept.source_urls, vault_path, folders)


REQUIRED_CONCEPT_SECTIONS = (
    "Overview",
    "Key Ideas",
    "Connections",
    "Open Questions",
    "Sources",
)


@dataclass(frozen=True, slots=True)
class _MarkdownSection:
    """An H2 and its content span, excluding fenced fake headings."""

    title: str
    start: int
    content_start: int
    end: int


def _markdown_h2_sections(body: str) -> list[_MarkdownSection]:
    """Locate real H2 sections with CommonMark-style fence awareness."""
    headings: list[tuple[str, int, int]] = []
    fence_char: str | None = None
    fence_length = 0
    offset = 0
    for line in body.splitlines(keepends=True):
        text = _markdown_line_text(line)
        if fence_char is not None:
            closer = re.fullmatch(rf" {{0,3}}({re.escape(fence_char)}+)[ \t]*", text)
            if closer and len(closer.group(1)) >= fence_length:
                fence_char = None
                fence_length = 0
            offset += len(line)
            continue

        opener = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", text)
        if opener:
            marker, remainder = opener.groups()
            if marker[0] == "~" or "`" not in remainder:
                fence_char = marker[0]
                fence_length = len(marker)
                offset += len(line)
                continue

        heading = re.match(r"^ {0,3}##[ \t]+(.+?)[ \t]*#*[ \t]*$", text)
        if heading:
            headings.append((heading.group(1).strip(), offset, offset + len(line)))
        offset += len(line)

    return [
        _MarkdownSection(
            title=title,
            start=start,
            content_start=content_start,
            end=headings[index + 1][1] if index + 1 < len(headings) else len(body),
        )
        for index, (title, start, content_start) in enumerate(headings)
    ]


def _ensure_required_sections(body: str) -> str:
    """Add missing real concept sections without trusting fenced headings."""
    result = body
    for index in range(len(REQUIRED_CONCEPT_SECTIONS) - 1, -1, -1):
        heading = REQUIRED_CONCEPT_SECTIONS[index]
        if any(section.title == heading for section in _markdown_h2_sections(result)):
            continue
        later_headings = list(REQUIRED_CONCEPT_SECTIONS[index + 1:])
        section = f"## {heading}\n"
        result = _insert_section_before(result, section, later_headings)
    return result


def _without_source_bullets(content: str) -> str:
    """Remove model-authored source bullets outside fenced regions."""
    retained: list[str] = []
    fence_char: str | None = None
    fence_length = 0
    for line in content.splitlines():
        if fence_char is not None:
            closer = re.fullmatch(rf" {{0,3}}({re.escape(fence_char)}+)[ \t]*", line)
            if closer and len(closer.group(1)) >= fence_length:
                fence_char = None
                fence_length = 0
            retained.append(line)
            continue

        opener = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if opener:
            marker, remainder = opener.groups()
            if marker[0] == "~" or "`" not in remainder:
                fence_char = marker[0]
                fence_length = len(marker)
                retained.append(line)
                continue
        if re.match(r"^\s*[-*+]\s+", line):
            continue
        retained.append(line)
    return "\n".join(retained).strip()


def _synchronize_sources_section(body: str, sources: list[str]) -> str:
    """Consolidate real Sources sections and emit managed citations once."""
    sections = [section for section in _markdown_h2_sections(body) if section.title == "Sources"]
    if not sections:  # Defensive: required-section enforcement normally adds it.
        return _append_section(body, "## Sources\n\n" + "\n".join(f"- {s}" for s in sources))

    retained = [
        content
        for section in sections
        if (content := _without_source_bullets(body[section.content_start:section.end]))
    ]
    canonical = "\n".join(f"- {source}" for source in sources)
    section_parts = [*retained, *([canonical] if canonical else [])]
    replacement = "## Sources\n\n" + "\n\n".join(section_parts) + "\n"

    # Replace the first section and remove every duplicate, working backwards
    # so the parser's original offsets remain valid.
    result = body
    for section in reversed(sections[1:]):
        result = result[:section.start] + result[section.end:]
    first = sections[0]
    return result[:first.start] + replacement + result[first.end:]


def _single_line_title(title: str, fallback: str = "Untitled Concept") -> str:
    """Collapse an untrusted title to one non-empty Markdown heading line."""
    normalized = re.sub(r"\s+", " ", title).strip()
    return normalized or fallback


def _normalize_concept_body(body: str, title: str, sources: list[str]) -> str:
    """Enforce the deterministic body portion of the concept-page contract."""
    model_content = _strip_model_page_wrappers(body)
    model_content = _ensure_required_sections(model_content)
    model_content = _synchronize_sources_section(model_content, sources)
    return f"# {_single_line_title(title)}\n\n{model_content}"


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

    accepted_sources = _bounded_source_references(
        source_urls,
        context=f"page:{slug}",
    )
    safe_title = _single_line_title(title, slug.replace("-", " ").title())
    content_body = _normalize_concept_body(body, safe_title, accepted_sources)
    import yaml

    class _IndentedSafeDumper(yaml.SafeDumper):
        def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
            return super().increase_indent(flow, False)

    frontmatter = {
        "title": safe_title,
        "vaultmind": True,
        "kind": "concept",
        "sources": accepted_sources,
    }
    serialized_frontmatter = yaml.dump(
        frontmatter,
        Dumper=_IndentedSafeDumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    ).rstrip()
    content_body = content_body.rstrip("\r\n")
    content = f"---\n{serialized_frontmatter}\n---\n\n{content_body}\n"

    # Match core.writer's same-directory temporary file + atomic replace
    # convention while retaining the established indented source-list format.
    import tempfile

    fd, tmp_path = tempfile.mkstemp(dir=article_path.parent, suffix=".md.tmp")
    try:
        with open(fd, "w", encoding="utf-8") as file:
            file.write(content)
        Path(tmp_path).replace(article_path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
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


# ---- Cross-page propagation (Stage 3) ----

PROPAGATION_CATALOG_LIMIT = 200
PROPAGATION_PACKET_CHARS = 4000


@dataclass(slots=True)
class _CatalogEntry:
    """Existing-concept summary used to seed the propagation prompt."""

    slug: str
    title: str
    summary: str
    existing_connections: str


def _build_catalog(
    vault_path: Path,
    folders: FolderConfig,
    exclude_slugs: set[str],
) -> list[_CatalogEntry]:
    """Build the on-disk concept catalog for propagation."""
    wiki_concepts_dir = vault_path / folders.wiki / folders.wiki_concepts
    if not wiki_concepts_dir.exists():
        return []

    entries: list[_CatalogEntry] = []
    for path in wiki_concepts_dir.glob("*.md"):
        slug = path.stem.lower()
        if slug in exclude_slugs:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("propagation_catalog_read_failed", path=str(path), error=str(exc))
            continue

        body = _strip_frontmatter_block(text)
        fm = _extract_frontmatter(text)
        title = _catalog_title(fm, body, slug)
        summary = _catalog_summary(body)
        connections = _extract_section(body, "Connections")

        entries.append(
            _CatalogEntry(
                slug=slug,
                title=title,
                summary=summary,
                existing_connections=connections.strip(),
            )
        )

    entries.sort(key=lambda entry: entry.slug)
    if len(entries) > PROPAGATION_CATALOG_LIMIT:
        log.warning("propagation_catalog_truncated", total=len(entries))
        entries = entries[:PROPAGATION_CATALOG_LIMIT]
    return entries


def _strip_frontmatter_block(markdown: str) -> str:
    """Return the markdown body without leading line-delimited frontmatter."""
    span = _frontmatter_span(markdown)
    if span is None:
        return markdown
    _content_start, _content_end, body_start = span
    return markdown[body_start:].lstrip("\r\n")


def _catalog_title(frontmatter: dict[str, object], body: str, slug: str) -> str:
    """Derive a catalog title from frontmatter, body H1, or slug."""
    title = frontmatter.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    h1 = _extract_h1_title(body)
    if h1:
        return h1
    return slug.replace("-", " ").title()


def _catalog_summary(body: str) -> str:
    """Pull the first non-empty, non-heading paragraph as a summary."""
    paragraph: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            if paragraph:
                break
            continue
        if stripped.startswith("#"):
            if paragraph:
                break
            continue
        paragraph.append(stripped)

    summary = " ".join(paragraph).strip()
    if len(summary) > 200:
        summary = summary[:200].rstrip() + "..."
    return summary


def _extract_section(body: str, heading: str) -> str:
    """Return the body of the section under `## {heading}`.

    Captures everything between the heading and the next `## ` heading or EOF.
    Returns an empty string if the section is missing.
    """
    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}\s*$(?P<body>.*?)(?=^##\s+|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(body)
    if not match:
        return ""
    return match.group("body")


def _format_catalog_block(entries: list[_CatalogEntry]) -> str:
    """Format catalog entries for inclusion in the prompt."""
    return "\n".join(
        f"- {entry.slug} | {entry.title} | {entry.summary}" for entry in entries
    )


def _has_wikilink_to_any(text: str, slugs: set[str]) -> bool:
    """Return True if `text` links to any slug using canonical normalization."""
    return bool(set(extract_wikilinks(text)) & slugs)


def _has_only_allowed_wikilinks(text: str, allowed_slugs: set[str]) -> bool:
    """Require at least one wikilink and constrain every target to `allowed_slugs`."""
    links = extract_wikilinks(text)
    return bool(links) and set(links) <= allowed_slugs


@dataclass(slots=True)
class _PropagationTouch:
    """A validated propagation touch ready to apply to disk."""

    target_slug: str
    connection_line: str
    relevance: int
    new_concept_slugs: tuple[str, ...]


async def _propagate_to_existing_concepts(
    *,
    sources: list[RawSourceRecord],
    slug_to_urls: dict[str, list[str]],
    provider: Provider,
    vault_path: Path,
    folders: FolderConfig,
    max_touches: int,
    result: CompileResult,
) -> None:
    """Stage 3: cross-link new sources into existing concept pages."""
    exclude = {slug.lower() for slug in slug_to_urls}
    catalog = _build_catalog(vault_path, folders, exclude)
    if not catalog:
        return

    catalog_block = _format_catalog_block(catalog)
    catalog_by_slug = {entry.slug: entry for entry in catalog}
    touched_slugs: set[str] = set()

    # Build per-source new-concept attribution from slug_to_urls.
    source_to_new_slugs: dict[str, list[str]] = {}
    for slug, urls in slug_to_urls.items():
        for url in urls:
            source_to_new_slugs.setdefault(url, []).append(slug)

    for source in sources:
        source_key = _source_key(source)
        new_slugs = source_to_new_slugs.get(source_key, [])
        if not new_slugs:
            continue
        new_slugs_lower = {slug.lower() for slug in new_slugs}

        prompt = COMPILE_GRAPH_TOUCH_PROMPT.format(
            new_source_packet=format_raw_source_packet(source, max_chars=PROPAGATION_PACKET_CHARS),
            new_concept_slugs=json.dumps(new_slugs),
            catalog=catalog_block,
        )

        try:
            response = await provider.complete(
                prompt,
                system="You are a precise librarian. Return only valid JSON.",
            )
        except Exception as exc:
            log.error("propagation_call_failed", source=source_key, error=str(exc))
            result.errors.append(
                f"Propagation call for '{source_key}' failed: {exc}"
            )
            result.propagation_failed_sources.add(source_key)
            continue

        touches = _parse_propagation_touches(
            response=response,
            source_key=source_key,
            catalog_slugs=set(catalog_by_slug.keys()),
            new_slugs_lower=new_slugs_lower,
            new_slugs=new_slugs,
        )

        kept = _filter_and_rank_touches(
            touches=touches,
            catalog_by_slug=catalog_by_slug,
            new_slugs_lower=new_slugs_lower,
            max_touches=max_touches,
            vault_path=vault_path,
            folders=folders,
        )

        for touch in kept:
            target_path = (
                vault_path
                / folders.wiki
                / folders.wiki_concepts
                / f"{touch.target_slug}.md"
            )
            try:
                _apply_touch(target_path, touch, source_key)
            except Exception as exc:
                log.error(
                    "propagation_touch_failed",
                    target=touch.target_slug,
                    source=source_key,
                    error=str(exc),
                )
                result.errors.append(
                    f"Propagation touch on '{touch.target_slug}' from '{source_key}' failed: {exc}"
                )
                result.propagation_failed_sources.add(source_key)
                continue
            touched_slugs.add(touch.target_slug)
            source_touches = result.propagation_touches_by_source.setdefault(source_key, [])
            if touch.target_slug not in source_touches:
                source_touches.append(touch.target_slug)

    result.articles_touched = len(touched_slugs)


def _parse_propagation_touches(
    *,
    response: str,
    source_key: str,
    catalog_slugs: set[str],
    new_slugs_lower: set[str],
    new_slugs: list[str],
) -> list[_PropagationTouch]:
    """Parse and per-touch-validate a propagation response."""
    cleaned = clean_json_response(response)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        log.warning(
            "propagation_parse_failed",
            source=source_key,
            response=response[:200],
        )
        return []

    raw_touches = data.get("touches") if isinstance(data, dict) else None
    if not isinstance(raw_touches, list):
        log.warning(
            "propagation_parse_failed",
            source=source_key,
            response=response[:200],
        )
        return []

    touches: list[_PropagationTouch] = []
    for item in raw_touches:
        if not isinstance(item, dict):
            log.debug("propagation_touch_invalid", source=source_key, item=str(item)[:120])
            continue

        target_slug = item.get("target_slug")
        if not isinstance(target_slug, str) or not target_slug.strip():
            log.debug("propagation_touch_invalid_slug", source=source_key)
            continue
        normalized_slug = target_slug.strip().lower()
        if normalized_slug not in catalog_slugs:
            log.debug(
                "propagation_touch_unknown_slug",
                source=source_key,
                slug=target_slug,
            )
            continue
        target_slug = normalized_slug

        connection_line = item.get("connection_line")
        if not isinstance(connection_line, str) or not connection_line.strip():
            log.debug("propagation_touch_invalid_line", source=source_key)
            continue
        if not _has_only_allowed_wikilinks(connection_line, new_slugs_lower):
            log.debug(
                "propagation_touch_unauthorized_link",
                source=source_key,
                slug=target_slug,
            )
            continue

        raw_relevance = item.get("relevance", 5)
        if isinstance(raw_relevance, int) and 1 <= raw_relevance <= 10:
            relevance = raw_relevance
        else:
            relevance = 5

        touches.append(
            _PropagationTouch(
                target_slug=target_slug,
                connection_line=connection_line.strip(),
                relevance=relevance,
                new_concept_slugs=tuple(new_slugs),
            )
        )

    return touches


def _filter_and_rank_touches(
    *,
    touches: list[_PropagationTouch],
    catalog_by_slug: dict[str, _CatalogEntry],
    new_slugs_lower: set[str],
    max_touches: int,
    vault_path: Path,
    folders: FolderConfig,
) -> list[_PropagationTouch]:
    """Apply current-disk idempotency, dedupe, sort, and trim."""
    survivors: list[_PropagationTouch] = []
    for touch in touches:
        if touch.target_slug not in catalog_by_slug:
            continue
        target_path = (
            vault_path / folders.wiki / folders.wiki_concepts / f"{touch.target_slug}.md"
        )
        try:
            _frontmatter, current_body = _split_page(target_path.read_text(encoding="utf-8"))
        except OSError:
            # Let the write path report the durable-state failure consistently.
            survivors.append(touch)
            continue
        current_connections = _extract_section(current_body, "Connections")
        if _has_wikilink_to_any(current_connections, new_slugs_lower):
            continue
        survivors.append(touch)

    # Dedupe by (target_slug, frozenset(new_concept_slugs)) — keep highest relevance.
    deduped: dict[tuple[str, frozenset[str]], _PropagationTouch] = {}
    for touch in survivors:
        key = (touch.target_slug, frozenset(touch.new_concept_slugs))
        existing = deduped.get(key)
        if existing is None or touch.relevance > existing.relevance:
            deduped[key] = touch

    ordered = sorted(
        deduped.values(),
        key=lambda touch: (-touch.relevance, touch.target_slug),
    )
    return ordered[:max_touches]


def _apply_touch(
    target_path: Path,
    touch: _PropagationTouch,
    source_key: str,
) -> None:
    """Apply a single touch to a concept page (frontmatter + body edits)."""
    text = target_path.read_text(encoding="utf-8")
    frontmatter, body = _split_page(text)

    new_sources = _merge_accepted_source_references(
        _frontmatter_sources(frontmatter),
        [source_key],
        context=f"propagation:{target_path.stem}",
    )
    new_frontmatter = _frontmatter_with_sources(frontmatter, new_sources)
    title = _single_line_title(
        _catalog_title(frontmatter, body, target_path.stem),
        target_path.stem.replace("-", " ").title(),
    )
    new_frontmatter["title"] = title
    new_frontmatter["vaultmind"] = True
    new_frontmatter["kind"] = "concept"

    body = _append_connections_bullet(body, touch.connection_line)
    body = _append_sources_bullet(body, source_key)
    body = _normalize_concept_body(body, title, new_sources)

    write_markdown_page(target_path, body=body, frontmatter=new_frontmatter)


def _split_page(text: str) -> tuple[dict[str, object], str]:
    """Split line-delimited frontmatter from a markdown page."""
    span = _frontmatter_span(text)
    if span is None:
        return {}, text.lstrip("\r\n")
    content_start, content_end, body_start = span

    import yaml

    try:
        fm_data = yaml.safe_load(text[content_start:content_end])
    except yaml.YAMLError:
        fm_data = {}
    frontmatter = fm_data if isinstance(fm_data, dict) else {}
    body = text[body_start:].lstrip("\r\n")
    return frontmatter, body


def _frontmatter_sources(frontmatter: dict[str, object]) -> list[str]:
    """Read existing source URLs/keys from a parsed frontmatter dict."""
    sources = frontmatter.get("sources")
    if isinstance(sources, list):
        return [str(item).strip() for item in sources if item and str(item).strip()]
    if isinstance(sources, str) and sources.strip():
        return [sources.strip()]
    return []


def _frontmatter_with_sources(
    frontmatter: dict[str, object],
    sources: list[str],
) -> dict[str, object]:
    """Return a new frontmatter dict with `sources` replaced, preserving order."""
    updated: dict[str, object] = {}
    saw_sources = False
    for key, value in frontmatter.items():
        if key == "sources":
            updated[key] = sources
            saw_sources = True
        else:
            updated[key] = value
    if not saw_sources:
        updated["sources"] = sources
    return updated


def _append_connections_bullet(body: str, connection_line: str) -> str:
    """Add a bullet to `## Connections`, creating the section if needed."""
    bullet = f"- {connection_line}"
    if re.search(r"^##\s+Connections\s*$", body, re.MULTILINE):
        return _append_bullet_to_section(body, "Connections", bullet)

    new_section = f"## Connections\n\n{bullet}\n"
    return _insert_section_before(body, new_section, ["Open Questions", "Sources"])


def _append_sources_bullet(body: str, source_key: str) -> str:
    """Add a bullet to `## Sources`, creating the section if needed."""
    bullet = f"- {source_key}"
    if re.search(r"^##\s+Sources\s*$", body, re.MULTILINE):
        section_body = _extract_section(body, "Sources")
        if _section_has_line(section_body, bullet):
            return body
        return _append_bullet_to_section(body, "Sources", bullet)

    new_section = f"## Sources\n\n{bullet}\n"
    return _append_section(body, new_section)


def _section_has_line(section_body: str, bullet: str) -> bool:
    """Return True if `bullet` already appears as a line in `section_body`."""
    target = bullet.strip()
    return any(line.strip() == target for line in section_body.splitlines())


def _append_bullet_to_section(body: str, heading: str, bullet: str) -> str:
    """Insert `bullet` at the end of `## {heading}` (before next `## ` or EOF)."""
    pattern = re.compile(
        rf"(^##\s+{re.escape(heading)}\s*$)(?P<body>.*?)(?=^##\s+|\Z)",
        re.MULTILINE | re.DOTALL,
    )

    def replace(match: re.Match[str]) -> str:
        heading_line = match.group(1)
        section_body = match.group("body")
        # Strip trailing whitespace/newlines, append bullet, restore spacing.
        trimmed = section_body.rstrip("\n")
        new_body = (
            f"{trimmed}\n{bullet}\n\n" if trimmed.strip() else f"\n{bullet}\n\n"
        )
        return heading_line + new_body

    return pattern.sub(replace, body, count=1)


def _insert_section_before(
    body: str,
    new_section: str,
    preferred_headings: list[str],
) -> str:
    """Insert `new_section` immediately before the first matching heading.

    Falls back to appending at the end if none of the headings appear.
    """
    sections = _markdown_h2_sections(body)
    for heading in preferred_headings:
        match = next((section for section in sections if section.title == heading), None)
        if match is None:
            continue
        prefix = body[:match.start].rstrip("\n")
        suffix = body[match.start:]
        return f"{prefix}\n\n{new_section}\n{suffix}"
    return _append_section(body, new_section)


def _append_section(body: str, new_section: str) -> str:
    """Append `new_section` to the end of the body with consistent spacing."""
    trimmed = body.rstrip("\n")
    if trimmed:
        return f"{trimmed}\n\n{new_section}"
    return new_section
