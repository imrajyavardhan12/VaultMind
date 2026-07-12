"""vm ask — the compound interest engine."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import structlog

from vaultmind.ai.json_utils import clean_json_response
from vaultmind.ai.prompts import (
    ASK_SELF_ASSESS_PROMPT,
    ASK_SYSTEM_PROMPT,
    ASK_USER_PROMPT,
)
from vaultmind.ai.providers.base import Provider
from vaultmind.core.raw_scanner import RawSourceRecord, format_raw_source_packet
from vaultmind.core.search import SearchMatch, search_notes
from vaultmind.core.vault_index import VaultNoteRecord, format_note_packet
from vaultmind.core.writer import slugify, write_markdown_page

log = structlog.get_logger()

MAX_ITERATIONS = 3
MAX_CONTEXT_NOTES = 30
MAX_CONTEXT_SOURCES = 20
MIN_WIKI_CONTEXT_SCORE = 35.0


@dataclass
class GatheredContext:
    """Accumulated search results across iterations."""
    wiki_notes: list[VaultNoteRecord] = field(default_factory=list)
    raw_sources: list[RawSourceRecord] = field(default_factory=list)

    @property
    def total_items(self) -> int:
        return len(self.wiki_notes) + len(self.raw_sources)


@dataclass
class AskResult:
    """Result of an ask run."""
    answer: str
    slug: str
    path: Path
    iterations: int
    gaps: list[str]


# ---- Prompt builders ----


def _build_context_text(question: str, ctx: GatheredContext) -> str:
    """Build the context string from gathered wiki notes and raw sources."""
    parts: list[str] = []

    if ctx.wiki_notes:
        parts.append("## 📚 Wiki Articles\n")
        for note in ctx.wiki_notes:
            parts.append(format_note_packet(note, max_chars=900))
        parts.append("\n---\n")

    if ctx.raw_sources:
        parts.append("## 📥 Raw Sources\n")
        for source in ctx.raw_sources:
            parts.append(format_raw_source_packet(source, max_chars=3500))
        parts.append("\n---\n")

    if not parts:
        return "No relevant notes or sources found."

    return "\n\n".join(parts)


# ---- LLM response parsing ----


def _extract_answer_text(response: str) -> str:
    """Extract the answer text from the LLM response, stripping any JSON wrapper."""
    try:
        cleaned = clean_json_response(response)
        data = json.loads(cleaned)
        answer = data.get("answer", "")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
    except (json.JSONDecodeError, TypeError):
        pass

    text = response.strip()
    if text.startswith("```"):
        try:
            _, _, text = text.split("```", 2)
            text = text.lstrip("json\n").lstrip("markdown\n").rstrip("`").strip()
        except ValueError:
            pass
    return text


def _extract_gaps_from_assessment(assessment_response: str) -> list[str]:
    """Parse LLM self-assessment to find knowledge gaps."""
    try:
        cleaned = clean_json_response(assessment_response)
        data = json.loads(cleaned)
        gaps = data.get("gaps", [])
        if isinstance(gaps, list):
            normalized: list[str] = []
            for gap in gaps:
                if not isinstance(gap, str):
                    continue
                clean_gap = " ".join(gap.split())
                if clean_gap and clean_gap not in normalized:
                    normalized.append(clean_gap)
            return normalized
        return []
    except (json.JSONDecodeError, TypeError):
        pass
    return []


# ---- Search ----


def _initial_search(
    question: str,
    vault_path: Path,
    folders_wiki: str,
    folders_wiki_concepts: str,
    folders_raw: str,
    folders_wiki_queries: str = "📊 Queries",
    excluded_wiki_paths: set[str] | None = None,
) -> GatheredContext:
    """Search concept and filed-query pages, falling back to Raw when needed."""
    excluded = excluded_wiki_paths or set()
    wiki_notes = [
        note
        for note in _load_searchable_wiki_notes(
            vault_path, folders_wiki, folders_wiki_concepts, folders_wiki_queries
        )
        if note.relative_path not in excluded
    ]
    wiki_matches = search_notes(wiki_notes, question, limit=MAX_CONTEXT_NOTES) if wiki_notes else []
    matched_wiki_notes = [match.note for match in wiki_matches]

    raw_sources = (
        []
        if _wiki_context_is_sufficient(wiki_matches)
        else _search_raw_sources(question, vault_path, folders_raw)
    )
    return GatheredContext(wiki_notes=matched_wiki_notes, raw_sources=raw_sources)


def _load_searchable_wiki_notes(
    vault_path: Path,
    folders_wiki: str,
    folders_wiki_concepts: str,
    folders_wiki_queries: str | None,
) -> list[VaultNoteRecord]:
    """Load concept and filed-query pages as ask-time note records."""
    wiki_notes: list[VaultNoteRecord] = []
    for folder in (folders_wiki_concepts, folders_wiki_queries):
        if folder is None:
            continue
        wiki_dir = vault_path / folders_wiki / folder
        if not wiki_dir.exists():
            continue
        for md_file in wiki_dir.glob("*.md"):
            frontmatter = _read_frontmatter(md_file)
            if not frontmatter:
                continue

            body = _strip_frontmatter(md_file.read_text(encoding="utf-8"))
            title = str(frontmatter.get("title") or md_file.stem.replace("-", " ").title())
            saved_at = _parse_datetime(frontmatter.get("saved") or frontmatter.get("created"))

            wiki_notes.append(VaultNoteRecord(
                path=md_file,
                relative_path=md_file.relative_to(vault_path).with_suffix("").as_posix(),
                title=title,
                saved_at=saved_at,
                tags=_parse_tags(frontmatter.get("tags")),
                source_type=str(frontmatter.get("kind")) if frontmatter.get("kind") else None,
                rating=None,
                read_time_minutes=None,
                status=None,
                canonical_url=None,
                source=None,
                vaultmind=bool(frontmatter.get("vaultmind")),
                body=body,
                summary=_extract_summary(body),
                raw_frontmatter=frontmatter,
            ))

    return wiki_notes


def _load_wiki_concepts(
    vault_path: Path,
    folders_wiki: str,
    folders_wiki_concepts: str,
) -> list[VaultNoteRecord]:
    """Load only concept pages (kept for callers that need the old helper)."""
    return _load_searchable_wiki_notes(vault_path, folders_wiki, folders_wiki_concepts, None)


def _wiki_context_is_sufficient(matches: list[SearchMatch]) -> bool:
    """Return True when wiki matches are strong enough to skip Raw fallback."""
    return bool(matches) and matches[0].score >= MIN_WIKI_CONTEXT_SCORE


def _follow_up_gap(
    gap: str,
    gathered: GatheredContext,
    vault_path: Path,
    folders_wiki: str,
    folders_wiki_concepts: str,
    folders_raw: str,
    folders_wiki_queries: str = "📊 Queries",
    excluded_wiki_paths: set[str] | None = None,
) -> None:
    """Search one gap and add bounded, deduplicated wiki/Raw context."""
    excluded = excluded_wiki_paths or set()
    all_wiki_notes = [
        note
        for note in _load_searchable_wiki_notes(
            vault_path, folders_wiki, folders_wiki_concepts, folders_wiki_queries
        )
        if note.relative_path not in excluded
    ]
    wiki_matches = search_notes(all_wiki_notes, gap, limit=MAX_CONTEXT_NOTES)
    existing_wiki = {note.relative_path for note in gathered.wiki_notes}
    existing_titles = {note.title.casefold() for note in gathered.wiki_notes}
    added_wiki_matches: list[SearchMatch] = []
    for match in wiki_matches:
        if len(gathered.wiki_notes) >= MAX_CONTEXT_NOTES:
            break
        if (
            match.note.relative_path not in existing_wiki
            and match.note.title.casefold() not in existing_titles
        ):
            gathered.wiki_notes.append(match.note)
            added_wiki_matches.append(match)
            existing_wiki.add(match.note.relative_path)
            existing_titles.add(match.note.title.casefold())

    # Only new evidence can satisfy a newly identified gap. A strong result that
    # was already gathered must not suppress the Raw fallback.
    if _wiki_context_is_sufficient(added_wiki_matches):
        return

    existing_raw = {source.relative_path for source in gathered.raw_sources}
    for source in _search_raw_sources(gap, vault_path, folders_raw):
        if len(gathered.raw_sources) >= MAX_CONTEXT_SOURCES:
            break
        if source.relative_path not in existing_raw:
            gathered.raw_sources.append(source)
            existing_raw.add(source.relative_path)


def _search_raw_sources(query: str, vault_path: Path, folders_raw: str) -> list[RawSourceRecord]:
    """Search Raw markdown sources with a small keyword scorer."""
    raw_dir = vault_path / folders_raw
    if not raw_dir.exists():
        legacy = vault_path / "Clippings"
        raw_dir = legacy if legacy.exists() else raw_dir
    if not raw_dir.exists():
        return []

    query_tokens = _tokenize_query(query)
    scored: list[tuple[int, RawSourceRecord]] = []
    for path in raw_dir.rglob("*.md"):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        frontmatter, body = _split_frontmatter(text)
        title = _extract_markdown_title(body) or path.stem
        source_url = _frontmatter_value(frontmatter, "source") or _frontmatter_value(
            frontmatter, "canonical_url"
        )
        raw_tags = _parse_tags(frontmatter.get("tags"))
        haystack = f"{title}\n{' '.join(raw_tags)}\n{body}".lower()
        score = sum(1 for token in query_tokens if token in haystack)
        if query.lower().strip() and query.lower().strip() in haystack:
            score += 4
        if score <= 0:
            continue

        try:
            relative_path = path.relative_to(vault_path).with_suffix("").as_posix()
        except ValueError:
            relative_path = path.name

        scored.append(
            (
                score,
                RawSourceRecord(
                    path=path,
                    relative_path=relative_path,
                    title=title,
                    source_url=source_url,
                    body=body.strip(),
                    content_hash="",
                    raw_tags=raw_tags,
                ),
            )
        )

    scored.sort(key=lambda item: (-item[0], item[1].title.lower()))
    return [record for _, record in scored[:MAX_CONTEXT_SOURCES]]


# ---- File I/O helpers ----


def _read_frontmatter(file_path: Path) -> dict[str, object]:
    """Parse frontmatter from a markdown file."""
    try:
        text = file_path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            return {}
        end = text.find("---", 3)
        if end == -1:
            return {}
        import yaml
        data = yaml.safe_load(text[3:end])
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _split_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Split markdown frontmatter and body."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("---", 3)
    if end == -1:
        return {}, text
    fm_text = text[3:end]
    body = text[end + 3 :]
    try:
        import yaml

        data = yaml.safe_load(fm_text)
    except Exception:
        data = {}
    return data if isinstance(data, dict) else {}, body


def _extract_markdown_title(body: str) -> str | None:
    """Extract the first H1 title from markdown."""
    match = re.search(r"^#\s+(.+?)$", body, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _frontmatter_value(frontmatter: dict[str, object], key: str) -> str | None:
    value = frontmatter.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _strip_frontmatter(text: str) -> str:
    """Remove frontmatter from markdown text."""
    if not text.startswith("---"):
        return text
    end = text.find("---", 3)
    if end == -1:
        return text
    return text[end + 3:].strip()


def _extract_summary(body: str) -> str:
    """Extract first paragraph as summary."""
    if not body:
        return ""
    for line in body.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:200]
    return ""


def _parse_tags(value: object) -> list[str]:
    """Parse frontmatter tags into a list of strings."""
    if isinstance(value, list):
        return [str(t).strip().lower() for t in value if t]
    if isinstance(value, str):
        return [t.strip().lower() for t in value.split(",") if t.strip()]
    return []


def _tokenize_query(query: str) -> set[str]:
    """Tokenize a Raw search query."""
    return {token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) >= 2}


def _parse_datetime(value: object) -> datetime | None:
    """Parse a datetime value from frontmatter."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None


# ---- Answer filing ----


def _validated_question(question: str) -> str:
    """Return a single-line non-empty question suitable for an exact H1."""
    clean = question.strip()
    if not clean:
        raise ValueError("question must not be empty")
    if "\n" in clean or "\r" in clean:
        raise ValueError("question must be a single line")
    return clean


def _slug_from_question(question: str) -> str:
    """Convert a question string to a query slug for filenames."""
    text = re.sub(r"[^\w\s]", "", question.strip().lower())
    text = re.sub(r"\s+", "-", text)
    return slugify(text)[:60] or "query"


def _sanitize_answer_headings(answer: str) -> str:
    """Demote provider H1/H2 headings without changing fenced examples."""
    lines = answer.strip().splitlines()
    sanitized: list[str] = []
    fence_char: str | None = None
    fence_length = 0
    index = 0

    while index < len(lines):
        line = lines[index]
        if fence_char is not None:
            sanitized.append(line)
            closing = re.match(rf"^ {{0,3}}{re.escape(fence_char)}{{{fence_length},}}[ \t]*$", line)
            if closing:
                fence_char = None
                fence_length = 0
            index += 1
            continue

        opening = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if opening:
            marker = opening.group(1)
            info = opening.group(2)
            # CommonMark only prohibits backticks in backtick-fence info strings.
            if marker[0] == "~" or "`" not in info:
                fence_char = marker[0]
                fence_length = len(marker)
                sanitized.append(line)
                index += 1
                continue

        if index + 1 < len(lines) and line.strip():
            setext = re.match(r"^ {0,3}(?:=+|-+)[ \t]*$", lines[index + 1])
            if setext:
                indent = line[: len(line) - len(line.lstrip(" "))]
                sanitized.append(f"{indent}### {line.strip()}")
                index += 2
                continue

        sanitized.append(re.sub(r"^( {0,3})#{1,2}[ \t]+", r"\1### ", line))
        index += 1

    return "\n".join(sanitized)


def _render_answer_markdown(
    question: str,
    answer: str,
    *,
    supporting_notes: list[str],
    supporting_sources: list[str],
    iterations: int,
    now: datetime,
    gaps: list[str] | None = None,
) -> str:
    """Render a query page that deterministically satisfies the page contract."""
    # Provider output is content, not page structure. Demoting headings prevents it
    # from introducing another H1 or colliding with the required H2 sections.
    safe_answer = _sanitize_answer_headings(answer)
    safe_question = _validated_question(question)
    safe_notes = [" ".join(path.split()) for path in supporting_notes]
    safe_sources = [" ".join(source.split()) for source in supporting_sources]
    safe_gaps = [" ".join(gap.split()) for gap in gaps or []]
    lines = [
        f"# {safe_question}",
        "",
        "## Answer",
        safe_answer,
        "",
        "## Supporting Wiki Pages",
    ]
    if safe_notes:
        for path in safe_notes:
            lines.append(f"- [[{path}]]")
    else:
        lines.append("*No wiki pages used.*")

    lines.extend(["", "## Supporting Raw Sources"])
    if safe_sources:
        for source in safe_sources:
            lines.append(f"- {source}")
    else:
        lines.append("*No raw sources used.*")

    lines.extend(["", "## Follow-up Questions"])
    if safe_gaps:
        lines.extend(f"- {gap}" for gap in safe_gaps)
    else:
        lines.append("*No follow-up questions.*")

    lines.extend([
        "",
        f"*Answered {now.strftime('%Y-%m-%d %H:%M')} | {iterations} iteration(s)*",
    ])

    return "\n".join(lines)


# ---- Main ask loop ----


async def ask_question(
    question: str,
    provider: Provider,
    vault_path: Path,
    folders_wiki: str,
    folders_wiki_concepts: str,
    folders_wiki_queries: str,
    folders_raw: str,
    *,
    depth: str = "deep",
    file_answer: bool = True,
) -> AskResult:
    """Run the ask loop: question → synthesize → self-assess gaps → follow-up → file.

    The answer is written to Wiki/📊 Queries/{slug}.md so future questions
    can read it and build on it — this is the compound interest engine.
    """
    question = _validated_question(question)
    max_iters = 1 if depth == "shallow" else MAX_ITERATIONS
    slug = _slug_from_question(question)
    query_relative_path = (
        Path(folders_wiki) / folders_wiki_queries / slug
    ).as_posix()
    excluded_wiki_paths = {query_relative_path}

    gathered = _initial_search(
        question,
        vault_path,
        folders_wiki,
        folders_wiki_concepts,
        folders_raw,
        folders_wiki_queries,
        excluded_wiki_paths,
    )
    log.info("ask_initial_search", wiki=len(gathered.wiki_notes), raw=len(gathered.raw_sources))

    answer = ""
    iteration = 0
    gaps: list[str] = []

    while iteration < max_iters:
        context = _build_context_text(question, gathered)
        user_prompt = ASK_USER_PROMPT.format(question=question, context=context)
        try:
            response = await provider.complete(user_prompt, system=ASK_SYSTEM_PROMPT)
        except Exception as exc:
            log.error("ask_llm_failed", iteration=iteration + 1, error=str(exc))
            raise

        answer = _extract_answer_text(response)
        iteration += 1

        # Shallow mode deliberately makes exactly one provider call.
        if depth == "shallow":
            break

        assess_prompt = ASK_SELF_ASSESS_PROMPT.format(question=question, answer=answer)
        try:
            assessment = await provider.complete(assess_prompt, system=ASK_SYSTEM_PROMPT)
            gaps = _extract_gaps_from_assessment(assessment)
        except Exception as exc:
            log.warning("ask_self_assess_failed", iteration=iteration, error=str(exc))
            gaps = []

        if not gaps:
            log.info("ask_no_gaps", iteration=iteration)
            break
        if iteration >= max_iters:
            break

        log.info("ask_follow_up", iteration=iteration, gaps=gaps)
        for gap in gaps:
            _follow_up_gap(
                gap,
                gathered,
                vault_path,
                folders_wiki,
                folders_wiki_concepts,
                folders_raw,
                folders_wiki_queries,
                excluded_wiki_paths,
            )

    now = datetime.now(UTC)

    note_paths_used = sorted({note.relative_path for note in gathered.wiki_notes})
    source_urls_used = sorted(
        {source.source_url or source.relative_path for source in gathered.raw_sources}
    )
    body = _render_answer_markdown(
        question,
        answer,
        supporting_notes=note_paths_used,
        supporting_sources=source_urls_used,
        iterations=iteration,
        now=now,
        gaps=gaps,
    )

    queries_dir = vault_path / folders_wiki / folders_wiki_queries
    path = queries_dir / f"{slug}.md"

    if file_answer:
        queries_dir.mkdir(parents=True, exist_ok=True)
        write_markdown_page(
            path,
            body=body,
            frontmatter={
                "title": question,
                "vaultmind": True,
                "kind": "query",
                "created": now.isoformat(),
            },
        )

    log.info("ask_complete", slug=slug, path=str(path), iterations=iteration)

    return AskResult(
        answer=answer,
        slug=slug,
        path=path,
        iterations=iteration,
        gaps=gaps,
    )
