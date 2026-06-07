"""Keyword and fuzzy search over indexed vault notes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher

from vaultmind.core.vault_index import VaultNoteRecord, truncate_for_ai

# Text-overlap helpers used by note scoring. (These previously lived in
# core.linker, which backed the removed save-pipeline related-notes feature.)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "what", "when",
    "why", "with",
}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _normalize_tags(tags: object) -> set[str]:
    """Normalize a frontmatter tags value into a lowercase set."""
    if not isinstance(tags, list):
        return set()
    normalized: set[str] = set()
    for tag in tags:
        if not isinstance(tag, str):
            continue
        clean = tag.strip().lower()
        if clean:
            normalized.add(clean)
    return normalized


def _tokenize_title(title: str) -> set[str]:
    """Tokenize and normalize note titles for overlap scoring."""
    tokens = re.findall(r"[a-z0-9]+", title.lower())
    return {token for token in tokens if len(token) >= 2 and token not in _STOPWORDS}


@dataclass(slots=True)
class SearchMatch:
    note: VaultNoteRecord
    score: float
    title_hits: list[str]
    tag_hits: list[str]
    excerpt: str


def search_notes(notes: list[VaultNoteRecord], query: str, *, limit: int = 50) -> list[SearchMatch]:
    """Search notes by keyword and fuzzy title matching."""
    normalized_query = query.strip()
    if not normalized_query:
        recent = sorted(notes, key=_recent_sort_key, reverse=True)[:limit]
        return [
            SearchMatch(
                note=note,
                score=0.0,
                title_hits=[],
                tag_hits=[],
                excerpt=truncate_for_ai(note.summary or note.body, max_chars=240),
            )
            for note in recent
        ]

    matches: list[SearchMatch] = []
    for note in notes:
        match = score_note_match(note, normalized_query)
        if match is not None:
            matches.append(match)

    matches.sort(key=lambda m: (m.score, _recent_sort_key(m.note)), reverse=True)
    return matches[:limit]


def score_note_match(note: VaultNoteRecord, query: str) -> SearchMatch | None:
    """Score a single note against a query. Returns None for weak/non-matches."""
    query_text = query.strip().lower()
    if not query_text:
        return None

    query_tokens = _tokenize_title(query_text)
    title_text = note.title.lower()
    body_text = note.body.lower()
    raw_tag_set = _normalize_tags(note.tags)
    tag_token_sets = {tag: _tokenize_title(tag) for tag in raw_tag_set}
    tag_set: set[str] = set().union(*tag_token_sets.values()) if tag_token_sets else set()

    score = 0.0
    title_hits: list[str] = []
    tag_hits: list[str] = []

    if query_text in title_text:
        score += 60
        title_hits.append(query_text)

    matching_tags = []
    for tag in note.tags:
        lowered = tag.lower()
        tag_tokens = _tokenize_title(tag)
        if query_text in lowered or (query_tokens and query_tokens.issubset(tag_tokens)):
            matching_tags.append(tag)
    if matching_tags:
        score += 40
        tag_hits.extend(matching_tags)

    if query_text in body_text:
        score += 20

    title_tokens = _tokenize_title(note.title)
    score += _jaccard(query_tokens, title_tokens) * 30
    score += _jaccard(query_tokens, tag_set) * 25

    ratio = SequenceMatcher(None, query_text, title_text).ratio()
    if ratio >= 0.72:
        score += ratio * 20

    if score < 15:
        return None

    if not title_hits:
        title_hits = sorted(query_tokens & title_tokens)
    if not tag_hits:
        tag_hits = sorted(query_tokens & tag_set)

    return SearchMatch(
        note=note,
        score=score,
        title_hits=title_hits,
        tag_hits=tag_hits,
        excerpt=build_match_excerpt(note.body, query_text),
    )


def build_match_excerpt(body: str, query: str, *, radius: int = 160) -> str:
    """Build a readable excerpt around the first query match."""
    text = " ".join(body.split())
    if not text:
        return ""

    if not query.strip():
        return truncate_for_ai(text, max_chars=2 * radius)

    lower_text = text.lower()
    lower_query = query.lower().strip()
    start = lower_text.find(lower_query)
    if start == -1:
        return truncate_for_ai(text, max_chars=2 * radius)

    begin = max(0, start - radius)
    end = min(len(text), start + len(lower_query) + radius)
    excerpt = text[begin:end].strip()
    if begin > 0:
        excerpt = "... " + excerpt
    if end < len(text):
        excerpt = excerpt + " ..."
    return excerpt


def _recent_sort_key(note: VaultNoteRecord) -> datetime:
    return note.saved_at or datetime(1970, 1, 1, tzinfo=UTC)
