"""Deterministic wiki-health linter — zero-LLM checks for vault integrity."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

DUPLICATE_SIMILARITY_THRESHOLD = 0.85


class LintSeverity(StrEnum):
    """Severity level for lint findings."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(slots=True)
class LintFinding:
    """A single lint finding."""

    code: str  # e.g. "broken_wikilink"
    severity: LintSeverity
    message: str  # human-readable, specific
    subject: str | None = None  # vault-relative path or slug the finding is about
    detail: str = ""


@dataclass(slots=True)
class ConceptPage:
    """A wiki concept page."""

    slug: str  # file stem, lowercased
    title: str
    relative_path: str  # vault-relative, posix, no .md
    sources: list[str]  # from frontmatter `sources`
    text: str  # full file text (for link scanning)


@dataclass(slots=True)
class WikiPage:
    """A wiki page (concept or other)."""

    relative_path: str  # vault-relative, posix, no .md
    text: str
    is_index: bool


@dataclass(slots=True)
class LintReport:
    """Result of a lint run."""

    findings: list[LintFinding] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == LintSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == LintSeverity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == LintSeverity.INFO)

    @property
    def total(self) -> int:
        return len(self.findings)

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0


def extract_wikilinks(text: str) -> list[str]:
    """Extract and normalize wikilinks from text.

    Handles [[target]], [[target|display]], paths, heading refs, block refs.
    Ignores links inside fenced code blocks.
    Returns lowercased basenames.
    """
    # Remove fenced code blocks
    text_without_code = re.sub(r'```[\s\S]*?```', '', text)

    # Find all wikilinks: [[...]]
    pattern = r'\[\[([^\]]+)\]\]'
    matches = re.findall(pattern, text_without_code)

    normalized = []
    for match in matches:
        # Split on | to get target part (before display text)
        target = match.split('|')[0].strip()

        # Remove heading and block refs
        target = re.sub(r'[#^].*$', '', target).strip()

        # Remove trailing .md
        if target.endswith('.md'):
            target = target[:-3]

        # Take the last path segment (basename)
        if '/' in target:
            target = target.split('/')[-1]

        # Lowercase and strip
        target = target.strip().lower()

        if target:
            normalized.append(target)

    return normalized


def check_uncompiled_raw(
    raw_sources: list[Any],
    manifest: Any,
) -> list[LintFinding]:
    """Check for uncompiled or changed raw sources.

    Severity: WARNING
    Code: uncompiled_raw
    """
    findings: list[LintFinding] = []

    for source in raw_sources:
        key = source.source_url or source.relative_path

        if key not in manifest.sources:
            findings.append(
                LintFinding(
                    code="uncompiled_raw",
                    severity=LintSeverity.WARNING,
                    message="Raw source never compiled",
                    subject=key,
                    detail="",
                )
            )
        elif manifest.sources[key].content_hash != source.content_hash:
            findings.append(
                LintFinding(
                    code="uncompiled_raw",
                    severity=LintSeverity.WARNING,
                    message="Raw source changed since last compile",
                    subject=key,
                    detail="",
                )
            )

    # Sort by subject for determinism
    findings.sort(key=lambda f: f.subject or "")
    return findings


def check_orphan_raw(
    raw_sources: list[Any],
    manifest: Any,
    concept_pages: list[ConceptPage],
) -> list[LintFinding]:
    """Check for compiled raw sources cited by no wiki article.

    Severity: INFO
    Code: orphan_raw

    Only considers sources that ARE compiled and current (not flagged by check #1).
    """
    findings: list[LintFinding] = []

    # Build set of raw sources that are cited by a wiki article's source list
    # or by a concept page's frontmatter sources.
    cited_keys = set()

    # Check manifest wiki_articles
    for _article_slug, entry in manifest.wiki_articles.items():
        for source_url in entry.source_urls:
            cited_keys.add(source_url)

    # Check concept_pages sources
    for page in concept_pages:
        for src in page.sources:
            cited_keys.add(src)

    # Check each raw source
    for source in raw_sources:
        key = source.source_url or source.relative_path

        manifest_entry = manifest.sources.get(key)

        # Skip if not compiled or changed since compile (flagged by check #1)
        if manifest_entry is None or manifest_entry.content_hash != source.content_hash:
            continue

        # Cited if the manifest back-reference lists any wiki article, or any
        # wiki article / concept page lists this source as a citation.
        if manifest_entry.wiki_articles:
            continue
        if key in cited_keys:
            continue

        findings.append(
            LintFinding(
                code="orphan_raw",
                severity=LintSeverity.INFO,
                message="Raw source compiled but cited by no wiki concept",
                subject=key,
                detail="",
            )
        )

    # Sort by subject for determinism
    findings.sort(key=lambda f: f.subject or "")
    return findings


def check_sourceless_wiki(
    concept_pages: list[ConceptPage],
    manifest: Any,
) -> list[LintFinding]:
    """Check for wiki concepts with no sources.

    Severity: WARNING
    Code: sourceless_wiki
    """
    findings: list[LintFinding] = []

    for page in concept_pages:
        # Check if page has no sources in frontmatter
        page_has_frontmatter_sources = bool(page.sources)

        # Check if manifest entry has source_urls
        entry = manifest.wiki_articles.get(page.slug)
        entry_has_sources = bool(entry and entry.source_urls)

        if not page_has_frontmatter_sources and not entry_has_sources:
            findings.append(
                LintFinding(
                    code="sourceless_wiki",
                    severity=LintSeverity.WARNING,
                    message="Wiki concept has no source citations",
                    subject=page.slug,
                    detail="",
                )
            )

    # Sort by subject for determinism
    findings.sort(key=lambda f: f.subject or "")
    return findings


def check_broken_wikilinks(
    wiki_pages: list[WikiPage],
    valid_targets: set[str],
) -> list[LintFinding]:
    """Check for broken wikilinks in wiki pages (except index).

    Severity: ERROR
    Code: broken_wikilink
    """
    findings: list[LintFinding] = []

    for page in wiki_pages:
        # Skip index (checked by check #5)
        if page.is_index:
            continue

        links = extract_wikilinks(page.text)
        seen = set()  # dedupe within page

        for target in links:
            if target in seen:
                continue
            seen.add(target)

            if target not in valid_targets:
                findings.append(
                    LintFinding(
                        code="broken_wikilink",
                        severity=LintSeverity.ERROR,
                        message=f"Wikilink target not found: {target}",
                        subject=page.relative_path,
                        detail=f"[[{target}]]",
                    )
                )

    # Sort by subject, then detail for determinism
    findings.sort(key=lambda f: (f.subject or "", f.detail or ""))
    return findings


def check_stale_index(
    index_text: str,
    concept_slugs: set[str],
) -> list[LintFinding]:
    """Check for index wikilinks to non-existent concepts.

    Severity: WARNING
    Code: stale_index_entry
    """
    findings: list[LintFinding] = []

    links = extract_wikilinks(index_text)
    seen = set()  # dedupe

    for target in links:
        if target in seen:
            continue
        seen.add(target)

        if target not in concept_slugs:
            findings.append(
                LintFinding(
                    code="stale_index_entry",
                    severity=LintSeverity.WARNING,
                    message=f"Index links non-existent concept: {target}",
                    subject="📇 Index",
                    detail=f"[[{target}]]",
                )
            )

    # Sort by detail for determinism
    findings.sort(key=lambda f: f.detail or "")
    return findings


def check_duplicate_concepts(
    concept_pages: list[ConceptPage],
) -> list[LintFinding]:
    """Check for suspiciously similar concept pages.

    Severity: WARNING
    Code: duplicate_concept

    Uses difflib.SequenceMatcher to compare slug and title (lowercased).
    Threshold: DUPLICATE_SIMILARITY_THRESHOLD (0.85).
    Reports one finding per pair (deduped).
    """
    findings: list[LintFinding] = []
    reported_pairs = set()  # track (slug_a, slug_b) to avoid duplicates

    for i, page_a in enumerate(concept_pages):
        for page_b in concept_pages[i + 1 :]:
            # Check slug similarity
            slug_ratio = difflib.SequenceMatcher(
                None, page_a.slug.lower(), page_b.slug.lower()
            ).ratio()

            # Check title similarity
            title_ratio = difflib.SequenceMatcher(
                None, page_a.title.lower(), page_b.title.lower()
            ).ratio()

            max_ratio = max(slug_ratio, title_ratio)

            if max_ratio >= DUPLICATE_SIMILARITY_THRESHOLD:
                pair_key = tuple(sorted([page_a.slug, page_b.slug]))
                if pair_key not in reported_pairs:
                    reported_pairs.add(pair_key)
                    findings.append(
                        LintFinding(
                            code="duplicate_concept",
                            severity=LintSeverity.WARNING,
                            message=f"Similar concept: {page_b.slug} (match: {max_ratio:.2f})",
                            subject=page_a.slug,
                            detail="",
                        )
                    )

    # Sort by subject, then message for determinism
    findings.sort(key=lambda f: (f.subject or "", f.message or ""))
    return findings


def lint_vault(
    *,
    raw_sources: list[Any],
    manifest: Any,
    concept_pages: list[ConceptPage],
    wiki_pages: list[WikiPage],
    index_text: str,
    valid_targets: set[str],
    concept_slugs: set[str],
) -> LintReport:
    """Run all lint checks and return a report."""
    findings: list[LintFinding] = []

    # Check 1: uncompiled_raw
    findings.extend(check_uncompiled_raw(raw_sources, manifest))

    # Check 2: orphan_raw
    findings.extend(check_orphan_raw(raw_sources, manifest, concept_pages))

    # Check 3: sourceless_wiki
    findings.extend(check_sourceless_wiki(concept_pages, manifest))

    # Check 4: broken_wikilink
    findings.extend(check_broken_wikilinks(wiki_pages, valid_targets))

    # Check 5: stale_index_entry
    findings.extend(check_stale_index(index_text, concept_slugs))

    # Check 6: duplicate_concept
    findings.extend(check_duplicate_concepts(concept_pages))

    return LintReport(findings=findings)


def render_lint_markdown(report: LintReport, *, date_label: str) -> str:
    """Render lint report as markdown body (no frontmatter).

    date_label: e.g. "2026-06-06"
    """
    lines = [f"# Wiki Lint Report — {date_label}"]

    if report.total == 0:
        lines.append("")
        lines.append("✅ No issues found. The wiki is healthy.")
        return "\n".join(lines)

    # Summary line
    lines.append("")
    lines.append(
        f"**{report.total} issue(s)** — {report.error_count} error, "
        f"{report.warning_count} warning, {report.info_count} info."
    )

    # Group findings by code, in fixed order
    code_order = [
        "broken_wikilink",
        "uncompiled_raw",
        "orphan_raw",
        "sourceless_wiki",
        "stale_index_entry",
        "duplicate_concept",
    ]

    findings_by_code: dict[str, list[LintFinding]] = {}
    for finding in report.findings:
        if finding.code not in findings_by_code:
            findings_by_code[finding.code] = []
        findings_by_code[finding.code].append(finding)

    for code in code_order:
        if code not in findings_by_code:
            continue

        group = findings_by_code[code]
        count = len(group)

        # Human title for code
        code_titles = {
            "broken_wikilink": "Broken Wikilinks",
            "uncompiled_raw": "Uncompiled Raw Sources",
            "orphan_raw": "Orphan Raw Sources",
            "sourceless_wiki": "Sourceless Wiki Concepts",
            "stale_index_entry": "Stale Index Entries",
            "duplicate_concept": "Duplicate Concepts",
        }
        title = code_titles.get(code, code)

        lines.append("")
        lines.append(f"## {title} ({count})")

        for finding in group:
            bullet = f"- {finding.subject or 'unknown'}: {finding.message}"
            if finding.detail:
                bullet += f" ({finding.detail})"
            lines.append(bullet)

    return "\n".join(lines)
