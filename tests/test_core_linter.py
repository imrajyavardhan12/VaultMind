"""Tests for the deterministic wiki linter."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from vaultmind.core.linter import (
    ConceptPage,
    LintFinding,
    LintSeverity,
    WikiPage,
    check_broken_wikilinks,
    check_duplicate_concepts,
    check_orphan_raw,
    check_sourceless_wiki,
    check_stale_index,
    check_stale_manifest,
    check_uncompiled_raw,
    extract_wikilinks,
    lint_vault,
    render_lint_markdown,
)
from vaultmind.schemas import Manifest, ManifestSource, ManifestWikiEntry

# ---- Fixtures ----


@pytest.fixture
def sample_raw_source():
    """Create a mock raw source record."""
    class RawSource:
        def __init__(self, relative_path, source_url, content_hash_val):
            self.relative_path = relative_path
            self.source_url = source_url
            self.content_hash = content_hash_val

    return RawSource


@pytest.fixture
def sample_concept_page():
    """Create a mock concept page."""
    def _make(slug="test-concept", title="Test Concept", sources=None, text=""):
        if sources is None:
            sources = []
        return ConceptPage(
            slug=slug,
            title=title,
            relative_path=f"🗺️ Wiki/🧠 Concepts/{slug}",
            sources=sources,
            text=text,
        )

    return _make


# ---- Tests for extract_wikilinks ----


def test_extract_wikilinks_simple():
    """Test extraction of simple wikilinks."""
    text = "[[foo]] and [[bar]]"
    links = extract_wikilinks(text)
    assert set(links) == {"foo", "bar"}


def test_extract_wikilinks_with_display():
    """Test extraction of wikilinks with display text."""
    text = "[[foo|Foo Title]] and [[bar|Bar Title]]"
    links = extract_wikilinks(text)
    assert set(links) == {"foo", "bar"}


def test_extract_wikilinks_with_paths():
    """Test extraction of wikilinks with paths."""
    text = "[[🗺️ Wiki/🧠 Concepts/foo]]"
    links = extract_wikilinks(text)
    assert links == ["foo"]


def test_extract_wikilinks_with_heading_ref():
    """Test extraction of wikilinks with heading refs."""
    text = "[[foo#section]]"
    links = extract_wikilinks(text)
    assert links == ["foo"]


def test_extract_wikilinks_with_block_ref():
    """Test extraction of wikilinks with block refs."""
    text = "[[foo^blockref]]"
    links = extract_wikilinks(text)
    assert links == ["foo"]


def test_extract_wikilinks_with_md_suffix():
    """Test extraction of wikilinks with .md suffix."""
    text = "[[foo.md]]"
    links = extract_wikilinks(text)
    assert links == ["foo"]


def test_extract_wikilinks_ignores_code_blocks():
    """Test that wikilinks inside code blocks are ignored."""
    text = """
Some text [[valid]].

```
[[invalid]]
```

More text [[another-valid]].
"""
    links = extract_wikilinks(text)
    assert set(links) == {"valid", "another-valid"}


def test_extract_wikilinks_case_normalization():
    """Test that links are lowercased."""
    text = "[[FOO]] and [[Bar]]"
    links = extract_wikilinks(text)
    assert set(links) == {"foo", "bar"}


def test_extract_wikilinks_deduplication_within_text():
    """Test that duplicates within text are preserved (caller dedupes)."""
    text = "[[foo]] and [[foo]]"
    links = extract_wikilinks(text)
    assert links == ["foo", "foo"]


# ---- Tests for check_uncompiled_raw ----


def test_check_uncompiled_raw_never_compiled(sample_raw_source):
    """Test detection of raw sources never compiled."""
    source = sample_raw_source("raw/a.md", "https://example.com/a", "hash-a")
    manifest = Manifest()

    findings = check_uncompiled_raw([source], manifest)

    assert len(findings) == 1
    assert findings[0].code == "uncompiled_raw"
    assert findings[0].severity == LintSeverity.WARNING
    assert findings[0].subject == "https://example.com/a"


def test_check_uncompiled_raw_hash_changed(sample_raw_source):
    """Test detection of raw sources with changed hash."""
    source = sample_raw_source("raw/a.md", "https://example.com/a", "hash-a-new")
    now = datetime.now(UTC)
    manifest = Manifest(
        sources={
            "https://example.com/a": ManifestSource(
                content_hash="hash-a-old",
                saved_at=now,
                compiled_at=None,
            )
        }
    )

    findings = check_uncompiled_raw([source], manifest)

    assert len(findings) == 1
    assert findings[0].code == "uncompiled_raw"
    assert findings[0].message == "Raw source changed since last compile"


def test_check_uncompiled_raw_no_issues(sample_raw_source):
    """Test that no findings when source is compiled and unchanged."""
    source = sample_raw_source("raw/a.md", "https://example.com/a", "hash-a")
    now = datetime.now(UTC)
    manifest = Manifest(
        sources={
            "https://example.com/a": ManifestSource(
                content_hash="hash-a",
                saved_at=now,
                compiled_at=None,
            )
        }
    )

    findings = check_uncompiled_raw([source], manifest)

    assert len(findings) == 0


def test_check_uncompiled_raw_uses_relative_path_as_fallback(sample_raw_source):
    """Test that relative_path is used as key when source_url is None."""
    source = sample_raw_source("raw/a.md", None, "hash-a")
    manifest = Manifest()

    findings = check_uncompiled_raw([source], manifest)

    assert len(findings) == 1
    assert findings[0].subject == "raw/a.md"


# ---- Tests for check_orphan_raw ----


def test_check_orphan_raw_no_orphans(sample_raw_source, sample_concept_page):
    """Test that compiled and cited sources are not orphaned."""
    source = sample_raw_source("raw/a.md", "https://example.com/a", "hash-a")
    concept = sample_concept_page(sources=["https://example.com/a"])
    now = datetime.now(UTC)
    manifest = Manifest(
        sources={
            "https://example.com/a": ManifestSource(
                content_hash="hash-a",
                saved_at=now,
                compiled_at=None,
                wiki_articles=["test-concept"],
            )
        }
    )

    findings = check_orphan_raw([source], manifest, [concept])

    assert len(findings) == 0


def test_check_orphan_raw_detects_orphaned(sample_raw_source, sample_concept_page):
    """Test detection of compiled but uncited sources."""
    source = sample_raw_source("raw/a.md", "https://example.com/a", "hash-a")
    concept = sample_concept_page()  # no sources
    now = datetime.now(UTC)
    manifest = Manifest(
        sources={
            "https://example.com/a": ManifestSource(
                content_hash="hash-a",
                saved_at=now,
                compiled_at=None,
                wiki_articles=[],
            )
        }
    )

    findings = check_orphan_raw([source], manifest, [concept])

    assert len(findings) == 1
    assert findings[0].code == "orphan_raw"
    assert findings[0].severity == LintSeverity.INFO
    assert findings[0].subject == "https://example.com/a"


def test_check_orphan_raw_cited_via_manifest_backref(sample_raw_source, sample_concept_page):
    """A source is cited if its manifest entry back-references a wiki article.

    Even when no wiki_articles.source_urls and no concept.sources list it, the
    `manifest.sources[key].wiki_articles` back-reference means it is cited.
    """
    source = sample_raw_source("raw/a.md", "https://example.com/a", "hash-a")
    concept = sample_concept_page()  # no frontmatter sources
    now = datetime.now(UTC)
    manifest = Manifest(
        sources={
            "https://example.com/a": ManifestSource(
                content_hash="hash-a",
                saved_at=now,
                compiled_at=None,
                wiki_articles=["concept-x"],
            )
        },
        # Note: no wiki_articles entry, so source_urls citation path is empty.
    )

    findings = check_orphan_raw([source], manifest, [concept])

    assert len(findings) == 0


def test_check_orphan_raw_skips_uncompiled(sample_raw_source):
    """Test that uncompiled sources are not reported (check #1 handles them)."""
    source = sample_raw_source("raw/a.md", "https://example.com/a", "hash-a")
    manifest = Manifest()  # source not in manifest

    findings = check_orphan_raw([source], manifest, [])

    assert len(findings) == 0


def test_check_orphan_raw_skips_changed(sample_raw_source):
    """Test that changed sources are not reported (check #1 handles them)."""
    source = sample_raw_source("raw/a.md", "https://example.com/a", "hash-a-new")
    now = datetime.now(UTC)
    manifest = Manifest(
        sources={
            "https://example.com/a": ManifestSource(
                content_hash="hash-a-old",
                saved_at=now,
                compiled_at=None,
            )
        }
    )

    findings = check_orphan_raw([source], manifest, [])

    assert len(findings) == 0


# ---- Tests for check_sourceless_wiki ----


def test_check_sourceless_wiki_with_sources(sample_concept_page):
    """Test that concepts with sources are not flagged."""
    concept = sample_concept_page(sources=["https://example.com/a"])
    manifest = Manifest()

    findings = check_sourceless_wiki([concept], manifest)

    assert len(findings) == 0


def test_check_sourceless_wiki_with_manifest_sources(sample_concept_page):
    """Test that concepts with manifest source_urls are not flagged."""
    concept = sample_concept_page(sources=[])
    now = datetime.now(UTC)
    manifest = Manifest(
        wiki_articles={
            "test-concept": ManifestWikiEntry(
                last_updated=now,
                source_urls=["https://example.com/a"],
            )
        }
    )

    findings = check_sourceless_wiki([concept], manifest)

    assert len(findings) == 0


def test_check_sourceless_wiki_detects_sourceless(sample_concept_page):
    """Test detection of concepts with no sources."""
    concept = sample_concept_page(sources=[])
    manifest = Manifest()

    findings = check_sourceless_wiki([concept], manifest)

    assert len(findings) == 1
    assert findings[0].code == "sourceless_wiki"
    assert findings[0].severity == LintSeverity.WARNING
    assert findings[0].subject == "test-concept"


# ---- Tests for check_broken_wikilinks ----


def test_check_broken_wikilinks_detects_broken():
    """Test detection of broken wikilinks."""
    page = WikiPage(
        relative_path="🗺️ Wiki/🧠 Concepts/concept-a",
        text="[[valid]] and [[invalid]]",
        is_index=False,
    )
    valid_targets = {"valid"}

    findings = check_broken_wikilinks([page], valid_targets)

    assert len(findings) == 1
    assert findings[0].code == "broken_wikilink"
    assert findings[0].severity == LintSeverity.ERROR
    assert findings[0].subject == "🗺️ Wiki/🧠 Concepts/concept-a"
    assert findings[0].detail == "[[invalid]]"


def test_check_broken_wikilinks_skips_index():
    """Test that index page is skipped (checked by check #5)."""
    page = WikiPage(
        relative_path="🗺️ Wiki/📇 Index",
        text="[[invalid]]",
        is_index=True,
    )
    valid_targets = set()

    findings = check_broken_wikilinks([page], valid_targets)

    assert len(findings) == 0


def test_check_broken_wikilinks_no_issues():
    """Test that no findings when all links are valid."""
    page = WikiPage(
        relative_path="🗺️ Wiki/🧠 Concepts/concept-a",
        text="[[valid-a]] and [[valid-b]]",
        is_index=False,
    )
    valid_targets = {"valid-a", "valid-b"}

    findings = check_broken_wikilinks([page], valid_targets)

    assert len(findings) == 0


def test_check_broken_wikilinks_dedupes_within_page():
    """Test that duplicate broken links in same page are deduped."""
    page = WikiPage(
        relative_path="🗺️ Wiki/🧠 Concepts/concept-a",
        text="[[invalid]] and [[invalid]]",
        is_index=False,
    )
    valid_targets = set()

    findings = check_broken_wikilinks([page], valid_targets)

    assert len(findings) == 1


# ---- Tests for check_stale_index ----


def test_check_stale_index_detects_stale():
    """Test detection of stale index entries."""
    index_text = "[[valid-concept]] and [[invalid-concept]]"
    concept_slugs = {"valid-concept"}

    findings = check_stale_index(index_text, concept_slugs)

    assert len(findings) == 1
    assert findings[0].code == "stale_index_entry"
    assert findings[0].severity == LintSeverity.WARNING
    assert findings[0].subject == "📇 Index"
    assert findings[0].detail == "[[invalid-concept]]"


def test_check_stale_index_no_issues():
    """Test that no findings when all index links are valid."""
    index_text = "[[concept-a]] and [[concept-b]]"
    concept_slugs = {"concept-a", "concept-b"}

    findings = check_stale_index(index_text, concept_slugs)

    assert len(findings) == 0


# ---- Tests for stale manifest state ----


def test_check_stale_manifest_is_deterministic(sample_concept_page):
    now = datetime.now(UTC)
    cited_history = "https://example.com/cited-history"
    missing_uncited = "https://example.com/missing-uncited"
    manifest = Manifest(
        sources={
            cited_history: ManifestSource(content_hash="a", saved_at=now),
            missing_uncited: ManifestSource(content_hash="b", saved_at=now),
        },
        wiki_articles={
            "z-missing": ManifestWikiEntry(last_updated=now),
            "a-missing": ManifestWikiEntry(last_updated=now),
        },
    )
    concept = sample_concept_page(slug="present", sources=[cited_history])

    findings = check_stale_manifest([], manifest, [concept])

    assert [(finding.code, finding.subject) for finding in findings] == [
        ("stale_manifest_concept", "a-missing"),
        ("stale_manifest_concept", "z-missing"),
        ("stale_manifest_source", missing_uncited),
    ]


# ---- Tests for check_duplicate_concepts ----


def test_check_duplicate_concepts_detects_similar_slugs(sample_concept_page):
    """Test detection of concepts with similar slugs."""
    concept_a = sample_concept_page(slug="attention-mechanisms")
    concept_b = sample_concept_page(slug="attention-mechanism")  # very similar

    findings = check_duplicate_concepts([concept_a, concept_b])

    assert len(findings) == 1
    assert findings[0].code == "duplicate_concept"
    assert findings[0].severity == LintSeverity.WARNING


def test_check_duplicate_concepts_detects_similar_titles(sample_concept_page):
    """Test detection of concepts with similar titles."""
    concept_a = sample_concept_page(slug="a", title="Attention Mechanisms")
    concept_b = sample_concept_page(slug="b", title="Attention Mechanism")  # very similar

    findings = check_duplicate_concepts([concept_a, concept_b])

    assert len(findings) == 1


def test_check_duplicate_concepts_no_false_positives(sample_concept_page):
    """Test that dissimilar concepts are not flagged."""
    concept_a = sample_concept_page(slug="attention-mechanisms", title="Attention Mechanisms")
    concept_b = sample_concept_page(slug="transformers", title="Transformer Models")

    findings = check_duplicate_concepts([concept_a, concept_b])

    assert len(findings) == 0


# ---- Tests for render_lint_markdown ----


def test_render_lint_markdown_empty_report():
    """Test rendering of empty report."""
    from vaultmind.core.linter import LintReport

    report = LintReport(findings=[])
    markdown = render_lint_markdown(report, date_label="2026-06-06")

    assert "No issues found" in markdown
    assert "Wiki Lint Report — 2026-06-06" in markdown


def test_render_lint_markdown_with_findings():
    """Test rendering of report with findings."""
    from vaultmind.core.linter import LintReport

    findings = [
        LintFinding(
            code="broken_wikilink",
            severity=LintSeverity.ERROR,
            message="Test error",
            subject="concept-a",
            detail="[[invalid]]",
        ),
        LintFinding(
            code="sourceless_wiki",
            severity=LintSeverity.WARNING,
            message="Test warning",
            subject="concept-b",
            detail="",
        ),
    ]
    report = LintReport(findings=findings)
    markdown = render_lint_markdown(report, date_label="2026-06-06")

    assert "2 issue(s)" in markdown
    assert "1 error" in markdown
    assert "1 warning" in markdown
    assert "Broken Wikilinks" in markdown
    assert "Sourceless Wiki Concepts" in markdown
    assert "concept-a" in markdown


# ---- Tests for lint_vault ----


def test_lint_vault_integration(sample_raw_source, sample_concept_page):
    """Test full lint_vault integration."""
    source = sample_raw_source("raw/a.md", "https://example.com/a", "hash-a")
    concept = sample_concept_page(sources=["https://example.com/a"])
    now = datetime.now(UTC)
    manifest = Manifest(
        sources={
            "https://example.com/a": ManifestSource(
                content_hash="hash-a",
                saved_at=now,
                compiled_at=None,
                wiki_articles=["test-concept"],
            )
        }
    )
    wiki_page = WikiPage(
        relative_path="🗺️ Wiki/🧠 Concepts/test-concept",
        text="[[test-concept]]",
        is_index=False,
    )
    valid_targets = {"test-concept"}
    concept_slugs = {"test-concept"}

    report = lint_vault(
        raw_sources=[source],
        manifest=manifest,
        concept_pages=[concept],
        wiki_pages=[wiki_page],
        index_text="",
        valid_targets=valid_targets,
        concept_slugs=concept_slugs,
    )

    assert report.total == 0
    assert report.error_count == 0
