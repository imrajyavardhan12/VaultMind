"""Deterministic quality metrics for persisted VaultMind wiki state."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from vaultmind.core.linter import ConceptPage, LintReport, extract_wikilinks
from vaultmind.core.raw_scanner import RawSourceRecord
from vaultmind.evaluation.models import (
    EvaluationMetrics,
    EvaluationScenario,
    QuerySupport,
)
from vaultmind.schemas import Manifest


def _ratio(numerator: int, denominator: int) -> float:
    """Return a stable vacuous-success ratio for an empty expectation set."""
    return 1.0 if denominator == 0 else numerator / denominator


def _strip_fenced_blocks(markdown: str) -> str:
    """Remove backtick and tilde fenced examples before graph inspection."""
    retained: list[str] = []
    marker: str | None = None
    marker_length = 0
    for line in markdown.splitlines(keepends=True):
        if marker is not None:
            closer = re.match(rf"^ {{0,3}}{re.escape(marker)}{{{marker_length},}}[ \t]*(?:\n)?$", line)
            if closer:
                marker = None
                marker_length = 0
            continue
        opener = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if opener and (opener.group(1)[0] == "~" or "`" not in opener.group(2)):
            marker = opener.group(1)[0]
            marker_length = len(opener.group(1))
            continue
        retained.append(line)
    return "".join(retained)


def index_quality(
    index_markdown: str | None,
    scenario: EvaluationScenario,
) -> tuple[int, int, float, int]:
    """Measure index presence, expected concept coverage, and unexpected links."""
    expected = {concept.slug.lower() for concept in scenario.expected_concepts}
    actual = set(
        extract_wikilinks(_strip_fenced_blocks(index_markdown or ""))
    )
    return (
        int(index_markdown is not None),
        len(expected),
        _ratio(len(actual & expected), len(expected)),
        len(actual - expected),
    )


def graph_quality(
    concept_pages: Sequence[ConceptPage], scenario: EvaluationScenario
) -> tuple[float, int]:
    """Return expected directed-edge recall and duplicate connection count."""
    real_edges: set[tuple[str, str]] = set()
    duplicate_connections = 0
    for page in sorted(concept_pages, key=lambda item: item.slug):
        cleaned = _strip_fenced_blocks(page.text)
        links = extract_wikilinks(cleaned)
        real_edges.update((page.slug, target) for target in links if target != page.slug)

        connections = _section(cleaned, "Connections")
        connection_links = extract_wikilinks(connections)
        duplicate_connections += len(connection_links) - len(set(connection_links))

    expected = {(edge.source.lower(), edge.target.lower()) for edge in scenario.expected_edges}
    return _ratio(len(real_edges & expected), len(expected)), duplicate_connections


def citation_inconsistencies(
    raw_sources: Sequence[RawSourceRecord],
    manifest: Manifest,
    concept_pages: Sequence[ConceptPage],
) -> list[str]:
    """Check concept/manifest citations and source/article backlinks both ways."""
    findings: set[str] = set()
    pages = {page.slug: page for page in concept_pages}

    for slug, page in sorted(pages.items()):
        page_sources = set(page.sources)
        entry = manifest.wiki_articles.get(slug)
        if entry is None:
            findings.add(f"concept_missing_manifest:{slug}")
            manifest_sources: set[str] = set()
        else:
            manifest_sources = set(entry.source_urls)
        for source in sorted(page_sources - manifest_sources):
            findings.add(f"page_source_missing_manifest_article:{slug}:{source}")
        for source in sorted(manifest_sources - page_sources):
            findings.add(f"manifest_article_source_missing_page:{slug}:{source}")
        for source in sorted(page_sources | manifest_sources):
            source_entry = manifest.sources.get(source)
            if source_entry is None:
                findings.add(f"article_source_missing_manifest_source:{slug}:{source}")
            elif slug not in source_entry.wiki_articles:
                findings.add(f"source_backlink_missing_article:{slug}:{source}")

    for source, source_manifest_entry in sorted(manifest.sources.items()):
        for slug in sorted(set(source_manifest_entry.wiki_articles)):
            linked_page = pages.get(slug)
            linked_article = manifest.wiki_articles.get(slug)
            if linked_page is None or source not in linked_page.sources:
                findings.add(f"source_backlink_missing_page_citation:{slug}:{source}")
            if linked_article is None or source not in linked_article.source_urls:
                findings.add(f"source_backlink_missing_article_citation:{slug}:{source}")

    raw_keys = {source.source_url or source.relative_path for source in raw_sources}
    for source in sorted(set(manifest.sources) - raw_keys):
        findings.add(f"manifest_source_missing_raw:{source}")
    return sorted(findings)


def source_concept_attribution_quality(
    scenario: EvaluationScenario,
    manifest: Manifest,
    concept_pages: Sequence[ConceptPage],
) -> tuple[int, float, int, float]:
    """Measure persisted source-to-concept assignments against fixture semantics."""
    source_keys_by_id = {source.id: source.source_url for source in scenario.sources}
    expected = {
        (source_keys_by_id[source_id], concept.slug.lower())
        for concept in scenario.expected_concepts
        for source_id in concept.source_ids
    }

    actual: set[tuple[str, str]] = set()
    for page in concept_pages:
        actual.update((source, page.slug.lower()) for source in page.sources)
    for slug, article_entry in manifest.wiki_articles.items():
        actual.update((source, slug.lower()) for source in article_entry.source_urls)
    for source, source_entry in manifest.sources.items():
        actual.update((source, slug.lower()) for slug in source_entry.wiki_articles)

    matched_count = len(actual & expected)
    unexpected_count = len(actual - expected)
    return (
        len(expected),
        _ratio(matched_count, len(expected)),
        unexpected_count,
        _ratio(matched_count, len(actual)),
    )


def current_compiled_source_count(
    raw_sources: Sequence[RawSourceRecord],
    manifest: Manifest,
    concept_pages: Sequence[ConceptPage],
) -> int:
    """Count current-hash Raw with at least one durable, reciprocal concept backlink."""
    pages = {page.slug: page for page in concept_pages}
    count = 0
    for source in raw_sources:
        key = source.source_url or source.relative_path
        entry = manifest.sources.get(key)
        if entry is None or entry.content_hash != source.content_hash:
            continue
        durable = any(
            slug in pages
            and key in pages[slug].sources
            and slug in manifest.wiki_articles
            and key in manifest.wiki_articles[slug].source_urls
            for slug in entry.wiki_articles
        )
        if durable:
            count += 1
    return count


def _section(markdown: str, heading: str) -> str:
    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}\s*$(?P<body>.*?)(?=^##\s+|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(markdown)
    return match.group("body") if match else ""


def _query_support(markdown: str, wiki_concepts: str, wiki_queries: str) -> set[QuerySupport]:
    support: set[QuerySupport] = set()
    wiki_section = _section(_strip_fenced_blocks(markdown), "Supporting Wiki Pages")
    for target in re.findall(r"\[\[([^\]|#^]+)", wiki_section):
        normalized = target.replace("\\", "/").strip().lower()
        if f"/{wiki_queries.lower()}/" in f"/{normalized}/":
            support.add(QuerySupport.FILED_QUERY)
        elif f"/{wiki_concepts.lower()}/" in f"/{normalized}/":
            support.add(QuerySupport.CONCEPT)

    raw_section = _section(_strip_fenced_blocks(markdown), "Supporting Raw Sources")
    if any(
        line.strip().startswith(("- ", "* ", "+ "))
        for line in raw_section.splitlines()
    ):
        support.add(QuerySupport.RAW)
    return support


def _duplicate_pairs(lint_report: LintReport) -> list[str]:
    pairs: set[str] = set()
    for finding in lint_report.findings:
        if finding.code != "duplicate_concept" or not finding.subject:
            continue
        match = re.search(r"Similar concept:\s+([^\s]+)", finding.message)
        other = match.group(1) if match else "unknown"
        pairs.add(" <-> ".join(sorted((finding.subject, other))))
    return sorted(pairs)


def calculate_metrics(
    *,
    scenario: EvaluationScenario,
    raw_sources: Sequence[RawSourceRecord],
    manifest: Manifest,
    concept_pages: Sequence[ConceptPage],
    lint_report: LintReport,
    index_markdown: str | None,
    query_pages: Mapping[str, str],
    wiki_concepts_folder: str,
    wiki_queries_folder: str,
    incremental_changed_file_count: int,
    incremental_owned_state_write_count: int,
    incremental_provider_call_count: int,
    reconciliation_probe_results: Sequence[str],
    provider_call_count: int,
) -> EvaluationMetrics:
    """Calculate all report metrics from durable vault state."""
    compiled = current_compiled_source_count(raw_sources, manifest, concept_pages)
    page_slugs = {page.slug for page in concept_pages}
    expected_slugs = {concept.slug for concept in scenario.expected_concepts}
    citation_findings = citation_inconsistencies(raw_sources, manifest, concept_pages)
    graph_recall, duplicate_connections = graph_quality(concept_pages, scenario)
    (
        index_present,
        expected_index_count,
        expected_index_recall,
        unexpected_index_count,
    ) = index_quality(index_markdown, scenario)
    (
        expected_attribution_count,
        expected_attribution_recall,
        unexpected_attribution_count,
        attribution_precision,
    ) = source_concept_attribution_quality(scenario, manifest, concept_pages)
    duplicate_pairs = _duplicate_pairs(lint_report)

    support_by_question = {
        question: _query_support(markdown, wiki_concepts_folder, wiki_queries_folder)
        for question, markdown in sorted(query_pages.items())
    }
    query_count = len(support_by_question)
    concept_supported = sum(
        QuerySupport.CONCEPT in support for support in support_by_question.values()
    )
    filed_supported = sum(
        QuerySupport.FILED_QUERY in support for support in support_by_question.values()
    )
    raw_supported = sum(QuerySupport.RAW in support for support in support_by_question.values())
    wiki_supported = sum(
        bool(support & {QuerySupport.CONCEPT, QuerySupport.FILED_QUERY})
        for support in support_by_question.values()
    )

    probe_results = list(reconciliation_probe_results)
    repaired_probe_count = sum(
        result.endswith(":repaired") for result in probe_results
    )

    expectation_passes = 0
    for expectation in scenario.queries:
        actual = support_by_question.get(expectation.question, set())
        expected = set(expectation.support)
        if expected <= actual and (
            (QuerySupport.FILED_QUERY in actual) == expectation.reuses_filed_query
        ):
            expectation_passes += 1

    return EvaluationMetrics(
        corpus_source_count=len(scenario.sources),
        scanned_source_count=len(raw_sources),
        current_compiled_source_count=compiled,
        current_compiled_coverage=_ratio(compiled, len(raw_sources)),
        concept_count=len(concept_pages),
        expected_concept_recall=_ratio(len(page_slugs & expected_slugs), len(expected_slugs)),
        expected_source_to_concept_attribution_count=expected_attribution_count,
        expected_source_to_concept_attribution_recall=expected_attribution_recall,
        unexpected_source_to_concept_attribution_count=unexpected_attribution_count,
        source_to_concept_attribution_precision=attribution_precision,
        suspicious_duplicate_pairs=duplicate_pairs,
        suspicious_duplicate_pair_count=len(duplicate_pairs),
        citation_provenance_inconsistencies=citation_findings,
        citation_provenance_inconsistency_count=len(citation_findings),
        expected_graph_edge_count=len(scenario.expected_edges),
        expected_graph_edge_recall=graph_recall,
        duplicate_connection_count=duplicate_connections,
        index_present=index_present,
        expected_concept_index_link_count=expected_index_count,
        expected_concept_index_link_recall=expected_index_recall,
        unexpected_index_link_count=unexpected_index_count,
        broken_wikilinks=sum(f.code == "broken_wikilink" for f in lint_report.findings),
        stale_manifest_findings=sum(
            f.code in {"stale_manifest_concept", "stale_manifest_source"}
            for f in lint_report.findings
        ),
        lint_finding_count=lint_report.total,
        query_count=query_count,
        concept_supported_query_count=concept_supported,
        filed_query_supported_query_count=filed_supported,
        raw_supported_query_count=raw_supported,
        wiki_supported_query_rate=_ratio(wiki_supported, query_count),
        raw_fallback_count=raw_supported,
        filed_query_reuse_count=filed_supported,
        query_expectation_pass_rate=_ratio(expectation_passes, len(scenario.queries)),
        incremental_changed_file_count=incremental_changed_file_count,
        incremental_owned_state_write_count=incremental_owned_state_write_count,
        incremental_provider_call_count=incremental_provider_call_count,
        reconciliation_probe_results=probe_results,
        reconciliation_probe_inconsistency_count=len(probe_results),
        reconciliation_probe_repaired_count=repaired_probe_count,
        reconciliation_probe_success_rate=_ratio(repaired_probe_count, len(probe_results)),
        provider_call_count=provider_call_count,
    )
