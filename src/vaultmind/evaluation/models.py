"""Typed contracts for the deterministic offline evaluation harness."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class QuerySupport(StrEnum):
    """Persisted evidence classes available to a filed query page."""

    CONCEPT = "concept"
    FILED_QUERY = "filed_query"
    RAW = "raw"


class CorpusSource(BaseModel):
    """One Raw document materialized in the isolated evaluation vault."""

    id: str
    title: str
    source_url: str
    tags: list[str] = Field(default_factory=list)
    body: str
    phase: int = Field(default=1, ge=1)


class ExpectedConcept(BaseModel):
    """A concept expected from the representative corpus."""

    slug: str
    title: str
    source_ids: list[str] = Field(default_factory=list)


class ExpectedEdge(BaseModel):
    """A directed concept wikilink expected in the compiled graph."""

    source: str
    target: str


class QueryExpectation(BaseModel):
    """Expected persisted support for one production Ask invocation."""

    question: str
    support: list[QuerySupport]
    reuses_filed_query: bool = False


class EvaluationScenario(BaseModel):
    """Complete corpus and quality expectations for one replay scenario."""

    name: str
    sources: list[CorpusSource]
    expected_concepts: list[ExpectedConcept]
    expected_edges: list[ExpectedEdge]
    queries: list[QueryExpectation]

    @model_validator(mode="after")
    def validate_unique_identity(self) -> EvaluationScenario:
        """Reject ambiguous fixture identities before any vault is mutated."""
        for label, values in (
            ("source id", [source.id for source in self.sources]),
            ("source URL", [source.source_url for source in self.sources]),
            ("concept slug", [concept.slug for concept in self.expected_concepts]),
            ("query", [query.question for query in self.queries]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {label} in evaluation scenario")
        source_ids = {source.id for source in self.sources}
        unknown_ids = sorted(
            source_id
            for concept in self.expected_concepts
            for source_id in concept.source_ids
            if source_id not in source_ids
        )
        if unknown_ids:
            raise ValueError(f"unknown expected concept source ids: {', '.join(unknown_ids)}")
        return self


class EvaluationThreshold(BaseModel):
    """An inclusive lower and/or upper quality bound for one numeric metric."""

    metric: str
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> EvaluationThreshold:
        if self.minimum is None and self.maximum is None:
            raise ValueError("threshold must define minimum or maximum")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("threshold minimum cannot exceed maximum")
        return self


class EvaluationThresholds(BaseModel):
    """Named threshold set checked by developer and CI evaluation runs."""

    name: str
    thresholds: list[EvaluationThreshold]


class EvaluationMetrics(BaseModel):
    """Stable machine-readable measurements from Compile, Ask, and Lint."""

    corpus_source_count: int
    scanned_source_count: int
    current_compiled_source_count: int
    current_compiled_coverage: float
    concept_count: int
    expected_concept_recall: float
    expected_source_to_concept_attribution_count: int
    expected_source_to_concept_attribution_recall: float
    unexpected_source_to_concept_attribution_count: int
    source_to_concept_attribution_precision: float
    suspicious_duplicate_pairs: list[str]
    suspicious_duplicate_pair_count: int
    citation_provenance_inconsistencies: list[str]
    citation_provenance_inconsistency_count: int
    expected_graph_edge_count: int
    expected_graph_edge_recall: float
    duplicate_connection_count: int
    index_present: int
    expected_concept_index_link_count: int
    expected_concept_index_link_recall: float
    unexpected_index_link_count: int
    broken_wikilinks: int
    stale_manifest_findings: int
    lint_finding_count: int
    query_count: int
    concept_supported_query_count: int
    filed_query_supported_query_count: int
    raw_supported_query_count: int
    wiki_supported_query_rate: float
    raw_fallback_count: int
    filed_query_reuse_count: int
    query_expectation_pass_rate: float
    incremental_changed_file_count: int
    incremental_owned_state_write_count: int
    incremental_provider_call_count: int
    reconciliation_probe_results: list[str]
    reconciliation_probe_inconsistency_count: int
    reconciliation_probe_repaired_count: int
    reconciliation_probe_success_rate: float
    provider_call_count: int


class ThresholdFailure(BaseModel):
    """One failed quality bound; reports retain all failures in stable order."""

    metric: str
    actual: float
    minimum: float | None = None
    maximum: float | None = None


class EvaluationReport(BaseModel):
    """Deterministic final evaluation artifact."""

    schema_version: int = 1
    scenario: str
    threshold_set: str
    metrics: EvaluationMetrics
    failures: list[ThresholdFailure]
    passed: bool


class EvaluationErrorReport(BaseModel):
    """Safe artifact emitted when evaluation cannot produce quality metrics."""

    schema_version: int = 1
    passed: bool = False
    error_type: str
    error_fingerprint: str
