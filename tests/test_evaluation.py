"""Rigorous integration and edge-case tests for the offline evaluation harness."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from vaultmind.commands import compile as compile_command
from vaultmind.core.linter import ConceptPage
from vaultmind.core.manifest import ManifestReconciliation
from vaultmind.core.raw_scanner import RawSourceRecord
from vaultmind.evaluation import (
    EvaluationThreshold,
    EvaluationThresholds,
    ExhaustedReplayResponseError,
    ReplayFixture,
    ReplayProvider,
    ReplayRule,
    UnmatchedReplayPromptError,
    evaluate_thresholds,
    render_report_json,
    run_evaluation,
)
from vaultmind.evaluation.metrics import (
    citation_inconsistencies,
    graph_quality,
    index_quality,
    source_concept_attribution_quality,
)
from vaultmind.evaluation.models import EvaluationScenario
from vaultmind.evaluation.replay import ReplayMatch
from vaultmind.evaluation.runner import DEFAULT_FIXTURE_DIR, OfflineNetworkError, _network_blocked
from vaultmind.schemas import Manifest, ManifestSource, ManifestWikiEntry


def _fixture_hashes(path: Path) -> dict[str, str]:
    return {
        item.relative_to(path).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


async def test_replay_matches_rules_under_concurrency_and_is_rule_local() -> None:
    provider = ReplayProvider(
        ReplayFixture(
            rules=[
                ReplayRule(
                    id="alpha",
                    match=ReplayMatch(prompt_contains=["alpha"]),
                    responses=["a1", "a2"],
                ),
                ReplayRule(
                    id="beta",
                    match=ReplayMatch(prompt_contains=["beta"]),
                    responses=["b1"],
                ),
            ]
        )
    )

    beta, alpha_one, alpha_two = await asyncio.gather(
        provider.complete("request beta"),
        provider.complete("request alpha"),
        provider.complete("another alpha"),
    )

    assert beta == "b1"
    assert {alpha_one, alpha_two} == {"a1", "a2"}
    assert provider.consumed_by_rule == {"alpha": 2, "beta": 1}


async def test_replay_rejects_unmatched_and_exhausted_without_prompt_disclosure() -> None:
    secret_prompt = "private prompt body token=do-not-disclose"
    provider = ReplayProvider(
        ReplayFixture(
            rules=[
                ReplayRule(
                    id="known",
                    match=ReplayMatch(prompt_contains=["known"]),
                    responses=["done"],
                )
            ]
        )
    )

    with pytest.raises(UnmatchedReplayPromptError) as unmatched:
        await provider.complete(secret_prompt)
    assert provider.call_count == 1
    assert secret_prompt not in str(unmatched.value)
    assert "prompt_sha256=" in str(unmatched.value)
    assert len(str(unmatched.value)) < 200

    assert await provider.complete("known") == "done"
    assert provider.call_count == 2
    with pytest.raises(ExhaustedReplayResponseError) as exhausted:
        await provider.complete("known")
    assert provider.call_count == 3
    assert "known" not in str(exhausted.value).replace("rule=known", "")
    assert len(str(exhausted.value)) < 220


def test_index_metrics_use_normalized_real_wikilinks() -> None:
    scenario = EvaluationScenario.model_validate(
        {
            "name": "index-edge-case",
            "sources": [],
            "expected_concepts": [
                {"slug": "alpha", "title": "Alpha", "source_ids": []},
                {"slug": "beta", "title": "Beta", "source_ids": []},
            ],
            "expected_edges": [],
            "queries": [],
        }
    )
    index_markdown = (
        "# Index\n\n- [[🧠 Concepts/Alpha.md|Alpha]]\n"
        "~~~markdown\n[[beta]]\n~~~\n"
        "- [[stale-concept#Overview|Stale]]\n"
    )

    present, expected_count, recall, unexpected_count = index_quality(
        index_markdown, scenario
    )

    assert present == 1
    assert expected_count == 2
    assert recall == 0.5
    assert unexpected_count == 1


def test_graph_metrics_ignore_fenced_edges_and_count_duplicate_connections() -> None:
    scenario = EvaluationScenario.model_validate(
        {
            "name": "edge-case",
            "sources": [],
            "expected_concepts": [],
            "expected_edges": [
                {"source": "alpha", "target": "beta"},
                {"source": "alpha", "target": "gamma"},
            ],
            "queries": [],
        }
    )
    page = ConceptPage(
        slug="alpha",
        title="Alpha",
        relative_path="Wiki/alpha",
        sources=[],
        text=(
            "## Connections\n[[beta]] and [[beta|again]]\n"
            "~~~markdown\n[[gamma]]\n~~~\n"
        ),
    )

    recall, duplicates = graph_quality([page], scenario)

    assert recall == 0.5
    assert duplicates == 1


def test_citation_metrics_check_both_manifest_directions_and_current_coverage() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    source = RawSourceRecord(
        path=Path("Raw/source.md"),
        relative_path="Raw/source",
        title="Source",
        source_url="https://example.test/source",
        body="body",
        content_hash="current",
        raw_tags=[],
    )
    page = ConceptPage(
        slug="alpha",
        title="Alpha",
        relative_path="Wiki/alpha",
        sources=["https://example.test/source"],
        text="# Alpha",
    )
    manifest = Manifest(
        sources={
            "https://example.test/source": ManifestSource(
                content_hash="current",
                saved_at=now,
                wiki_articles=[],
            )
        },
        wiki_articles={
            "alpha": ManifestWikiEntry(
                last_updated=now,
                source_urls=[],
                content_hash="hash",
            )
        },
    )

    findings = citation_inconsistencies([source], manifest, [page])

    assert any(item.startswith("page_source_missing_manifest_article") for item in findings)
    assert any(item.startswith("source_backlink_missing_article") for item in findings)


def test_network_guard_rejects_socket_connections() -> None:
    with _network_blocked(), pytest.raises(OfflineNetworkError):
        socket.create_connection(("example.com", 443))


def test_committed_fixture_shape_and_hygiene() -> None:
    corpus = json.loads((DEFAULT_FIXTURE_DIR / "corpus.json").read_text(encoding="utf-8"))
    fixture_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(DEFAULT_FIXTURE_DIR.glob("*.json"))
    )

    assert len(corpus["sources"]) == 20
    assert len(corpus["expected_concepts"]) >= 4
    assert all(len(concept["source_ids"]) >= 2 for concept in corpus["expected_concepts"])
    assert corpus["expected_edges"]
    assert not re.search(r"(?:/Users/|/home/|[A-Za-z]:\\\\)", fixture_text)
    assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", fixture_text)
    assert "api_key" not in fixture_text.lower()


def test_complete_fixture_is_isolated_deterministic_and_passes() -> None:
    before = _fixture_hashes(DEFAULT_FIXTURE_DIR)

    first = run_evaluation()
    second = run_evaluation()
    first_json = render_report_json(first)

    assert first.passed
    assert first_json == render_report_json(second)
    assert _fixture_hashes(DEFAULT_FIXTURE_DIR) == before
    assert first.metrics.corpus_source_count == 20
    assert first.metrics.scanned_source_count == 20
    assert first.metrics.concept_count == 4
    assert first.metrics.current_compiled_coverage == 1.0
    assert first.metrics.expected_source_to_concept_attribution_recall == 1.0
    assert first.metrics.unexpected_source_to_concept_attribution_count == 0
    assert first.metrics.source_to_concept_attribution_precision == 1.0
    assert first.metrics.expected_graph_edge_recall == 1.0
    assert first.metrics.index_present == 1
    assert first.metrics.expected_concept_index_link_recall == 1.0
    assert first.metrics.unexpected_index_link_count == 0
    assert first.metrics.reconciliation_probe_success_rate == 1.0
    assert first.metrics.reconciliation_probe_results == [
        "concept_hash:repaired",
        "ghost_wiki_entry:repaired",
        "source_back_reference:repaired",
    ]
    assert first.metrics.filed_query_reuse_count == 1
    assert first.metrics.raw_fallback_count == 1
    assert first.metrics.incremental_provider_call_count == 0
    assert first.metrics.incremental_changed_file_count == 0
    assert first.metrics.incremental_owned_state_write_count == 0
    assert not re.search(r"(?:/Users/|/home/|/var/folders/|[A-Za-z]:\\\\)", first_json)
    assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", first_json)
    assert "/tmp/" not in first_json
    assert "vaultmind-evaluation-" not in first_json
    assert "api_key" not in first_json.lower()


def test_evaluation_invokes_production_compile_for_each_phase_and_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production_compile = compile_command._compile
    invocations = 0

    def observe_compile(*, full: bool, dry_run: bool, verbose: bool, max_touches: int) -> None:
        nonlocal invocations
        invocations += 1
        production_compile(
            full=full,
            dry_run=dry_run,
            verbose=verbose,
            max_touches=max_touches,
        )

    monkeypatch.setattr(compile_command, "_compile", observe_compile)

    report = run_evaluation()

    assert report.passed
    assert invocations == 3


@pytest.mark.parametrize("mutation", ["missing", "incomplete"])
def test_index_rebuild_mutations_fail_the_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = tmp_path / "evaluation"
    shutil.copytree(DEFAULT_FIXTURE_DIR, fixture)
    replay_path = fixture / "replay.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    index_rule = next(rule for rule in replay["rules"] if rule["id"] == "index-rebuild")
    index_rule["responses"].append(index_rule["responses"][-1])
    replay_path.write_text(json.dumps(replay, indent=2) + "\n", encoding="utf-8")

    production_rebuild = compile_command._rebuild_wiki_index

    if mutation == "missing":
        monkeypatch.setattr(
            compile_command,
            "_rebuild_wiki_index",
            lambda config, manifest, provider: None,
        )
    else:
        def incomplete_rebuild(config, manifest, provider) -> None:  # type: ignore[no-untyped-def]
            production_rebuild(config, manifest, provider)
            index_path = (
                config.vault_path
                / config.folders.wiki
                / f"{config.folders.wiki_index}.md"
            )
            retained = [
                line
                for line in index_path.read_text(encoding="utf-8").splitlines()
                if "[[retrieval-grounding" not in line
            ]
            index_path.write_text("\n".join(retained) + "\n", encoding="utf-8")

        monkeypatch.setattr(compile_command, "_rebuild_wiki_index", incomplete_rebuild)

    report = run_evaluation(fixture)

    assert not report.passed
    failure_metrics = {failure.metric for failure in report.failures}
    expected_failure = (
        "index_present"
        if mutation == "missing"
        else "expected_concept_index_link_recall"
    )
    assert expected_failure in failure_metrics


def test_disabled_production_reconciliation_fails_the_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_reconciliation(manifest, *, concepts_dir, current_raw_keys):  # type: ignore[no-untyped-def]
        return ManifestReconciliation(
            manifest=manifest,
            repairs=(),
            concept_membership_changed=False,
        )

    monkeypatch.setattr(compile_command, "reconcile_manifest", no_reconciliation)

    report = run_evaluation()

    assert not report.passed
    assert report.metrics.reconciliation_probe_success_rate < 1.0
    assert report.metrics.reconciliation_probe_results == [
        "concept_hash:unrepaired",
        "ghost_wiki_entry:unrepaired",
        "source_back_reference:unrepaired",
    ]
    assert "reconciliation_probe_success_rate" in {
        failure.metric for failure in report.failures
    }


def test_selectively_disabled_hash_reconciliation_fails_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production_reconcile = compile_command.reconcile_manifest

    def skip_hash_repair(manifest, *, concepts_dir, current_raw_keys):  # type: ignore[no-untyped-def]
        result = production_reconcile(
            manifest,
            concepts_dir=concepts_dir,
            current_raw_keys=current_raw_keys,
        )
        slug = "citation-provenance"
        entry = result.manifest.wiki_articles.get(slug)
        if entry is not None and "reconciliation-probe-ghost" in manifest.wiki_articles:
            result.manifest.wiki_articles[slug] = entry.model_copy(
                update={"content_hash": "reconciliation-probe-incorrect-hash"}
            )
        return result

    monkeypatch.setattr(compile_command, "reconcile_manifest", skip_hash_repair)

    report = run_evaluation()

    assert not report.passed
    assert report.metrics.reconciliation_probe_results == [
        "concept_hash:unrepaired",
        "ghost_wiki_entry:repaired",
        "source_back_reference:repaired",
    ]
    assert "reconciliation_probe_success_rate" in {
        failure.metric for failure in report.failures
    }


def test_unchanged_compile_counts_caught_unmatched_provider_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production_compile = compile_command._compile
    invocation = 0

    def attempt_unmatched(*, full: bool, dry_run: bool, verbose: bool, max_touches: int) -> None:
        nonlocal invocation
        invocation += 1
        if invocation == 3:
            config = compile_command.load_config()
            provider = compile_command.get_provider(config, tier="deep")
            with pytest.raises(UnmatchedReplayPromptError):
                asyncio.run(provider.complete("deliberately unmatched incremental request"))
        production_compile(
            full=full,
            dry_run=dry_run,
            verbose=verbose,
            max_touches=max_touches,
        )

    monkeypatch.setattr(compile_command, "_compile", attempt_unmatched)

    report = run_evaluation()

    assert report.metrics.incremental_provider_call_count == 1
    assert "incremental_provider_call_count" in {
        failure.metric for failure in report.failures
    }


def test_unchanged_compile_detects_same_byte_owned_state_rewrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production_compile = compile_command._compile

    def rewrite_manifest(*, full: bool, dry_run: bool, verbose: bool, max_touches: int) -> None:
        production_compile(
            full=full,
            dry_run=dry_run,
            verbose=verbose,
            max_touches=max_touches,
        )
        manifest_path = compile_command.load_config().vault_path / "vault.manifest.json"
        if manifest_path.exists():
            manifest_path.write_bytes(manifest_path.read_bytes())

    monkeypatch.setattr(compile_command, "_compile", rewrite_manifest)

    report = run_evaluation()

    assert report.metrics.incremental_changed_file_count == 0
    assert report.metrics.incremental_owned_state_write_count > 0
    assert "incremental_owned_state_write_count" in {
        failure.metric for failure in report.failures
    }


def test_cyclic_wrong_source_concept_attribution_fails_quality_bounds() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    source_one = "https://example.test/one"
    source_two = "https://example.test/two"
    scenario = EvaluationScenario.model_validate(
        {
            "name": "wrong-attribution",
            "sources": [
                {
                    "id": "one",
                    "title": "One",
                    "source_url": source_one,
                    "body": "one",
                },
                {
                    "id": "two",
                    "title": "Two",
                    "source_url": source_two,
                    "body": "two",
                },
            ],
            "expected_concepts": [
                {"slug": "alpha", "title": "Alpha", "source_ids": ["one"]},
                {"slug": "beta", "title": "Beta", "source_ids": ["two"]},
            ],
            "expected_edges": [],
            "queries": [],
        }
    )
    pages = [
        ConceptPage("alpha", "Alpha", "Wiki/alpha", [source_two], "# Alpha"),
        ConceptPage("beta", "Beta", "Wiki/beta", [source_one], "# Beta"),
    ]
    manifest = Manifest(
        sources={
            source_one: ManifestSource(
                content_hash="one", saved_at=now, wiki_articles=["beta"]
            ),
            source_two: ManifestSource(
                content_hash="two", saved_at=now, wiki_articles=["alpha"]
            ),
        },
        wiki_articles={
            "alpha": ManifestWikiEntry(
                last_updated=now, source_urls=[source_two], content_hash="alpha"
            ),
            "beta": ManifestWikiEntry(
                last_updated=now, source_urls=[source_one], content_hash="beta"
            ),
        },
    )

    expected_count, recall, unexpected_count, precision = (
        source_concept_attribution_quality(scenario, manifest, pages)
    )

    assert expected_count == 2
    assert recall == 0.0
    assert unexpected_count == 2
    assert precision == 0.0

    metrics = run_evaluation().metrics.model_copy(
        update={
            "expected_source_to_concept_attribution_recall": recall,
            "unexpected_source_to_concept_attribution_count": unexpected_count,
            "source_to_concept_attribution_precision": precision,
        }
    )
    failures = evaluate_thresholds(
        metrics,
        EvaluationThresholds(
            name="attribution-bounds",
            thresholds=[
                EvaluationThreshold(
                    metric="expected_source_to_concept_attribution_recall", minimum=1.0
                ),
                EvaluationThreshold(
                    metric="unexpected_source_to_concept_attribution_count", maximum=0
                ),
                EvaluationThreshold(
                    metric="source_to_concept_attribution_precision", minimum=1.0
                ),
            ],
        ),
    )
    assert [failure.metric for failure in failures] == [
        "expected_source_to_concept_attribution_recall",
        "source_to_concept_attribution_precision",
        "unexpected_source_to_concept_attribution_count",
    ]


def test_threshold_evaluation_aggregates_all_failures_deterministically() -> None:
    metrics = run_evaluation().metrics.model_copy(
        update={"current_compiled_coverage": 0.25, "broken_wikilinks": 3}
    )
    thresholds = EvaluationThresholds(
        name="deliberate-failure",
        thresholds=[
            EvaluationThreshold(metric="current_compiled_coverage", minimum=1.0),
            EvaluationThreshold(metric="broken_wikilinks", maximum=0),
            EvaluationThreshold(metric="concept_count", minimum=1),
        ],
    )

    failures = evaluate_thresholds(metrics, thresholds)

    assert [failure.metric for failure in failures] == [
        "broken_wikilinks",
        "current_compiled_coverage",
    ]


def test_check_fails_for_stale_index_lint_finding(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    shutil.copytree(DEFAULT_FIXTURE_DIR, fixture)
    replay_path = fixture / "replay.json"
    replay_data = json.loads(replay_path.read_text(encoding="utf-8"))
    index_rule = next(rule for rule in replay_data["rules"] if rule["id"] == "index-rebuild")
    index_rule["responses"] = [
        response + "\n- [[deleted-concept]] — stale.\n"
        for response in index_rule["responses"]
    ]
    replay_path.write_text(json.dumps(replay_data), encoding="utf-8")
    output = tmp_path / "stale-index-report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_fixture.py",
            "--fixture-dir",
            str(fixture),
            "--output",
            str(output),
            "--check",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["metrics"]["lint_finding_count"] == 1
    assert report["metrics"]["unexpected_index_link_count"] == 1
    failure_metrics = {failure["metric"] for failure in report["failures"]}
    assert "lint_finding_count" in failure_metrics
    assert "unexpected_index_link_count" in failure_metrics


def test_cli_writes_safe_report_when_evaluation_raises(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    shutil.copytree(DEFAULT_FIXTURE_DIR, fixture)
    replay_path = fixture / "replay.json"
    replay_path.write_text('{"rules": []}', encoding="utf-8")
    output = tmp_path / "error-report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_fixture.py",
            "--fixture-dir",
            str(fixture),
            "--output",
            str(output),
            "--check",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report == {
        "schema_version": 1,
        "passed": False,
        "error_type": "EvaluationRunError",
        "error_fingerprint": hashlib.sha256(b"EvaluationRunError").hexdigest()[:24],
    }
    serialized = output.read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    assert "prompt" not in serialized.lower()


def test_check_writes_report_before_nonzero_exit(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    shutil.copytree(DEFAULT_FIXTURE_DIR, fixture)
    threshold_path = fixture / "thresholds.json"
    threshold_data = json.loads(threshold_path.read_text(encoding="utf-8"))
    threshold_data["thresholds"].append(
        {"metric": "expected_graph_edge_recall", "minimum": 2.0}
    )
    threshold_path.write_text(json.dumps(threshold_data), encoding="utf-8")
    output = tmp_path / "failed-report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_fixture.py",
            "--fixture-dir",
            str(fixture),
            "--output",
            str(output),
            "--check",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert not report["passed"]
    assert [failure["metric"] for failure in report["failures"]] == [
        "expected_graph_edge_recall"
    ]
