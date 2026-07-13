"""Isolated end-to-end runner for the checked-in offline evaluation fixture."""

from __future__ import annotations

import asyncio
import builtins
import hashlib
import json
import os
import shutil
import socket
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import typer
import yaml

import vaultmind.commands.compile as compile_command
from vaultmind.ai.asker import ask_question
from vaultmind.commands.lint import _scan_all_targets, _scan_concept_pages, _scan_wiki_pages
from vaultmind.config import AppConfig, EnvSettings, FolderConfig
from vaultmind.core.linter import LintReport, lint_vault
from vaultmind.core.manifest import read_manifest, write_manifest
from vaultmind.core.raw_scanner import scan_raw_sources
from vaultmind.evaluation.metrics import calculate_metrics
from vaultmind.evaluation.models import (
    EvaluationReport,
    EvaluationScenario,
    EvaluationThresholds,
    ThresholdFailure,
)
from vaultmind.evaluation.replay import ReplayProvider
from vaultmind.schemas import Manifest
from vaultmind.utils.hashing import content_hash

DEFAULT_FIXTURE_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "evaluation"
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "vault" / "VAULTMIND.md"


class EvaluationRunError(Exception):
    """The fixture could not complete through the real production paths."""


class OfflineNetworkError(EvaluationRunError):
    """Code attempted network I/O during an offline evaluation."""


@dataclass(frozen=True, slots=True)
class _ReconciliationProbe:
    """Deterministic manifest-only drift injected between production phases."""

    concept_slug: str
    source_key: str
    ghost_slug: str = "reconciliation-probe-ghost"


def load_scenario(path: Path) -> EvaluationScenario:
    """Load and validate an evaluation corpus fixture."""
    return EvaluationScenario.model_validate(json.loads(path.read_text(encoding="utf-8")))


def load_thresholds(path: Path) -> EvaluationThresholds:
    """Load and validate quality bounds."""
    return EvaluationThresholds.model_validate(json.loads(path.read_text(encoding="utf-8")))


def evaluate_thresholds(
    metrics: object, thresholds: EvaluationThresholds
) -> list[ThresholdFailure]:
    """Evaluate every threshold and return all failures sorted by metric name."""
    failures: list[ThresholdFailure] = []
    for threshold in sorted(thresholds.thresholds, key=lambda item: item.metric):
        raw_actual = getattr(metrics, threshold.metric, None)
        if isinstance(raw_actual, bool) or not isinstance(raw_actual, (int, float)):
            raise ValueError(f"threshold metric is not numeric: {threshold.metric}")
        actual = float(raw_actual)
        below = threshold.minimum is not None and actual < threshold.minimum
        above = threshold.maximum is not None and actual > threshold.maximum
        if below or above:
            failures.append(
                ThresholdFailure(
                    metric=threshold.metric,
                    actual=actual,
                    minimum=threshold.minimum,
                    maximum=threshold.maximum,
                )
            )
    return failures


def render_report_json(report: EvaluationReport) -> str:
    """Render canonical report JSON with fixed formatting and a final newline."""
    return report.model_dump_json(indent=2) + "\n"


@contextmanager
def _network_blocked() -> Iterator[None]:
    def reject(*args: object, **kwargs: object) -> None:
        raise OfflineNetworkError("network access is disabled during offline evaluation")

    with (
        patch.object(socket.socket, "connect", reject),
        patch.object(socket.socket, "connect_ex", reject),
        patch("socket.create_connection", reject),
        patch("socket.getaddrinfo", reject),
    ):
        yield


def _materialize_sources(
    vault_path: Path, scenario: EvaluationScenario, *, phase: int
) -> None:
    """Add one declared corpus phase without rewriting earlier Raw files."""
    raw_dir = vault_path / FolderConfig().raw
    raw_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(scenario.sources, start=1):
        if source.phase != phase:
            continue
        frontmatter = yaml.safe_dump(
            {"source": source.source_url, "tags": source.tags},
            allow_unicode=True,
            sort_keys=False,
        ).rstrip()
        body = source.body.rstrip()
        content = f"---\n{frontmatter}\n---\n\n# {source.title}\n\n{body}\n"
        (raw_dir / f"{index:02d}-{source.id}.md").write_text(content, encoding="utf-8")


def _materialize_vault(
    vault_path: Path, scenario: EvaluationScenario, schema_path: Path
) -> AppConfig:
    """Create the first corpus phase in a fresh temporary vault."""
    folders = FolderConfig()
    _materialize_sources(vault_path, scenario, phase=1)
    shutil.copyfile(schema_path, vault_path / "VAULTMIND.md")
    return AppConfig(
        vault_path=vault_path,
        folders=folders,
        env=EnvSettings(
            anthropic_api_key="",
            openai_api_key="",
            ollama_base_url="",
        ),
    )


def _discard_command_output(*args: object, **kwargs: object) -> None:
    """Keep production command display side effects out of deterministic evaluation output."""


def _run_production_compile(config: AppConfig, provider: ReplayProvider) -> None:
    """Invoke the production compile command with only external dependencies replaced."""
    try:
        with (
            patch.object(compile_command, "load_config", return_value=config),
            patch.object(compile_command, "get_provider", return_value=provider),
            patch.object(compile_command, "setup_logging", _discard_command_output),
            patch.object(compile_command, "print_error", _discard_command_output),
            patch.object(compile_command, "print_info", _discard_command_output),
            patch.object(compile_command, "print_success", _discard_command_output),
            patch.object(compile_command, "print_warning", _discard_command_output),
        ):
            compile_command._compile(
                full=False,
                dry_run=False,
                verbose=False,
                max_touches=5,
            )
    except typer.Exit as exc:
        raise EvaluationRunError(
            f"production compile exited with status {exc.exit_code}"
        ) from exc


async def _run_queries(
    config: AppConfig,
    scenario: EvaluationScenario,
    provider: ReplayProvider,
) -> dict[str, str]:
    pages: dict[str, str] = {}
    for expectation in scenario.queries:
        result = await ask_question(
            expectation.question,
            provider,
            config.vault_path,
            config.folders.wiki,
            config.folders.wiki_concepts,
            config.folders.wiki_queries,
            config.folders.raw,
            depth="shallow",
            file_answer=True,
        )
        pages[expectation.question] = result.path.read_text(encoding="utf-8")
    return pages


def _inject_reconciliation_probe(
    config: AppConfig,
    scenario: EvaluationScenario,
) -> _ReconciliationProbe:
    """Persist safe manifest drift whose canonical values already exist on disk."""
    phase_one_ids = {source.id for source in scenario.sources if source.phase == 1}
    candidates = sorted(
        [
            concept
            for concept in scenario.expected_concepts
            if phase_one_ids.intersection(concept.source_ids)
        ],
        key=lambda concept: concept.slug,
    )
    if not candidates:
        raise EvaluationRunError("reconciliation probe requires a phase-one concept")

    concept = candidates[0]
    sources_by_id = {source.id: source for source in scenario.sources}
    source_id = min(source_id for source_id in concept.source_ids if source_id in phase_one_ids)
    source_key = sources_by_id[source_id].source_url
    probe = _ReconciliationProbe(concept_slug=concept.slug, source_key=source_key)

    manifest = read_manifest(config.vault_path)
    article_entry = manifest.wiki_articles.get(probe.concept_slug)
    source_entry = manifest.sources.get(probe.source_key)
    if article_entry is None or source_entry is None:
        raise EvaluationRunError("phase one did not create reconciliation probe targets")

    manifest.wiki_articles[probe.concept_slug] = article_entry.model_copy(
        update={"content_hash": "reconciliation-probe-incorrect-hash"}
    )
    manifest.wiki_articles[probe.ghost_slug] = article_entry.model_copy(
        update={"content_hash": "reconciliation-probe-ghost-hash"}
    )
    manifest.sources[probe.source_key] = source_entry.model_copy(
        update={"wiki_articles": [probe.ghost_slug]}
    )
    write_manifest(config.vault_path, manifest)
    return probe


def _verify_reconciliation_probe(
    config: AppConfig,
    probe: _ReconciliationProbe,
    manifest: Manifest | None = None,
) -> list[str]:
    """Report each probe repair against canonical concept files on disk."""
    manifest = manifest if manifest is not None else read_manifest(config.vault_path)
    article_entry = manifest.wiki_articles.get(probe.concept_slug)
    article_path = _concept_path(config, probe.concept_slug)
    hash_repaired = (
        article_entry is not None
        and article_path.is_file()
        and article_entry.content_hash
        == content_hash(article_path.read_text(encoding="utf-8"))
    )

    canonical_backlinks = sorted(
        page.slug
        for page in _scan_concept_pages(config)
        if probe.source_key in page.sources
    )
    source_entry = manifest.sources.get(probe.source_key)
    backlink_repaired = (
        source_entry is not None
        and source_entry.wiki_articles == canonical_backlinks
    )
    results = [
        ("concept_hash", hash_repaired),
        ("ghost_wiki_entry", probe.ghost_slug not in manifest.wiki_articles),
        ("source_back_reference", backlink_repaired),
    ]
    return [f"{name}:{'repaired' if repaired else 'unrepaired'}" for name, repaired in results]


@contextmanager
def _observe_reconciliation_probe(
    config: AppConfig,
    probe: _ReconciliationProbe,
) -> Iterator[list[list[str]]]:
    """Capture probe state immediately after each production reconciliation call."""
    production_reconcile = compile_command.reconcile_manifest
    observations: list[list[str]] = []

    def observed_reconcile(
        manifest: Any,
        *,
        concepts_dir: Path,
        current_raw_keys: set[str],
    ) -> Any:
        result = production_reconcile(
            manifest,
            concepts_dir=concepts_dir,
            current_raw_keys=current_raw_keys,
        )
        observations.append(_verify_reconciliation_probe(config, probe, result.manifest))
        return result

    with patch.object(compile_command, "reconcile_manifest", observed_reconcile):
        yield observations


def _concept_path(config: AppConfig, slug: str) -> Path:
    return (
        config.vault_path
        / config.folders.wiki
        / config.folders.wiki_concepts
        / f"{slug}.md"
    )


def _run_lint(config: AppConfig) -> LintReport:
    """Exercise canonical deterministic lint logic over production scanners."""
    raw_sources = scan_raw_sources(config)
    manifest = read_manifest(config.vault_path)
    concept_pages = _scan_concept_pages(config)
    wiki_pages = _scan_wiki_pages(config)
    index_path = config.vault_path / config.folders.wiki / f"{config.folders.wiki_index}.md"
    index_text = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
    concept_slugs = {page.slug for page in concept_pages}
    return lint_vault(
        raw_sources=raw_sources,
        manifest=manifest,
        concept_pages=concept_pages,
        wiki_pages=wiki_pages,
        index_text=index_text,
        valid_targets=_scan_all_targets(config),
        concept_slugs=concept_slugs,
    )


def _owned_state_hashes(config: AppConfig) -> dict[str, str]:
    """Hash every vault-owned state file, excluding human Raw/schema and external logs."""
    owned: list[Path] = []
    manifest_path = config.vault_path / "vault.manifest.json"
    if manifest_path.is_file():
        owned.append(manifest_path)
    wiki_path = config.vault_path / config.folders.wiki
    if wiki_path.exists():
        owned.extend(path for path in wiki_path.rglob("*") if path.is_file())
    return {
        path.relative_to(config.vault_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(owned, key=lambda item: item.relative_to(config.vault_path).as_posix())
    }


def _changed_hash_count(before: dict[str, str], after: dict[str, str]) -> int:
    return sum(before.get(path) != after.get(path) for path in sorted(set(before) | set(after)))


def _is_owned_state_path(config: AppConfig, value: object) -> bool:
    if isinstance(value, int):
        return False
    try:
        path = Path(value)  # type: ignore[arg-type]
        relative = path.resolve().relative_to(config.vault_path.resolve())
    except (OSError, TypeError, ValueError):
        return False
    if not relative.parts:
        return False
    return relative.parts[0] == config.folders.wiki or relative.name.startswith(
        "vault.manifest.json"
    )


@contextmanager
def _observe_owned_state_writes(config: AppConfig) -> Iterator[list[str]]:
    """Record attempted writes, temporary creates, and replacements in owned state."""
    writes: list[str] = []
    nested_path_write = 0
    original_path_open = Path.open
    original_write_text = Path.write_text
    original_write_bytes = Path.write_bytes
    original_builtin_open = builtins.open
    original_os_open = os.open
    original_os_replace = os.replace

    def record(operation: str, value: object) -> None:
        if _is_owned_state_path(config, value):
            writes.append(f"{operation}:{Path(value).name}")  # type: ignore[arg-type]

    def observed_path_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        mode = args[0] if args else kwargs.get("mode", "r")
        if nested_path_write == 0 and any(flag in str(mode) for flag in "wax+"):
            record("open", path)
        return original_path_open(path, *args, **kwargs)

    def observed_write_text(path: Path, *args: Any, **kwargs: Any) -> int:
        nonlocal nested_path_write
        record("write_text", path)
        nested_path_write += 1
        try:
            return original_write_text(path, *args, **kwargs)
        finally:
            nested_path_write -= 1

    def observed_write_bytes(path: Path, *args: Any, **kwargs: Any) -> int:
        nonlocal nested_path_write
        record("write_bytes", path)
        nested_path_write += 1
        try:
            return original_write_bytes(path, *args, **kwargs)
        finally:
            nested_path_write -= 1

    def observed_builtin_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if any(flag in mode for flag in "wax+"):
            record("open", file)
        return original_builtin_open(file, mode, *args, **kwargs)

    def observed_os_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            record("os.open", path)
        return original_os_open(path, flags, *args, **kwargs)

    def observed_os_replace(source: Any, destination: Any, *args: Any, **kwargs: Any) -> None:
        record("replace", destination)
        original_os_replace(source, destination, *args, **kwargs)

    with (
        patch.object(Path, "open", observed_path_open),
        patch.object(Path, "write_text", observed_write_text),
        patch.object(Path, "write_bytes", observed_write_bytes),
        patch.object(builtins, "open", observed_builtin_open),
        patch.object(os, "open", observed_os_open),
        patch.object(os, "replace", observed_os_replace),
    ):
        yield writes


def _run_unchanged_incremental(
    config: AppConfig, provider: ReplayProvider
) -> tuple[int, int, int]:
    """Run unchanged production Compile and observe calls, writes, and final content."""
    before = _owned_state_hashes(config)
    calls_before = provider.call_count
    with _observe_owned_state_writes(config) as writes:
        _run_production_compile(config, provider)
    after = _owned_state_hashes(config)
    return (
        _changed_hash_count(before, after),
        provider.call_count - calls_before,
        len(writes),
    )


def run_evaluation(
    fixture_dir: Path = DEFAULT_FIXTURE_DIR,
    *,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> EvaluationReport:
    """Run the checked-in scenario in a temporary, network-disabled vault."""
    scenario = load_scenario(fixture_dir / "corpus.json")
    thresholds = load_thresholds(fixture_dir / "thresholds.json")
    provider = ReplayProvider.from_path(fixture_dir / "replay.json")

    with tempfile.TemporaryDirectory(prefix="vaultmind-evaluation-") as directory, _network_blocked():
        config = _materialize_vault(Path(directory), scenario, schema_path)
        calls_before = provider.call_count
        _run_production_compile(config, provider)
        if provider.call_count == calls_before:
            raise EvaluationRunError("first corpus phase produced no provider calls")

        probe = _inject_reconciliation_probe(config, scenario)
        probe_results: list[str] | None = None
        for phase in range(2, max(source.phase for source in scenario.sources) + 1):
            _materialize_sources(config.vault_path, scenario, phase=phase)
            calls_before = provider.call_count
            if phase == 2:
                with _observe_reconciliation_probe(config, probe) as observations:
                    _run_production_compile(config, provider)
                if not observations:
                    raise EvaluationRunError("production compile did not reconcile manifest state")
                # The first observation is the pre-compile reconciliation. Later
                # article updates must not be allowed to mask a missed repair.
                probe_results = observations[0]
            else:
                _run_production_compile(config, provider)
            if provider.call_count == calls_before:
                raise EvaluationRunError(f"corpus phase {phase} produced no provider calls")
        if probe_results is None:
            raise EvaluationRunError("reconciliation probe requires production compile phase 2")

        query_pages = asyncio.run(_run_queries(config, scenario, provider))
        lint_report = _run_lint(config)
        incremental_changed, incremental_calls, incremental_writes = (
            _run_unchanged_incremental(config, provider)
        )

        final_raw = scan_raw_sources(config)
        final_manifest = read_manifest(config.vault_path)
        concept_pages = _scan_concept_pages(config)
        index_path = (
            config.vault_path
            / config.folders.wiki
            / f"{config.folders.wiki_index}.md"
        )
        index_markdown = (
            index_path.read_text(encoding="utf-8") if index_path.is_file() else None
        )
        metrics = calculate_metrics(
            scenario=scenario,
            raw_sources=final_raw,
            manifest=final_manifest,
            concept_pages=concept_pages,
            lint_report=lint_report,
            index_markdown=index_markdown,
            query_pages=query_pages,
            wiki_concepts_folder=config.folders.wiki_concepts,
            wiki_queries_folder=config.folders.wiki_queries,
            incremental_changed_file_count=incremental_changed,
            incremental_owned_state_write_count=incremental_writes,
            incremental_provider_call_count=incremental_calls,
            reconciliation_probe_results=probe_results,
            provider_call_count=provider.call_count,
        )

    failures = evaluate_thresholds(metrics, thresholds)
    return EvaluationReport(
        scenario=scenario.name,
        threshold_set=thresholds.name,
        metrics=metrics,
        failures=failures,
        passed=not failures,
    )
