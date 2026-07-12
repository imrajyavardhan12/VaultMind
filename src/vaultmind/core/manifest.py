"""Vault manifest persistence and deterministic disk reconciliation."""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import ValidationError

from vaultmind.schemas import Manifest, ManifestSource, ManifestWikiEntry
from vaultmind.utils.hashing import content_hash

MANIFEST_FILENAME = "vault.manifest.json"


class ManifestReadError(Exception):
    """The existing manifest cannot be read safely."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Cannot read manifest {path}: {reason}")


@dataclass(frozen=True, slots=True)
class ManifestReconciliation:
    """A reconciled manifest and a deterministic description of its repairs."""

    manifest: Manifest
    repairs: tuple[str, ...]
    concept_membership_changed: bool

    @property
    def changed(self) -> bool:
        return bool(self.repairs)


@dataclass(frozen=True, slots=True)
class _DiskConcept:
    slug: str
    content_hash: str
    source_urls: tuple[str, ...]
    modified_at: datetime


def manifest_path(vault_path: Path) -> Path:
    return vault_path / MANIFEST_FILENAME


def read_manifest(vault_path: Path) -> Manifest:
    """Load a manifest, returning an empty v1 manifest only when it is absent."""
    path = manifest_path(vault_path)
    try:
        path.lstat()
    except FileNotFoundError:
        return Manifest()
    except OSError as exc:
        raise ManifestReadError(path, str(exc)) from exc

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("manifest must be a JSON object")
        raw_version = data.get("version")
        if type(raw_version) is not int or raw_version != 1:
            raise ValueError(
                f"invalid manifest version {raw_version!r}; expected integer 1"
            )
        return Manifest.model_validate(data)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ManifestReadError(path, str(exc)) from exc


def write_manifest(vault_path: Path, manifest: Manifest) -> None:
    """Write the manifest to disk atomically."""
    path = manifest_path(vault_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def reconcile_manifest(
    manifest: Manifest,
    *,
    concepts_dir: Path,
    current_raw_keys: Collection[str],
) -> ManifestReconciliation:
    """Reconcile persisted state against canonical concept pages and Raw inventory.

    Concept membership, hashes, citations, and source back-references come from
    concept files on disk. Current Raw files are used only to decide whether an
    uncited historical source is stale; reconciliation never marks Raw as compiled.
    """
    reconciled = manifest.model_copy(deep=True)
    disk_concepts = _scan_disk_concepts(concepts_dir)
    old_slugs = set(reconciled.wiki_articles)
    new_slugs = set(disk_concepts)
    repairs: list[str] = []

    for slug in sorted(old_slugs - new_slugs):
        repairs.append(f"removed missing concept entry: {slug}")

    wiki_articles: dict[str, ManifestWikiEntry] = {}
    for slug in sorted(disk_concepts):
        concept = disk_concepts[slug]
        existing = reconciled.wiki_articles.get(slug)
        source_urls = list(concept.source_urls)
        if existing is None:
            repairs.append(f"added concept entry from disk: {slug}")
            last_updated = concept.modified_at
        else:
            last_updated = existing.last_updated
            if existing.content_hash != concept.content_hash:
                repairs.append(f"repaired concept hash: {slug}")
            if existing.source_urls != source_urls:
                repairs.append(f"repaired concept sources: {slug}")
        wiki_articles[slug] = ManifestWikiEntry(
            last_updated=last_updated,
            source_urls=source_urls,
            content_hash=concept.content_hash,
        )
    reconciled.wiki_articles = wiki_articles

    backlinks: dict[str, list[str]] = {}
    for slug, concept in disk_concepts.items():
        for source_key in concept.source_urls:
            backlinks.setdefault(source_key, []).append(slug)

    raw_keys = set(current_raw_keys)
    sources: dict[str, ManifestSource] = {}
    for source_key in sorted(reconciled.sources):
        entry = reconciled.sources[source_key]
        canonical_backlinks = sorted(backlinks.get(source_key, []))
        if source_key not in raw_keys and not canonical_backlinks:
            repairs.append(f"removed missing uncited Raw source: {source_key}")
            continue
        if entry.wiki_articles != canonical_backlinks:
            repairs.append(f"repaired source back-references: {source_key}")
            entry = entry.model_copy(update={"wiki_articles": canonical_backlinks})
        sources[source_key] = entry
    reconciled.sources = sources

    return ManifestReconciliation(
        manifest=reconciled,
        repairs=tuple(repairs),
        concept_membership_changed=old_slugs != new_slugs,
    )


def _scan_disk_concepts(concepts_dir: Path) -> dict[str, _DiskConcept]:
    if not concepts_dir.exists():
        return {}

    concepts: dict[str, _DiskConcept] = {}
    for path in sorted(concepts_dir.glob("*.md"), key=lambda item: item.name):
        text = path.read_text(encoding="utf-8")
        concepts[path.stem] = _DiskConcept(
            slug=path.stem,
            content_hash=content_hash(text),
            source_urls=tuple(_extract_sources(text)),
            modified_at=datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
        )
    return concepts


def _extract_sources(markdown: str) -> list[str]:
    if not markdown.startswith("---"):
        return []
    end = markdown.find("---", 3)
    if end == -1:
        return []
    try:
        data = yaml.safe_load(markdown[3:end])
    except yaml.YAMLError:
        return []
    if not isinstance(data, dict):
        return []
    value = data.get("sources")
    candidates = value if isinstance(value, list) else [value] if isinstance(value, str) else []
    sources: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        normalized = candidate.strip()
        if normalized and normalized not in sources:
            sources.append(normalized)
    return sources


def upsert_source(
    manifest: Manifest,
    *,
    url: str,
    content_hash: str,
    saved_at: datetime,
    wiki_articles: list[str] | None = None,
) -> None:
    """Add or update a source entry in the manifest."""
    now = datetime.now(UTC)
    existing = manifest.sources.get(url)

    if existing is not None:
        merged_articles = existing.wiki_articles
        if wiki_articles is not None:
            merged_articles = list(dict.fromkeys([*existing.wiki_articles, *wiki_articles]))
        manifest.sources[url] = ManifestSource(
            content_hash=content_hash,
            saved_at=existing.saved_at,
            compiled_at=now,
            wiki_articles=merged_articles,
        )
    else:
        manifest.sources[url] = ManifestSource(
            content_hash=content_hash,
            saved_at=saved_at,
            compiled_at=now,
            wiki_articles=wiki_articles or [],
        )


def upsert_wiki_article(
    manifest: Manifest,
    *,
    slug: str,
    content_hash: str,
    source_urls: list[str],
) -> None:
    """Add or update a wiki article entry in the manifest."""
    manifest.wiki_articles[slug] = ManifestWikiEntry(
        last_updated=datetime.now(UTC),
        source_urls=source_urls,
        content_hash=content_hash,
    )


def is_source_new_or_changed(manifest: Manifest, url: str, current_hash: str) -> bool:
    """Return True if the source is new or its content hash has changed."""
    existing = manifest.sources.get(url)
    return existing is None or existing.content_hash != current_hash


def get_changed_sources(manifest: Manifest, source_hashes: dict[str, str]) -> list[str]:
    """Return source keys that are new or have changed content hashes."""
    return [
        url
        for url, current_hash in source_hashes.items()
        if is_source_new_or_changed(manifest, url, current_hash)
    ]


def update_compiled_at(manifest: Manifest) -> None:
    """Update the last_compiled timestamp."""
    manifest.last_compiled = datetime.now(UTC)
