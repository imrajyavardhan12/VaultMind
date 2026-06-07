"""Typed data models for the VaultMind compile pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

# ---- Manifest models (for vm compile) ----


class ManifestSource(BaseModel):
    """A source note tracked in the manifest."""

    content_hash: str
    saved_at: datetime
    compiled_at: datetime | None = None
    wiki_articles: list[str] = Field(default_factory=list)


class ManifestWikiEntry(BaseModel):
    """A wiki article tracked in the manifest."""

    last_updated: datetime
    source_urls: list[str] = Field(default_factory=list)
    content_hash: str = ""


class Manifest(BaseModel):
    """Source of truth for the compile loop — tracks what has been compiled."""

    version: int = 1
    last_compiled: datetime | None = None
    sources: dict[str, ManifestSource] = Field(default_factory=dict)
    wiki_articles: dict[str, ManifestWikiEntry] = Field(default_factory=dict)


# ---- Concept triage models ----


class ConceptStatus(StrEnum):
    NEW = "new"
    EXISTING = "existing"
    MERGE = "merge"


class WikiConceptEntry(BaseModel):
    """A concept extracted during concept triage."""

    name: str
    status: ConceptStatus
    description: str = ""
    source_urls: list[str] = Field(default_factory=list)
    merge_target: str | None = None
