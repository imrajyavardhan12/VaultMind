# VaultMind Schema

This file defines the ownership contract for VaultMind's AI-maintained wiki layer.

## Directory Ownership

- `📥 Raw/` — Human or Obsidian Web Clipper owned. VaultMind reads only.
- `🗺️ Wiki/` — VaultMind owned. User reviews. The AI-maintained knowledge base.
- `VAULTMIND.md` — Human-owned schema. Optional scaffold by VaultMind.
- `vault.manifest.json` — VaultMind owned. Machine-readable state.

## Page Contracts

### Concept Pages

Concept pages live in `🗺️ Wiki/🧠 Concepts/{slug}.md`.

Required frontmatter:
```yaml
---
title: "Human Title"
vaultmind: true
kind: concept
sources:
  - https://example.com/source
---
```

Structure:
- `# Title` (H1, same as frontmatter title)
- `## Overview`
- `## Key Ideas`
- `## Connections`
- `## Open Questions`
- `## Sources`

### Query Pages

Query pages live in `🗺️ Wiki/📊 Queries/{question-slug}.md`.

Required frontmatter:
```yaml
---
title: "Question?"
vaultmind: true
kind: query
created: 2026-05-16T00:00:00+00:00
---
```

## Wikilink Style

Use Obsidian `[[slug|display text]]` syntax for all internal links.

## Citation Rules

Source URLs are listed in the `sources` frontmatter field and in the Sources section. Prefer the original canonical URL.

## Review Policy

- Wiki pages are VaultMind owned — user reviews, never overwrites directly.
- Raw sources are immutable ground truth — VaultMind never modifies them.
- Run `vm compile` after adding new raw sources.
- Run `vm lint` periodically to catch wiki decay.

## What VaultMind May Edit

VaultMind may create and update pages in:
- `🗺️ Wiki/🧠 Concepts/` — concept articles
- `🗺️ Wiki/📊 Queries/` — answered queries
- `🗺️ Wiki/📇 Index.md` — wiki master index
- `🗺️ Wiki/📋 Log.md` — compile activity log
- `🗺️ Wiki/📋 Inbox/lint-YYYY-MM-DD.md` — lint reports
- `vault.manifest.json` — compile state tracking

VaultMind must never modify content in `📥 Raw/` or other human-owned directories.
