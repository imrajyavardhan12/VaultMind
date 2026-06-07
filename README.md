# VaultMind

> Clip anything into Obsidian. Run `vm compile`. Ask your living wiki.

VaultMind is a local-first CLI for building a personal LLM-maintained wiki inside an Obsidian vault.

It is inspired by Andrej Karpathy's LLM Wiki pattern:

https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f

## Product Thesis

VaultMind is not a web clipper, bookmark manager, or generic RAG chatbot.

Obsidian Web Clipper is already good at capturing web pages, images, assets, and source material. VaultMind should not compete with that. VaultMind is the intelligence layer that reads your clipped source material and maintains a durable wiki from it.

The core idea:

> Knowledge should be compiled once, maintained continuously, and reused repeatedly.

Most AI document tools retrieve chunks from raw files every time you ask a question. VaultMind instead builds a persistent markdown wiki. Every source added, every answer filed, and every maintenance pass should make the next interaction smarter.

## What VaultMind Does

VaultMind has one primary workflow:

```text
Obsidian Web Clipper -> 📥 Raw/ -> vm compile -> 🗺️ Wiki/
```

You save original source documents into Obsidian. VaultMind reads those sources, identifies concepts, creates and updates wiki pages, maintains an index, records what changed, and lets you ask questions against the compiled knowledge base.

In simple terms:

- Obsidian captures the material.
- VaultMind organizes and synthesizes it.
- The wiki compounds over time.

## The Three Layers

### 1. Raw Sources

Raw sources are original documents saved as markdown.

Examples:

- clipped articles,
- papers converted to markdown,
- transcripts,
- meeting notes,
- manually pasted source documents.

Rules:

- Raw sources are the ground truth.
- VaultMind reads them.
- VaultMind never rewrites them.
- The user or Obsidian Web Clipper owns them.

Default folder:

```text
{vault}/📥 Raw/
```

### 2. Wiki

The wiki is the LLM-authored layer.

It contains concept pages, query answers, weekly summaries, lint reports, an index, and a log. VaultMind owns this layer. The user reviews it.

Default folder:

```text
{vault}/🗺️ Wiki/
```

### 3. Schema

The schema is the contract that tells the LLM how to maintain the wiki.

It should live in the vault root as:

```text
VAULTMIND.md
```

It defines directory ownership, page formats, citation rules, wikilink style, review policy, and what VaultMind may edit.

## Core Workflows

### Ingest

Ingest means adding source material to `📥 Raw/`.

Primary path:

```text
Obsidian Web Clipper -> 📥 Raw/
```

VaultMind does not need to own capture. It can keep helper commands, but the product center is source markdown already in the vault.

### Compile

Compile is the main product loop.

```bash
vm compile
```

It should:

1. Scan `📥 Raw/`.
2. Detect new or changed source files.
3. Read the current wiki index and known concept pages.
4. Ask the LLM which concepts should be created or updated.
5. Create or update pages in `🗺️ Wiki/🧠 Concepts/`.
6. Update `🗺️ Wiki/📇 Index.md`.
7. Append to `🗺️ Wiki/📋 Log.md`.
8. Update `vault.manifest.json`.

Compile should be conservative. Updating an existing concept page is usually better than creating a duplicate page.

### Ask

Ask is the second compounding loop.

```bash
vm ask "What is the difference between RLHF and DPO?"
```

It should:

1. Search the compiled wiki first.
2. Search Raw only when the wiki is insufficient.
3. Produce a grounded answer.
4. In normal mode, file the answer to `🗺️ Wiki/📊 Queries/`.
5. In preview mode, print without writing.

### Lint

Lint is the health-maintenance loop.

```bash
vm lint
```

It should write a reviewable report to:

```text
🗺️ Wiki/📋 Inbox/lint-YYYY-MM-DD.md
```

Checks should include orphan raw sources, concept duplicates, broken wikilinks, stale index entries, wiki pages with no sources, and raw material that has not been compiled.

## Vault Layout

Canonical layout:

```text
{vault}/
├── 📥 Raw/
│   └── assets/
├── 🗺️ Wiki/
│   ├── 🧠 Concepts/
│   ├── 📊 Queries/
│   ├── 📋 Inbox/
│   ├── 📅 Weekly/
│   ├── 📇 Index.md
│   └── 📋 Log.md
├── VAULTMIND.md
└── vault.manifest.json
```

Ownership:

- `📥 Raw/`: human or Obsidian Web Clipper owned; VaultMind read-only.
- `🗺️ Wiki/`: VaultMind owned; user reviews.
- `VAULTMIND.md`: human-owned schema, optionally scaffolded by VaultMind.
- `vault.manifest.json`: VaultMind owned.

## Page Contracts

Concept pages live in:

```text
🗺️ Wiki/🧠 Concepts/{slug}.md
```

Recommended shape:

```markdown
---
title: "Human Title"
vaultmind: true
kind: concept
sources:
  - https://example.com/source
---

# Human Title

## Overview

## Key Ideas

## Connections

## Open Questions

## Sources
```

Query pages live in:

```text
🗺️ Wiki/📊 Queries/{question-slug}.md
```

Recommended shape:

```markdown
---
title: "Question?"
vaultmind: true
kind: query
created: 2026-05-16T00:00:00+00:00
---

# Question?

## Answer

## Supporting Wiki Pages

## Supporting Raw Sources

## Follow-up Questions
```

## CLI

The complete command surface:

```bash
vm init        # scaffold the vault (📥 Raw + 🗺️ Wiki) and write config
vm compile     # compile new/changed Raw sources into the Wiki
vm ask "..."   # answer from the Wiki, falling back to Raw when needed
vm lint        # deterministic wiki-health report → 🗺️ Wiki/📋 Inbox/
vm version     # print the version
```

No-write/preview modes keep every command safe to dry-run:

```bash
vm compile --dry-run   # show what would compile, write nothing
vm ask "..." --preview # print the answer without filing it
vm lint --preview      # print the health report without writing it
vm lint --strict       # exit non-zero if any error-severity findings
```

See `vm <command> --help` for all flags.

## Installation

```bash
pipx install vaultmind
```

Or from this repository:

```bash
uv sync
uv run vm init
```

## Configuration

`vm init` creates:

```text
~/.config/vaultmind/config.yaml
~/.config/vaultmind/.env
```

The config stores vault paths, folder names, and AI provider preferences. The `.env` stores API keys.

## Architecture Principles

- Prefer markdown files over hidden state.
- Prefer a readable JSON manifest over a database.
- Keep Raw immutable.
- Keep Wiki reviewable.
- Make preview modes truly no-write.
- Make changes inspectable with git diffs.
- Do not add vector databases, LangChain, SQLite, or background daemons until the core loop proves it needs them.

## Current Engineering Priorities

1. Make `vm compile` robust: existing concept awareness, multi-concept manifest mappings, index rebuild, log writes.
2. Make `vm ask` robust: wiki-first search, Raw fallback, true preview mode, filed query metadata.
3. `vm lint` shipped: deterministic wiki-health report (orphan/uncompiled raw, sourceless pages, broken wikilinks, stale index, duplicate concepts). Next: surface its findings into `compile`/`ask`, and consider opt-in autofix.
4. Honor the page contracts end-to-end: concept pages should carry Connections + Open Questions; query pages should fill Follow-up Questions from `vm ask`'s self-assessed gaps (currently computed but discarded).

## Success Criteria

VaultMind is working when:

- adding 20 raw sources produces a coherent wiki instead of 20 isolated summaries,
- concept pages improve rather than duplicate as more sources arrive,
- `vm ask` mostly answers from the wiki and only uses Raw when needed,
- useful answers become durable query pages,
- the index and log make the system navigable,
- lint catches wiki decay before the user loses trust,
- the vault feels smarter, not merely larger.
