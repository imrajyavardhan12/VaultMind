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
5. Create or update pages in `🗺️ Wiki/🧠 Concepts/` from bounded packets of the attributed Raw source text.
6. Update `🗺️ Wiki/📇 Index.md`.
7. Append to `🗺️ Wiki/📋 Log.md`.
8. Update `vault.manifest.json`.

Compile should be conservative. Updating an existing concept page is usually better than creating a duplicate page. Article generation and updates are grounded in bounded Raw source packets, and VaultMind deterministically enforces the concept-page headings and synchronizes source citations before each atomic write.

Before diffing Raw sources, compile reconciles the manifest with concept pages on disk. Concept membership, content hashes, frontmatter citations, and source back-references are repaired from those pages. Missing uncited Raw entries are removed, while historical sources still cited by a concept are preserved. Reconciliation never marks a current Raw file as compiled. Repair-only runs persist the repaired manifest and log the repair; concept membership changes also rebuild the index.

### Ask

Ask is the second compounding loop.

```bash
vm ask "What is the difference between RLHF and DPO?"
```

It:

1. Searches compiled concept pages and previously filed query answers first.
2. Searches Raw only when the combined wiki results are insufficient.
3. In deep mode, performs up to three grounded synthesis passes, searching each self-assessed gap between passes and stopping early when no gaps remain.
4. Files the answer and its final follow-up gaps to `🗺️ Wiki/📊 Queries/` in normal mode, making the answer searchable context for later questions.
5. Prints without creating files, folders, manifest changes, or log entries in preview mode.

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

Required shape:

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

Required shape:

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
vm compile --dry-run          # show compilation and reconciliation; write nothing
vm compile --full             # force all current Raw through compile; preserve Wiki and history
vm compile --max-touches 5    # cap existing pages touched per source (0 disables propagation)
vm ask "..." --preview        # print the answer without filing it
vm lint --preview      # print the health report without writing it
vm lint --strict       # exit non-zero if any error-severity findings
```

After creating or updating concepts, `vm compile` may propagate links into existing
concept pages. Each source can touch at most `--max-touches` pages (default: 5),
adding a constrained `Connections` wikilink and source provenance. Repeated
propagation is idempotent against the page currently on disk. If a propagation
provider call or page write fails, that source's current hash is not recorded;
the command exits non-zero and the next incremental compile retries the source.
Touches that were written before a later failure remain recorded in both manifest
directions. `--dry-run` performs neither propagation calls nor writes. `--full` is
non-destructive: it forces every current Raw source through compilation without
resetting manifest provenance or deleting Wiki pages.

A missing `vault.manifest.json` starts as an empty version-1 manifest. If an
existing manifest is malformed, unreadable, schema-invalid, or has an unsupported
version, compile and lint stop with a non-zero exit before writing anything. The
file is never silently replaced; repair it or restore a recoverable copy first.

See `vm <command> --help` for all flags.

## Installation

The current source and GitHub release version is 0.2.0. Check that PyPI lists it,
then install and verify the isolated CLI:

```bash
python -m pip index versions vaultmind
pipx install vaultmind==0.2.0
vm version  # VaultMind v0.2.0
```

If PyPI does not list 0.2.0 yet, follow the authorized manual publication procedure
rather than installing from an unverified artifact. Or run from this repository:

```bash
uv sync
uv run vm init
uv run vm version
```

Maintainers: see [Releasing VaultMind](docs/RELEASING.md) for Trusted Publisher
setup, protected-environment approval, release verification, and recovery guidance.

## Configuration

`vm init` creates:

```text
~/.config/vaultmind/config.yaml
~/.config/vaultmind/.env
```

The config stores vault paths, folder names, and AI provider preferences. The `.env` stores API keys.

### Runtime provider fallback (v0.2.0)

Every completion follows `ai.fallback_chain` in order. VaultMind constructs all
configured providers that have their required credentials (Ollama requires a base
URL), then tries them at runtime until one returns a non-empty response. A single
configured provider uses this same chain abstraction.

Each provider performs bounded retries only for transient connection, timeout,
rate-limit, and server failures. Authentication, permission, malformed-request,
and other permanent failures advance immediately to the next provider. Empty
responses also advance. Cancellation and terminal interrupts are never swallowed.

If the chain is exhausted, `vm ask` and `vm compile` exit non-zero with a concise,
sanitized list of attempted provider/model pairs. Use `--verbose` for chained
diagnostics. Structured logs record provider/model attempts and selection, but do
not include prompts, API keys, or response bodies.

## Architecture Principles

- Prefer markdown files over hidden state.
- Prefer a readable JSON manifest over a database.
- Keep Raw immutable.
- Keep Wiki reviewable.
- Make preview modes truly no-write.
- Make changes inspectable with git diffs.
- Do not add vector databases, LangChain, SQLite, or background daemons until the core loop proves it needs them.

## Current Engineering Priorities

1. Strengthen `vm compile` conservatism and recovery for larger, evolving vaults.
2. Improve Ask retrieval quality without sacrificing wiki-first behavior, bounded context, or preview safety.
3. Surface deterministic `vm lint` findings in compile and Ask workflows, and evaluate opt-in autofix.
4. Improve citation verification while preserving readable, contract-compliant concept and query pages.

## Measured Quality Gate

The core quality claim is checked by a deterministic, offline 20-source evaluation. It exercises production Compile, manifest reconciliation, index rebuilding, Ask filing and reuse, and Lint with prompt-matched replay responses. CI fails on regressions in current compiled coverage, concept/graph coherence, reciprocal provenance, lint health, query support/reuse, or unchanged incremental stability.

Run it locally with:

```bash
uv run python scripts/evaluate_fixture.py --output evaluation-report.json --check
```

See [Offline Evaluation](docs/EVALUATION.md) for metric definitions, fixture maintenance, threshold governance, and live/offline boundaries.

## Success Criteria

VaultMind is working when:

- adding 20 raw sources produces a coherent wiki instead of 20 isolated summaries,
- concept pages improve rather than duplicate as more sources arrive,
- `vm ask` mostly answers from the wiki and only uses Raw when needed,
- useful answers become durable query pages,
- the index and log make the system navigable,
- lint catches wiki decay before the user loses trust,
- the vault feels smarter, not merely larger.
