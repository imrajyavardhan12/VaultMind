# Changelog

## Unreleased

### Added

- First-class OpenRouter runtime and setup support, including independent credentials, configurable OpenAI-compatible endpoints, model slugs, and ordered fallback diagnostics.

## 0.2.0 - 2026-07-12

### Added

- Grounded compilation from bounded, source-attributed Raw packets, with deterministic concept-page contracts, synchronized citations, and atomic page writes.
- Cross-page propagation after compilation: new knowledge can add constrained, provenance-preserving Connections to existing concept pages, with an adjustable `--max-touches` limit.
- Compounding Ask retrieval across both concept pages and previously filed query answers, with Raw fallback only when wiki context is insufficient.
- Deep Ask mode with up to three grounded synthesis passes, gap-directed retrieval between passes, early stopping, and final follow-up gaps saved with the answer.
- Manifest reconciliation against concept pages on disk, repairing concept membership, hashes, citations, and source back-references before incremental compilation.
- Ordered runtime failover across every available provider in `fallback_chain`, including sanitized exhaustion errors and provider/model selection observability.
- Wheel and source-distribution builds plus isolated installed-package CLI verification in CI.

### Changed

- `vm compile --full` now safely forces every current Raw source through compilation without resetting manifest provenance, deleting wiki pages, or discarding history.
- Compilation propagation is idempotent and failure-aware: durable touches retain bidirectional provenance while failed sources remain eligible for incremental retry.
- Missing manifests initialize as empty version-1 state, while malformed, unreadable, schema-invalid, or unsupported manifests now stop compile and lint instead of being silently replaced.
- Repair-only compile runs persist reconciled state, log the repair, and rebuild the index when concept membership changes; dry runs remain write-free.
- Ask preview mode performs no query, manifest, wiki-log, folder, or application-log writes.
- Anthropic, OpenAI, and Ollama retry only transient connection, timeout, rate-limit, and server failures before failing over.
- Empty AI completions trigger fallback; permanent request and credential errors fail over immediately, while cancellation and terminal interrupts propagate.
- `vm ask` and `vm compile` show concise provider errors normally while `--verbose` retains chained diagnostics.
- Package metadata, runtime metadata, and the installed `vm version` command now report 0.2.0 and are verified together during packaging.
