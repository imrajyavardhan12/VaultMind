# Changelog

## 0.2.0 - 2026-07-12

### Added

- Ordered runtime failover across every available provider in `fallback_chain`.
- Sanitized provider-chain exhaustion errors and provider/model selection observability.
- Wheel and source-distribution builds plus an isolated installed-CLI smoke test in CI.

### Changed

- Anthropic, OpenAI, and Ollama retry only transient connection, timeout, rate-limit, and server failures before failing over.
- Empty AI completions trigger fallback; permanent request and credential errors fail over immediately.
- `vm ask` and `vm compile` show concise provider errors normally while `--verbose` retains chained diagnostics.
- Package and runtime versions are now 0.2.0.
