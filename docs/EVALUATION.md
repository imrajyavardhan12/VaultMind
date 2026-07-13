# Offline Evaluation

VaultMind has a deterministic developer quality gate for the claim that a representative corpus becomes a grounded, connected, reusable wiki. It is not a public `vm` command and does not alter production provider or workflow semantics.

## Run it

```bash
uv run python scripts/evaluate_fixture.py --output evaluation-report.json --check
```

`--output` writes stable JSON. `--check` exits 1 after writing the report when any threshold fails. If evaluation cannot complete, the script exits 2 and still writes a safe artifact containing only the exception type and a stable fingerprint—never raw exception text, prompts, paths, or secrets. Without `--check`, completed threshold failures remain visible in `failures` but the process exits normally. CI runs with no provider credentials and requires `evaluation-report.json` to exist before retaining it as an artifact.

The harness deliberately blocks socket connections. It uses only `ReplayProvider`, creates a temporary vault, copies the corpus into that vault, and deletes it after measurement. Never point the harness at a personal vault. Live-provider experimentation belongs outside this benchmark: live output is nondeterministic, costs money, can expose private data, and is not comparable to the committed baseline.

## Production paths exercised

The runner invokes the production Compile command orchestration for both corpus phases and a third unchanged invocation, including its changed-source selection, manifest reconciliation/persistence, concept create/update, and index-trigger decisions. Between phases one and two, it injects three deterministic manifest-only inconsistencies in the temporary vault: a ghost article entry, an incorrect canonical article hash, and an incorrect source back-reference. Production Compile phase two must reconcile all three against concept files on disk. The probe uses fixed labels and existing manifest values; it adds no timestamp or path to the fixture or report, and repaired probe state is absent before final quality measurement.

The runner also uses the production Raw scanner, atomic markdown writer, Ask search/filing loop, command scanners, and canonical lint logic. Replay replaces provider completion; configuration and human command output are injected at the command boundary. During the unchanged invocation, the harness instruments writes, temporary creates, and atomic replacements to VaultMind-owned files under `🗺️ Wiki/` plus `vault.manifest.json`. It also compares before/after content hashes as a secondary assertion; Raw, `VAULTMIND.md`, and application logs outside the vault are excluded.

The report intentionally contains no timestamps, temporary paths, response text, prompts, environment values, or file hashes. Repeated runs produce byte-identical JSON even though production files inside each disposable vault use normal runtime timestamps.

## Metrics

All ratios are in `[0, 1]`; an empty expectation set has ratio `1.0`.

- **Corpus/scanned source count**: declared fixture documents and Raw records observed by the production scanner.
- **Current compiled coverage**: current-hash Raw records with at least one durable reciprocal back-reference: manifest source → concept, concept frontmatter → source, and manifest article → source must all exist.
- **Concept count / expected concept recall**: persisted concept pages and the fraction of declared concept slugs present.
- **Source-to-concept attribution**: expected fixture source IDs are resolved to their canonical source URLs and compared with persisted concept citations and manifest back-references. The report includes expected-pair recall, unexpected-pair count, and assignment precision.
- **Suspicious duplicate pairs**: canonical `duplicate_concept` lint pairs.
- **Citation/provenance inconsistencies**: differences between concept and manifest article source lists, missing source records, and missing links in either direction between manifest sources and articles.
- **Expected graph-edge recall**: found directed `(source concept, target concept)` wikilinks divided by explicit expected edges. Targets are normalized with production wikilink rules. Backtick and tilde fenced examples do not count.
- **Duplicate connection count**: repeated normalized targets beyond the first occurrence in each real `Connections` section.
- **Index quality**: whether the production index file exists, expected concept-link count and recall, and unique unexpected/stale link count. Links use canonical wikilink normalization; aliases, paths, headings, and fenced examples cannot evade the measurement.
- **Broken wikilinks / stale manifest findings / lint findings**: counts from canonical deterministic lint output.
- **Query support**: parsed only from the persisted query-page contract. `Supporting Wiki Pages` is classified as concept-page or previously filed-query support by path; `Supporting Raw Sources` records Raw fallback. Wiki-supported rate accepts either concept or filed-query evidence. The report separately counts concept support, filed-query support, Raw support, Raw fallback, and filed-query reuse.
- **Query expectation pass rate**: fraction of fixture queries containing every declared support class and the declared filed-query reuse behavior.
- **Incremental writes/changed files/provider calls**: observed owned-state write attempts, changed owned-state hashes, and replay calls during the unchanged production Compile invocation. The zero-write gate detects same-byte rewrites and temporary writes that a final content snapshot cannot.
- **Reconciliation probe**: stable per-inconsistency results plus repaired count and success rate for the ghost article, incorrect article hash, and incorrect source back-reference injected before production Compile phase two. The committed gate requires full success.

`provider_call_count` is diagnostic, not a cost estimate.

## Fixture design

`tests/fixtures/evaluation/corpus.json` contains exactly 20 synthetic Raw sources. Five explicitly attributed sources substantially support each of four overlapping concepts: Retrieval Grounding, Citation Provenance, Knowledge Graphs, and Incremental Compilation. Sixteen phase-one sources create the concepts; four phase-two sources exercise production article updates and compounding provenance. Bodies cross-reference neighboring ideas so the desired result is four compounding pages rather than twenty isolated summaries. Eight directed graph edges are explicit. Three queries cover concept support, reuse of a filed answer, and one intentional Raw fallback.

The corpus uses reserved `evaluation.example` URLs and contains no private text, generated timestamp, absolute path, credential, or API key.

## Replay maintenance

`tests/fixtures/evaluation/replay.json` holds conjunctive prompt-match rules for triage, deduplication, concurrent article generation, index rebuild, and Ask. Matching is by rule, not global response order. Every rule has a finite response queue. An unmatched, ambiguous, or exhausted request raises a typed error containing only bounded SHA-256 fingerprints and character counts—not prompt content.

When a production prompt changes:

1. Run the evaluation and identify the safe rule ID/fingerprint that failed.
2. Update the narrowest stable match fragments; do not copy secrets or machine paths.
3. Review response grounding against the corpus and preserve explicit citations/edges.
4. Run the full quality command and confirm two report renders are byte-identical.
5. Review fixture diffs as benchmark changes, not snapshots to update blindly.

Avoid broad fragments that can match multiple call shapes. Article rules must remain concept-specific so concurrent generation cannot consume another concept's response.

## Threshold governance

`tests/fixtures/evaluation/thresholds.json` is reviewed policy. Threshold evaluation is inclusive and reports every failure sorted by metric. Baseline thresholds should change only with an explained product-quality decision or an intentional fixture redesign. Do not automatically ratchet or rewrite them from current output. A regression should normally fix production behavior or replay grounding, not weaken the gate.

Interpret failures by layer:

- coverage or provenance: inspect article citations and manifest reciprocity;
- graph recall/duplicates: inspect real `Connections` links and deduplication;
- index: inspect index existence and normalized expected/unexpected links;
- reconciliation probe: inspect phase-two manifest repair behavior against canonical disk state;
- lint: inspect broken targets or stale state;
- query support/reuse: inspect persisted Supporting sections and wiki-first retrieval;
- incremental writes/calls: inspect source hashes, reconciliation, or unconditional orchestration;
- replay mismatch: review the production prompt shape and the narrow fixture rule.

The benchmark does not judge prose with another LLM, compare vendors, call live providers, measure load, account for tokens, or replace human review.
