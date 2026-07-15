"""Tests for the compile pipeline internals."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path

import yaml
from structlog.testing import capture_logs

from vaultmind.ai.compiler import (
    ARTICLE_PROMPT_TOTAL_CHARS,
    ARTICLE_SOURCE_PER_SOURCE_CHARS,
    ARTICLE_SOURCE_REFERENCE_LIMIT,
    ARTICLE_SOURCE_TOTAL_CHARS,
    REQUIRED_CONCEPT_SECTIONS,
    _create_article,
    _deduplicate_concepts,
    _extract_frontmatter,
    _format_article_source_context,
    _markdown_h2_sections,
    _normalize_concept_body,
    _sanitize_article_wikilinks,
    _split_page,
    _strip_frontmatter_block,
    _strip_model_page_wrappers,
    compile_sources,
)
from vaultmind.commands.compile import _run_compile_async
from vaultmind.core.linter import WikiPage, check_broken_wikilinks, extract_wikilinks
from vaultmind.core.manifest import read_manifest
from vaultmind.core.raw_scanner import RawSourceRecord, scan_raw_sources
from vaultmind.schemas import ConceptStatus, Manifest, WikiConceptEntry


class StubProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []
        self.model = "stub-model"

    async def complete(self, prompt: str, system: str = "") -> str:
        del system
        self.prompts.append(prompt)
        return self.responses.pop(0)


def _raw_source(*, slug: str, source_url: str | None = None) -> RawSourceRecord:
    return RawSourceRecord(
        path=Path(f"/tmp/{slug}.md"),
        relative_path=f"Clippings/{slug}",
        title=slug,
        source_url=source_url,
        body=f"# {slug}\n\nBody text from {slug}",
        content_hash=f"hash-{slug}",
        raw_tags=[],
    )


def test_compile_sources_create_prompt_contains_resolvable_raw_packets_and_unresolved_key(
    test_config,
):
    source_a = _raw_source(slug="ground-a", source_url="https://example.com/a")
    source_b = _raw_source(slug="ground-b", source_url="https://example.com/b")
    unresolved = "https://example.com/missing"
    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": "Grounded Concept",
                    "status": "new",
                    "description": "Grounded in multiple sources",
                    "source_urls": [source_a.source_url, source_b.source_url, unresolved],
                }
            ]
        }
    )
    provider = StubProvider([triage, "## Overview\n\nGrounded article."])

    result, _ = asyncio.run(
        compile_sources(
            [source_a, source_b],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
            max_touches=0,
        )
    )

    assert result.articles_created == 1
    create_prompt = provider.prompts[1]
    assert format(source_a.body) in create_prompt
    assert format(source_b.body) in create_prompt
    assert f"Source: {unresolved}\n\n[Raw source unavailable]" in create_prompt


def test_compile_sources_update_prompt_exceeds_old_excerpt_and_bounds_raw_context(
    test_config,
):
    marker = "CONTENT_AFTER_THE_OLD_800_CHARACTER_BOUNDARY"
    sources = [
        RawSourceRecord(
            path=Path(f"/tmp/bounded-{index}.md"),
            relative_path=f"Clippings/bounded-{index}",
            title=f"Bounded {index}",
            source_url=f"https://example.com/bounded-{index}",
            body=(str(index) * 900) + marker + (str(index) * 9000),
            content_hash=f"bounded-hash-{index}",
            raw_tags=[],
        )
        for index in range(5)
    ]
    article_dir = test_config.vault_path / test_config.folders.wiki / test_config.folders.wiki_concepts
    article_dir.mkdir(parents=True, exist_ok=True)
    (article_dir / "bounded.md").write_text("# Bounded\n\nExisting", encoding="utf-8")
    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": "Bounded",
                    "status": "existing:bounded",
                    "description": "Bounded source context",
                    "source_urls": [source.source_url for source in sources],
                    "merge_target": "bounded",
                }
            ]
        }
    )
    provider = StubProvider([triage, "# Bounded\n\nUpdated"])

    asyncio.run(
        compile_sources(
            sources,
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
            max_touches=0,
        )
    )

    update_prompt = provider.prompts[1]
    assert update_prompt.count(marker) == len(sources)
    expected_per_source = min(
        ARTICLE_SOURCE_PER_SOURCE_CHARS,
        ARTICLE_SOURCE_TOTAL_CHARS // len(sources),
    )
    assert expected_per_source > 800
    assert all((str(index) * (expected_per_source + 1)) not in update_prompt for index in range(5))

    source_lookup = {source.source_url: source for source in sources}
    context = _format_article_source_context(
        [source.source_url for source in sources], source_lookup
    )
    assert len(context) <= ARTICLE_SOURCE_TOTAL_CHARS
    assert context == _format_article_source_context(
        [source.source_url for source in sources], source_lookup
    )


def test_source_reference_limit_is_deterministic_and_consistent_across_artifacts(
    test_config,
):
    source = _raw_source(slug="accepted", source_url="https://example.com/accepted")
    long_keys = [
        f"https://missing.example/{index:03d}/" + (chr(97 + index % 26) * 400)
        for index in range(ARTICLE_SOURCE_REFERENCE_LIMIT + 7)
    ]
    submitted = [source.source_url, source.source_url, *long_keys]
    expected = [source.source_url, *long_keys[: ARTICLE_SOURCE_REFERENCE_LIMIT - 1]]
    dropped = long_keys[ARTICLE_SOURCE_REFERENCE_LIMIT - 1 :]
    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": "Bounded Attribution",
                    "status": "new",
                    "description": "Bounded and consistent",
                    "source_urls": submitted,
                }
            ]
        }
    )
    provider = StubProvider([triage, "## Overview\n\nArticle."])

    with capture_logs() as logs:
        result, slug_to_urls = asyncio.run(
            compile_sources(
                [source],
                Manifest(),
                provider,
                test_config.vault_path,
                test_config.folders,
                max_touches=0,
            )
        )

    truncations = [
        entry for entry in logs if entry["event"] == "article_source_references_truncated"
    ]
    assert truncations
    assert truncations[0]["accepted"] == ARTICLE_SOURCE_REFERENCE_LIMIT
    assert truncations[0]["dropped"] == len(submitted) - 1 - len(expected)
    assert result.articles_created == 1
    assert slug_to_urls == {"bounded-attribution": expected}
    context = provider.prompts[1].partition("Attributed source citations:\n")[2].partition(
        "\n\nRaw source packets:"
    )[0]
    assert len(context.splitlines()) == ARTICLE_SOURCE_REFERENCE_LIMIT
    for key in expected[1:]:
        assert hashlib.sha256(key.encode()).hexdigest()[:12] in context
    for key in dropped:
        assert hashlib.sha256(key.encode()).hexdigest()[:12] not in context
    lookup = {source.source_url: source}
    assert _format_article_source_context(submitted, lookup) == _format_article_source_context(
        submitted, lookup
    )

    page_path = (
        test_config.vault_path
        / test_config.folders.wiki
        / test_config.folders.wiki_concepts
        / "bounded-attribution.md"
    )
    page = page_path.read_text(encoding="utf-8")
    frontmatter, body = page.split("---", 2)[1:]
    assert yaml.safe_load(frontmatter)["sources"] == expected
    sources_body = body.partition("## Sources\n")[2]
    assert [line[2:] for line in sources_body.splitlines() if line.startswith("- ")] == expected


def test_article_prompt_and_context_are_hard_bounded_for_pathological_attribution():
    resolved = [
        RawSourceRecord(
            path=Path(f"/tmp/pathological-{index}.md"),
            relative_path=f"Clippings/{index}",
            title="T" * 2000,
            source_url=f"https://example.com/resolved/{index}/" + ("r" * 1000),
            body=f"RAW-{index}-" + ("body" * 3000),
            content_hash=f"hash-{index}",
            raw_tags=["tag" * 1000],
        )
        for index in range(120)
    ]
    unresolved = [f"https://missing.example/{index}/" + ("u" * 1000) for index in range(120)]
    keys = [source.source_url for source in resolved] + unresolved
    lookup = {source.source_url: source for source in resolved}

    context = _format_article_source_context(keys, lookup)

    assert len(context) <= ARTICLE_SOURCE_TOTAL_CHARS
    citation_block, _, packet_block = context.partition("\n\nRaw source packets:\n")
    assert len(citation_block.splitlines()) == ARTICLE_SOURCE_REFERENCE_LIMIT + 1
    retained_keys = keys[:ARTICLE_SOURCE_REFERENCE_LIMIT]
    for key in retained_keys:
        assert hashlib.sha256(key.encode()).hexdigest()[:12] in citation_block
    for key in keys[ARTICLE_SOURCE_REFERENCE_LIMIT:]:
        assert hashlib.sha256(key.encode()).hexdigest()[:12] not in citation_block
    for packet in packet_block.split("\n\n---\n\n"):
        if packet and "[Raw source unavailable]" not in packet:
            assert "Raw body:\n" in packet
            assert packet.partition("Raw body:\n")[2]

    concept = WikiConceptEntry(
        name="Pathological prompt",
        status=ConceptStatus.NEW,
        description="description" * 100_000,
        source_urls=keys,
    )
    provider = StubProvider(["article"])
    asyncio.run(_create_article(concept, lookup, provider))
    assert len(provider.prompts[0]) <= ARTICLE_PROMPT_TOTAL_CHARS


def test_resolvable_source_is_retained_ahead_of_unresolved_reference_cap(test_config):
    source = _raw_source(slug="grounded-last", source_url="https://example.com/grounded-last")
    unresolved = [
        f"https://missing.example/{index}" for index in range(ARTICLE_SOURCE_REFERENCE_LIMIT)
    ]
    submitted = [*unresolved, source.source_url]
    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": "Grounded Priority",
                    "status": "new",
                    "description": "Must retain Raw grounding",
                    "source_urls": submitted,
                }
            ]
        }
    )
    provider = StubProvider([triage, "## Overview\n\nGrounded."])

    _result, slug_to_urls = asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
            max_touches=0,
        )
    )

    accepted = slug_to_urls["grounded-priority"]
    assert accepted[0] == source.source_url
    assert len(accepted) == ARTICLE_SOURCE_REFERENCE_LIMIT
    assert "Raw body:\n# grounded-last" in provider.prompts[1]


def test_wrapper_scanner_tracks_fence_marker_length_and_valid_closers():
    model = (
        "# Remove before\n"
        "````python\n"
        "# Keep in four-backtick fence\n"
        "```\n"
        "# Keep after shorter closer\n"
        "~~~~\n"
        "# Keep after mismatched closer\n"
        "````   \n"
        "   # Remove after real closer ###\n"
        "~~~text\n"
        "# Keep in tilde fence\n"
        "````\n"
        "# Keep after mismatched backticks\n"
        "~~~~\n"
        "# Remove after tilde closer\n"
        "```\n"
        "# Keep in unclosed fence\n"
    )

    cleaned = _strip_model_page_wrappers(model)

    assert "Remove before" not in cleaned
    assert "Remove after real closer" not in cleaned
    assert "Remove after tilde closer" not in cleaned
    assert "# Keep in four-backtick fence" in cleaned
    assert "# Keep after shorter closer" in cleaned
    assert "# Keep after mismatched closer" in cleaned
    assert "# Keep in tilde fence" in cleaned
    assert "# Keep after mismatched backticks" in cleaned
    assert "# Keep in unclosed fence" in cleaned


def test_wrapper_scanner_preserves_four_space_pseudo_fences_and_body_content():
    model = (
        "    ```\n"
        "    # Four-space indented content\n"
        "    ```\n"
        "  # Remove indented H1\n"
        "\n"
        "  preserved body indentation  \n"
    )

    preserved = (
        "    ```\n"
        "    # Four-space indented content\n"
        "    ```\n"
        "\n"
        "  preserved body indentation  \n"
    )
    assert _strip_model_page_wrappers(model) == preserved

    normalized = _normalize_concept_body(model, "Canonical", [])
    assert normalized.startswith("# Canonical\n\n")
    assert preserved in normalized


def test_wrapper_scanner_removes_crlf_frontmatter_atx_and_setext_h1s():
    model = (
        "---\r\n"
        "title: Model title\r\n"
        "sources: [hallucinated]\r\n"
        "---\r\n"
        "\r\n"
        "   # Indented ATX title ###\r\n"
        "Keep this paragraph exactly.  \r\n"
        "Model setext title\r\n"
        "   ======   \r\n"
        "After.\r\n"
    )

    assert _strip_model_page_wrappers(model) == (
        "\r\nKeep this paragraph exactly.  \r\nAfter.\r\n"
    )


def test_wrapper_scanner_accepts_longer_closer_but_not_closer_with_content():
    model = """```lang
# Keep one
```` trailing
# Keep two
````
# Remove outside
"""

    cleaned = _strip_model_page_wrappers(model)

    assert "# Keep one" in cleaned
    assert "# Keep two" in cleaned
    assert "Remove outside" not in cleaned


def test_h2_scanner_rejects_fence_closer_with_trailing_content():
    body = """````markdown
## Sources
```` trailing content
## Open Questions
````
## Overview

Real overview.
"""

    assert [section.title for section in _markdown_h2_sections(body)] == ["Overview"]

    normalized = _normalize_concept_body(body, "Fence Contract", ["https://example.com/source"])
    titles = [section.title for section in _markdown_h2_sections(normalized)]
    for heading in REQUIRED_CONCEPT_SECTIONS:
        assert titles.count(heading) == 1


def test_normalization_sanitizes_multiline_title_to_exactly_one_h1():
    normalized = _normalize_concept_body(
        "## Overview\n\nBody.",
        "Good Title\n# Injected Title",
        [],
    )

    assert normalized.startswith("# Good Title # Injected Title\n")
    assert len(re.findall(r"^ {0,3}#(?:[ \t]+|$)", normalized, re.MULTILINE)) == 1


def test_frontmatter_parser_requires_line_delimited_closer():
    page = (
        "---\r\n"
        "title: foo---bar\r\n"
        "sources:\r\n"
        "  - https://example.com/original\r\n"
        "---\r\n"
        "\r\n"
        "# Foo\r\n"
    )

    assert _extract_frontmatter(page) == {
        "title": "foo---bar",
        "sources": ["https://example.com/original"],
    }
    frontmatter, body = _split_page(page)
    assert frontmatter["sources"] == ["https://example.com/original"]
    assert body == "# Foo\r\n"
    assert _strip_frontmatter_block(page) == "# Foo\r\n"


def test_normalization_ignores_fenced_headings_and_consolidates_sources():
    source = "https://example.com/managed"
    model_body = f"""```markdown
## Key Ideas
## Connections
## Open Questions
## Sources
- {source}
```

## Overview

Real overview.

## Sources

- {source}
First source note.

## Sources

- {source}
Second source note.
"""

    normalized = _normalize_concept_body(model_body, "Fence Aware", [source])
    sections = _markdown_h2_sections(normalized)
    real_titles = [section.title for section in sections]

    for heading in REQUIRED_CONCEPT_SECTIONS:
        assert real_titles.count(heading) == 1
    assert "First source note." in normalized
    assert "Second source note." in normalized
    outside_fence = normalized[normalized.index("```", normalized.index("```") + 3) + 3 :]
    assert outside_fence.count(f"- {source}") == 1


def test_article_wikilink_sanitizer_preserves_canonical_approved_forms_and_fences():
    body = (
        "Keep [[Approved]], [[Wiki/Concepts/APPROVED.md#Heading|Shown]], "
        "[[approved^block]], [[approved.md]], and [[approved|Alias]].\n"
        "Drop [[invented]], [[path/invented.md#Heading]], [[invented^block]], "
        "and [[invented|Display Text]] while keeping sibling [[approved]].\n\n"
        "~~~markdown\n"
        "[[invented|Tilde Example]]\n"
        "~~~\n"
        "`````markdown\n"
        "[[invented|Long Fence Example]]\n"
        "```\n"
        "[[still-in-long-fence]]\n"
        "`````\n"
    )

    sanitized = _sanitize_article_wikilinks(body, {"approved"})

    assert sanitized == (
        "Keep [[Approved]], [[Wiki/Concepts/APPROVED.md#Heading|Shown]], "
        "[[approved^block]], [[approved.md]], and [[approved|Alias]].\n"
        "Drop invented, invented, invented, and Display Text while keeping sibling "
        "[[approved]].\n\n"
        "~~~markdown\n"
        "[[invented|Tilde Example]]\n"
        "~~~\n"
        "`````markdown\n"
        "[[invented|Long Fence Example]]\n"
        "```\n"
        "[[still-in-long-fence]]\n"
        "`````\n"
    )
    assert set(extract_wikilinks(sanitized)) == {"approved"}


def test_article_wikilink_sanitizer_rescans_nested_and_malformed_aliases():
    body = (
        "Nested [[bad|[[evil]]]], malformed [[[[bad]]]], surrounding [[[evil]]], "
        "and alias [[bad|Display [[evil]] text]]; keep [[approved]], drop [[bad]]."
    )

    sanitized = _sanitize_article_wikilinks(body, {"approved"})

    assert extract_wikilinks(sanitized) == ["approved"]
    assert "[[evil]]" not in sanitized
    assert "[[bad]]" not in sanitized
    assert "Nested evil" in sanitized
    assert "keep [[approved]], drop bad" in sanitized


def test_article_prompts_bound_existing_concept_allowlist_and_prohibit_links_when_empty():
    concept = WikiConceptEntry(
        name="Prompt Contract",
        status=ConceptStatus.NEW,
        description="Prompt rules",
        source_urls=[],
    )
    empty_provider = StubProvider(["article"])
    asyncio.run(
        _create_article(
            concept,
            {},
            empty_provider,
            approved_concept_slugs=set(),
        )
    )

    assert "No concept slugs are approved" in empty_provider.prompts[0]
    assert "Do not use any Obsidian wikilinks" in empty_provider.prompts[0]

    bounded_provider = StubProvider(["article"])
    asyncio.run(
        _create_article(
            concept,
            {},
            bounded_provider,
            approved_concept_slugs={f"concept-{index:04d}" for index in range(1000)},
        )
    )

    bounded_prompt = bounded_provider.prompts[0]
    assert len(bounded_prompt) <= ARTICLE_PROMPT_TOTAL_CHARS
    assert "concept-0000" in bounded_prompt
    assert "concept-0999" not in bounded_prompt


def test_compile_create_write_sanitizes_against_precompile_disk_concepts(test_config):
    source = _raw_source(slug="wikilink-create", source_url="https://example.com/create")
    _seed_existing_concept(
        test_config,
        slug="approved-existing",
        title="Approved Existing",
        body="# Approved Existing\n\n## Overview\n\nExisting.\n",
    )
    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": "Fresh A",
                    "status": "new",
                    "description": "First concurrent concept",
                    "source_urls": [source.source_url],
                },
                {
                    "name": "Fresh B",
                    "status": "new",
                    "description": "Second concurrent concept",
                    "source_urls": [source.source_url],
                },
            ]
        }
    )
    provider = StubProvider(
        [
            triage,
            triage,
            (
                "# Fresh A\n\n## Overview\n\n"
                "Use [[APPROVED-EXISTING|Approved]], not [[fresh-b|Fresh B]] or "
                "[[kv-caching]]. Nested [[bad|[[evil]]]] and [[[[bad]]]].\n\n"
                "~~~markdown\n[[invented-tilde-example]]\n~~~\n"
            ),
            "# Fresh B\n\n## Overview\n\nFresh B.",
        ]
    )

    result, _ = asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
            existing_concepts=[("approved-existing", "Approved Existing")],
            max_touches=0,
        )
    )

    fresh_a = (
        test_config.vault_path
        / test_config.folders.wiki
        / test_config.folders.wiki_concepts
        / "fresh-a.md"
    ).read_text(encoding="utf-8")
    assert result.articles_created == 2
    assert "[[APPROVED-EXISTING|Approved]]" in fresh_a
    assert "not Fresh B or kv-caching" in fresh_a
    assert "Nested evil and bad" in fresh_a
    assert "[[fresh-b" not in fresh_a
    assert "~~~markdown\n[[invented-tilde-example]]\n~~~" in fresh_a
    assert extract_wikilinks(fresh_a) == ["approved-existing"]
    persisted_page = WikiPage(relative_path="fresh-a", text=fresh_a, is_index=False)
    assert check_broken_wikilinks(
        [persisted_page], {"approved-existing", "fresh-a", "fresh-b"}
    ) == []
    assert all("approved-existing" in prompt for prompt in provider.prompts[2:])
    assert all("Only use wikilinks whose target" in prompt for prompt in provider.prompts[2:])


def test_compile_create_write_sanitizes_model_title_before_frontmatter_and_h1(
    test_config,
):
    source = _raw_source(slug="title-wikilink", source_url="https://example.com/title")
    model_title = "Fresh [[invented|Readable [[nested]]]]"
    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": model_title,
                    "status": "new",
                    "description": "Untrusted model title",
                    "source_urls": [source.source_url],
                }
            ]
        }
    )
    provider = StubProvider([triage, "## Overview\n\nArticle body."])

    result, _ = asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
            max_touches=0,
        )
    )

    article_path = (
        test_config.vault_path
        / test_config.folders.wiki
        / test_config.folders.wiki_concepts
        / "fresh-inventedreadable-nested.md"
    )
    persisted = article_path.read_text(encoding="utf-8")
    assert result.articles_created == 1
    assert _extract_frontmatter(persisted)["title"] == "Fresh Readable nested"
    assert "\n# Fresh Readable nested\n" in persisted
    assert "[[invented" not in persisted
    assert "[[nested" not in persisted
    assert check_broken_wikilinks(
        [
            WikiPage(
                relative_path="fresh-inventedreadable-nested",
                text=persisted,
                is_index=False,
            )
        ],
        {"fresh-inventedreadable-nested"},
    ) == []


def test_compile_create_write_sanitizes_after_lone_cr_fence_before_persistence(
    test_config,
):
    source = _raw_source(slug="cr-fence", source_url="https://example.com/cr-fence")
    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": "CR Fence",
                    "status": "new",
                    "description": "Lone CR fence endings",
                    "source_urls": [source.source_url],
                }
            ]
        }
    )
    model_article = (
        "## Overview\n\nExample:\n"
        "```text\r"
        "[[fenced-example]]\r"
        "```\r"
        "After [[invented-after-cr|readable fallback]].\r"
    )
    assert extract_wikilinks(model_article) == ["invented-after-cr"]
    provider = StubProvider([triage, model_article])

    result, _ = asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
            max_touches=0,
        )
    )

    article_path = (
        test_config.vault_path
        / test_config.folders.wiki
        / test_config.folders.wiki_concepts
        / "cr-fence.md"
    )
    persisted = article_path.read_text(encoding="utf-8")
    assert result.articles_created == 1
    assert "[[fenced-example]]" in persisted
    assert "After readable fallback." in persisted
    assert extract_wikilinks(persisted) == []
    assert check_broken_wikilinks(
        [WikiPage(relative_path="cr-fence", text=persisted, is_index=False)],
        {"cr-fence"},
    ) == []


def test_compile_update_write_sanitizes_links_and_preserves_readable_markdown(test_config):
    source = _raw_source(slug="wikilink-update", source_url="https://example.com/update")
    target_path = _seed_existing_concept(
        test_config,
        slug="update-target",
        title="Human Update Title",
        body="# Human Update Title\n\n## Overview\n\nOld.\n",
    )
    _seed_existing_concept(
        test_config,
        slug="approved-related",
        title="Approved Related",
        body="# Approved Related\n\n## Overview\n\nRelated.\n",
    )
    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": "Update Target",
                    "status": "existing:update-target",
                    "description": "Update existing article",
                    "source_urls": [source.source_url],
                    "merge_target": "update-target",
                }
            ]
        }
    )
    model_update = (
        "# Wrong Model Title\n\n## Overview\n\n"
        "Keep **[[Wiki/Concepts/APPROVED-RELATED.md#Overview|Related]]**, "
        "replace [[kv-caching|KV caching]] and [[path/linear-attention.md^detail]]. "
        "Do not reconstruct [[bad|[[evil]]]] or [[[[bad]]]].\n\n"
        "`````markdown\n"
        "[[invented-long-fence-example]]\n"
        "```\n"
        "[[invented-after-short-run]]\n"
        "`````\n"
    )
    provider = StubProvider([triage, model_update])

    result, _ = asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
            existing_concepts=[
                ("approved-related", "Approved Related"),
                ("update-target", "Human Update Title"),
            ],
            max_touches=0,
        )
    )

    updated = target_path.read_text(encoding="utf-8")
    assert result.articles_updated == 1
    assert "# Human Update Title" in updated
    assert "**[[Wiki/Concepts/APPROVED-RELATED.md#Overview|Related]]**" in updated
    assert "replace KV caching and linear-attention." in updated
    assert "Do not reconstruct evil or bad" in updated
    assert (
        "`````markdown\n"
        "[[invented-long-fence-example]]\n"
        "```\n"
        "[[invented-after-short-run]]\n"
        "`````"
    ) in updated
    assert extract_wikilinks(updated) == ["approved-related"]
    persisted_page = WikiPage(relative_path="update-target", text=updated, is_index=False)
    assert check_broken_wikilinks(
        [persisted_page], {"approved-related", "update-target"}
    ) == []
    assert "approved-related" in provider.prompts[1]


def test_written_concept_enforces_page_contract_and_synchronizes_sources(test_config):
    source = _raw_source(slug="contract", source_url="https://example.com/contract")
    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": "Human Contract Title",
                    "status": "new",
                    "description": "Contract test",
                    "source_urls": [source.source_url, source.source_url],
                }
            ]
        }
    )
    model_page = """---
title: Model Title
sources:
  - https://model.example/hallucinated
---

# Model Title
# Duplicate Leading Title

Model-authored prose is retained.

## Overview

An overview.

## Sources

- https://model.example/hallucinated-body-source
- https://example.com/contract
- https://example.com/contract
"""
    provider = StubProvider([triage, model_page])

    asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
            max_touches=0,
        )
    )

    path = (
        test_config.vault_path
        / test_config.folders.wiki
        / test_config.folders.wiki_concepts
        / "human-contract-title.md"
    )
    page = path.read_text(encoding="utf-8")
    frontmatter, body = page.split("---", 2)[1:]
    parsed = yaml.safe_load(frontmatter)
    assert parsed == {
        "title": "Human Contract Title",
        "vaultmind": True,
        "kind": "concept",
        "sources": [source.source_url],
    }
    assert re.findall(r"^#\s+.+$", body, re.MULTILINE) == ["# Human Contract Title"]
    for heading in ("Overview", "Key Ideas", "Connections", "Open Questions", "Sources"):
        assert f"## {heading}" in body
    assert "Model-authored prose is retained." in body
    assert body.count(f"- {source.source_url}") == 1
    assert "Model Title" not in body
    assert "https://model.example/hallucinated" not in page
    assert "https://model.example/hallucinated-body-source" not in page


def test_compile_sources_updates_existing_article_with_relative_path_raw_body(test_config):
    source = _raw_source(slug="raw-a", source_url=None)
    article_dir = test_config.vault_path / test_config.folders.wiki / test_config.folders.wiki_concepts
    article_dir.mkdir(parents=True, exist_ok=True)
    (article_dir / "concept-a.md").write_text("# Concept A\n\nOld content", encoding="utf-8")

    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": "Concept A",
                    "status": "existing:concept-a",
                    "description": "Existing concept",
                    "source_urls": [source.relative_path],
                    "merge_target": "concept-a",
                }
            ]
        }
    )
    provider = StubProvider([triage, "# Concept A\n\nUpdated content"])

    result, slug_to_urls = asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
        )
    )

    assert result.articles_updated == 1
    assert result.articles_created == 0
    assert slug_to_urls == {"concept-a": [source.relative_path]}
    assert "Body text from raw-a" in provider.prompts[1]


def test_compile_sources_preserves_existing_frontmatter_sources_on_update(test_config):
    source = _raw_source(slug="raw-b", source_url="https://example.com/new-source")
    old_source_url = "https://example.com/old-source"
    article_dir = test_config.vault_path / test_config.folders.wiki / test_config.folders.wiki_concepts
    article_dir.mkdir(parents=True, exist_ok=True)
    article_path = article_dir / "concept-a.md"
    article_path.write_text(
        "\n".join(
            [
                "---",
                "title: Human Concept A",
                "vaultmind: true",
                "kind: concept",
                "sources:",
                f"  - {old_source_url}",
                "---",
                "",
                "# Concept A",
                "",
                "Old content",
            ]
        ),
        encoding="utf-8",
    )

    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": "Concept A",
                    "status": "existing:concept-a",
                    "description": "Existing concept with new source",
                    "source_urls": [source.source_url],
                    "merge_target": "concept-a",
                }
            ]
        }
    )
    provider = StubProvider([triage, "# Concept A\n\nUpdated content"])

    result, slug_to_urls = asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
        )
    )

    updated = article_path.read_text(encoding="utf-8")
    assert result.articles_updated == 1
    assert slug_to_urls == {"concept-a": [old_source_url, source.source_url]}
    assert "title: Human Concept A" in updated
    assert f"  - {old_source_url}" in updated
    assert f"  - {source.source_url}" in updated


def test_fixture_vault_compile_updates_existing_concept_and_preserves_sources(fixture_vault):
    records = scan_raw_sources(fixture_vault)
    source = next(
        record
        for record in records
        if record.source_url == "https://blog.research.google/2023/03/encouraging-sparse-attention-in.html"
    )
    old_source_url = "https://lena-voita.github.io/posts/annotated_transformers/attention.html"

    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": "Attention Mechanisms",
                    "status": "existing:attention-mechanisms",
                    "description": "Existing attention concept expanded with sparse attention trade-offs",
                    "source_urls": [source.source_url],
                    "merge_target": "attention-mechanisms",
                }
            ]
        }
    )
    provider = StubProvider([triage, "# Attention Mechanisms\n\nUpdated with sparse attention."])
    manifest = read_manifest(fixture_vault.vault_path)

    result, slug_to_urls = asyncio.run(
        _run_compile_async(
            [source],
            manifest,
            fixture_vault,
            provider,
            dry_run=False,
        )
    )

    article_path = (
        fixture_vault.vault_path
        / fixture_vault.folders.wiki
        / fixture_vault.folders.wiki_concepts
        / "attention-mechanisms.md"
    )
    updated = article_path.read_text(encoding="utf-8")

    assert result.articles_updated == 1
    assert slug_to_urls == {"attention-mechanisms": [old_source_url, source.source_url]}
    assert source.source_url in manifest.sources
    assert manifest.sources[source.source_url].wiki_articles == ["attention-mechanisms"]
    assert old_source_url in manifest.wiki_articles["attention-mechanisms"].source_urls
    assert source.source_url in manifest.wiki_articles["attention-mechanisms"].source_urls
    assert f"  - {old_source_url}" in updated
    assert f"  - {source.source_url}" in updated


def test_compile_sources_dry_run_does_not_write_concept_pages(test_config):
    source = _raw_source(slug="dry-run-source", source_url="https://example.com/dry-run")
    article_dir = test_config.vault_path / test_config.folders.wiki / test_config.folders.wiki_concepts
    article_path = article_dir / "dry-run-concept.md"
    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": "Dry Run Concept",
                    "status": "new",
                    "description": "A concept that should not be written",
                    "source_urls": [source.source_url],
                }
            ]
        }
    )
    provider = StubProvider([triage])

    result, slug_to_urls = asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
            dry_run=True,
        )
    )

    assert result.articles_created == 0
    assert result.articles_updated == 0
    assert slug_to_urls == {"dry-run-concept": [source.source_url]}
    assert provider.responses == []
    assert not article_path.exists()


def test_deduplicate_concepts_preserves_existing_status_when_response_omits_status():
    concepts = [
        WikiConceptEntry(
            name="Attention",
            status=ConceptStatus.EXISTING,
            description="Existing concept",
            source_urls=["https://example.com/a"],
            merge_target="attention",
        ),
        WikiConceptEntry(
            name="Self Attention",
            status=ConceptStatus.NEW,
            description="New overlap",
            source_urls=["https://example.com/b"],
        ),
    ]
    provider = StubProvider(
        [
            json.dumps(
                {
                    "concepts": [
                        {
                            "name": "Attention Mechanisms",
                            "description": "Merged concept",
                            "source_urls": ["https://example.com/a", "https://example.com/b"],
                        }
                    ]
                }
            )
        ]
    )

    deduped = asyncio.run(_deduplicate_concepts(concepts, provider))

    assert len(deduped) == 1
    assert deduped[0].status == ConceptStatus.EXISTING
    assert deduped[0].merge_target == "attention"


def test_compile_sources_continues_when_one_concept_create_fails_and_names_it_in_errors(
    test_config, tmp_path
):
    """When one concept's article create fails, other concepts still succeed and error is named."""
    success_source = _raw_source(slug="success-source", source_url="https://example.com/success")
    failing_source = _raw_source(slug="fail-source", source_url="https://example.com/fail")

    # Setup wiki concepts directory for article writes
    article_dir = test_config.vault_path / test_config.folders.wiki / test_config.folders.wiki_concepts
    article_dir.mkdir(parents=True, exist_ok=True)

    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": "Success Concept",
                    "status": "new",
                    "description": "This one will succeed",
                    "source_urls": [success_source.source_url],
                },
                {
                    "name": "Failing Concept",
                    "status": "new",
                    "description": "This one will fail",
                    "source_urls": [failing_source.source_url],
                },
            ]
        }
    )

    # Dedup response — keep the two concepts as-is
    dedup = json.dumps(
        {
            "concepts": [
                {
                    "name": "Success Concept",
                    "status": "new",
                    "description": "This one will succeed",
                    "source_urls": [success_source.source_url],
                },
                {
                    "name": "Failing Concept",
                    "status": "new",
                    "description": "This one will fail",
                    "source_urls": [failing_source.source_url],
                },
            ]
        }
    )

    class FailingProvider(StubProvider):
        """Provider that fails on specific concept names."""

        def __init__(self):
            # Responses for triage, dedup, and one article creation (second will fail)
            super().__init__([
                triage,
                dedup,
                "# Success Concept\n\nSuccess content",
            ])

        async def complete(self, prompt: str, system: str = "") -> str:
            del system
            self.prompts.append(prompt)
            # Fail only on article creation prompts for "Failing Concept"
            # Article creation prompts contain "Write a new wiki article for the concept"
            if "Write a new wiki article" in prompt and "Failing Concept" in prompt:
                raise ValueError("Simulated article creation failure")
            response = self.responses.pop(0)
            return response

    provider = FailingProvider()

    result, slug_to_urls = asyncio.run(
        compile_sources(
            [success_source, failing_source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
        )
    )

    # Success concept should have been created
    assert result.articles_created == 1
    # One error should be recorded with the concept name
    assert len(result.errors) == 1
    assert "Concept 'Failing Concept' failed:" in result.errors[0]
    # Success concept should be in slug_to_urls
    assert "success-concept" in slug_to_urls
    # Article file should exist for success concept
    success_article = article_dir / "success-concept.md"
    assert success_article.exists()


def test_compile_run_skips_manifest_update_for_failed_source(test_config):
    """When one concept fails, only successful sources update the manifest."""
    success_source = _raw_source(slug="success-source", source_url="https://example.com/success")
    failing_source = _raw_source(slug="fail-source", source_url="https://example.com/fail")

    # Setup wiki concepts directory for article writes
    article_dir = test_config.vault_path / test_config.folders.wiki / test_config.folders.wiki_concepts
    article_dir.mkdir(parents=True, exist_ok=True)

    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": "Success Concept",
                    "status": "new",
                    "description": "This one will succeed",
                    "source_urls": [success_source.source_url],
                },
                {
                    "name": "Failing Concept",
                    "status": "new",
                    "description": "This one will fail",
                    "source_urls": [failing_source.source_url],
                },
            ]
        }
    )

    # Dedup response — keep the two concepts as-is
    dedup = json.dumps(
        {
            "concepts": [
                {
                    "name": "Success Concept",
                    "status": "new",
                    "description": "This one will succeed",
                    "source_urls": [success_source.source_url],
                },
                {
                    "name": "Failing Concept",
                    "status": "new",
                    "description": "This one will fail",
                    "source_urls": [failing_source.source_url],
                },
            ]
        }
    )

    class FailingProvider(StubProvider):
        """Provider that fails on specific concept names."""

        def __init__(self):
            # Responses for triage, dedup, and one article creation (second will fail)
            super().__init__([
                triage,
                dedup,
                "# Success Concept\n\nSuccess content",
            ])

        async def complete(self, prompt: str, system: str = "") -> str:
            del system
            self.prompts.append(prompt)
            # Fail only on article creation prompts for "Failing Concept"
            if "Write a new wiki article" in prompt and "Failing Concept" in prompt:
                raise ValueError("Simulated article creation failure")
            response = self.responses.pop(0)
            return response

    provider = FailingProvider()
    manifest = Manifest()

    # Call _run_compile_async directly to test manifest update behavior
    _result, _slug_to_urls = asyncio.run(
        _run_compile_async(
            [success_source, failing_source],
            manifest,
            test_config,
            provider,
            dry_run=False,
        )
    )

    # Verify the manifest was updated only for the successful source
    assert success_source.source_url in manifest.sources
    assert failing_source.source_url not in manifest.sources
    assert "success-concept" in manifest.wiki_articles


# ---- Cross-page propagation (Stage 3) ----


def _seed_existing_concept(
    test_config,
    *,
    slug: str,
    title: str,
    body: str,
    existing_sources: list[str] | None = None,
) -> Path:
    """Write a concept page to the vault and return its path."""
    article_dir = (
        test_config.vault_path / test_config.folders.wiki / test_config.folders.wiki_concepts
    )
    article_dir.mkdir(parents=True, exist_ok=True)
    fm_lines = [
        "---",
        f'title: "{title}"',
        "vaultmind: true",
        "kind: concept",
        "sources:",
    ]
    for source in existing_sources or []:
        fm_lines.append(f"  - {source}")
    fm_lines.append("---")
    fm_block = "\n".join(fm_lines)

    path = article_dir / f"{slug}.md"
    path.write_text(f"{fm_block}\n\n{body}\n", encoding="utf-8")
    return path


def _triage_for_new_concept(name: str, slug: str, source_key: str) -> str:
    return json.dumps(
        {
            "concepts": [
                {
                    "name": name,
                    "status": "new",
                    "description": f"New concept {name}",
                    "source_urls": [source_key],
                    "merge_target": None,
                }
            ]
        }
    )


def _touch_response(target_slug: str, new_slug: str, relevance: int = 7) -> str:
    return json.dumps(
        {
            "touches": [
                {
                    "target_slug": target_slug,
                    "connection_line": (
                        f"[[{new_slug}|New Concept]] — relates to {target_slug}."
                    ),
                    "relevance": relevance,
                }
            ]
        }
    )


def test_propagation_writes_connections_bullet_to_existing_concept(test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    _seed_existing_concept(
        test_config,
        slug="existing-concept",
        title="Existing Concept",
        body=(
            "# Existing Concept\n\n"
            "## Overview\n\n"
            "Some overview text.\n\n"
            "## Key Ideas\n\n"
            "- Idea one\n\n"
            "## Connections\n\n"
            "- [[other-concept]] — pre-existing relationship.\n\n"
            "## Sources\n\n"
            "- https://example.com/old-source\n"
        ),
        existing_sources=["https://example.com/old-source"],
    )

    triage = _triage_for_new_concept("New Concept", "new-concept", source.source_url)
    touch = _touch_response("existing-concept", "new-concept", relevance=8)
    provider = StubProvider(
        [triage, "# New Concept\n\nBody", touch]
    )

    result, _ = asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
        )
    )

    assert result.articles_touched == 1
    assert result.propagation_touches_by_source == {
        source.source_url: ["existing-concept"]
    }
    target_path = (
        test_config.vault_path
        / test_config.folders.wiki
        / test_config.folders.wiki_concepts
        / "existing-concept.md"
    )
    updated = target_path.read_text(encoding="utf-8")
    # Original sections preserved
    assert "## Overview\n\nSome overview text." in updated
    assert "- Idea one" in updated
    # Connections section now includes both old + new bullet
    assert "[[other-concept]] — pre-existing relationship." in updated
    assert "[[new-concept|New Concept]]" in updated
    # Sources section updated
    assert "- https://example.com/old-source" in updated
    assert "- https://example.com/raw-a" in updated


def test_propagation_is_idempotent_when_target_already_links_new_slug(test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    target_path = _seed_existing_concept(
        test_config,
        slug="existing-concept",
        title="Existing Concept",
        body=(
            "# Existing Concept\n\n"
            "## Connections\n\n"
            "- [[new-concept|New Concept]] — already linked.\n"
        ),
    )
    original = target_path.read_text(encoding="utf-8")

    triage = _triage_for_new_concept("New Concept", "new-concept", source.source_url)
    # The LLM still tries to touch — propagation must skip it via the idempotency guard.
    touch = _touch_response("existing-concept", "new-concept", relevance=9)
    provider = StubProvider([triage, "# New Concept\n\nBody", touch])

    result, _ = asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
        )
    )

    assert result.articles_touched == 0
    assert target_path.read_text(encoding="utf-8") == original


def test_propagation_caps_touches_at_max_touches_choosing_highest_relevance(test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    for slug in ("target-a", "target-b", "target-c", "target-d"):
        _seed_existing_concept(
            test_config,
            slug=slug,
            title=slug.replace("-", " ").title(),
            body=f"# {slug}\n\n## Overview\n\nSummary for {slug}.\n",
        )

    triage = _triage_for_new_concept("New Concept", "new-concept", source.source_url)
    touch_response = json.dumps(
        {
            "touches": [
                {
                    "target_slug": "target-a",
                    "connection_line": "[[new-concept|New Concept]] — A relates.",
                    "relevance": 9,
                },
                {
                    "target_slug": "target-b",
                    "connection_line": "[[new-concept|New Concept]] — B relates.",
                    "relevance": 4,
                },
                {
                    "target_slug": "target-c",
                    "connection_line": "[[new-concept|New Concept]] — C relates.",
                    "relevance": 7,
                },
                {
                    "target_slug": "target-d",
                    "connection_line": "[[new-concept|New Concept]] — D relates.",
                    "relevance": 6,
                },
            ]
        }
    )
    provider = StubProvider(
        [triage, "# New Concept\n\nBody", touch_response]
    )

    result, _ = asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
            max_touches=2,
        )
    )

    assert result.articles_touched == 2
    concepts_dir = (
        test_config.vault_path / test_config.folders.wiki / test_config.folders.wiki_concepts
    )
    assert "[[new-concept|New Concept]]" in (concepts_dir / "target-a.md").read_text(encoding="utf-8")
    assert "[[new-concept|New Concept]]" in (concepts_dir / "target-c.md").read_text(encoding="utf-8")
    assert "[[new-concept|New Concept]]" not in (concepts_dir / "target-b.md").read_text(encoding="utf-8")
    assert "[[new-concept|New Concept]]" not in (concepts_dir / "target-d.md").read_text(encoding="utf-8")


def test_propagation_drops_touch_with_unknown_target_slug(test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    _seed_existing_concept(
        test_config,
        slug="existing-concept",
        title="Existing Concept",
        body="# Existing Concept\n\n## Overview\n\nSomething.\n",
    )

    triage = _triage_for_new_concept("New Concept", "new-concept", source.source_url)
    bad_touch = json.dumps(
        {
            "touches": [
                {
                    "target_slug": "not-in-catalog",
                    "connection_line": "[[new-concept|New Concept]] — unrelated.",
                    "relevance": 8,
                }
            ]
        }
    )
    provider = StubProvider([triage, "# New Concept\n\nBody", bad_touch])

    result, _ = asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
        )
    )

    assert result.articles_touched == 0
    assert result.errors == []


def test_propagation_drops_touch_without_wikilink_to_a_new_slug(test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    _seed_existing_concept(
        test_config,
        slug="existing-concept",
        title="Existing Concept",
        body="# Existing Concept\n\n## Overview\n\nSomething.\n",
    )

    triage = _triage_for_new_concept("New Concept", "new-concept", source.source_url)
    bad_touch = json.dumps(
        {
            "touches": [
                {
                    "target_slug": "existing-concept",
                    "connection_line": "[[unrelated-slug|Other]] — wrong target link.",
                    "relevance": 8,
                }
            ]
        }
    )
    provider = StubProvider([triage, "# New Concept\n\nBody", bad_touch])

    result, _ = asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
        )
    )

    assert result.articles_touched == 0


def test_propagation_inserts_connections_section_when_missing_before_open_questions(test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    target_path = _seed_existing_concept(
        test_config,
        slug="existing-concept",
        title="Existing Concept",
        body=(
            "# Existing Concept\n\n"
            "## Overview\n\n"
            "An overview.\n\n"
            "## Open Questions\n\n"
            "- What about X?\n\n"
            "## Sources\n\n"
            "- https://example.com/old-source\n"
        ),
        existing_sources=["https://example.com/old-source"],
    )

    triage = _triage_for_new_concept("New Concept", "new-concept", source.source_url)
    touch = _touch_response("existing-concept", "new-concept", relevance=8)
    provider = StubProvider([triage, "# New Concept\n\nBody", touch])

    result, _ = asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
        )
    )

    assert result.articles_touched == 1
    body = target_path.read_text(encoding="utf-8")
    connections_index = body.find("## Connections")
    open_questions_index = body.find("## Open Questions")
    sources_index = body.find("## Sources")
    assert connections_index != -1
    assert connections_index < open_questions_index < sources_index
    assert "[[new-concept|New Concept]]" in body


def test_propagation_inserts_connections_section_at_end_when_no_open_questions_or_sources(test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    target_path = _seed_existing_concept(
        test_config,
        slug="existing-concept",
        title="Existing Concept",
        body=(
            "# Existing Concept\n\n"
            "## Overview\n\n"
            "An overview.\n\n"
            "## Key Ideas\n\n"
            "- Idea one\n"
        ),
    )

    triage = _triage_for_new_concept("New Concept", "new-concept", source.source_url)
    touch = _touch_response("existing-concept", "new-concept", relevance=8)
    provider = StubProvider([triage, "# New Concept\n\nBody", touch])

    result, _ = asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
        )
    )

    assert result.articles_touched == 1
    body = target_path.read_text(encoding="utf-8")
    overview_index = body.find("## Overview")
    key_ideas_index = body.find("## Key Ideas")
    connections_index = body.find("## Connections")
    sources_index = body.find("## Sources")
    assert overview_index < key_ideas_index < connections_index < sources_index
    assert "[[new-concept|New Concept]]" in body
    assert "- https://example.com/raw-a" in body


def test_propagation_adds_source_url_to_frontmatter_and_sources_section_once(test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    target_path = _seed_existing_concept(
        test_config,
        slug="existing-concept",
        title="Existing Concept",
        body=(
            "# Existing Concept\n\n"
            "## Overview\n\n"
            "Body.\n"
        ),
    )

    triage = _triage_for_new_concept("New Concept", "new-concept", source.source_url)
    touch = _touch_response("existing-concept", "new-concept", relevance=8)
    provider = StubProvider([triage, "# New Concept\n\nBody", touch])

    result_one, _ = asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
        )
    )
    assert result_one.articles_touched == 1
    after_first = target_path.read_text(encoding="utf-8")
    # The line `- {url}` appears once in the frontmatter list and once in the
    # `## Sources` section.
    assert after_first.count(f"- {source.source_url}") == 2
    # And frontmatter contains exactly one sources list entry for this URL.
    _, _, body_first = after_first.partition("---\n\n")
    assert body_first.count(f"- {source.source_url}") == 1

    # Second compile — idempotency guard must skip touch.
    triage_two = _triage_for_new_concept("New Concept", "new-concept", source.source_url)
    touch_two = _touch_response("existing-concept", "new-concept", relevance=8)
    provider_two = StubProvider(
        [triage_two, "# New Concept\n\nBody", touch_two]
    )
    result_two, _ = asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider_two,
            test_config.vault_path,
            test_config.folders,
        )
    )
    assert result_two.articles_touched == 0
    after_second = target_path.read_text(encoding="utf-8")
    # Idempotent: should be byte-identical to the first run.
    assert after_second == after_first


def test_propagation_excludes_self_for_concepts_created_this_run(test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    # Pre-create a concept page with the SAME slug as the new concept the LLM
    # will return — this simulates a concept that was created earlier in the
    # same compile run.
    _seed_existing_concept(
        test_config,
        slug="new-concept",
        title="New Concept",
        body="# New Concept\n\n## Overview\n\nExisting.\n",
    )
    _seed_existing_concept(
        test_config,
        slug="other-concept",
        title="Other Concept",
        body="# Other Concept\n\n## Overview\n\nOther summary.\n",
    )

    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": "New Concept",
                    "status": "existing:new-concept",
                    "description": "Reusing the existing slug",
                    "source_urls": [source.source_url],
                    "merge_target": "new-concept",
                }
            ]
        }
    )
    # No touches needed — return empty list.
    touch = json.dumps({"touches": []})
    # update path runs the article-update prompt instead of create
    provider = StubProvider([triage, "# New Concept\n\nUpdated", touch])

    asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
        )
    )

    # The 3rd prompt (index 2) is the propagation call.
    assert len(provider.prompts) >= 3
    propagation_prompt = provider.prompts[2]
    # Catalog must contain other-concept but NOT new-concept.
    assert "other-concept |" in propagation_prompt
    assert "- new-concept |" not in propagation_prompt


def test_propagation_dry_run_makes_no_call_and_no_write(test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    target_path = _seed_existing_concept(
        test_config,
        slug="existing-concept",
        title="Existing Concept",
        body="# Existing Concept\n\n## Overview\n\nBody.\n",
    )
    original = target_path.read_text(encoding="utf-8")

    triage = _triage_for_new_concept("New Concept", "new-concept", source.source_url)
    provider = StubProvider([triage])  # ONLY triage — no touch response

    result, _ = asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
            dry_run=True,
            max_touches=5,
        )
    )

    assert result.articles_touched == 0
    assert provider.responses == []
    assert target_path.read_text(encoding="utf-8") == original


def test_propagation_zero_max_touches_disables_stage(test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    target_path = _seed_existing_concept(
        test_config,
        slug="existing-concept",
        title="Existing Concept",
        body="# Existing Concept\n\n## Overview\n\nBody.\n",
    )
    original = target_path.read_text(encoding="utf-8")

    triage = _triage_for_new_concept("New Concept", "new-concept", source.source_url)
    # Only triage + article create — no touch response.
    provider = StubProvider([triage, "# New Concept\n\nBody"])

    result, _ = asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
            max_touches=0,
        )
    )

    assert result.articles_touched == 0
    assert provider.responses == []
    assert target_path.read_text(encoding="utf-8") == original


def test_propagation_error_recorded_and_does_not_abort_other_sources(
    test_config, monkeypatch
):
    source_a = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    source_b = _raw_source(slug="raw-b", source_url="https://example.com/raw-b")
    _seed_existing_concept(
        test_config,
        slug="target-a",
        title="Target A",
        body="# Target A\n\n## Overview\n\nA body.\n",
    )
    target_b_path = _seed_existing_concept(
        test_config,
        slug="target-b",
        title="Target B",
        body="# Target B\n\n## Overview\n\nB body.\n",
    )

    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": "Concept A",
                    "status": "new",
                    "description": "A new",
                    "source_urls": [source_a.source_url],
                    "merge_target": None,
                },
                {
                    "name": "Concept B",
                    "status": "new",
                    "description": "B new",
                    "source_urls": [source_b.source_url],
                    "merge_target": None,
                },
            ]
        }
    )
    dedup = triage  # Same — no merge.
    touch_a = _touch_response("target-a", "concept-a", relevance=8)
    touch_b = _touch_response("target-b", "concept-b", relevance=8)
    provider = StubProvider(
        [
            triage,
            dedup,
            "# Concept A\n\nA body",
            "# Concept B\n\nB body",
            touch_a,
            touch_b,
        ]
    )

    # Monkeypatch write_markdown_page to fail ONCE — for target-a only.
    from vaultmind.ai import compiler as compiler_module

    original_write = compiler_module.write_markdown_page

    def flaky_write(path, *, body, frontmatter=None):
        if path.stem == "target-a":
            raise RuntimeError("boom")
        return original_write(path, body=body, frontmatter=frontmatter)

    monkeypatch.setattr(compiler_module, "write_markdown_page", flaky_write)

    result, _ = asyncio.run(
        compile_sources(
            [source_a, source_b],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
        )
    )

    assert result.articles_touched == 1
    assert result.propagation_failed_sources == {source_a.source_url}
    assert result.propagation_touches_by_source == {source_b.source_url: ["target-b"]}
    assert any(
        "Propagation touch on 'target-a' from 'https://example.com/raw-a' failed:" in err
        for err in result.errors
    )
    # target-b was written successfully.
    body_b = target_b_path.read_text(encoding="utf-8")
    assert "[[concept-b|New Concept]]" in body_b


def test_propagation_idempotency_ignores_wikilink_inside_code_block(test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    target_path = _seed_existing_concept(
        test_config,
        slug="existing-concept",
        title="Existing Concept",
        body=(
            "# Existing Concept\n\n"
            "## Connections\n\n"
            "Here is how you would write a wikilink:\n\n"
            "```markdown\n"
            "[[new-slug]] (this is doc syntax, not a real link)\n"
            "```\n"
        ),
    )

    triage = _triage_for_new_concept("New Slug", "new-slug", source.source_url)
    touch = _touch_response("existing-concept", "new-slug", relevance=8)
    provider = StubProvider([triage, "# New Slug\n\nBody", touch])

    result, _ = asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
        )
    )

    assert result.articles_touched == 1
    updated = target_path.read_text(encoding="utf-8")
    # The fenced code block must be preserved verbatim.
    assert "```markdown\n[[new-slug]] (this is doc syntax, not a real link)\n```" in updated
    # The real propagation bullet was appended exactly once (display text is
    # the touch_response helper's hard-coded "New Concept").
    bullet = "- [[new-slug|New Concept]] — relates to existing-concept."
    assert updated.count(bullet) == 1


def test_propagation_normalizes_target_slug_case_and_whitespace(test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    target_path = _seed_existing_concept(
        test_config,
        slug="target",
        title="Target",
        body="# Target\n\n## Overview\n\nTarget body.\n",
    )

    # Concept name slugifies to "new-slug" so the new_slugs set matches the
    # connection_line wikilink target below.
    triage = _triage_for_new_concept("New Slug", "new-slug", source.source_url)
    touch = json.dumps(
        {
            "touches": [
                {
                    "target_slug": " Target  ",
                    "connection_line": "[[new-slug|New]] — relates.",
                    "relevance": 7,
                }
            ]
        }
    )
    provider = StubProvider([triage, "# New Slug\n\nBody", touch])

    result, _ = asyncio.run(
        compile_sources(
            [source],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
        )
    )

    assert result.articles_touched == 1
    updated = target_path.read_text(encoding="utf-8")
    assert "- [[new-slug|New]] — relates." in updated


def test_compile_result_articles_touched_counts_distinct_targets(test_config):
    source_a = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    source_b = _raw_source(slug="raw-b", source_url="https://example.com/raw-b")
    _seed_existing_concept(
        test_config,
        slug="shared-target",
        title="Shared Target",
        body="# Shared Target\n\n## Overview\n\nShared body.\n",
    )

    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": "Concept A",
                    "status": "new",
                    "description": "A new",
                    "source_urls": [source_a.source_url],
                    "merge_target": None,
                },
                {
                    "name": "Concept B",
                    "status": "new",
                    "description": "B new",
                    "source_urls": [source_b.source_url],
                    "merge_target": None,
                },
            ]
        }
    )
    dedup = triage
    touch_a = _touch_response("shared-target", "concept-a", relevance=8)
    touch_b = _touch_response("shared-target", "concept-b", relevance=8)
    provider = StubProvider(
        [
            triage,
            dedup,
            "# Concept A\n\nA body",
            "# Concept B\n\nB body",
            touch_a,
            touch_b,
        ]
    )

    result, _ = asyncio.run(
        compile_sources(
            [source_a, source_b],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
        )
    )

    assert result.articles_touched == 1


def test_propagation_same_concept_multi_source_uses_current_disk_idempotency(test_config):
    source_a = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    source_b = _raw_source(slug="raw-b", source_url="https://example.com/raw-b")
    target_path = _seed_existing_concept(
        test_config,
        slug="shared-target",
        title="Shared Target",
        body="# Shared Target\n\n## Connections\n",
    )
    triage = json.dumps(
        {
            "concepts": [
                {
                    "name": "Shared New Concept",
                    "status": "new",
                    "description": "Shared by two sources",
                    "source_urls": [source_a.source_url, source_b.source_url],
                    "merge_target": None,
                }
            ]
        }
    )
    touch = _touch_response("shared-target", "shared-new-concept", relevance=8)
    provider = StubProvider([triage, "# Shared New Concept\n\nBody", touch, touch])

    result, _ = asyncio.run(
        compile_sources(
            [source_a, source_b],
            Manifest(),
            provider,
            test_config.vault_path,
            test_config.folders,
        )
    )

    updated = target_path.read_text(encoding="utf-8")
    assert result.articles_touched == 1
    assert updated.count("[[shared-new-concept|New Concept]]") == 1


def test_propagation_accepts_canonical_allowed_wikilink_forms(test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    target_path = _seed_existing_concept(
        test_config,
        slug="target",
        title="Target",
        body="# Target\n\n## Overview\n\nBody.\n",
    )
    triage = _triage_for_new_concept("New Concept", "new-concept", source.source_url)
    touch = json.dumps(
        {
            "touches": [
                {
                    "target_slug": "target",
                    "connection_line": (
                        "[[new-concept|alias]], [[Wiki/Concepts/new-concept#Heading]], "
                        "[[new-concept^block]], and [[new-concept.md]] are related."
                    ),
                    "relevance": 8,
                }
            ]
        }
    )
    provider = StubProvider([triage, "# New Concept\n\nBody", touch])

    result, _ = asyncio.run(
        compile_sources(
            [source], Manifest(), provider, test_config.vault_path, test_config.folders
        )
    )

    assert result.articles_touched == 1
    assert "[[Wiki/Concepts/new-concept#Heading]]" in target_path.read_text(encoding="utf-8")


def test_propagation_rejects_line_with_any_unauthorized_wikilink(test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    target_path = _seed_existing_concept(
        test_config,
        slug="target",
        title="Target",
        body="# Target\n\n## Overview\n\nBody.\n",
    )
    original = target_path.read_text(encoding="utf-8")
    triage = _triage_for_new_concept("New Concept", "new-concept", source.source_url)
    touch = json.dumps(
        {
            "touches": [
                {
                    "target_slug": "target",
                    "connection_line": (
                        "[[new-concept|allowed]] but also "
                        "[[Wiki/Concepts/unauthorized.md#Heading|not allowed]]."
                    ),
                    "relevance": 8,
                }
            ]
        }
    )
    provider = StubProvider([triage, "# New Concept\n\nBody", touch])

    result, _ = asyncio.run(
        compile_sources(
            [source], Manifest(), provider, test_config.vault_path, test_config.folders
        )
    )

    assert result.articles_touched == 0
    assert target_path.read_text(encoding="utf-8") == original


def test_propagation_call_failure_marks_source_for_retry(test_config):
    source = _raw_source(slug="raw-a", source_url="https://example.com/raw-a")
    _seed_existing_concept(
        test_config,
        slug="target",
        title="Target",
        body="# Target\n\n## Overview\n\nBody.\n",
    )
    triage = _triage_for_new_concept("New Concept", "new-concept", source.source_url)

    class FailingPropagationProvider(StubProvider):
        async def complete(self, prompt: str, system: str = "") -> str:
            if len(self.prompts) == 2:
                self.prompts.append(prompt)
                raise RuntimeError("provider unavailable")
            return await super().complete(prompt, system)

    provider = FailingPropagationProvider([triage, "# New Concept\n\nBody"])

    result, _ = asyncio.run(
        compile_sources(
            [source], Manifest(), provider, test_config.vault_path, test_config.folders
        )
    )

    assert result.propagation_failed_sources == {source.source_url}
    assert any("Propagation call" in error for error in result.errors)
