"""Tests for the vm ask compound-interest engine."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from vaultmind.ai.asker import (
    AskResult,
    GatheredContext,
    _build_context_text,
    _extract_answer_text,
    _extract_gaps_from_assessment,
    _follow_up_gap,
    _initial_search,
    _render_answer_markdown,
    _slug_from_question,
    ask_question,
)
from vaultmind.core.vault_index import VaultNoteRecord


class StubProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []
        self.model = "stub-model"

    async def complete(self, prompt: str, system: str = "") -> str:
        del system
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else ""


def _wiki_note(title: str, path_slug: str, body: str = "Sample body content") -> VaultNoteRecord:
    return VaultNoteRecord(
        path=Path(f"/tmp/{path_slug}.md"),
        relative_path=f"🗺️ Wiki/🧠 Concepts/{path_slug}",
        title=title,
        saved_at=datetime.now(UTC),
        tags=["ai"],
        source_type="concept",
        rating=None,
        read_time_minutes=None,
        status=None,
        canonical_url=None,
        source=None,
        vaultmind=True,
        body=body,
        summary="Summary text",
        raw_frontmatter={"title": title, "vaultmind": True},
    )


class TestSlugFromQuestion:
    def test_basic_slug(self):
        assert _slug_from_question("What is attention?") == "what-is-attention"

    def test_strips_punctuation(self):
        assert _slug_from_question('"How does RLHF work?"') == "how-does-rlhf-work"

    def test_handles_long_questions(self):
        long_q = " ".join(["word"] * 30)
        slug = _slug_from_question(long_q)
        assert len(slug) <= 64  # slugify caps at 80, but _slug_from_question further limits

    def test_fallback_when_empty(self):
        assert _slug_from_question("---") == "untitled"


class TestBuildContextText:
    def test_empty_context(self):
        ctx = GatheredContext(wiki_notes=[], raw_sources=[])
        result = _build_context_text("test", ctx)
        assert "No relevant notes" in result

    def test_wiki_notes_included(self):
        note = _wiki_note("Attention", "attention")
        ctx = GatheredContext(wiki_notes=[note], raw_sources=[])
        result = _build_context_text("attention", ctx)
        assert "📚 Wiki Articles" in result
        assert "Attention" in result

    def test_context_caps_at_max_chars(self):
        long_body = "x" * 5000
        note = _wiki_note("Title", "note", body=long_body)
        ctx = GatheredContext(wiki_notes=[note], raw_sources=[])
        result = _build_context_text("test", ctx)
        assert len(result) < 50000  # Should be truncated


class TestExtractAnswerText:
    def test_parses_json_answer(self):
        response = '{"answer": "Attention is a mechanism."}'
        assert _extract_answer_text(response) == "Attention is a mechanism."

    def test_strips_code_fences(self):
        response = '```json\n{"answer": "Test answer."}\n```'
        # The code successfully parses JSON even inside code fences
        assert _extract_answer_text(response) == "Test answer."

    def test_fallback_on_non_json(self):
        response = "This is a plain text answer."
        assert _extract_answer_text(response) == "This is a plain text answer."


class TestExtractGapsFromAssessment:
    def test_parses_valid_gaps_json(self):
        response = '{"gaps": ["gap one", "gap two"]}'
        assert _extract_gaps_from_assessment(response) == ["gap one", "gap two"]

    def test_empty_gaps_list(self):
        response = '{"gaps": []}'
        assert _extract_gaps_from_assessment(response) == []

    def test_filters_non_string_gaps(self):
        response = '{"gaps": ["valid", 123, null, "also valid"]}'
        assert _extract_gaps_from_assessment(response) == ["valid", "also valid"]

    def test_invalid_json_returns_empty(self):
        response = "not json at all"
        assert _extract_gaps_from_assessment(response) == []


class TestRenderAnswerMarkdown:
    def test_includes_question(self):
        body = _render_answer_markdown(
            "What is attention?",
            "It is a mechanism.",
            supporting_notes=[],
            supporting_sources=[],
            iterations=1,
            now=datetime.now(UTC),
        )
        assert "# What is attention?" in body
        assert "## Answer" in body
        assert "It is a mechanism." in body

    def test_supporting_notes_section(self):
        body = _render_answer_markdown(
            "Question?",
            "Answer.",
            supporting_notes=["path/to/note1", "path/to/note2"],
            supporting_sources=[],
            iterations=1,
            now=datetime.now(UTC),
        )
        assert "## Supporting Wiki Pages" in body
        assert "[[path/to/note1]]" in body
        assert "[[path/to/note2]]" in body

    def test_supporting_sources_section(self):
        body = _render_answer_markdown(
            "Question?",
            "Answer.",
            supporting_notes=[],
            supporting_sources=["https://example.com/article"],
            iterations=1,
            now=datetime.now(UTC),
        )
        assert "## Supporting Raw Sources" in body
        assert "https://example.com/article" in body

    def test_iterations_in_footer(self):
        body = _render_answer_markdown(
            "Q", "A", supporting_notes=[], supporting_sources=[], iterations=3, now=datetime.now(UTC)
        )
        assert "3 iteration(s)" in body


class TestFollowUpGap:
    def test_finds_matching_note(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        wiki_dir = vault / "🗺️ Wiki" / "🧠 Concepts"
        wiki_dir.mkdir(parents=True)

        note_file = wiki_dir / "attention.md"
        note_file.write_text("---\ntitle: Attention\n---\n\n# Attention\n\nContent about attention mechanisms.")

        ctx = GatheredContext(wiki_notes=[], raw_sources=[])
        _follow_up_gap("attention", ctx, vault, "🗺️ Wiki", "🧠 Concepts", "📥 Raw")
        assert len(ctx.wiki_notes) == 1
        assert ctx.wiki_notes[0].title == "Attention"

    def test_skips_existing_notes_by_title(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        wiki_dir = vault / "🗺️ Wiki" / "🧠 Concepts"
        wiki_dir.mkdir(parents=True)

        note_file = wiki_dir / "attention.md"
        note_file.write_text("---\ntitle: Attention\n---\n\n# Attention\n\nContent.")

        # Same title but different path (simulating an already-tracked note)
        existing = _wiki_note("Attention", "existing-attention")
        ctx = GatheredContext(wiki_notes=[existing], raw_sources=[])
        _follow_up_gap("attention", ctx, vault, "🗺️ Wiki", "🧠 Concepts", "📥 Raw")
        # Gap found a note with same title "Attention" — deduplication prevents adding a second
        titles = [n.title for n in ctx.wiki_notes]
        assert titles.count("Attention") == 1

    def test_searches_raw_when_only_strong_wiki_match_is_already_gathered(
        self, tmp_path: Path
    ):
        vault = tmp_path / "vault"
        wiki_dir = vault / "Wiki" / "Concepts"
        raw_dir = vault / "Raw"
        wiki_dir.mkdir(parents=True)
        raw_dir.mkdir(parents=True)
        (wiki_dir / "capacity.md").write_text(
            "---\ntitle: Capacity\nvaultmind: true\n---\n\n# Capacity\n\nCapacity overview.",
            encoding="utf-8",
        )
        (raw_dir / "capacity-details.md").write_text(
            "# Capacity details\n\nCapacity evidence missing from the overview.", encoding="utf-8"
        )
        context = _initial_search("capacity", vault, "Wiki", "Concepts", "Raw", "Queries")
        assert [note.title for note in context.wiki_notes] == ["Capacity"]
        assert context.raw_sources == []

        _follow_up_gap("capacity", context, vault, "Wiki", "Concepts", "Raw", "Queries")

        assert [source.title for source in context.raw_sources] == ["Capacity details"]

    def test_new_strong_wiki_match_skips_raw(self, tmp_path: Path):
        vault = tmp_path / "vault"
        wiki_dir = vault / "Wiki" / "Concepts"
        raw_dir = vault / "Raw"
        wiki_dir.mkdir(parents=True)
        raw_dir.mkdir(parents=True)
        (wiki_dir / "capacity.md").write_text(
            "---\ntitle: Capacity\nvaultmind: true\n---\n\n# Capacity\n\nCapacity overview.",
            encoding="utf-8",
        )
        (raw_dir / "capacity.md").write_text(
            "# Capacity raw\n\nCapacity source evidence.", encoding="utf-8"
        )
        context = GatheredContext()

        _follow_up_gap("capacity", context, vault, "Wiki", "Concepts", "Raw", "Queries")

        assert [note.title for note in context.wiki_notes] == ["Capacity"]
        assert context.raw_sources == []


class TestInitialSearch:
    def test_falls_back_to_raw_when_wiki_has_no_matches(self, tmp_path: Path):
        vault = tmp_path / "vault"
        wiki_dir = vault / "🗺️ Wiki" / "🧠 Concepts"
        raw_dir = vault / "📥 Raw"
        wiki_dir.mkdir(parents=True)
        raw_dir.mkdir(parents=True)

        (wiki_dir / "attention.md").write_text(
            "---\ntitle: Attention\nvaultmind: true\nkind: concept\n---\n\n# Attention\n\nTransformer attention notes.",
            encoding="utf-8",
        )
        (raw_dir / "rlhf.md").write_text(
            "---\nsource: https://example.com/rlhf\n---\n\n# RLHF\n\nRLHF uses human feedback.",
            encoding="utf-8",
        )

        ctx = _initial_search("What is RLHF?", vault, "🗺️ Wiki", "🧠 Concepts", "📥 Raw")

        assert ctx.wiki_notes == []
        assert len(ctx.raw_sources) == 1
        assert ctx.raw_sources[0].title == "RLHF"

    def test_skips_raw_when_wiki_match_is_strong(self, tmp_path: Path):
        vault = tmp_path / "vault"
        wiki_dir = vault / "🗺️ Wiki" / "🧠 Concepts"
        raw_dir = vault / "📥 Raw"
        wiki_dir.mkdir(parents=True)
        raw_dir.mkdir(parents=True)

        (wiki_dir / "rlhf.md").write_text(
            "---\ntitle: RLHF\nvaultmind: true\nkind: concept\n---\n\n# RLHF\n\nRLHF notes.",
            encoding="utf-8",
        )
        (raw_dir / "rlhf-source.md").write_text(
            "# RLHF source\n\nRLHF uses human feedback.",
            encoding="utf-8",
        )

        ctx = _initial_search("RLHF", vault, "🗺️ Wiki", "🧠 Concepts", "📥 Raw")

        assert [note.title for note in ctx.wiki_notes] == ["RLHF"]
        assert ctx.raw_sources == []

    def test_searches_filed_queries_and_uses_combined_wiki_strength(self, tmp_path: Path):
        vault = tmp_path / "vault"
        query_dir = vault / "🗺️ Wiki" / "📊 Queries"
        raw_dir = vault / "📥 Raw"
        query_dir.mkdir(parents=True)
        raw_dir.mkdir(parents=True)
        (query_dir / "dpo.md").write_text(
            "---\ntitle: DPO\nvaultmind: true\nkind: query\ncreated: 2026-01-01T00:00:00+00:00\n---\n\n"
            "# DPO\n\n## Answer\nDPO directly optimizes preferences.",
            encoding="utf-8",
        )
        (raw_dir / "dpo.md").write_text("# DPO raw\n\nDPO source.", encoding="utf-8")

        ctx = _initial_search(
            "DPO", vault, "🗺️ Wiki", "🧠 Concepts", "📥 Raw", "📊 Queries"
        )

        assert [note.source_type for note in ctx.wiki_notes] == ["query"]
        assert ctx.wiki_notes[0].relative_path == "🗺️ Wiki/📊 Queries/dpo"
        assert ctx.raw_sources == []


class TestAskResult:
    def test_ask_result_dataclass(self):
        result = AskResult(
            answer="Test answer.",
            slug="test-slug",
            path=Path("/tmp/test-slug.md"),
            iterations=2,
            gaps=["gap one"],
        )
        assert result.answer == "Test answer."
        assert result.slug == "test-slug"
        assert result.iterations == 2
        assert result.gaps == ["gap one"]


def test_ask_preview_does_not_write_query_file(tmp_path: Path):
    vault = tmp_path / "vault"
    raw_dir = vault / "📥 Raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "rlhf.md").write_text("# RLHF\n\nRLHF uses human feedback.", encoding="utf-8")

    provider = StubProvider(['{"answer": "RLHF uses human feedback.", "gaps": []}'])

    result = asyncio.run(
        ask_question(
            question="What is RLHF?",
            provider=provider,
            vault_path=vault,
            folders_wiki="🗺️ Wiki",
            folders_wiki_concepts="🧠 Concepts",
            folders_wiki_queries="📊 Queries",
            folders_raw="📥 Raw",
            depth="shallow",
            file_answer=False,
        )
    )

    assert result.answer == "RLHF uses human feedback."
    assert not result.path.exists()
    assert not (vault / "🗺️ Wiki").exists()
    assert not (vault / "vault.manifest.json").exists()
    assert "RLHF uses human feedback" in provider.prompts[0]


def test_deep_ask_runs_three_syntheses_and_returns_latest_gaps(tmp_path: Path):
    vault = tmp_path / "vault"
    raw_dir = vault / "📥 Raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "alpha.md").write_text("# Alpha\n\nalpha evidence", encoding="utf-8")
    (raw_dir / "beta.md").write_text("# Beta\n\nbeta evidence", encoding="utf-8")
    provider = StubProvider([
        '{"answer": "first"}',
        '{"gaps": ["alpha evidence"]}',
        '{"answer": "second"}',
        '{"gaps": ["beta evidence"]}',
        '{"answer": "third"}',
        '{"gaps": ["fresh final gap"]}',
    ])

    result = asyncio.run(ask_question(
        "Uncovered topic",
        provider,
        vault,
        "🗺️ Wiki",
        "🧠 Concepts",
        "📊 Queries",
        "📥 Raw",
        depth="deep",
        file_answer=False,
    ))

    assert result.answer == "third"
    assert result.iterations == 3
    assert result.gaps == ["fresh final gap"]
    assert len(provider.prompts) == 6
    assert "alpha evidence" in provider.prompts[2]
    assert "alpha evidence" in provider.prompts[4]
    assert "beta evidence" in provider.prompts[4]


def test_deep_ask_stops_after_latest_assessment_reports_no_gaps(tmp_path: Path):
    provider = StubProvider(['{"answer": "complete"}', '{"gaps": []}'])

    result = asyncio.run(ask_question(
        "Question",
        provider,
        tmp_path,
        "Wiki",
        "Concepts",
        "Queries",
        "Raw",
        depth="deep",
        file_answer=False,
    ))

    assert result.iterations == 1
    assert result.gaps == []
    assert len(provider.prompts) == 2


def test_follow_up_context_is_deduplicated_and_capped(tmp_path: Path):
    vault = tmp_path / "vault"
    concepts = vault / "Wiki" / "Concepts"
    raw = vault / "Raw"
    concepts.mkdir(parents=True)
    raw.mkdir(parents=True)
    for index in range(35):
        (concepts / f"note-{index}.md").write_text(
            f"---\ntitle: Note {index}\nvaultmind: true\nkind: concept\n---\n\n"
            f"# Note {index}\n\ncapacity evidence {index}",
            encoding="utf-8",
        )
    for index in range(25):
        (raw / f"source-{index}.md").write_text(
            f"# Source {index}\n\ncapacity raw evidence {index}", encoding="utf-8"
        )

    context = GatheredContext()
    _follow_up_gap("capacity", context, vault, "Wiki", "Concepts", "Raw", "Queries")
    _follow_up_gap("capacity", context, vault, "Wiki", "Concepts", "Raw", "Queries")

    assert len(context.wiki_notes) == 30
    assert len(context.raw_sources) == 20
    assert len({note.relative_path for note in context.wiki_notes}) == 30
    assert len({source.relative_path for source in context.raw_sources}) == 20


def test_query_page_contract_and_final_gap_rendering():
    body = _render_answer_markdown(
        "Question?",
        "# Unsafe provider heading\n\nAnswer.",
        supporting_notes=["Wiki/Concepts/one", "Wiki/Queries/two"],
        supporting_sources=["Raw/source"],
        iterations=3,
        now=datetime(2026, 1, 1, tzinfo=UTC),
        gaps=["First gap", "Second gap"],
    )

    assert [line for line in body.splitlines() if line.startswith("# ")] == ["# Question?"]
    assert [line for line in body.splitlines() if line.startswith("## ")] == [
        "## Answer",
        "## Supporting Wiki Pages",
        "## Supporting Raw Sources",
        "## Follow-up Questions",
    ]
    assert "- [[Wiki/Concepts/one]]" in body
    assert "- [[Wiki/Queries/two]]" in body
    assert "- First gap\n- Second gap" in body


def test_query_page_sanitizes_setext_headings_but_preserves_fenced_examples():
    answer = """Provider title
==============

Answer text.

Provider section
----------------

```markdown
Fenced title
============
# Fenced ATX title
```

~~~markdown
Fenced section
--------------
## Fenced ATX section
~~~"""

    body = _render_answer_markdown(
        "Question?",
        answer,
        supporting_notes=[],
        supporting_sources=[],
        iterations=1,
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert "### Provider title" in body
    assert "### Provider section" in body
    assert "Provider title\n==============" not in body
    assert "Provider section\n----------------" not in body
    assert "Fenced title\n============\n# Fenced ATX title" in body
    assert "Fenced section\n--------------\n## Fenced ATX section" in body
    assert "Answer text." in body


def test_query_page_fences_follow_commonmark_marker_and_closer_rules():
    answer = """~~~`markdown`
# Keep in tilde fence with backtick info
~~~
~~~ `markdown`
## Keep in tilde fence with spaced backtick info
~~~
# Demote after tilde fence

````markdown
# Keep before shorter pseudo-closer
```
# Keep after shorter pseudo-closer
~~~~
# Keep after mismatched marker
````
# Demote after backtick fence

~~~markdown
# Keep before trailing-content pseudo-closer
~~~ not-a-closer
## Keep after trailing-content pseudo-closer
~~~
## Demote after final fence

```markdown `invalid-info`
# Demote after invalid backtick opener"""

    body = _render_answer_markdown(
        "Question?",
        answer,
        supporting_notes=[],
        supporting_sources=[],
        iterations=1,
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert "# Keep in tilde fence with backtick info" in body
    assert "## Keep in tilde fence with spaced backtick info" in body
    assert "### Demote after tilde fence" in body
    assert "# Keep after shorter pseudo-closer" in body
    assert "# Keep after mismatched marker" in body
    assert "### Demote after backtick fence" in body
    assert "## Keep after trailing-content pseudo-closer" in body
    assert "### Demote after final fence" in body
    assert "### Demote after invalid backtick opener" in body


def test_query_page_uses_explicit_no_gaps_marker():
    body = _render_answer_markdown(
        "Question?",
        "Answer.",
        supporting_notes=[],
        supporting_sources=[],
        iterations=1,
        now=datetime(2026, 1, 1, tzinfo=UTC),
        gaps=[],
    )
    assert "## Follow-up Questions\n*No follow-up questions.*" in body


def test_filed_answer_is_reused_by_a_later_question(tmp_path: Path):
    vault = tmp_path / "vault"
    first_provider = StubProvider(['{"answer": "RLHF alignment uses preference feedback."}'])
    first = asyncio.run(ask_question(
        "How does RLHF improve alignment?",
        first_provider,
        vault,
        "Wiki",
        "Concepts",
        "Queries",
        "Raw",
        depth="shallow",
    ))
    second_provider = StubProvider(['{"answer": "It reuses filed knowledge."}'])

    second = asyncio.run(ask_question(
        "RLHF alignment",
        second_provider,
        vault,
        "Wiki",
        "Concepts",
        "Queries",
        "Raw",
        depth="shallow",
    ))

    assert "RLHF alignment uses preference feedback." in second_provider.prompts[0]
    second_text = second.path.read_text(encoding="utf-8")
    assert f"[[Wiki/Queries/{first.slug}]]" in second_text


@pytest.mark.parametrize("question", ["", "   ", "First line\nSecond line", "First\rSecond"])
def test_ask_rejects_questions_that_cannot_be_an_exact_h1(
    tmp_path: Path,
    question: str,
):
    provider = StubProvider(['{"answer": "must not run"}'])

    with pytest.raises(ValueError):
        asyncio.run(
            ask_question(
                question,
                provider,
                tmp_path / "vault",
                "Wiki",
                "Concepts",
                "Queries",
                "Raw",
                depth="shallow",
            )
        )

    assert provider.prompts == []
    assert not (tmp_path / "vault").exists()


def test_repeated_query_refresh_excludes_its_own_page_from_support(tmp_path: Path):
    vault = tmp_path / "vault"
    first_provider = StubProvider(['{"answer": "Original evidence."}'])
    first = asyncio.run(
        ask_question(
            "What is RLHF?",
            first_provider,
            vault,
            "Wiki",
            "Concepts",
            "Queries",
            "Raw",
            depth="shallow",
        )
    )
    first_relative_path = f"Wiki/Queries/{first.slug}"

    refresh_provider = StubProvider(['{"answer": "Refreshed answer."}'])
    refreshed = asyncio.run(
        ask_question(
            "What is RLHF?",
            refresh_provider,
            vault,
            "Wiki",
            "Concepts",
            "Queries",
            "Raw",
            depth="shallow",
        )
    )

    assert "Original evidence." not in refresh_provider.prompts[0]
    refreshed_text = refreshed.path.read_text(encoding="utf-8")
    assert f"[[{first_relative_path}]]" not in refreshed_text
    assert "Refreshed answer." in refreshed_text


def test_ask_files_query_when_enabled(tmp_path: Path):
    vault = tmp_path / "vault"
    wiki_dir = vault / "🗺️ Wiki" / "🧠 Concepts"
    wiki_dir.mkdir(parents=True)
    (wiki_dir / "rlhf.md").write_text(
        "---\ntitle: RLHF\nvaultmind: true\nkind: concept\n---\n\n# RLHF\n\nRLHF notes.",
        encoding="utf-8",
    )

    provider = StubProvider(['{"answer": "RLHF notes answer.", "gaps": []}'])

    result = asyncio.run(
        ask_question(
            question="What is RLHF?",
            provider=provider,
            vault_path=vault,
            folders_wiki="🗺️ Wiki",
            folders_wiki_concepts="🧠 Concepts",
            folders_wiki_queries="📊 Queries",
            folders_raw="📥 Raw",
            depth="shallow",
            file_answer=True,
        )
    )

    assert result.path.exists()
    text = result.path.read_text(encoding="utf-8")
    assert "kind: query" in text
    assert "RLHF notes answer." in text
