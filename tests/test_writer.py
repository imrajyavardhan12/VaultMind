"""Tests for the vault markdown writer."""

from __future__ import annotations

from pathlib import Path

from vaultmind.core.writer import parse_frontmatter, slugify, write_markdown_page


def test_slugify():
    assert slugify("The Attention Economy Is Broken") == "the-attention-economy-is-broken"
    assert slugify("Hello! World? 🌍") == "hello-world"
    assert slugify("") == "untitled"


def test_slugify_max_length():
    long_title = "a" * 200
    slug = slugify(long_title)
    assert len(slug) <= 80


def test_write_markdown_page_roundtrip(tmp_path: Path):
    path = tmp_path / "sub" / "page.md"
    written = write_markdown_page(
        path,
        body="# Title\n\nBody text.",
        frontmatter={"title": "Title", "vaultmind": True, "kind": "concept"},
    )
    assert written == path
    content = path.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert "kind: concept" in content
    assert "# Title" in content


def test_write_markdown_page_no_frontmatter(tmp_path: Path):
    path = tmp_path / "plain.md"
    write_markdown_page(path, body="just a body")
    content = path.read_text(encoding="utf-8")
    assert not content.startswith("---")
    assert content.strip() == "just a body"


def test_parse_frontmatter(tmp_path: Path):
    path = tmp_path / "fm.md"
    path.write_text("---\ntitle: Hello\nkind: query\n---\n\nbody", encoding="utf-8")
    fm = parse_frontmatter(path)
    assert fm == {"title": "Hello", "kind": "query"}


def test_parse_frontmatter_none_when_absent(tmp_path: Path):
    path = tmp_path / "nofm.md"
    path.write_text("no frontmatter here", encoding="utf-8")
    assert parse_frontmatter(path) is None
