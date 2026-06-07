"""Vault writer — atomic markdown writes to the Obsidian vault."""

from __future__ import annotations

import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import structlog
import yaml

log = structlog.get_logger()

MAX_FILENAME_LENGTH = 80


def slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    # Remove emoji and special characters
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    text = text.strip("-")
    return text[:MAX_FILENAME_LENGTH] if text else "untitled"


def write_markdown_page(
    path: Path,
    *,
    body: str,
    frontmatter: dict[str, Any] | None = None,
) -> Path:
    """Atomically write a generic markdown page with optional frontmatter."""
    path.parent.mkdir(parents=True, exist_ok=True)

    content_body = body.rstrip() + "\n"
    if frontmatter is not None:
        fm = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True, sort_keys=False)
        content = f"---\n{fm}---\n\n{content_body}"
    else:
        content = content_body

    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".md.tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(content)
        Path(tmp_path).replace(path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    log.info("markdown_page_written", path=str(path))
    return path


def parse_frontmatter(file_path: Path) -> dict[str, Any] | None:
    """Parse YAML frontmatter from a markdown file. Returns None on failure."""
    try:
        text = file_path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            return None
        end = text.find("---", 3)
        if end == -1:
            return None
        fm_data = yaml.safe_load(text[3:end])
        return fm_data if isinstance(fm_data, dict) else None
    except (OSError, yaml.YAMLError):
        return None
