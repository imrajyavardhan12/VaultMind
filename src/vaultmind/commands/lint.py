"""vm lint — deterministic wiki-health report (zero-LLM).

Usage:
    vm lint              # scan vault and write lint-YYYY-MM-DD.md to 📋 Inbox/
    vm lint --preview    # print report without writing
    vm lint --strict     # exit non-zero if any ERROR findings
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog
import typer
import yaml
from rich.panel import Panel
from rich.table import Table

from vaultmind.config import AppConfig, load_config
from vaultmind.core.linter import (
    ConceptPage,
    WikiPage,
    lint_vault,
    render_lint_markdown,
)
from vaultmind.core.manifest import read_manifest
from vaultmind.core.raw_scanner import scan_raw_sources
from vaultmind.core.writer import write_markdown_page
from vaultmind.utils.display import console, print_info
from vaultmind.utils.logging import setup_logging

log = structlog.get_logger()


def lint(
    preview: bool = typer.Option(False, "--preview", help="Print the report without writing it"),
    strict: bool = typer.Option(False, "--strict", help="Exit non-zero if any error-severity findings"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """Scan the vault and generate a wiki-health report.

    The report is written to 🗺️ Wiki/📋 Inbox/lint-YYYY-MM-DD.md unless --preview is set.
    Use --strict to exit non-zero if any ERROR findings exist.
    """
    setup_logging(verbose=verbose)
    config = load_config()

    # Gather materials from disk
    raw_sources = scan_raw_sources(config)
    manifest = read_manifest(config.vault_path)

    # Build concept_pages from 🗺️ Wiki/🧠 Concepts/*.md
    concept_pages = _scan_concept_pages(config)

    # Build wiki_pages from all .md under 🗺️ Wiki/
    wiki_pages = _scan_wiki_pages(config)

    # Get index text
    index_path = (
        config.vault_path
        / config.folders.wiki
        / f"{config.folders.wiki_index}.md"
    )
    index_text = index_path.read_text(encoding="utf-8") if index_path.exists() else ""

    # Build valid_targets (lowercased stems of ALL .md files in vault)
    valid_targets = _scan_all_targets(config)

    # Build concept_slugs (lowercased stems of concept files)
    concept_slugs = {page.slug for page in concept_pages}

    # Run lint
    report = lint_vault(
        raw_sources=raw_sources,
        manifest=manifest,
        concept_pages=concept_pages,
        wiki_pages=wiki_pages,
        index_text=index_text,
        valid_targets=valid_targets,
        concept_slugs=concept_slugs,
    )

    # Render terminal summary
    _render_terminal_summary(report)

    # Get markdown body
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    markdown_body = render_lint_markdown(report, date_label=today)

    if preview:
        # Print the markdown body without writing
        console.print(markdown_body)
    else:
        # Write to vault
        inbox_dir = (
            config.vault_path
            / config.folders.wiki
            / config.folders.wiki_inbox
        )
        inbox_dir.mkdir(parents=True, exist_ok=True)

        lint_path = inbox_dir / f"lint-{today}.md"
        frontmatter = {
            "title": f"Lint Report {today}",
            "vaultmind": True,
            "kind": "lint",
            "created": datetime.now(UTC).isoformat(),
        }

        write_markdown_page(lint_path, body=markdown_body, frontmatter=frontmatter)
        print_info(f"Lint report written to {lint_path.relative_to(config.vault_path)}")

    # Exit with error code if strict mode and has errors
    if strict and report.has_errors:
        raise typer.Exit(1)


def _scan_concept_pages(config: AppConfig) -> list[ConceptPage]:
    """Scan concept pages from 🗺️ Wiki/🧠 Concepts/*.md."""
    concepts_dir = (
        config.vault_path
        / config.folders.wiki
        / config.folders.wiki_concepts
    )

    concept_pages: list[ConceptPage] = []

    if not concepts_dir.exists():
        return concept_pages

    for md_file in concepts_dir.glob("*.md"):
        try:
            text = md_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        # Parse frontmatter
        frontmatter = _parse_frontmatter(text)

        slug = md_file.stem.lower()
        title = frontmatter.get("title", slug.replace("-", " ").title())
        sources = _coerce_sources_list(frontmatter.get("sources", []))
        relative_path = md_file.relative_to(config.vault_path).with_suffix("").as_posix()

        concept_pages.append(
            ConceptPage(
                slug=slug,
                title=title,
                relative_path=relative_path,
                sources=sources,
                text=text,
            )
        )

    return concept_pages


def _scan_wiki_pages(config: AppConfig) -> list[WikiPage]:
    """Scan wiki pages from 🗺️ Wiki/**/*.md, excluding generated Inbox reports.

    Reports written to 📋 Inbox/ are VaultMind output, not authored wiki content.
    Scanning them would let a lint report's own ``[[...]]`` finding text pollute
    the next run (and falsely flag the report as having broken wikilinks).
    """
    wiki_dir = config.vault_path / config.folders.wiki
    wiki_index_path = wiki_dir / f"{config.folders.wiki_index}.md"
    inbox_dir = wiki_dir / config.folders.wiki_inbox

    wiki_pages: list[WikiPage] = []

    if not wiki_dir.exists():
        return wiki_pages

    for md_file in wiki_dir.rglob("*.md"):
        if md_file.is_relative_to(inbox_dir):
            continue
        try:
            text = md_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        relative_path = md_file.relative_to(config.vault_path).with_suffix("").as_posix()
        is_index = md_file == wiki_index_path

        wiki_pages.append(
            WikiPage(
                relative_path=relative_path,
                text=text,
                is_index=is_index,
            )
        )

    return wiki_pages


def _scan_all_targets(config: AppConfig) -> set[str]:
    """Get lowercased stems of all .md files in the vault, excluding Inbox reports.

    Generated lint reports are not link destinations; excluding them keeps the
    valid-target set stable across runs.
    """
    inbox_dir = config.vault_path / config.folders.wiki / config.folders.wiki_inbox
    targets = set()

    for md_file in config.vault_path.rglob("*.md"):
        if md_file.is_relative_to(inbox_dir):
            continue
        targets.add(md_file.stem.lower())

    return targets


def _parse_frontmatter(text: str) -> dict[str, Any]:
    """Parse YAML frontmatter from markdown text."""
    if not text.startswith("---"):
        return {}

    end = text.find("---", 3)
    if end == -1:
        return {}

    fm_text = text[3:end].strip()
    if not fm_text:
        return {}

    try:
        data = yaml.safe_load(fm_text)
        return data if isinstance(data, dict) else {}
    except yaml.YAMLError:
        return {}


def _coerce_sources_list(value: object) -> list[str]:
    """Coerce a frontmatter sources value to list[str]."""
    if isinstance(value, list):
        return [str(v).strip() for v in value if isinstance(v, str) and v.strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []


def _render_terminal_summary(report: Any) -> None:
    """Render a rich terminal summary of the lint report."""
    # Summary panel
    summary_lines = [
        f"Total Issues: {report.total}",
        f"Errors: {report.error_count}",
        f"Warnings: {report.warning_count}",
        f"Info: {report.info_count}",
    ]

    console.print(
        Panel(
            "\n".join(summary_lines),
            title="📋 Lint Summary",
            border_style="cyan" if report.error_count == 0 else "red",
        )
    )

    if not report.findings:
        return

    # Findings table
    table = Table(title="Findings")
    table.add_column("Severity", justify="left")
    table.add_column("Code", justify="left")
    table.add_column("Subject", justify="left")
    table.add_column("Message", justify="left")

    # Cap at ~40 rows for readability
    for finding in report.findings[:40]:
        severity_color = {
            "error": "red",
            "warning": "yellow",
            "info": "dim",
        }.get(finding.severity, "dim")

        severity_str = f"[{severity_color}]{finding.severity}[/{severity_color}]"
        table.add_row(
            severity_str,
            finding.code,
            finding.subject or "-",
            finding.message,
        )

    if len(report.findings) > 40:
        table.add_row("", "", "", f"... and {len(report.findings) - 40} more")

    console.print(table)
