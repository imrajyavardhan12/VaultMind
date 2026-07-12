"""vm ask — the compound interest engine."""

from __future__ import annotations

import asyncio

import typer

from vaultmind.ai.asker import ask_question
from vaultmind.ai.providers import ProviderExhaustedError, get_provider
from vaultmind.config import load_config
from vaultmind.core.wiki_log import append_wiki_log
from vaultmind.utils.display import console, print_error, print_info
from vaultmind.utils.logging import setup_logging


def ask(
    question: str,
    depth: str = typer.Option("deep", "--depth", help="shallow (1 pass) or deep (up to 3 passes)"),
    preview: bool = typer.Option(False, "--preview", help="Print answer to stdout without filing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """Ask a question and have it answered by synthesizing your wiki and raw sources.

    The answer is filed to Wiki/📊 Queries/ as a markdown note so future questions
    can build on it — every answer compounds into the knowledge base.

    Use --preview to print the answer without filing it.
    """
    if not preview:
        setup_logging(verbose=verbose)
    config = load_config()

    if depth not in ("shallow", "deep"):
        print_info("--depth must be 'shallow' or 'deep'")
        return

    provider = get_provider(config, tier="deep")

    try:
        result = asyncio.run(
            ask_question(
                question=question,
                provider=provider,
                vault_path=config.vault_path,
                folders_wiki=config.folders.wiki,
                folders_wiki_concepts=config.folders.wiki_concepts,
                folders_wiki_queries=config.folders.wiki_queries,
                folders_raw=config.folders.raw,
                depth=depth,
                file_answer=not preview,
            )
        )
    except ProviderExhaustedError as exc:
        print_error(str(exc))
        if verbose:
            raise
        raise typer.Exit(1) from None

    if preview:
        console.print(result.answer)
    else:
        append_wiki_log(config, event="ask", detail=question)
        print_info(f"Answer filed to:\n{result.path}")
        console.print(result.answer)


if __name__ == "__main__":
    typer.run(ask)
