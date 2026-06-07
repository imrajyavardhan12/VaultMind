"""CLI entry point — typer app."""

from __future__ import annotations

import typer

from vaultmind.commands import register_commands

app = typer.Typer(
    name="vm",
    help="VaultMind — a local-first CLI for an LLM-maintained Obsidian wiki.",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Show VaultMind version."""
    from vaultmind import __version__

    typer.echo(f"VaultMind v{__version__}")


register_commands(app)


if __name__ == "__main__":
    app()
