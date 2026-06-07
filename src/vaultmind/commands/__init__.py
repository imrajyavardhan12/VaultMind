"""Command registration for modular CLI wiring."""

from __future__ import annotations

import typer


def register_commands(app: typer.Typer) -> None:
    """Register all commands on the main Typer app."""
    from vaultmind.commands.ask import ask
    from vaultmind.commands.compile import compile
    from vaultmind.commands.init import init
    from vaultmind.commands.lint import lint

    app.command("ask")(ask)
    app.command("init")(init)
    app.command("compile")(compile)
    app.command("lint")(lint)
