"""Release artifact and version contract tests."""

from __future__ import annotations

import tomllib
from importlib.metadata import version
from pathlib import Path

from typer.testing import CliRunner

from vaultmind import __version__
from vaultmind.main import app


def test_package_and_runtime_versions_are_consistent() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["version"] == "0.2.0"
    assert __version__ == "0.2.0"
    assert version("vaultmind") == "0.2.0"


def test_vm_version_reports_release_version() -> None:
    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "VaultMind v0.2.0"
