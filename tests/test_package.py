"""Release artifact and version contract tests."""

from __future__ import annotations

import tomllib
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml
from typer.testing import CliRunner

from vaultmind import __version__
from vaultmind.main import app


def test_package_and_runtime_versions_are_consistent() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["version"] == "0.2.2"
    assert __version__ == "0.2.2"
    assert version("vaultmind") == "0.2.2"


def test_vm_version_reports_release_version() -> None:
    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "VaultMind v0.2.2"


def _workflow(path: str) -> dict[str, Any]:
    """Load workflow keys as strings instead of YAML 1.1 booleans."""
    parsed = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(parsed, dict)
    return parsed


def _steps_by_name(job: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {step["name"]: step for step in job["steps"]}


SETUP_UV_V8_3_2_SHA = "11f9893b081a58869d3b5fccaea48c9e9e46f990"


def test_ci_uses_maintained_actions_and_committed_root_lock() -> None:
    workflow = _workflow(".github/workflows/ci.yml")
    steps = _steps_by_name(workflow["jobs"]["quality"])
    gitignore_lines = Path(".gitignore").read_text(encoding="utf-8").splitlines()

    assert steps["Check out repository"]["uses"] == "actions/checkout@v7"
    assert steps["Set up Python"]["uses"] == "actions/setup-python@v6"
    assert steps["Set up uv"]["uses"] == f"astral-sh/setup-uv@{SETUP_UV_V8_3_2_SHA}"
    assert "# v8.3.2" in Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert steps["Set up uv"]["with"]["enable-cache"] == "true"
    assert steps["Set up uv"]["with"]["cache-dependency-glob"] == "uv.lock"
    assert steps["Install dependencies"]["run"] == "uv sync --locked --group dev"
    assert Path("uv.lock").is_file()
    assert "uv.lock" not in {
        line.strip() for line in gitignore_lines if not line.lstrip().startswith("#")
    }


def test_ci_preserves_quality_build_and_dynamic_wheel_smoke() -> None:
    workflow = _workflow(".github/workflows/ci.yml")
    steps = _steps_by_name(workflow["jobs"]["quality"])
    commands = "\n".join(step.get("run", "") for step in steps.values())
    smoke = steps["Smoke-test installed wheel CLI"]["run"]

    for required in (
        "uv run ruff check",
        "uv run mypy src/vaultmind",
        "uv run pytest",
        "uv build",
        "pip install dist/*.whl",
        "/vm version",
    ):
        assert required in commands
    assert "pyproject.toml" in smoke
    assert "tomllib" in smoke
    assert "EXPECTED_VERSION" in smoke
    assert "VaultMind v0.2.2" not in smoke


def test_publish_workflow_has_release_and_required_manual_triggers() -> None:
    workflow = _workflow(".github/workflows/publish.yml")
    triggers = workflow["on"]

    assert triggers["release"]["types"] == ["published"]
    tag_input = triggers["workflow_dispatch"]["inputs"]["tag"]
    assert tag_input["required"] == "true"
    assert tag_input["type"] == "string"


def test_publish_workflow_validates_exact_tag_and_versions() -> None:
    workflow = _workflow(".github/workflows/publish.yml")
    build = workflow["jobs"]["build"]
    steps = _steps_by_name(build)
    validation = steps["Validate release tag format"]["run"]
    consistency = steps["Verify checkout and version consistency"]["run"]
    checkout = steps["Check out exact release tag"]

    assert build["env"]["RELEASE_TAG"] == (
        "${{ inputs.tag || github.event.release.tag_name }}"
    )
    assert "^v[0-9]+\\.[0-9]+\\.[0-9]+$" in validation
    assert checkout["uses"] == "actions/checkout@v7"
    assert checkout["with"]["ref"].startswith("refs/tags/${{")
    assert checkout["with"]["persist-credentials"] == "false"
    assert "refs/tags/$RELEASE_TAG" in consistency
    assert "git rev-parse HEAD" in consistency
    assert "pyproject.toml" in consistency
    assert "src/vaultmind/__init__.py" in consistency
    assert "tag_version" in consistency
    assert "project_version" in consistency
    assert "runtime_version" in consistency


def test_publish_builds_once_verifies_and_retains_exact_distributions() -> None:
    workflow = _workflow(".github/workflows/publish.yml")
    jobs = workflow["jobs"]
    build_steps = _steps_by_name(jobs["build"])
    publish_steps = _steps_by_name(jobs["publish"])
    workflow_text = Path(".github/workflows/publish.yml").read_text(encoding="utf-8")
    smoke = build_steps["Smoke-test installed wheel CLI"]["run"]

    assert workflow_text.count("uv build") == 1
    assert build_steps["Set up uv"]["uses"] == (
        f"astral-sh/setup-uv@{SETUP_UV_V8_3_2_SHA}"
    )
    assert workflow_text.count("# v8.3.2") == 1
    assert "pip install dist/*.whl" in smoke
    assert "pyproject.toml" in smoke
    assert "/vm version" in smoke
    upload = build_steps["Retain built distributions"]
    assert upload["uses"] == "actions/upload-artifact@v7"
    assert upload["with"]["name"] == "python-distributions"
    assert upload["with"]["path"] == "dist/"
    assert upload["with"]["if-no-files-found"] == "error"
    assert jobs["publish"]["needs"] == "build"
    download = publish_steps["Download verified distributions"]
    assert download["uses"] == "actions/download-artifact@v8"
    assert download["with"]["name"] == upload["with"]["name"]


def test_publish_job_is_least_privilege_oidc_without_secrets() -> None:
    workflow = _workflow(".github/workflows/publish.yml")
    publish = workflow["jobs"]["publish"]
    steps = _steps_by_name(publish)
    workflow_text = Path(".github/workflows/publish.yml").read_text(encoding="utf-8").lower()

    assert publish["environment"]["name"] == "pypi"
    assert publish["permissions"] == {"contents": "read", "id-token": "write"}
    assert steps["Publish distributions with Trusted Publishing"]["uses"] == (
        "pypa/gh-action-pypi-publish@release/v1"
    )
    for forbidden in ("password:", "api-token", "skip-existing"):
        assert forbidden not in workflow_text


def test_dependabot_preserves_python_schedule_and_monitors_actions_monthly() -> None:
    config = _workflow(".github/dependabot.yml")
    updates = {update["package-ecosystem"]: update for update in config["updates"]}

    assert updates["pip"]["directory"] == "/"
    assert updates["pip"]["schedule"]["interval"] == "weekly"
    assert updates["pip"]["open-pull-requests-limit"] == "5"
    assert updates["github-actions"]["directory"] == "/"
    assert updates["github-actions"]["schedule"]["interval"] == "monthly"


def test_release_docs_identify_trusted_publisher_and_protection() -> None:
    docs = Path("docs/RELEASING.md").read_text(encoding="utf-8")

    for expected in (
        "`imrajyavardhan12`",
        "`VaultMind`",
        "`publish.yml`",
        "`pypi`",
        "required reviewers",
        "v0.2.0",
        "Yank",
        "`uv.lock`",
        "uv sync --locked --group dev",
        "Dependabot",
    ):
        assert expected in docs
    assert "[Releasing VaultMind](docs/RELEASING.md)" in Path("README.md").read_text(
        encoding="utf-8"
    )
