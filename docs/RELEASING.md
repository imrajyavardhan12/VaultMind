# Releasing VaultMind

VaultMind publishes to PyPI from `.github/workflows/publish.yml` with GitHub OIDC. No PyPI API token or password is stored in GitHub. The workflow builds once, verifies the wheel in an isolated environment, retains the distributions as a workflow artifact, and then passes those exact files to the protected publication job.

## One-time setup

### 1. Configure the PyPI Trusted Publisher

An owner of the `vaultmind` project on PyPI must add a GitHub Trusted Publisher with this exact tuple:

| PyPI field | Value |
| --- | --- |
| Owner | `imrajyavardhan12` |
| Repository | `VaultMind` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

Configure this at <https://pypi.org/manage/project/vaultmind/settings/publishing/>. Field values are case-sensitive. Do not create a PyPI API token as a substitute.

### 2. Protect the GitHub environment

In **Repository settings → Environments**, create an environment named `pypi`. Add required reviewers so that every publication requires maintainer approval. Limit deployment branches/tags to the release policy where practical. Do not add publishing secrets to the environment.

The build job does not receive an OIDC token. Only the job gated by `pypi` has `id-token: write`, alongside `contents: read`.

## Normal release sequence

1. Confirm `main` is clean and the release notes are complete.
2. Set the same `X.Y.Z` version in `pyproject.toml` and `src/vaultmind/__init__.py`. Update `CHANGELOG.md`.
3. Run the complete local gate:

   ```bash
   uv run ruff check && uv run mypy src/vaultmind && uv run pytest && uv build
   python -m venv /tmp/vaultmind-release-check
   /tmp/vaultmind-release-check/bin/pip install --force-reinstall dist/vaultmind-X.Y.Z-py3-none-any.whl
   test "$(/tmp/vaultmind-release-check/bin/vm version)" = "VaultMind vX.Y.Z"
   ```

4. Merge the reviewed release change and wait for CI to pass on `main`.
5. Create and push an annotated `vX.Y.Z` tag from that exact commit.
6. Publish the GitHub release for `vX.Y.Z`. The `release.published` event starts the Publish to PyPI workflow.
7. Review the validation/build job and its `python-distributions` artifact. Approve the protected `pypi` environment only after the tag, versions, and files are correct.
8. Confirm the PyPI publication and perform the verification below.

The workflow rejects tags that do not match `vX.Y.Z`, checks out `refs/tags/vX.Y.Z`, and requires the tag, project metadata, and runtime version to agree. Existing filenames are not skipped: a version conflict fails visibly rather than appearing successful.

## Dependency lock policy

The root `uv.lock` is committed and is the reproducibility source for development and CI. CI installs the development group with `uv sync --locked --group dev`; a stale lock therefore fails instead of silently resolving different versions. Keep `pyproject.toml` and `uv.lock` changes in the same reviewed change: after an intentional dependency edit, run `uv lock`, then `uv sync --locked --group dev` and the complete local gate.

Dependabot's weekly `pip` updates must include a corresponding `uv.lock` update when resolution changes. If a Dependabot PR changes dependency declarations without refreshing the lock, update it with `uv lock` before merging. The monthly `github-actions` updates maintain action pins; retain full commit-SHA pins and their release-version comments when reviewing those PRs.

## Manual publication and recovery

Use manual dispatch only for an existing release tag whose GitHub release did not trigger publication or whose run failed before PyPI accepted files. This is also the path for publishing the existing `v0.2.0` tag after its Trusted Publisher authorization is configured.

1. Ensure the Trusted Publisher and protected `pypi` environment are configured.
2. Open **Actions → Publish to PyPI → Run workflow**.
3. Select the workflow from the default branch and enter the required existing tag, for example `v0.2.0`.
4. Inspect the validation/build result and retained artifact.
5. Approve the `pypi` environment deployment.

Never recreate or move a released tag to recover a publication. Fix workflow-only problems on the default branch and dispatch again for the immutable tag. Because the workflow itself is loaded from the selected/default branch while source is checked out from the requested tag, its validation always confirms the artifact version against that exact tag.

If PyPI already accepted one or more files, do not rerun expecting overwrite behavior; PyPI distributions are immutable. Inspect the project page first and follow the rollback guidance.

## Verification

Verify both metadata and a clean installation after publication:

```bash
python -m pip index versions vaultmind
python -m venv /tmp/vaultmind-pypi-check
/tmp/vaultmind-pypi-check/bin/pip install --no-cache-dir vaultmind==X.Y.Z
test "$(/tmp/vaultmind-pypi-check/bin/vm version)" = "VaultMind vX.Y.Z"
```

Also check <https://pypi.org/project/vaultmind/> for both the wheel and source distribution, and compare their filenames with the retained `python-distributions` workflow artifact.

## Rollback and yanking

PyPI artifacts cannot be replaced or deleted as a normal rollback. If a release is defective but not malicious:

1. Yank the affected release from the PyPI project’s release page and provide a clear reason. Yanking prevents new unconstrained resolution while preserving reproducible installs that pin the exact version.
2. Mark the GitHub release as affected and document the issue; do not move or reuse its tag.
3. Prepare, validate, tag, and publish a new patch release through the normal sequence.

Delete a PyPI release only for an exceptional security or legal incident, understanding that deletion is irreversible and the version filename still cannot be reused. For compromised GitHub/PyPI access, revoke the Trusted Publisher first, disable the `pypi` environment, preserve workflow evidence, and follow the relevant account and package incident procedures.
