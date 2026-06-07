## Implementation workflow

For any non-trivial change (more than ~20 lines, any logic change, or anything touching multiple files), you MUST delegate to subagents. Do not implement directly.

### Step 1: Write a spec

Before invoking the coder, write a spec in this exact format:

```
## Goal
<one sentence — what the change accomplishes>

## Context
<2-4 sentences — why this change is needed, what depends on it>

## Files to modify
- path/to/a.ext — <what changes here>
- path/to/b.ext — <what changes here>

## Files to read (do not modify)
- path/to/interface.ext — <why relevant>
- path/to/tests/test_a.ext — <why relevant>

## Acceptance criteria
- <concrete, checkable statement>
- <concrete, checkable statement>

## Test plan
Command: `<exact command to run>`
Must pass: <list of test names or behaviors that must succeed>

## Out of scope
- <explicit list of things NOT to touch>
```

The "Files to read" and "Out of scope" sections are mandatory, not optional. They prevent the most common failures.

### Step 2: Delegate to the coder subagent

Invoke the `coder` subagent with the full spec as the prompt.

### Step 3: Delegate to the adversarial-code-reviewer subagent

After the coder reports completion, invoke the `adversarial-code-reviewer` subagent with the spec and the coder's output.

### Step 4: Act on the verdict

- **PASS** → present the change to the user with a one-line summary.
- **REVISE** → send the adversarial-code-reviewer's critical issues back to the coder. Maximum 2 retries. If still failing after 2 retries, take over the implementation directly rather than continuing to retry.
- **REJECT** → rethink the plan before delegating again. Do not just retry the coder with the same spec.

### Trivial changes

Single-file edits under ~20 lines, typo fixes, comment-only changes, and config tweaks may skip delegation and be done directly.

### Project test command

Default test command for this project: `uv run pytest`

Full quality gate (matches CI): `uv run ruff check && uv run mypy src/vaultmind && uv run pytest`