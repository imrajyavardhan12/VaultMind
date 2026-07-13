#!/usr/bin/env python3
"""Developer/CI entry point for the deterministic offline evaluation fixture."""

from __future__ import annotations

import argparse
import hashlib
import sys
from contextlib import redirect_stdout
from pathlib import Path

from vaultmind.evaluation import render_report_json, run_evaluation
from vaultmind.evaluation.models import EvaluationErrorReport
from vaultmind.evaluation.runner import DEFAULT_FIXTURE_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the deterministic JSON report here")
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=DEFAULT_FIXTURE_DIR,
        help="evaluation fixture directory (defaults to the committed baseline)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero when one or more committed thresholds fail",
    )
    args = parser.parse_args()

    # Production paths emit human logs to stdout; keep JSON stdout machine-readable.
    try:
        with redirect_stdout(sys.stderr):
            report = run_evaluation(args.fixture_dir)
        rendered = render_report_json(report)
        exit_code = 1 if args.check and not report.passed else 0
    except Exception as exc:
        # Never serialize arbitrary exception text: provider/production failures
        # may contain temporary paths or sensitive input. A stable fingerprint is
        # enough to correlate the artifact with protected CI diagnostics.
        error_type = type(exc).__name__[:120]
        error_report = EvaluationErrorReport(
            error_type=error_type,
            error_fingerprint=hashlib.sha256(error_type.encode("utf-8")).hexdigest()[:24],
        )
        rendered = error_report.model_dump_json(indent=2) + "\n"
        print(f"evaluation failed: {error_type}", file=sys.stderr)
        exit_code = 2

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
