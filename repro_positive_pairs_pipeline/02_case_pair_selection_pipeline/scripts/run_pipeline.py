#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from case_pair_selection_pipeline.pipeline import prepare_run, verify_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select evidence-backed PMC case pairs and emit the immutable "
            "Pipeline-03 handoff."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare",
        help=(
            "Build candidates and summaries, apply human/animal screening, match "
            "citations, and emit the eligible-pair handoff."
        ),
    )
    prepare.add_argument(
        "--retrieval-run",
        required=True,
        help=(
            "One completed Pipeline-01 v3 run with the unified dataset contract."
        ),
    )
    prepare.add_argument("--config", required=True)
    prepare.add_argument(
        "--species-adjudications",
        default=None,
        help=(
            "Optional JSONL overrides for current-run species labels; every row "
            "must contain schema_version, patient_uid, label, and reason."
        ),
    )
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--limit", type=int, default=None)
    prepare.add_argument("--workers", type=int, default=1)
    prepare.add_argument(
        "--allow-upstream-failures",
        action="store_true",
        help=(
            "Explicitly accept a completed_with_failures Pipeline-01 run after "
            "reviewing its recorded coverage loss."
        ),
    )

    verify = commands.add_parser(
        "verify",
        help="Verify artifacts, selection alignment, and the handoff contract.",
    )
    verify.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    if args.command == "prepare":
        manifest = prepare_run(
            args.retrieval_run,
            args.output_dir,
            args.config,
            species_adjudications=args.species_adjudications,
            limit=args.limit,
            workers=args.workers,
            allow_upstream_failures=args.allow_upstream_failures,
        )
        _print(
            {
                "run_spec_id": manifest["run_spec_id"],
                "status": manifest["status"],
                "handoff": manifest["handoff_contract"],
            }
        )
        return
    if args.command == "verify":
        _print(verify_run(args.output_dir))
        return
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
