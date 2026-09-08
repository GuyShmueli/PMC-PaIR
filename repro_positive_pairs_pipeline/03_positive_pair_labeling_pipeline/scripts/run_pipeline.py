#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from positive_pair_labeling_pipeline.config import STAGES
from positive_pair_labeling_pipeline.handoff import initialize_run
from positive_pair_labeling_pipeline.locking import run_lock
from positive_pair_labeling_pipeline.pipeline import (
    build_model_stage,
    download_and_ingest_model_stage,
    finalize_run,
    ingest_model_responses,
    resolve_ambiguous_submission,
    retry_model_stage,
    status_model_stage,
    submit_model_stage,
    verify_run,
)


def _stage_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", choices=STAGES, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Pipeline 03 positive image-pair labeling and validation."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    initialize = commands.add_parser(
        "initialize",
        aliases=["import"],
        help="Validate and snapshot one completed Pipeline 02 handoff.",
    )
    initialize.add_argument(
        "--selection-run",
        required=True,
        help="One completed 02_case_pair_selection_pipeline run.",
    )
    initialize.add_argument("--output-dir", required=True)
    initialize.add_argument("--config", required=True)

    build = commands.add_parser("build", help="Build one model stage's request shards.")
    build.add_argument("--output-dir", required=True)
    _stage_argument(build)

    ingest = commands.add_parser(
        "ingest",
        help="Snapshot and strictly reconcile offline Batch response JSONL files.",
    )
    ingest.add_argument("--output-dir", required=True)
    _stage_argument(ingest)
    ingest.add_argument(
        "--response-file",
        action="append",
        required=True,
        help="A response JSONL file or directory; repeat for multiple shards.",
    )

    for name, help_text in (
        ("submit", "Upload and submit every request shard exactly once."),
        ("status", "Retrieve remote status for every submitted shard."),
        ("download", "Download all completed shards and strictly ingest them."),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--output-dir", required=True)
        _stage_argument(command)
        command.add_argument(
            "--attempt",
            type=int,
            default=1,
            help="Managed attempt generation (default: 1). Use retry for generations >=2.",
        )
        command.add_argument(
            "--acknowledge-external-api",
            action="store_true",
            help="Confirm that this command may contact OpenAI and incur processing/cost.",
        )

    retry = commands.add_parser(
        "retry",
        help="Submit a new attempt after the prior downloaded attempt failed validation.",
    )
    retry.add_argument("--output-dir", required=True)
    _stage_argument(retry)
    retry.add_argument("--attempt", type=int, required=True)
    retry.add_argument("--acknowledge-external-api", action="store_true")

    resolve = commands.add_parser(
        "resolve",
        help="Bind a manually verified batch ID after an ambiguous create call.",
    )
    resolve.add_argument("--output-dir", required=True)
    _stage_argument(resolve)
    resolve.add_argument("--shard-sha256", required=True)
    resolve.add_argument("--batch-id", required=True)
    resolve.add_argument("--attempt", type=int, default=1)
    resolve.add_argument("--acknowledge-external-api", action="store_true")

    finalize = commands.add_parser("finalize", help="Validate and write final products.")
    finalize.add_argument("--output-dir", required=True)

    verify = commands.add_parser("verify", help="Verify hashes and cross-stage alignment.")
    verify.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _live_client(acknowledged: bool) -> Any:
    if not acknowledged:
        raise ValueError(
            "Live commands require --acknowledge-external-api before a client is created"
        )
    from openai import OpenAI

    return OpenAI(max_retries=0)


def main() -> None:
    args = parse_args()
    if args.command in {"initialize", "import"}:
        manifest = initialize_run(
            args.selection_run,
            args.output_dir,
            args.config,
        )
        _print(
            {
                "run_spec_id": manifest["run_spec_id"],
                "status": manifest["status"],
                "handoff": manifest["stages"]["00_handoff"]["details"],
            }
        )
        return
    if args.command == "build":
        with run_lock(args.output_dir):
            _print(build_model_stage(args.output_dir, args.stage))
        return
    if args.command == "ingest":
        with run_lock(args.output_dir):
            records, diagnostics = ingest_model_responses(
                args.output_dir, args.stage, args.response_file
            )
            _print({"records": len(records), "diagnostics": diagnostics})
        return
    if args.command == "finalize":
        with run_lock(args.output_dir):
            _print(finalize_run(args.output_dir))
        return
    if args.command == "verify":
        with run_lock(args.output_dir):
            _print(verify_run(args.output_dir))
        return

    client = _live_client(args.acknowledge_external_api)
    try:
        with run_lock(args.output_dir):
            if args.command == "submit":
                result = submit_model_stage(
                    args.output_dir,
                    args.stage,
                    client=client,
                    acknowledge_external_api=True,
                    attempt=args.attempt,
                )
            elif args.command == "retry":
                result = retry_model_stage(
                    args.output_dir,
                    args.stage,
                    attempt=args.attempt,
                    client=client,
                    acknowledge_external_api=True,
                )
            elif args.command == "status":
                result = status_model_stage(
                    args.output_dir,
                    args.stage,
                    client=client,
                    acknowledge_external_api=True,
                    attempt=args.attempt,
                )
            elif args.command == "download":
                records, diagnostics = download_and_ingest_model_stage(
                    args.output_dir,
                    args.stage,
                    client=client,
                    acknowledge_external_api=True,
                    attempt=args.attempt,
                )
                result = {"records": len(records), "diagnostics": diagnostics}
            elif args.command == "resolve":
                result = resolve_ambiguous_submission(
                    args.output_dir,
                    args.stage,
                    shard_sha256=args.shard_sha256,
                    batch_id=args.batch_id,
                    client=client,
                    acknowledge_external_api=True,
                    attempt=args.attempt,
                )
            else:
                raise AssertionError(f"Unhandled command: {args.command}")
            _print(result)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
