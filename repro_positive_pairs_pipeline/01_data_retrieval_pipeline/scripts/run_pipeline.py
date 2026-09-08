#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval_pipeline.run import run_pipeline_from_local_sources
from retrieval_pipeline.cli import SingleUseOption


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all reproducible PMC data-retrieval stages."
    )
    parser.add_argument(
        "--input-csv",
        action=SingleUseOption,
        required=True,
        help="One canonical unified PMC-Patients CSV.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--oa-responses-dir",
        default=None,
        help="Use cached OA XML responses instead of the network.",
    )
    parser.add_argument(
        "--archives-dir",
        default=None,
        help="Use local TGZ archives instead of FTP/HTTPS downloads.",
    )
    parser.add_argument("--oa-endpoint", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--metadata-retries", type=int, default=3)
    parser.add_argument("--archive-retries", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--user-agent",
        default="repro-pmc-data-retrieval/1.0",
        help="Set a descriptive NCBI request User-Agent; include contact information.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_pipeline_from_local_sources(
        args.input_csv,
        args.output_dir,
        oa_responses_dir=args.oa_responses_dir,
        archives_dir=args.archives_dir,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        jpeg_quality=args.jpeg_quality,
        limit=args.limit,
        oa_endpoint=args.oa_endpoint,
        timeout_seconds=args.timeout_seconds,
        metadata_retries=args.metadata_retries,
        archive_retries=args.archive_retries,
        user_agent=args.user_agent,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
