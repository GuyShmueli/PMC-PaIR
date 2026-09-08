#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval_pipeline.artifacts import artifact_path
from retrieval_pipeline.extraction import ArchiveClient, extract_figure_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/extract PMC packages into canonical JPG/TXT/NXML assets."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--files-info-json",
        default=None,
        help="Defaults to <output-dir>/02_files_info.json.",
    )
    parser.add_argument(
        "--archives-dir",
        default=None,
        help="Offline/replay mode: directory containing the named TGZ archives.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff-seconds", type=float, default=2.0)
    parser.add_argument(
        "--user-agent",
        default="repro-pmc-data-retrieval/1.0",
        help="Set a descriptive NCBI download User-Agent.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packages_path = args.files_info_json or artifact_path(
        args.output_dir,
        "packages_json",
        create_parent=False,
    )
    client = ArchiveClient(
        archives_dir=args.archives_dir,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        backoff_seconds=args.backoff_seconds,
        user_agent=args.user_agent,
    )
    try:
        stats = extract_figure_assets(
            packages_path,
            args.output_dir,
            client=client,
            resume=args.resume,
            checkpoint_every=args.checkpoint_every,
            jpeg_quality=args.jpeg_quality,
            limit=args.limit,
            source_label=(
                "local archive cache"
                if args.archives_dir
                else "remote OA packages"
            ),
        )
    finally:
        client.close()
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
