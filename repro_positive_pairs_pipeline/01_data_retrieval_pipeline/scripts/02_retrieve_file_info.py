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
from retrieval_pipeline.oa import (
    DEFAULT_OA_ENDPOINT,
    OAClient,
    resolve_oa_packages,
    response_directory_fetcher,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve selected PMC IDs to NCBI OA TGZ package records."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--article-ids-json",
        default=None,
        help="Defaults to <output-dir>/01_unique_article_ids.json.",
    )
    parser.add_argument(
        "--oa-responses-dir",
        default=None,
        help="Offline/replay mode: directory containing <PMCID>.xml OA responses.",
    )
    parser.add_argument("--endpoint", default=DEFAULT_OA_ENDPOINT)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff-seconds", type=float, default=1.0)
    parser.add_argument(
        "--user-agent",
        default="repro-pmc-data-retrieval/1.0",
    )
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ids_path = args.article_ids_json or artifact_path(
        args.output_dir,
        "article_ids_json",
        create_parent=False,
    )

    client = None
    if args.oa_responses_dir:
        fetch_xml = response_directory_fetcher(args.oa_responses_dir)
        source_label = "local OA response cache"
    else:
        client = OAClient(
            endpoint=args.endpoint,
            timeout_seconds=args.timeout_seconds,
            retries=args.retries,
            backoff_seconds=args.backoff_seconds,
            user_agent=args.user_agent,
        )
        fetch_xml = client.fetch_xml
        source_label = "NCBI OA API"

    try:
        stats = resolve_oa_packages(
            ids_path,
            args.output_dir,
            fetch_xml=fetch_xml,
            resume=args.resume,
            checkpoint_every=args.checkpoint_every,
            limit=args.limit,
            source_label=source_label,
        )
    finally:
        if client is not None:
            client.close()
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
