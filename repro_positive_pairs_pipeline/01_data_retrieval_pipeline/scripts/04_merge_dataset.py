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
from retrieval_pipeline.merge import build_merged_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge figure rows with similarity data and validate the handoff."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--images-csv", default=None)
    parser.add_argument("--unique-articles-csv", default=None)
    parser.add_argument(
        "--allow-file-problems",
        action="store_true",
        help="Write the dataset despite missing/misplaced local assets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_csv = args.images_csv or artifact_path(
        args.output_dir,
        "images_captions_csv",
        create_parent=False,
    )
    patients_csv = args.unique_articles_csv or artifact_path(
        args.output_dir,
        "patients_csv",
        create_parent=False,
    )
    report = build_merged_dataset(
        images_csv,
        patients_csv,
        args.output_dir,
        strict_files=not args.allow_file_problems,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
