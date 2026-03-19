from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

from utils.artifacts import batch_stage_path
from utils.batch_api import download_batch_files
from utils.io_utils import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the output and error files for a submitted OpenAI batch."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", required=True, choices=["citation", "question", "answer", "positive"])
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Optional batch ID override. Otherwise the script reads the stage metadata file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.batch_id:
        batch_id = args.batch_id
        batch_meta = None
    else:
        batch_meta = load_json(batch_stage_path(output_dir, args.stage, "batch_meta", create_parent=False))
        batch_id = batch_meta["batch"]["id"]

    payload = download_batch_files(batch_id)
    batch = payload["batch"]
    output_text = payload["output_text"]
    error_text = payload["error_text"]

    merged_meta = {"batch": batch}
    if batch_meta is not None:
        merged_meta.update({key: value for key, value in batch_meta.items() if key != "batch"})
    save_json(merged_meta, batch_stage_path(output_dir, args.stage, "batch_meta"))

    if output_text is not None:
        batch_stage_path(output_dir, args.stage, "raw_responses").write_text(output_text, encoding="utf-8")
    if error_text is not None:
        batch_stage_path(output_dir, args.stage, "error_responses").write_text(error_text, encoding="utf-8")

    print(f"Batch status: {batch.get('status')}")
    print(f"Requests completed: {(batch.get('request_counts') or {}).get('completed')}")
    print(f"Requests failed: {(batch.get('request_counts') or {}).get('failed')}")
    if output_text is not None:
        print(batch_stage_path(output_dir, args.stage, "raw_responses"))
    if error_text is not None:
        print(batch_stage_path(output_dir, args.stage, "error_responses"))


if __name__ == "__main__":
    main()
