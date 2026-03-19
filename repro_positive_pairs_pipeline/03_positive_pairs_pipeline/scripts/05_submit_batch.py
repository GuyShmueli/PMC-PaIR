from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

from utils.artifacts import batch_stage_path
from utils.batch_api import submit_batch
from utils.io_utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a prepared batch request file to OpenAI.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", required=True, choices=["citation", "question", "answer", "positive"])
    parser.add_argument("--description", default=None, help="Optional batch description override")
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--completion-window", default="24h")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    request_path = batch_stage_path(output_dir, args.stage, "requests", create_parent=False)
    description = args.description or f"{args.stage} batch"

    payload = submit_batch(
        request_jsonl_path=request_path,
        description=description,
        endpoint=args.endpoint,
        completion_window=args.completion_window,
        metadata={"description": description, "stage": args.stage},
    )
    save_json(payload, batch_stage_path(output_dir, args.stage, "batch_meta"))

    batch_id = payload["batch"]["id"]
    print(f"Submitted batch: {batch_id}")
    print(batch_stage_path(output_dir, args.stage, "batch_meta"))


if __name__ == "__main__":
    main()
