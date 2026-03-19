from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

from utils.artifacts import batch_stage_path
from utils.io_utils import extract_aligned_batch_texts, load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract assistant message texts from a raw OpenAI batch response file."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", required=True, choices=["citation", "question", "answer", "positive"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    request_path = batch_stage_path(output_dir, args.stage, "requests", create_parent=False)
    raw_response_path = batch_stage_path(output_dir, args.stage, "raw_responses", create_parent=False)

    texts, diagnostics = extract_aligned_batch_texts(request_path, raw_response_path)
    save_json(texts, batch_stage_path(output_dir, args.stage, "texts"))

    meta_path = batch_stage_path(output_dir, args.stage, "batch_meta", create_parent=False)
    meta = load_json(meta_path)
    meta["text_extraction"] = diagnostics
    save_json(meta, meta_path)

    print(f"Aligned texts written: {len(texts)}")
    print(f"Missing responses: {len(diagnostics['missing_custom_ids'])}")
    print(batch_stage_path(output_dir, args.stage, "texts"))


if __name__ == "__main__":
    main()
