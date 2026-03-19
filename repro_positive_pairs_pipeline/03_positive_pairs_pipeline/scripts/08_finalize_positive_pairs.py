from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

from utils.artifacts import artifact_path, batch_stage_path
from utils.io_utils import load_json, save_json
from utils.positive_parser import process_positive_responses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and bucket the final positive-pair model outputs."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--legacy-radiology-bucket",
        action="store_true",
        help="Preserve the notebook's older modality bucket order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    pair_caption_dicts = load_json(artifact_path(output_dir, "pair_caption_dicts_json", create_parent=False))
    request_indices = load_json(batch_stage_path(output_dir, "positive", "request_indices", create_parent=False))
    positive_texts = load_json(batch_stage_path(output_dir, "positive", "texts", create_parent=False))

    result = process_positive_responses(
        positive_texts,
        request_indices,
        pair_caption_dicts,
        legacy_modality_bucket=args.legacy_radiology_bucket,
    )

    radiology_bucket = result["by_bucket"].get(
        "radiology",
        {"labeled": [], "pairs_clean": [], "text_pairs": []},
    )

    save_json(result["labeled_all"], artifact_path(output_dir, "labeled_positive_pairs_json"))
    save_json(result["pairs_clean_all"], artifact_path(output_dir, "positive_pairs_clean_json"))
    save_json(result["text_pairs_all"], artifact_path(output_dir, "text_pairs_json"))
    save_json(result["by_bucket"], artifact_path(output_dir, "positive_pairs_by_bucket_json"))
    save_json(
        radiology_bucket["labeled"],
        artifact_path(output_dir, "radiology_labeled_positive_pairs_json"),
    )
    save_json(
        radiology_bucket["pairs_clean"],
        artifact_path(output_dir, "radiology_pairs_clean_json"),
    )
    save_json(
        radiology_bucket["text_pairs"],
        artifact_path(output_dir, "radiology_text_pairs_json"),
    )
    save_json(result["bad_rows"], artifact_path(output_dir, "bad_rows_json"))
    save_json(result["stats"], artifact_path(output_dir, "final_stats_json"))

    print(f"All positive pairs kept: {len(result['pairs_clean_all'])}")
    print(f"Radiology positive pairs kept: {len(radiology_bucket['pairs_clean'])}")
    print(f"Bad rows logged: {len(result['bad_rows'])}")
    print(artifact_path(output_dir, "radiology_pairs_clean_json"))


if __name__ == "__main__":
    main()
