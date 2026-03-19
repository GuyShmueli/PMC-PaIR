from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

from utils.artifacts import artifact_path
from utils.case_summaries import (
    extract_case_summary_pairs,
    find_valid_case_summary_indices,
)
from utils.io_utils import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract case-summary text from each candidate XML pair."
    )
    parser.add_argument("--output-dir", required=True, help="Directory created by stage 01")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes. Use 1 for serial execution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    pair_xml_paths = load_json(artifact_path(output_dir, "pair_xml_paths_json", create_parent=False))
    case_summary_pairs = extract_case_summary_pairs(pair_xml_paths, workers=args.workers)
    valid_indices, invalid_indices = find_valid_case_summary_indices(case_summary_pairs)

    save_json(case_summary_pairs, artifact_path(output_dir, "case_summary_pairs_json"))
    save_json(valid_indices, artifact_path(output_dir, "summary_valid_indices_json"))
    save_json(invalid_indices, artifact_path(output_dir, "summary_invalid_indices_json"))

    stats = {
        "pairs_processed": len(pair_xml_paths),
        "valid_case_summary_pairs": len(valid_indices),
        "invalid_case_summary_pairs": len(invalid_indices),
    }
    save_json(stats, artifact_path(output_dir, "stage2_stats_json"))

    print(f"Processed XML pairs: {len(pair_xml_paths)}")
    print(f"Valid case-summary pairs: {len(valid_indices)}")
    print(f"Invalid case-summary pairs: {len(invalid_indices)}")


if __name__ == "__main__":
    main()
