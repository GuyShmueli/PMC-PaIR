from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

from utils.artifacts import artifact_path
from utils.citations import find_common_indices, process_xml_pairs
from utils.io_utils import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find citing/cited case-pair relationships inside the candidate XML pairs."
    )
    parser.add_argument("--output-dir", required=True, help="Directory created by stage 01")
    parser.add_argument(
        "--restrict-to-summary-valid",
        action="store_true",
        help="Process only the summary-valid indices from stage 02.",
    )
    parser.add_argument(
        "--title-fallback",
        action="store_true",
        default=False,
        help="Allow title-substring fallback when DOI/PMID/PMCID do not match.",
    )
    parser.add_argument(
        "--title-min-cited-chars",
        type=int,
        default=0,
        help="Minimum normalized cited-title length for title fallback.",
    )
    parser.add_argument(
        "--title-min-cited-tokens",
        type=int,
        default=0,
        help="Minimum normalized cited-title token count for title fallback.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    pair_xml_paths = load_json(artifact_path(output_dir, "pair_xml_paths_json", create_parent=False))

    if args.restrict_to_summary_valid:
        summary_valid_indices = load_json(
            artifact_path(output_dir, "summary_valid_indices_json", create_parent=False)
        )
        indices_to_process = list(summary_valid_indices)
        xml_pairs_to_process = [pair_xml_paths[idx] for idx in indices_to_process]
    else:
        summary_valid_indices = None
        indices_to_process = list(range(len(pair_xml_paths)))
        xml_pairs_to_process = pair_xml_paths

    citation_records, citation_valid_indices = process_xml_pairs(
        xml_pairs_to_process,
        pair_indices=indices_to_process,
        title_fallback=args.title_fallback,
        title_min_cited_chars=args.title_min_cited_chars,
        title_min_cited_tokens=args.title_min_cited_tokens,
    )

    save_json(citation_records, artifact_path(output_dir, "citation_records_json"))
    save_json(citation_valid_indices, artifact_path(output_dir, "citation_valid_indices_json"))

    common_indices = None
    if summary_valid_indices is not None:
        common_indices = find_common_indices(summary_valid_indices, citation_valid_indices)
        save_json(common_indices, artifact_path(output_dir, "common_indices_json"))

    stats = {
        "pairs_processed": len(xml_pairs_to_process),
        "citation_pairs_found": len(citation_valid_indices),
        "used_summary_valid_restriction": bool(args.restrict_to_summary_valid),
        "common_indices_count": len(common_indices) if common_indices is not None else None,
        "title_fallback": bool(args.title_fallback),
        "title_min_cited_chars": args.title_min_cited_chars,
        "title_min_cited_tokens": args.title_min_cited_tokens,
    }
    save_json(stats, artifact_path(output_dir, "stage3_stats_json"))

    print(f"Pairs processed: {len(xml_pairs_to_process)}")
    print(f"Citation pairs found: {len(citation_valid_indices)}")
    if common_indices is not None:
        print(f"Common indices (summary + citation): {len(common_indices)}")


if __name__ == "__main__":
    main()
