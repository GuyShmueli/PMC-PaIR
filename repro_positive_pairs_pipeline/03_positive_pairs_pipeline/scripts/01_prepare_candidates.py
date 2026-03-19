from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

from utils.artifacts import artifact_path
from utils.data_prep import (
    build_pair_caption_artifacts,
    build_patient_caption_map,
    build_xml_pairs,
    create_patient_uid_pairs,
    load_and_filter_rows,
)
from utils.io_utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare candidate case pairs and aligned caption / image / XML artifacts."
    )
    parser.add_argument("--dataset-csv", required=True, help="Path to final_csv2.csv")
    parser.add_argument("--diagram-paths-json", required=True, help="Path to imgs_paths_diagrams.json")
    parser.add_argument("--data2-base", required=True, help="Path to the data2 directory")
    parser.add_argument("--output-dir", required=True, help="Directory for all stage artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_filter_rows(args.dataset_csv, args.diagram_paths_json)
    df.to_csv(artifact_path(output_dir, "filtered_rows_csv"), index=False)

    patient_uid_pairs = create_patient_uid_pairs(df)
    save_json(patient_uid_pairs, artifact_path(output_dir, "pair_patient_uids_json"))

    patient_uid_to_captions = build_patient_caption_map(df)
    (
        pair_text_paths,
        pair_image_paths,
        pair_caption_dicts,
        caption_stats,
    ) = build_pair_caption_artifacts(
        patient_uid_pairs,
        args.data2_base,
        patient_uid_to_captions,
    )
    save_json(pair_text_paths, artifact_path(output_dir, "pair_text_paths_json"))
    save_json(pair_image_paths, artifact_path(output_dir, "pair_image_paths_json"))
    save_json(pair_caption_dicts, artifact_path(output_dir, "pair_caption_dicts_json"))

    pair_xml_paths, xml_valid_indices, missing_xml = build_xml_pairs(patient_uid_pairs, args.data2_base)
    save_json(pair_xml_paths, artifact_path(output_dir, "pair_xml_paths_json"))
    save_json(xml_valid_indices, artifact_path(output_dir, "pair_xml_valid_indices_json"))
    save_json(missing_xml, artifact_path(output_dir, "pair_xml_missing_json"))

    stats = {
        "rows_loaded": int(len(df)),
        "candidate_case_pairs": len(patient_uid_pairs),
        "pairs_with_both_xml_files": len(xml_valid_indices),
        "pairs_missing_at_least_one_xml": len(missing_xml),
        **caption_stats,
    }
    save_json(stats, artifact_path(output_dir, "stage1_stats_json"))

    print(f"Filtered rows: {len(df)}")
    print(f"Candidate case pairs: {len(patient_uid_pairs)}")
    print(f"Pairs with both XML files: {len(xml_valid_indices)}")
    print(f"Pairs missing at least one XML file: {len(missing_xml)}")
    print(f"Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
