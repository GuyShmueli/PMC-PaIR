#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval_pipeline.patients import prepare_patient_cohort
from retrieval_pipeline.artifacts import artifact_path
from retrieval_pipeline.cli import SingleUseOption
from retrieval_pipeline.unification import materialize_unified_patient_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic single-patient PMC article cohort."
    )
    parser.add_argument(
        "--input-csv",
        action=SingleUseOption,
        required=True,
        help="One canonical unified PMC-Patients CSV.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    materialize_unified_patient_input(args.input_csv, args.output_dir)
    stats = prepare_patient_cohort(
        artifact_path(args.output_dir, "unified_patients_csv", create_parent=False),
        args.output_dir,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
