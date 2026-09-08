from __future__ import annotations

from pathlib import Path


RUN_MANIFEST = "run_manifest.json"

SELECTION_PIPELINE_ID = "case_pair_selection_pipeline"
SELECTION_PIPELINE_VERSION = "6.1.0"
SELECTION_HANDOFF_SCHEMA_VERSION = 4
SOURCE_ELIGIBLE_CASES = "03_eligible_cases.jsonl"
SOURCE_ELIGIBLE_PAIRS = "03_eligible_pairs.jsonl"
SOURCE_HANDOFF_MANIFEST = "03_handoff_manifest.json"
SOURCE_SPECIES_CLASSIFICATIONS = "02_species_classifications.jsonl"
SOURCE_PAIR_SPECIES = "02_pair_species.jsonl"
SOURCE_SPECIES_STATS = "02_species_stats.json"

# Pipeline 03 snapshots the immutable Pipeline 02 handoff under these names.
# The normalized compatibility views below are derived from these snapshots;
# Pipeline 03 never reads the Level-1 retrieval tree.
HANDOFF_CASES = "00_eligible_cases.jsonl"
HANDOFF_PAIRS = "00_eligible_pairs.jsonl"
HANDOFF_MANIFEST = "00_selection_handoff.json"
HANDOFF_SOURCE_MANIFEST = "00_selection_run_manifest.json"
HANDOFF_SPECIES_CLASSIFICATIONS = "00_species_classifications.jsonl"
HANDOFF_PAIR_SPECIES = "00_pair_species.jsonl"
HANDOFF_SPECIES_STATS = "00_species_stats.json"
HANDOFF_STAGE = "00_handoff"

# Derived, normalized views used by the existing stage 04-08 implementation.
CASES = "01_cases.jsonl"
SUMMARIES = "02_case_summaries.jsonl"
CITATIONS = "03_citations.jsonl"
SELECTION = "03_selection.json"
FINAL_LABELED = "labeled_positive_pairs.jsonl"
FINAL_PAIRS = "08_positive_pairs.json"
FINAL_TEXT = "08_text_pairs.json"
FINAL_BUCKETS = "08_pairs_by_bucket.json"
FINAL_STATS = "08_stats.json"


def batch_stage_dir(output_dir: str | Path, stage: str) -> Path:
    return Path(output_dir) / "batch" / stage
