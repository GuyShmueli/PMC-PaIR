from __future__ import annotations

RUN_MANIFEST = "run_manifest.json"

# Required handoff contract from 01_data_retrieval_pipeline. ``new_data`` is
# the canonical asset-tree name in a unified retrieval run.
UPSTREAM_PIPELINE_VERSION = "3.1.0"
UPSTREAM_DATASET_CONTRACT_SCHEMA_VERSION = 1
UPSTREAM_DATASET_MODE = "unified"
UPSTREAM_MATERIALIZED_INPUT_COUNT = 1
UPSTREAM_PATIENTS = "00_unified_patients.csv"
UPSTREAM_MERGED = "04_merged.csv"
UPSTREAM_ASSETS = "new_data"

DIAGRAM_CLASSIFICATIONS = "00_diagram_classifications.jsonl"
DIAGRAM_BLACKLIST = "00_diagram_blacklist.json"
DIAGRAM_STATS = "00_diagram_stats.json"
CASES = "01_cases.jsonl"
PAIRS = "01_pairs.jsonl"
STAGE_1_STATS = "01_stats.json"
SUMMARIES = "02_case_summaries.jsonl"
STAGE_2_STATS = "02_stats.json"
SPECIES_CLASSIFICATIONS = "02_species_classifications.jsonl"
PAIR_SPECIES = "02_pair_species.jsonl"
SPECIES_STATS = "02_species_stats.json"
CITATIONS = "03_citations.jsonl"
SELECTION = "03_selection.json"
STAGE_3_STATS = "03_stats.json"

# Stable, self-contained downstream handoff. Pipeline 03 needs only these
# artifacts and the completed producer run manifest; it never needs to reopen
# Level-1 assets.
ELIGIBLE_CASES = "03_eligible_cases.jsonl"
ELIGIBLE_PAIRS = "03_eligible_pairs.jsonl"
HANDOFF_MANIFEST = "03_handoff_manifest.json"
HANDOFF_SCHEMA_VERSION = 4
