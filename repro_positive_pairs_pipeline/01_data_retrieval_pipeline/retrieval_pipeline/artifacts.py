from __future__ import annotations

from pathlib import Path


PIPELINE_VERSION = "3.1.0"


ARTIFACTS = {
    # Stage 00: one canonical cohort materialized from one source CSV.
    "unified_patients_csv": "00_unified_patients.csv",
    "unification_provenance_csv": "00_unification_provenance.csv",
    "unification_report_json": "00_unification_report.json",
    # Stage 01: patient/article cohort.
    "patients_csv": "01_unique_articles.csv",
    "article_ids_json": "01_unique_article_ids.json",
    "patient_stats_json": "01_stats.json",
    # Stage 02: NCBI OA package resolution.
    "packages_json": "02_files_info.json",
    "package_failures_json": "02_failed_ids.json",
    "package_state_json": "02_retrieval_state.json",
    "package_stats_json": "02_stats.json",
    # Stage 03: canonical local assets and figure manifest.
    # `new_data` is the canonical, run-relative asset directory name shared by
    # the three pipeline contracts.
    "assets_dir": "new_data",
    "images_captions_csv": "03_images_captions.csv",
    "article_failures_json": "03_article_failures.json",
    "figure_failures_json": "03_figure_failures.json",
    "extraction_state_json": "03_extraction_state.json",
    "extraction_stats_json": "03_stats.json",
    # Stage 04: downstream-ready dataset and audit files.
    "merged_csv": "04_merged.csv",
    "image_paths_json": "04_image_paths.json",
    "handoff_report_json": "04_handoff_report.json",
    "run_manifest_json": "run_manifest.json",
}


def artifact_path(
    output_dir: str | Path,
    key: str,
    *,
    create_parent: bool = True,
) -> Path:
    if key not in ARTIFACTS:
        raise KeyError(f"Unknown artifact key: {key}")

    path = Path(output_dir) / ARTIFACTS[key]
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path
