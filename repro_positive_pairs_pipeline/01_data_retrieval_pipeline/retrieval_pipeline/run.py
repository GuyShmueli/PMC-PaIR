from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path
from typing import Any, Callable

from .artifacts import ARTIFACTS, PIPELINE_VERSION, artifact_path
from .extraction import ArchiveClient, extract_figure_assets
from .io_utils import load_json, save_json, sha256_file, sha256_tree
from .merge import build_merged_dataset
from .oa import OAClient, resolve_oa_packages, response_directory_fetcher
from .patients import prepare_patient_cohort
from .unification import (
    materialize_unified_patient_input,
    normalize_input_csvs,
)


def _dependency_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _artifact_hashes(output_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in sorted(set(ARTIFACTS.values())):
        if name == ARTIFACTS["run_manifest_json"]:
            continue
        path = output_dir / name
        if path.is_file():
            hashes[name] = sha256_file(path)
        elif path.is_dir():
            hashes[name + "/"] = sha256_tree(path)
    return hashes


def _validated_stage_00_resume(
    input_csvs: str | Path,
    run_dir: Path,
) -> dict[str, Any]:
    report_path = artifact_path(
        run_dir,
        "unification_report_json",
        create_parent=False,
    )
    if not report_path.is_file():
        raise ValueError(
            "Cannot resume a nonempty run directory without a complete "
            f"{report_path.name} contract"
        )
    report = load_json(report_path)
    if report.get("pipeline_version") != PIPELINE_VERSION:
        raise ValueError("Cannot resume: Stage 00 pipeline version has changed")

    source_paths = normalize_input_csvs(input_csvs)
    expected_sources = report.get("sources", [])
    actual_hashes = [sha256_file(path) for path in source_paths]
    expected_hashes = [source.get("sha256") for source in expected_sources]
    if actual_hashes != expected_hashes:
        raise ValueError(
            "Cannot resume: canonical --input-csv content has changed"
        )

    expected_outputs = report.get("outputs", {})
    for key in ("unified_patients_csv", "unification_provenance_csv"):
        path = artifact_path(run_dir, key, create_parent=False)
        expected_hash = expected_outputs.get(path.name)
        if not path.is_file() or not expected_hash:
            raise ValueError(
                f"Cannot resume: Stage 00 artifact is missing: {path}"
            )
        if sha256_file(path) != expected_hash:
            raise ValueError(
                f"Cannot resume: Stage 00 artifact integrity failed: {path}"
            )
    return report


def _validated_stage_01_resume(
    unified_input_csv: str | Path,
    run_dir: Path,
) -> dict[str, Any]:
    stats_path = artifact_path(
        run_dir,
        "patient_stats_json",
        create_parent=False,
    )
    if not stats_path.is_file():
        raise ValueError(
            "Cannot resume a nonempty run directory without a complete "
            f"{stats_path.name} contract"
        )

    stats = load_json(stats_path)
    if stats.get("pipeline_version") != PIPELINE_VERSION:
        raise ValueError("Cannot resume: Stage 01 pipeline version has changed")
    if stats.get("configuration", {}).get("uniqueness_scope") != "all_rows":
        raise ValueError("Cannot resume: Stage 01 does not use the unified cohort rule")
    if stats.get("input", {}).get("sha256") != sha256_file(unified_input_csv):
        raise ValueError("Cannot resume: materialized unified input has changed")

    expected_outputs = stats.get("outputs", {})
    for key in ("patients_csv", "article_ids_json"):
        path = artifact_path(run_dir, key, create_parent=False)
        expected_hash = expected_outputs.get(path.name)
        if not path.is_file() or not expected_hash:
            raise ValueError(
                f"Cannot resume: Stage 01 artifact is missing: {path}"
            )
        if sha256_file(path) != expected_hash:
            raise ValueError(
                f"Cannot resume: Stage 01 artifact integrity failed: {path}"
            )
    return stats


def run_pipeline(
    input_csv: str | Path,
    output_dir: str | Path,
    *,
    oa_fetch_xml: Callable[[str], bytes] | None = None,
    oa_client: OAClient | None = None,
    archive_fetcher: Callable[[dict[str, str]], bytes] | None = None,
    archive_client: ArchiveClient | None = None,
    resume: bool = False,
    checkpoint_every: int = 100,
    jpeg_quality: int = 95,
    limit: int | None = None,
    oa_source_label: str = "NCBI OA API",
    archive_source_label: str = "remote OA packages",
) -> dict[str, Any]:
    run_dir = Path(output_dir)
    has_existing_content = run_dir.exists() and any(run_dir.iterdir())
    if not resume and has_existing_content:
        raise FileExistsError(
            "Output directory is not empty; use --resume or choose a new "
            f"run directory: {run_dir}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    if resume and has_existing_content:
        stage_00 = _validated_stage_00_resume(input_csv, run_dir)
        unified_input_path = artifact_path(
            run_dir, "unified_patients_csv", create_parent=False
        )
        stage_01 = _validated_stage_01_resume(
            unified_input_path,
            run_dir,
        )
    else:
        stage_00 = materialize_unified_patient_input(input_csv, run_dir)
        unified_input_path = artifact_path(
            run_dir, "unified_patients_csv", create_parent=False
        )
        stage_01 = prepare_patient_cohort(
            unified_input_path,
            run_dir,
        )
    stage_02 = resolve_oa_packages(
        artifact_path(run_dir, "article_ids_json", create_parent=False),
        run_dir,
        fetch_xml=oa_fetch_xml,
        client=oa_client,
        resume=resume,
        checkpoint_every=checkpoint_every,
        limit=limit,
        source_label=oa_source_label,
    )
    stage_03 = extract_figure_assets(
        artifact_path(run_dir, "packages_json", create_parent=False),
        run_dir,
        fetch_archive=archive_fetcher,
        client=archive_client,
        resume=resume,
        checkpoint_every=checkpoint_every,
        jpeg_quality=jpeg_quality,
        limit=limit,
        source_label=archive_source_label,
    )
    stage_04 = build_merged_dataset(
        artifact_path(run_dir, "images_captions_csv", create_parent=False),
        artifact_path(run_dir, "patients_csv", create_parent=False),
        run_dir,
        strict_files=True,
    )

    failure_summary = {
        "oa_resolution_failures": int(stage_02["failures"]),
        "article_extraction_failures": int(stage_03["article_failures"]),
        "figure_extraction_failures": int(stage_03["figure_failures"]),
        "handoff_file_problems": int(stage_04["file_problem_count"]),
        "selected_patients_without_images": len(
            stage_04["patient_uids_without_images"]
        ),
    }

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "dataset_contract": {
            "schema_version": 1,
            "mode": "unified",
            "materialized_input_count": 1,
            "materialized_input": ARTIFACTS["unified_patients_csv"],
            "merged_csv": ARTIFACTS["merged_csv"],
            "assets_dir": ARTIFACTS["assets_dir"],
        },
        "status": (
            "complete"
            if not any(failure_summary.values())
            else "completed_with_failures"
        ),
        "failure_summary": failure_summary,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "dependencies": {
            "pandas": _dependency_version("pandas"),
            "Pillow": _dependency_version("Pillow"),
            "requests": _dependency_version("requests"),
        },
        "configuration": {
            "uniqueness_scope": "all_rows",
            "checkpoint_every": checkpoint_every,
            "jpeg_quality": jpeg_quality,
            "limit": limit,
        },
        "input": {
            "name": unified_input_path.name,
            "sha256": sha256_file(unified_input_path),
            "source_file_count": int(stage_00["source_file_count"]),
            "sources": stage_00["sources"],
        },
        "stage_summaries": {
            "00": stage_00,
            "01": stage_01,
            "02": stage_02,
            "03": stage_03,
            "04": stage_04,
        },
        "artifact_sha256": _artifact_hashes(run_dir),
    }
    save_json(manifest, artifact_path(run_dir, "run_manifest_json"))
    return manifest


def run_pipeline_from_local_sources(
    input_csv: str | Path,
    output_dir: str | Path,
    *,
    oa_responses_dir: str | Path | None = None,
    archives_dir: str | Path | None = None,
    resume: bool = False,
    checkpoint_every: int = 100,
    jpeg_quality: int = 95,
    limit: int | None = None,
    oa_endpoint: str | None = None,
    timeout_seconds: float = 30.0,
    metadata_retries: int = 3,
    archive_retries: int = 3,
    user_agent: str = "repro-pmc-data-retrieval/1.0",
) -> dict[str, Any]:
    oa_client: OAClient | None = None
    archive_client: ArchiveClient | None = None
    oa_fetch_xml = None
    archive_fetcher = None

    if oa_responses_dir is not None:
        oa_fetch_xml = response_directory_fetcher(oa_responses_dir)
        oa_source_label = "local OA response cache"
    else:
        oa_kwargs = {
            "timeout_seconds": timeout_seconds,
            "retries": metadata_retries,
            "user_agent": user_agent,
        }
        if oa_endpoint is not None:
            oa_kwargs["endpoint"] = oa_endpoint
        oa_client = OAClient(**oa_kwargs)
        oa_source_label = "NCBI OA API"

    archive_client = ArchiveClient(
        archives_dir=archives_dir,
        timeout_seconds=max(timeout_seconds, 120.0),
        retries=archive_retries,
        user_agent=user_agent,
    )
    archive_source_label = (
        "local archive cache" if archives_dir is not None else "remote OA packages"
    )

    try:
        return run_pipeline(
            input_csv,
            output_dir,
            oa_fetch_xml=oa_fetch_xml,
            oa_client=oa_client,
            archive_fetcher=archive_fetcher,
            archive_client=archive_client,
            resume=resume,
            checkpoint_every=checkpoint_every,
            jpeg_quality=jpeg_quality,
            limit=limit,
            oa_source_label=oa_source_label,
            archive_source_label=archive_source_label,
        )
    finally:
        if oa_client is not None:
            oa_client.close()
        if archive_client is not None:
            archive_client.close()
