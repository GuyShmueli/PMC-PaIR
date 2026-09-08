from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path
from typing import Any, Iterable

from .artifacts import RUN_MANIFEST
from .config import PIPELINE_ID, PIPELINE_VERSION, SelectionConfig
from .errors import ArtifactIntegrityError
from .io_utils import (
    canonical_json_bytes,
    file_metadata,
    load_json,
    save_json,
    sha256_bytes,
    sha256_file,
    verify_file_metadata,
)


MANIFEST_SCHEMA_VERSION = 1


def implementation_metadata() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    source_paths = sorted(
        [
            *project_root.glob("case_pair_selection_pipeline/*.py"),
            *project_root.glob("scripts/*.py"),
        ],
        key=lambda path: path.relative_to(project_root).as_posix(),
    )
    for name in (
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-lock.txt",
        "environment.yml",
    ):
        path = project_root / name
        if not path.is_file():
            raise ArtifactIntegrityError(
                f"Implementation contract file is missing: {path}"
            )
        source_paths.append(path)

    files = {
        path.relative_to(project_root).as_posix(): sha256_file(path)
        for path in sorted(
            set(source_paths),
            key=lambda item: item.relative_to(project_root).as_posix(),
        )
    }
    dependencies: dict[str, str] = {}
    for distribution in (
        "Pillow",
        "lxml",
        "numpy",
        "pandas",
        "pytesseract",
        "python-dateutil",
        "pytz",
        "six",
        "tzdata",
    ):
        try:
            dependencies[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependencies[distribution] = "not-installed"
    return {
        "schema_version": 1,
        "python": (
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
        "dependencies": dependencies,
        "files": files,
        "source_sha256": sha256_bytes(canonical_json_bytes(files)),
    }


def external_input_metadata(path: str | Path, *, label: str) -> dict[str, Any]:
    item = Path(path)
    if not item.is_file():
        raise FileNotFoundError(f"Required input is missing: {item}")
    return {
        "label": label,
        "name": item.name,
        "bytes": item.stat().st_size,
        "sha256": sha256_file(item),
    }


def new_manifest(
    *,
    config: SelectionConfig,
    inputs: list[dict[str, Any]],
    limit: int | None,
    workers: int,
    diagram_filter_runtime: dict[str, Any] | None,
) -> dict[str, Any]:
    implementation = implementation_metadata()
    run_spec = {
        "pipeline_id": PIPELINE_ID,
        "pipeline_version": PIPELINE_VERSION,
        "implementation": implementation,
        "config": config.as_dict(),
        "inputs": inputs,
        "limit": limit,
        "workers": workers,
        "diagram_filter_runtime": diagram_filter_runtime,
    }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "pipeline_id": PIPELINE_ID,
        "pipeline_version": PIPELINE_VERSION,
        "implementation": implementation,
        "run_spec_id": sha256_bytes(canonical_json_bytes(run_spec)),
        "status": "initializing",
        "config": config.as_dict(),
        "config_sha256": config.sha256,
        "limit": limit,
        "workers": workers,
        "diagram_filter_runtime": diagram_filter_runtime,
        "inputs": inputs,
        "stages": {},
        "artifacts": {},
    }


def manifest_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / RUN_MANIFEST


def save_manifest(output_dir: str | Path, manifest: dict[str, Any]) -> None:
    save_json(manifest, manifest_path(output_dir))


def load_manifest(output_dir: str | Path) -> dict[str, Any]:
    path = manifest_path(output_dir)
    if not path.is_file():
        raise ArtifactIntegrityError(f"Run manifest is missing: {path}")
    manifest = load_json(path)
    if not isinstance(manifest, dict):
        raise ArtifactIntegrityError("Run manifest is not an object")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ArtifactIntegrityError("Unsupported run manifest schema")
    if manifest.get("pipeline_id") != PIPELINE_ID:
        raise ArtifactIntegrityError("Run was created by a different pipeline")
    if manifest.get("pipeline_version") != PIPELINE_VERSION:
        raise ArtifactIntegrityError(
            "Run was created by a different pipeline version"
        )
    return manifest


def record_stage(
    output_dir: str | Path,
    manifest: dict[str, Any],
    stage: str,
    *,
    status: str,
    artifact_paths: Iterable[str | Path],
    details: dict[str, Any] | None = None,
) -> None:
    run_dir = Path(output_dir)
    recorded: list[str] = []
    for path in artifact_paths:
        metadata = file_metadata(path, root=run_dir)
        relative = metadata["path"]
        manifest["artifacts"][relative] = metadata
        recorded.append(relative)
    manifest["stages"][stage] = {
        "status": status,
        "artifacts": sorted(recorded),
        "details": details or {},
    }
    save_manifest(run_dir, manifest)


def verify_manifest(output_dir: str | Path) -> dict[str, Any]:
    manifest = load_manifest(output_dir)
    if manifest.get("implementation") != implementation_metadata():
        raise ArtifactIntegrityError(
            "Executable source or runtime dependencies changed after run initialization"
        )
    if sha256_bytes(canonical_json_bytes(manifest.get("config"))) != manifest.get(
        "config_sha256"
    ):
        raise ArtifactIntegrityError("Canonical configuration fingerprint changed")
    expected_spec = {
        "pipeline_id": manifest["pipeline_id"],
        "pipeline_version": manifest["pipeline_version"],
        "implementation": manifest["implementation"],
        "config": manifest["config"],
        "inputs": manifest["inputs"],
        "limit": manifest["limit"],
        "workers": manifest["workers"],
        "diagram_filter_runtime": manifest["diagram_filter_runtime"],
    }
    if sha256_bytes(canonical_json_bytes(expected_spec)) != manifest.get(
        "run_spec_id"
    ):
        raise ArtifactIntegrityError("Run specification fingerprint changed")

    artifacts = manifest.get("artifacts")
    stages = manifest.get("stages")
    if not isinstance(artifacts, dict) or not isinstance(stages, dict):
        raise ArtifactIntegrityError(
            "Run manifest artifact/stage tables are malformed"
        )
    for stage_name, stage_record in stages.items():
        if not isinstance(stage_record, dict) or not isinstance(
            stage_record.get("artifacts"), list
        ):
            raise ArtifactIntegrityError(
                f"Stage record is malformed: {stage_name}"
            )
        for relative in stage_record["artifacts"]:
            if relative not in artifacts:
                raise ArtifactIntegrityError(
                    f"Stage {stage_name!r} references unrecorded artifact "
                    f"{relative!r}"
                )
    for relative, metadata in artifacts.items():
        if not isinstance(metadata, dict) or metadata.get("path") != relative:
            raise ArtifactIntegrityError(
                f"Artifact table entry is malformed: {relative!r}"
            )
        verify_file_metadata(metadata, root=output_dir)
    return manifest


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "external_input_metadata",
    "implementation_metadata",
    "load_manifest",
    "new_manifest",
    "record_stage",
    "save_manifest",
    "verify_manifest",
]
