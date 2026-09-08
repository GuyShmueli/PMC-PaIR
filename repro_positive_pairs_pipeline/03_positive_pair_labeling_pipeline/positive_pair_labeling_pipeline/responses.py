"""Strict reconciliation of asynchronous Batch responses."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .errors import ArtifactIntegrityError, ResponseValidationError
from .io_utils import iter_jsonl, load_json, load_jsonl, save_json, save_jsonl, sha256_file


def _safe_artifact_path(stage_dir: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str):
        raise ArtifactIntegrityError("Manifest artifact path is not a string")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ArtifactIntegrityError(
            f"Manifest contains an unsafe artifact path: {relative_path!r}"
        )
    root = stage_dir.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ArtifactIntegrityError(
            f"Manifest artifact escapes stage directory: {relative_path!r}"
        ) from exc
    return path


def _verify_metadata(stage_dir: Path, metadata: Mapping[str, Any]) -> Path:
    path = _safe_artifact_path(stage_dir, metadata.get("path"))
    if not path.is_file():
        raise ArtifactIntegrityError(f"Manifest artifact is missing: {path}")
    if path.stat().st_size != metadata.get("bytes"):
        raise ArtifactIntegrityError(f"Manifest byte count does not match: {path}")
    if sha256_file(path) != metadata.get("sha256"):
        raise ArtifactIntegrityError(f"Manifest SHA-256 does not match: {path}")
    return path


def _load_request_contract(
    stage_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    manifest_path = stage_dir / "request_manifest.json"
    if not manifest_path.is_file():
        raise ArtifactIntegrityError(f"Request manifest is missing: {manifest_path}")
    try:
        manifest = load_json(manifest_path)
    except Exception as exc:
        raise ArtifactIntegrityError(
            f"Request manifest cannot be parsed: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ArtifactIntegrityError("Unsupported or malformed request manifest")
    if manifest.get("endpoint") != "/v1/chat/completions":
        raise ArtifactIntegrityError("Request manifest has an unsupported endpoint")

    index_metadata = manifest.get("request_index")
    if not isinstance(index_metadata, Mapping):
        raise ArtifactIntegrityError("Request manifest has no valid request_index")
    index_path = _verify_metadata(stage_dir, index_metadata)
    try:
        index = load_jsonl(index_path)
    except Exception as exc:
        raise ArtifactIntegrityError(
            f"Request index cannot be parsed: {index_path}"
        ) from exc

    request_count = manifest.get("request_count")
    if (
        isinstance(request_count, bool)
        or not isinstance(request_count, int)
        or request_count <= 0
        or request_count != len(index)
        or index_metadata.get("count") != len(index)
    ):
        raise ArtifactIntegrityError("Request counts disagree with the request index")

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ArtifactIntegrityError("Request manifest has no shard metadata")
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ArtifactIntegrityError("Malformed shard metadata")
        _verify_metadata(stage_dir, shard)

    custom_ids: set[str] = set()
    pair_indices: set[int] = set()
    ordinals: set[int] = set()
    for row_number, item in enumerate(index, start=1):
        custom_id = item.get("custom_id")
        pair_index = item.get("pair_index")
        ordinal = item.get("ordinal")
        if not isinstance(custom_id, str) or not custom_id:
            raise ArtifactIntegrityError(
                f"Invalid custom_id in request index row {row_number}"
            )
        if isinstance(pair_index, bool) or not isinstance(pair_index, int) or pair_index < 0:
            raise ArtifactIntegrityError(
                f"Invalid pair_index in request index row {row_number}"
            )
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ArtifactIntegrityError(
                f"Invalid ordinal in request index row {row_number}"
            )
        if custom_id in custom_ids:
            raise ArtifactIntegrityError(f"Duplicate request custom_id: {custom_id}")
        if pair_index in pair_indices:
            raise ArtifactIntegrityError(f"Duplicate request pair_index: {pair_index}")
        if ordinal in ordinals:
            raise ArtifactIntegrityError(f"Duplicate request ordinal: {ordinal}")
        custom_ids.add(custom_id)
        pair_indices.add(pair_index)
        ordinals.add(ordinal)

    if ordinals != set(range(len(index))):
        raise ArtifactIntegrityError("Request ordinals are not contiguous from zero")
    return manifest, index, manifest_path


def normalize_response_files(
    response_files: str | Path | Iterable[str | Path],
) -> list[Path]:
    if isinstance(response_files, (str, Path)):
        response_files = [response_files]
    paths: list[Path] = []
    seen: set[Path] = set()
    for source in response_files:
        candidate = Path(source)
        expanded = sorted(candidate.glob("*.jsonl")) if candidate.is_dir() else [candidate]
        for path in expanded:
            resolved = path.resolve()
            if resolved in seen:
                raise ResponseValidationError(f"Duplicate response file path: {path}")
            seen.add(resolved)
            paths.append(path)
    if not paths:
        raise ResponseValidationError("No response JSONL files were provided")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ResponseValidationError(f"Response files are missing: {missing}")
    return paths


def _portable_artifact_path(path: Path, stage_dir: Path) -> str:
    """Use a run-relative path internally and only a basename externally."""

    try:
        return path.resolve().relative_to(stage_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def _assistant_text(row: Mapping[str, Any]) -> str:
    error = row.get("error")
    if error is not None:
        raise ResponseValidationError(f"Batch row contains an error: {error!r}")

    response = row.get("response")
    if not isinstance(response, Mapping):
        raise ResponseValidationError("Batch row has no response object")
    status_code = response.get("status_code")
    if (
        isinstance(status_code, bool)
        or not isinstance(status_code, int)
        or not 200 <= status_code < 300
    ):
        raise ResponseValidationError(
            f"Batch row has a non-success status code: {status_code!r}"
        )

    body = response.get("body")
    if not isinstance(body, Mapping):
        raise ResponseValidationError("Batch response has no body object")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ResponseValidationError(
            "Batch response must contain exactly one choice"
        )
    choice = choices[0]
    if not isinstance(choice, Mapping) or choice.get("index") != 0:
        raise ResponseValidationError(
            "Batch response must contain exactly one choice at index 0"
        )
    if choice.get("finish_reason") != "stop":
        raise ResponseValidationError(
            f"Batch response did not finish normally: {choice.get('finish_reason')!r}"
        )
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise ResponseValidationError("Batch response choice has no message object")
    if message.get("refusal") not in (None, ""):
        raise ResponseValidationError("Batch response contains an explicit refusal")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ResponseValidationError("Batch response assistant content is empty")
    return content


def reconcile_responses(
    stage_dir: str | Path,
    response_files: str | Path | Iterable[str | Path],
    output_path: str | Path | None = None,
    diagnostics_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate and align raw output, then atomically save canonical records.

    ``stage_dir`` must contain ``request_manifest.json`` and its recorded
    ``request_index.jsonl`` and shards. By default, validated records are saved
    as ``responses.jsonl`` and diagnostics as ``response_manifest.json`` in the
    stage directory. No derived response file is written unless every expected
    custom ID has exactly one successful, non-empty response.

    Returns ``(records, diagnostics)``. Records are ordered by the request
    ordinal and contain ``ordinal``, ``pair_index``, ``custom_id``, and ``text``.
    """

    stage_root = Path(stage_dir)
    manifest, request_index, manifest_path = _load_request_contract(stage_root)
    paths = normalize_response_files(response_files)
    expected = {item["custom_id"]: item for item in request_index}

    by_custom_id: dict[str, str] = {}
    seen_custom_ids: set[str] = set()
    problems: list[str] = []
    response_file_metadata: list[dict[str, Any]] = []
    rows_seen = 0

    for path in paths:
        row_count = 0
        try:
            for line_number, row in enumerate(iter_jsonl(path), start=1):
                row_count += 1
                rows_seen += 1
                custom_id = row.get("custom_id")
                location = f"{path}:{line_number}"
                if not isinstance(custom_id, str) or not custom_id:
                    problems.append(f"{location}: missing custom_id")
                    continue
                if custom_id not in expected:
                    problems.append(f"{location}: unexpected custom_id {custom_id!r}")
                    continue
                if custom_id in seen_custom_ids:
                    problems.append(f"{location}: duplicate custom_id {custom_id!r}")
                    continue
                seen_custom_ids.add(custom_id)
                try:
                    by_custom_id[custom_id] = _assistant_text(row)
                except ResponseValidationError as exc:
                    problems.append(f"{location}: {exc}")
        except Exception as exc:
            if isinstance(exc, ResponseValidationError):
                raise
            raise ResponseValidationError(
                f"Malformed response JSONL: {path}"
            ) from exc
        response_file_metadata.append(
            {
                "bytes": path.stat().st_size,
                "path": _portable_artifact_path(path, stage_root),
                "row_count": row_count,
                "sha256": sha256_file(path),
            }
        )

    missing = sorted(set(expected) - set(by_custom_id))
    if missing:
        preview = missing[:20]
        suffix = " ..." if len(missing) > len(preview) else ""
        problems.append(f"missing custom_ids: {preview}{suffix}")
    if problems:
        preview = problems[:20]
        suffix = f" (+{len(problems) - len(preview)} more)" if len(problems) > len(preview) else ""
        raise ResponseValidationError("; ".join(preview) + suffix)

    records = [
        {
            "custom_id": item["custom_id"],
            "ordinal": item["ordinal"],
            "pair_index": item["pair_index"],
            "text": by_custom_id[item["custom_id"]],
        }
        for item in sorted(request_index, key=lambda record: record["ordinal"])
    ]

    destination = Path(output_path) if output_path is not None else stage_root / "responses.jsonl"
    diagnostics_destination = (
        Path(diagnostics_path)
        if diagnostics_path is not None
        else stage_root / "response_manifest.json"
    )
    save_jsonl(records, destination)
    diagnostics: dict[str, Any] = {
        "output": {
            "bytes": destination.stat().st_size,
            "path": _portable_artifact_path(destination, stage_root),
            "record_count": len(records),
            "sha256": sha256_file(destination),
        },
        "request_count": manifest["request_count"],
        "request_manifest": {
            "path": _portable_artifact_path(manifest_path, stage_root),
            "sha256": sha256_file(manifest_path),
        },
        "response_count": len(records),
        "response_files": response_file_metadata,
        "rows_seen": rows_seen,
        "schema_version": 1,
        "stage": manifest.get("stage"),
        "validation": "complete",
    }
    save_json(diagnostics, diagnostics_destination)
    return records, diagnostics


__all__ = ["normalize_response_files", "reconcile_responses"]
