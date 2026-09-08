"""Deterministic, fail-closed construction of OpenAI Batch request shards."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .config import (
    BATCH_ENDPOINT as CHAT_COMPLETIONS_ENDPOINT,
    BATCH_FILE_LIMIT_BYTES as OPENAI_BATCH_MAX_BYTES,
    DEFAULT_MAX_SHARD_BYTES,
    DEFAULT_MAX_SHARD_REQUESTS,
    PIPELINE_MAX_SHARD_REQUESTS,
)
from .errors import ArtifactIntegrityError, ConfigurationError
from .io_utils import (
    canonical_json_bytes,
    load_json,
    save_bytes,
    save_json,
    save_jsonl,
    sha256_bytes,
    sha256_file,
)


_STAGE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_ALLOWED_INPUT_KEYS = {"pair_index", "body", "method", "url", "custom_id"}


def _validate_limits(max_shard_bytes: int, max_shard_requests: int) -> None:
    if isinstance(max_shard_bytes, bool) or not isinstance(max_shard_bytes, int):
        raise ConfigurationError("max_shard_bytes must be an integer")
    if not 0 < max_shard_bytes < OPENAI_BATCH_MAX_BYTES:
        raise ConfigurationError(
            f"max_shard_bytes must be positive and strictly below "
            f"{OPENAI_BATCH_MAX_BYTES:,}"
        )
    if isinstance(max_shard_requests, bool) or not isinstance(max_shard_requests, int):
        raise ConfigurationError("max_shard_requests must be an integer")
    if not 0 < max_shard_requests <= PIPELINE_MAX_SHARD_REQUESTS:
        raise ConfigurationError(
            "max_shard_requests must be between 1 and the pipeline safety "
            f"ceiling of {PIPELINE_MAX_SHARD_REQUESTS:,}"
        )


def _normalize_request(
    record: Mapping[str, Any], *, stage: str, ordinal: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    unknown_keys = set(record) - _ALLOWED_INPUT_KEYS
    if unknown_keys:
        raise ValueError(
            f"Request {ordinal} has unsupported keys: {sorted(unknown_keys)}"
        )

    pair_index = record.get("pair_index")
    if isinstance(pair_index, bool) or not isinstance(pair_index, int) or pair_index < 0:
        raise ValueError(
            f"Request {ordinal} must contain a non-negative integer pair_index"
        )

    body = record.get("body")
    if not isinstance(body, Mapping) or not body:
        raise ValueError(f"Request {ordinal} must contain a non-empty object body")
    body = dict(body)

    method = record.get("method", "POST")
    endpoint = record.get("url", CHAT_COMPLETIONS_ENDPOINT)
    if method != "POST":
        raise ValueError(f"Request {ordinal} uses unsupported method {method!r}")
    if endpoint != CHAT_COMPLETIONS_ENDPOINT:
        raise ValueError(
            f"Request {ordinal} uses endpoint {endpoint!r}; every shard must use "
            f"{CHAT_COMPLETIONS_ENDPOINT!r}"
        )

    # The hash deliberately excludes custom_id: it identifies the API operation.
    request_core = {
        "body": body,
        "method": "POST",
        "url": CHAT_COMPLETIONS_ENDPOINT,
    }
    request_sha256 = sha256_bytes(canonical_json_bytes(request_core))
    custom_id = f"{stage}-{ordinal:08d}-{request_sha256[:16]}"
    envelope = {"custom_id": custom_id, **request_core}
    index_record = {
        "custom_id": custom_id,
        "ordinal": ordinal,
        "pair_index": pair_index,
        "request_sha256": request_sha256,
        "stage": stage,
    }
    return envelope, index_record


def _metadata_matches(path: Path, metadata: Mapping[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == metadata.get("bytes")
        and sha256_file(path) == metadata.get("sha256")
    )


def _verify_existing_directory(
    output_dir: Path, expected_manifest: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return an identical complete manifest, otherwise fail closed."""

    manifest_path = output_dir / "request_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        existing = load_json(manifest_path)
    except Exception as exc:
        raise ArtifactIntegrityError(
            f"Existing request manifest cannot be read: {manifest_path}"
        ) from exc
    if existing != expected_manifest:
        raise ArtifactIntegrityError(
            f"Refusing to replace non-identical request artifacts in {output_dir}"
        )

    artifacts = [existing.get("request_index"), *existing.get("shards", [])]
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ArtifactIntegrityError(
                f"Existing request manifest is malformed: {manifest_path}"
            )
        relative = artifact.get("path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ArtifactIntegrityError(
                f"Unsafe artifact path in existing manifest: {relative!r}"
            )
        if not _metadata_matches(output_dir / relative, artifact):
            raise ArtifactIntegrityError(
                f"Existing request artifact does not match its manifest: {relative}"
            )
    return existing


def _publish_staging_directory(staging_dir: Path, output_dir: Path) -> None:
    """Publish a complete directory with one same-filesystem rename."""

    if output_dir.exists():
        if output_dir.is_dir() and not any(output_dir.iterdir()):
            output_dir.rmdir()
        else:
            raise ArtifactIntegrityError(
                f"Refusing to overwrite existing artifacts in {output_dir}"
            )
    os.replace(staging_dir, output_dir)


def write_request_shards(
    requests: Iterable[Mapping[str, Any]],
    *,
    stage: str,
    output_dir: str | Path,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    max_shard_requests: int = DEFAULT_MAX_SHARD_REQUESTS,
) -> dict[str, Any]:
    """Write canonical Batch JSONL shards and their authoritative index.

    Each input record must contain ``pair_index`` (a non-negative integer) and
    ``body`` (a Chat Completions request body). Optional ``method`` and ``url``
    fields are accepted only when they are exactly ``POST`` and
    ``/v1/chat/completions``. Any caller-provided ``custom_id`` is intentionally
    replaced by a deterministic ID derived from the stage, ordinal, and request
    hash. The ``pair_index`` never leaves the local request index.

    The function is idempotent for byte-identical artifacts. It will not replace
    a different or incomplete non-empty output directory.
    """

    if not isinstance(stage, str) or not _STAGE_RE.fullmatch(stage):
        raise ConfigurationError(
            "stage must match ^[a-z0-9][a-z0-9_-]{0,31}$"
        )
    _validate_limits(max_shard_bytes, max_shard_requests)

    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-", dir=destination.parent
        )
    )

    index_records: list[dict[str, Any]] = []
    shard_metadata: list[dict[str, Any]] = []
    current_payload = bytearray()
    current_record_indices: list[int] = []
    seen_pair_indices: set[int] = set()

    def close_shard() -> None:
        nonlocal current_payload, current_record_indices
        if current_record_indices:
            shard_number = len(shard_metadata)
            shard_name = f"requests-{shard_number:05d}.jsonl"
            shard_path = staging_dir / shard_name
            payload = current_payload
            save_bytes(payload, shard_path)
            shard_sha256 = sha256_bytes(payload)
            for line_number, index_position in enumerate(
                current_record_indices, start=1
            ):
                index_records[index_position].update(
                    {
                        "line_number": line_number,
                        "shard_number": shard_number,
                        "shard_path": shard_name,
                        "shard_sha256": shard_sha256,
                    }
                )
            shard_metadata.append(
                {
                    "bytes": len(payload),
                    "first_ordinal": index_records[current_record_indices[0]][
                        "ordinal"
                    ],
                    "last_ordinal": index_records[current_record_indices[-1]][
                        "ordinal"
                    ],
                    "path": shard_name,
                    "request_count": len(current_record_indices),
                    "sha256": shard_sha256,
                    "shard_number": shard_number,
                }
            )
            current_payload = bytearray()
            current_record_indices = []

    try:
        for ordinal, raw_record in enumerate(requests):
            if not isinstance(raw_record, Mapping):
                raise ValueError(f"Request {ordinal} is not an object")
            envelope, index_record = _normalize_request(
                raw_record, stage=stage, ordinal=ordinal
            )
            pair_index = index_record["pair_index"]
            if pair_index in seen_pair_indices:
                raise ValueError(f"Duplicate pair_index in requests: {pair_index}")
            seen_pair_indices.add(pair_index)

            line = canonical_json_bytes(envelope) + b"\n"
            if len(line) > max_shard_bytes:
                raise ValueError(
                    f"Request {ordinal} is {len(line):,} bytes and cannot fit in a "
                    f"{max_shard_bytes:,}-byte shard"
                )

            would_exceed_bytes = bool(current_record_indices) and (
                len(current_payload) + len(line) > max_shard_bytes
            )
            would_exceed_count = (
                len(current_record_indices) >= max_shard_requests
            )
            if would_exceed_bytes or would_exceed_count:
                close_shard()

            index_record["line_bytes"] = len(line)
            index_record["line_sha256"] = sha256_bytes(line)
            index_records.append(index_record)
            current_record_indices.append(len(index_records) - 1)
            current_payload.extend(line)

        close_shard()
        if not index_records:
            raise ValueError("At least one request is required")

        index_path = staging_dir / "request_index.jsonl"
        save_jsonl(index_records, index_path)
        index_metadata = {
            "bytes": index_path.stat().st_size,
            "count": len(index_records),
            "path": index_path.name,
            "sha256": sha256_file(index_path),
        }
        manifest: dict[str, Any] = {
            "endpoint": CHAT_COMPLETIONS_ENDPOINT,
            "max_shard_bytes": max_shard_bytes,
            "max_shard_requests": max_shard_requests,
            "request_count": len(index_records),
            "request_index": index_metadata,
            "schema_version": 1,
            "shard_count": len(shard_metadata),
            "shards": shard_metadata,
            "stage": stage,
        }
        save_json(manifest, staging_dir / "request_manifest.json")

        if destination.exists() and not destination.is_dir():
            raise ArtifactIntegrityError(
                f"Request artifact destination is not a directory: {destination}"
            )
        if destination.exists() and any(destination.iterdir()):
            existing = _verify_existing_directory(destination, manifest)
            if existing is None:
                raise ArtifactIntegrityError(
                    f"Refusing to replace incomplete request artifacts in {destination}"
                )
            shutil.rmtree(staging_dir)
            return existing

        _publish_staging_directory(staging_dir, destination)
        return manifest
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


shard_requests = write_request_shards


__all__ = [
    "CHAT_COMPLETIONS_ENDPOINT",
    "DEFAULT_MAX_SHARD_BYTES",
    "DEFAULT_MAX_SHARD_REQUESTS",
    "OPENAI_BATCH_MAX_BYTES",
    "PIPELINE_MAX_SHARD_REQUESTS",
    "shard_requests",
    "write_request_shards",
]
