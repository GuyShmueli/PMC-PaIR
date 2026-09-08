"""Fail-closed, idempotent OpenAI Batch lifecycle management.

The module imports no concrete OpenAI client. Callers inject an initialized
client, which keeps tests offline and makes every external action explicit.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .errors import ArtifactIntegrityError, BatchStateError, ConfigurationError
from .io_utils import load_json, save_bytes, save_json, sha256_bytes, sha256_file
from .sharding import CHAT_COMPLETIONS_ENDPOINT, OPENAI_BATCH_MAX_BYTES


COMPLETION_WINDOW = "24h"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REMOTE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_AMBIGUOUS_STATES = {"ambiguous_create", "create_in_progress"}


def _require_acknowledgement(value: bool) -> None:
    if value is not True:
        raise ConfigurationError(
            "External OpenAI API access was not acknowledged; pass "
            "acknowledge_external_api=True explicitly"
        )


def _to_serializable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except TypeError:
            return value.model_dump()
    if isinstance(value, Mapping):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            str(key): _to_serializable(item)
            for key, item in attributes.items()
            if not str(key).startswith("_")
        }
    return str(value)


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _validate_sha256(value: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"Invalid SHA-256 digest: {value!r}")
    return value


def _validate_remote_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _REMOTE_ID_RE.fullmatch(value):
        raise BatchStateError(f"Invalid {label}: {value!r}")
    return value


def _content_bytes(response: Any) -> bytes:
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content
    if isinstance(content, bytearray):
        return bytes(content)
    read = getattr(response, "read", None)
    if callable(read):
        value = read()
        if isinstance(value, str):
            return value.encode("utf-8")
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text.encode("utf-8")
    if isinstance(response, (bytes, bytearray)):
        return bytes(response)
    raise BatchStateError("OpenAI file-content response did not contain bytes")


def _publish_directory(staging_dir: Path, destination: Path) -> None:
    if destination.exists():
        if destination.is_dir() and not any(destination.iterdir()):
            destination.rmdir()
        else:
            raise ArtifactIntegrityError(
                f"Refusing to overwrite existing batch download: {destination}"
            )
    os.replace(staging_dir, destination)


class BatchManager:
    """Manage one durable submission record per exact shard SHA-256."""

    def __init__(self, *, client: Any, state_dir: str | Path) -> None:
        if client is None:
            raise ConfigurationError("An initialized OpenAI client must be injected")
        self.client = client
        self.state_dir = Path(state_dir)
        self.submission_dir = self.state_dir / "submissions"
        self.download_dir = self.state_dir / "downloads"

    def _record_path(self, shard_sha256: str) -> Path:
        return self.submission_dir / f"{_validate_sha256(shard_sha256)}.json"

    def _load_record(self, shard_sha256: str) -> dict[str, Any]:
        path = self._record_path(shard_sha256)
        if not path.is_file():
            raise BatchStateError(
                f"No submission record exists for shard {shard_sha256}"
            )
        try:
            record = load_json(path)
        except Exception as exc:
            raise ArtifactIntegrityError(
                f"Submission record cannot be parsed: {path}"
            ) from exc
        if not isinstance(record, dict) or record.get("shard_sha256") != shard_sha256:
            raise ArtifactIntegrityError(f"Submission record is inconsistent: {path}")
        return record

    def _save_record(self, record: dict[str, Any]) -> None:
        save_json(record, self._record_path(record["shard_sha256"]))

    def submit_shard(
        self,
        shard_path: str | Path,
        *,
        expected_sha256: str | None = None,
        stage: str | None = None,
        description: str | None = None,
        metadata: Mapping[str, str] | None = None,
        acknowledge_external_api: bool = False,
    ) -> dict[str, Any]:
        """Upload and submit a shard once, durably blocking ambiguous retries."""

        _require_acknowledgement(acknowledge_external_api)
        path = Path(shard_path)
        if not path.is_file():
            raise ArtifactIntegrityError(f"Batch shard is missing: {path}")
        if path.suffix != ".jsonl":
            raise ArtifactIntegrityError(f"Batch shard must be JSONL: {path}")
        size = path.stat().st_size
        if size <= 0 or size >= OPENAI_BATCH_MAX_BYTES:
            raise ArtifactIntegrityError(
                f"Batch shard must be non-empty and below "
                f"{OPENAI_BATCH_MAX_BYTES:,} bytes: {path}"
            )
        shard_sha256 = sha256_file(path)
        if expected_sha256 is not None and _validate_sha256(expected_sha256) != shard_sha256:
            raise ArtifactIntegrityError(f"Batch shard digest changed: {path}")

        record_path = self._record_path(shard_sha256)
        record: dict[str, Any] | None = None
        if record_path.exists():
            record = self._load_record(shard_sha256)
            state = record.get("state")
            if state in _AMBIGUOUS_STATES:
                raise BatchStateError(
                    f"Submission for shard {shard_sha256} is in ambiguous-create "
                    "state. Do not resubmit automatically; resolve it using the "
                    "recorded upload file ID and the OpenAI dashboard/API."
                )
            if record.get("batch_id"):
                # Idempotent success: a durable batch ID always wins over retries.
                return record
            if state not in {"upload_pending", "upload_failed", "uploaded"}:
                raise BatchStateError(
                    f"Submission record has unsupported state {state!r}: {record_path}"
                )

        base_record: dict[str, Any] = {
            "completion_window": COMPLETION_WINDOW,
            "endpoint": CHAT_COMPLETIONS_ENDPOINT,
            "schema_version": 1,
            "shard_bytes": size,
            # Content identity is authoritative; a basename is sufficient context
            # without leaking a machine-specific absolute run path.
            "shard_path": path.name,
            "shard_sha256": shard_sha256,
            "stage": stage,
        }

        uploaded_file_id = record.get("uploaded_file_id") if record else None
        uploaded_file_snapshot = record.get("uploaded_file") if record else None
        if not uploaded_file_id:
            pending = {**base_record, "state": "upload_pending"}
            self._save_record(pending)
            try:
                with path.open("rb") as handle:
                    uploaded = self.client.files.create(file=handle, purpose="batch")
            except Exception as exc:
                self._save_record(
                    {**pending, "state": "upload_failed", "last_error": str(exc)}
                )
                raise BatchStateError(f"Batch shard upload failed: {path}") from exc
            uploaded_file_id = _validate_remote_id(
                _field(uploaded, "id"), label="uploaded file ID"
            )
            uploaded_file_snapshot = _to_serializable(uploaded)
            record = {
                **base_record,
                "state": "uploaded",
                "uploaded_file": uploaded_file_snapshot,
                "uploaded_file_id": uploaded_file_id,
            }
            self._save_record(record)

        batch_metadata: dict[str, str] = {
            "shard_sha256": shard_sha256,
        }
        if stage:
            batch_metadata["stage"] = str(stage)
        if description:
            batch_metadata["description"] = str(description)
        if metadata:
            for key, value in metadata.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise ConfigurationError("Batch metadata keys and values must be strings")
                if key in batch_metadata and batch_metadata[key] != value:
                    raise ConfigurationError(
                        f"Batch metadata cannot override {key!r}"
                    )
                batch_metadata[key] = value
        if len(batch_metadata) > 16:
            raise ConfigurationError("Batch metadata cannot contain more than 16 entries")

        # This durable marker is intentionally written *before* the create call.
        # A process/network failure after this point may have created the remote
        # batch, so an automatic retry could duplicate a costly workload.
        ambiguous = {
            **base_record,
            "metadata": batch_metadata,
            "state": "ambiguous_create",
            "uploaded_file": uploaded_file_snapshot,
            "uploaded_file_id": uploaded_file_id,
        }
        self._save_record(ambiguous)
        try:
            batch = self.client.batches.create(
                input_file_id=uploaded_file_id,
                endpoint=CHAT_COMPLETIONS_ENDPOINT,
                completion_window=COMPLETION_WINDOW,
                metadata=batch_metadata,
            )
        except Exception as exc:
            self._save_record({**ambiguous, "last_error": str(exc)})
            raise BatchStateError(
                "Batch create returned an error after submission may have reached "
                "OpenAI; the shard is now ambiguous and will not be resubmitted"
            ) from exc

        batch_id = _validate_remote_id(_field(batch, "id"), label="batch ID")
        returned_endpoint = _field(batch, "endpoint")
        if returned_endpoint not in (None, CHAT_COMPLETIONS_ENDPOINT):
            self._save_record(
                {
                    **ambiguous,
                    "batch": _to_serializable(batch),
                    "batch_id": batch_id,
                    "last_error": f"Unexpected endpoint {returned_endpoint!r}",
                }
            )
            raise BatchStateError("Created batch reports an unexpected endpoint")
        submitted = {
            **base_record,
            "batch": _to_serializable(batch),
            "batch_id": batch_id,
            "metadata": batch_metadata,
            "remote_status": _field(batch, "status"),
            "state": "submitted",
            "uploaded_file": uploaded_file_snapshot,
            "uploaded_file_id": uploaded_file_id,
        }
        self._save_record(submitted)
        return submitted

    def retrieve_status(
        self,
        shard_sha256: str,
        *,
        acknowledge_external_api: bool = False,
    ) -> dict[str, Any]:
        """Retrieve and durably record the remote status for a known shard."""

        _require_acknowledgement(acknowledge_external_api)
        record = self._load_record(_validate_sha256(shard_sha256))
        if record.get("state") in _AMBIGUOUS_STATES:
            raise BatchStateError(
                "Cannot retrieve by shard while batch creation is ambiguous; "
                "resolve the batch ID first"
            )
        batch_id = _validate_remote_id(record.get("batch_id"), label="batch ID")
        try:
            batch = self.client.batches.retrieve(batch_id)
        except Exception as exc:
            raise BatchStateError(f"Could not retrieve batch {batch_id}") from exc
        if _field(batch, "id") not in (None, batch_id):
            raise BatchStateError("Retrieved batch ID does not match submission record")
        endpoint = _field(batch, "endpoint")
        if endpoint not in (None, CHAT_COMPLETIONS_ENDPOINT):
            raise BatchStateError("Retrieved batch has an unexpected endpoint")
        input_file_id = _field(batch, "input_file_id")
        if input_file_id not in (None, record.get("uploaded_file_id")):
            raise BatchStateError("Retrieved batch has an unexpected input file")

        snapshot = _to_serializable(batch)
        status = _field(batch, "status")
        updated = {
            **record,
            "batch": snapshot,
            "remote_status": status,
            "state": "completed" if status == "completed" else "submitted",
        }
        self._save_record(updated)
        return snapshot

    def resolve_ambiguous_create(
        self,
        shard_sha256: str,
        *,
        batch_id: str,
        acknowledge_external_api: bool = False,
    ) -> dict[str, Any]:
        """Bind a manually discovered batch ID to an ambiguous shard safely."""

        _require_acknowledgement(acknowledge_external_api)
        shard_sha256 = _validate_sha256(shard_sha256)
        batch_id = _validate_remote_id(batch_id, label="batch ID")
        record = self._load_record(shard_sha256)
        if record.get("state") not in _AMBIGUOUS_STATES:
            raise BatchStateError("Submission is not in ambiguous-create state")
        try:
            batch = self.client.batches.retrieve(batch_id)
        except Exception as exc:
            raise BatchStateError(f"Could not retrieve candidate batch {batch_id}") from exc
        if _field(batch, "id") not in (None, batch_id):
            raise BatchStateError("Candidate batch ID does not match")
        if _field(batch, "endpoint") not in (None, CHAT_COMPLETIONS_ENDPOINT):
            raise BatchStateError("Candidate batch uses an unexpected endpoint")
        if _field(batch, "input_file_id") != record.get("uploaded_file_id"):
            raise BatchStateError(
                "Candidate batch input file does not match the uploaded shard"
            )
        resolved = {
            **record,
            "batch": _to_serializable(batch),
            "batch_id": batch_id,
            "remote_status": _field(batch, "status"),
            "state": (
                "completed" if _field(batch, "status") == "completed" else "submitted"
            ),
        }
        resolved.pop("last_error", None)
        self._save_record(resolved)
        return resolved

    def _verified_existing_download(
        self, destination: Path, batch_id: str, shard_sha256: str
    ) -> dict[str, Any] | None:
        manifest_path = destination / "download_manifest.json"
        if not manifest_path.is_file():
            return None
        manifest = load_json(manifest_path)
        if (
            not isinstance(manifest, dict)
            or manifest.get("batch_id") != batch_id
            or manifest.get("shard_sha256") != shard_sha256
        ):
            raise ArtifactIntegrityError(
                f"Existing download manifest is inconsistent: {manifest_path}"
            )
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise ArtifactIntegrityError("Download manifest has no file metadata")
        recorded_names: set[str] = set()
        for metadata in files:
            if not isinstance(metadata, Mapping):
                raise ArtifactIntegrityError("Malformed download file metadata")
            relative = metadata.get("path")
            if not isinstance(relative, str) or Path(relative).name != relative:
                raise ArtifactIntegrityError("Unsafe download artifact path")
            path = destination / relative
            if relative in recorded_names:
                raise ArtifactIntegrityError(
                    f"Duplicate artifact in download manifest: {relative}"
                )
            recorded_names.add(relative)
            if (
                not path.is_file()
                or path.stat().st_size != metadata.get("bytes")
                or sha256_file(path) != metadata.get("sha256")
            ):
                raise ArtifactIntegrityError(
                    f"Existing downloaded artifact changed: {path}"
                )
        if "output.jsonl" not in recorded_names or "batch.json" not in recorded_names:
            raise ArtifactIntegrityError(
                "Download manifest omits required output.jsonl or batch.json"
            )
        return manifest

    def download_completed(
        self,
        shard_sha256: str,
        *,
        acknowledge_external_api: bool = False,
    ) -> dict[str, Any]:
        """Download a completed batch into an atomic, batch-ID-scoped directory."""

        _require_acknowledgement(acknowledge_external_api)
        shard_sha256 = _validate_sha256(shard_sha256)
        batch = self.retrieve_status(
            shard_sha256, acknowledge_external_api=True
        )
        if _field(batch, "status") != "completed":
            raise BatchStateError(
                f"Batch is not completed (status={_field(batch, 'status')!r})"
            )
        batch_id = _validate_remote_id(_field(batch, "id"), label="batch ID")
        destination = self.download_dir / batch_id
        if destination.exists() and not destination.is_dir():
            raise ArtifactIntegrityError(
                f"Batch download destination is not a directory: {destination}"
            )
        if destination.exists() and any(destination.iterdir()):
            existing = self._verified_existing_download(
                destination, batch_id, shard_sha256
            )
            if existing is None:
                raise ArtifactIntegrityError(
                    f"Existing batch download is incomplete: {destination}"
                )
            return existing

        output_file_id = _field(batch, "output_file_id")
        if output_file_id is not None:
            output_file_id = _validate_remote_id(output_file_id, label="output file ID")
        error_file_id = _field(batch, "error_file_id")
        if error_file_id is not None:
            error_file_id = _validate_remote_id(error_file_id, label="error file ID")
        if output_file_id is None and error_file_id is None:
            raise BatchStateError(f"Completed batch {batch_id} has no output or error file")

        try:
            output_payload = (
                _content_bytes(self.client.files.content(output_file_id))
                if output_file_id is not None
                else b""
            )
            error_payload = (
                _content_bytes(self.client.files.content(error_file_id))
                if error_file_id
                else None
            )
        except Exception as exc:
            if isinstance(exc, BatchStateError):
                raise
            raise BatchStateError(f"Could not download files for batch {batch_id}") from exc
        self.download_dir.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{batch_id}.staging-", dir=self.download_dir
            )
        )
        try:
            files: list[dict[str, Any]] = []

            def write_artifact(name: str, payload: bytes) -> None:
                save_bytes(payload, staging_dir / name)
                files.append(
                    {
                        "bytes": len(payload),
                        "path": name,
                        "sha256": sha256_bytes(payload),
                    }
                )

            write_artifact("output.jsonl", output_payload)
            if error_payload is not None:
                write_artifact("errors.jsonl", error_payload)
            save_json(batch, staging_dir / "batch.json")
            batch_payload_path = staging_dir / "batch.json"
            files.append(
                {
                    "bytes": batch_payload_path.stat().st_size,
                    "path": "batch.json",
                    "sha256": sha256_file(batch_payload_path),
                }
            )
            manifest = {
                "batch_id": batch_id,
                "files": files,
                "output_file_id": output_file_id,
                "error_file_id": error_file_id,
                "schema_version": 1,
                "shard_sha256": shard_sha256,
            }
            save_json(manifest, staging_dir / "download_manifest.json")
            _publish_directory(staging_dir, destination)
        except Exception:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            raise

        record = self._load_record(shard_sha256)
        record["download"] = {
            "batch_id": batch_id,
            "manifest_path": f"downloads/{batch_id}/download_manifest.json",
        }
        self._save_record(record)
        return manifest


def submit_shard(
    *,
    client: Any,
    state_dir: str | Path,
    shard_path: str | Path,
    acknowledge_external_api: bool,
    **kwargs: Any,
) -> dict[str, Any]:
    """Functional wrapper around :meth:`BatchManager.submit_shard`."""

    return BatchManager(client=client, state_dir=state_dir).submit_shard(
        shard_path,
        acknowledge_external_api=acknowledge_external_api,
        **kwargs,
    )


def retrieve_batch_status(
    *,
    client: Any,
    state_dir: str | Path,
    shard_sha256: str,
    acknowledge_external_api: bool,
) -> dict[str, Any]:
    """Functional wrapper around :meth:`BatchManager.retrieve_status`."""

    return BatchManager(client=client, state_dir=state_dir).retrieve_status(
        shard_sha256, acknowledge_external_api=acknowledge_external_api
    )


def download_completed_batch(
    *,
    client: Any,
    state_dir: str | Path,
    shard_sha256: str,
    acknowledge_external_api: bool,
) -> dict[str, Any]:
    """Functional wrapper around :meth:`BatchManager.download_completed`."""

    return BatchManager(client=client, state_dir=state_dir).download_completed(
        shard_sha256, acknowledge_external_api=acknowledge_external_api
    )


__all__ = [
    "BatchManager",
    "COMPLETION_WINDOW",
    "download_completed_batch",
    "retrieve_batch_status",
    "submit_shard",
]
