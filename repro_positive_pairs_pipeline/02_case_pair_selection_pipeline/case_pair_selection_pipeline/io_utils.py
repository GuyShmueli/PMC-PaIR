from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

from .errors import ArtifactIntegrityError


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    """Hash a tree using the exact portable contract of the Level-1 pipeline."""

    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Tree root is missing: {root}")
    digest = hashlib.sha256()
    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_replace(data: bytes, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def save_bytes(data: bytes, path: str | Path) -> None:
    _atomic_replace(data, path)


def save_text(text: str, path: str | Path) -> None:
    save_bytes(text.encode("utf-8"), path)


def save_json(value: Any, path: str | Path) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    save_bytes(payload, path)


def save_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for record in records:
                handle.write(canonical_json_bytes(record))
                handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            yield value


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def relative_posix(path: str | Path, root: str | Path) -> str:
    resolved_root = Path(root).resolve()
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path escapes run root: {path}") from exc


def resolve_relative(relative_path: str, root: str | Path) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Artifact path must be run-relative: {relative_path!r}")
    resolved_root = Path(root).resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Artifact path escapes run root: {relative_path!r}") from exc
    return resolved


def file_metadata(path: str | Path, *, root: str | Path) -> dict[str, Any]:
    item = Path(path)
    if not item.is_file():
        raise ArtifactIntegrityError(f"Expected artifact is missing: {item}")
    return {
        "path": relative_posix(item, root),
        "bytes": item.stat().st_size,
        "sha256": sha256_file(item),
    }


def verify_file_metadata(metadata: dict[str, Any], *, root: str | Path) -> Path:
    path = resolve_relative(str(metadata.get("path", "")), root)
    if not path.is_file():
        raise ArtifactIntegrityError(f"Recorded artifact is missing: {path}")
    actual_size = path.stat().st_size
    actual_hash = sha256_file(path)
    if actual_size != metadata.get("bytes") or actual_hash != metadata.get("sha256"):
        raise ArtifactIntegrityError(f"Recorded artifact changed: {path}")
    return path
