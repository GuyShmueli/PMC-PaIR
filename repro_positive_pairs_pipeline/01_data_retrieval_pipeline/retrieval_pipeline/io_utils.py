from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


_PMC_ID_RE = re.compile(r"^(?:PMC)?(?P<id>\d+)$", flags=re.IGNORECASE)
_PATIENT_UID_RE = re.compile(
    r"^(?:PMC)?(?P<id>\d+)(?:[-_](?P<case>\d+))?$",
    flags=re.IGNORECASE,
)


def normalize_pmc_id(value: Any) -> str:
    text = str(value).strip()
    match = _PMC_ID_RE.fullmatch(text)
    if not match:
        raise ValueError(f"Invalid PMC ID: {value!r}")
    return str(int(match.group("id")))


def normalize_patient_uid(value: Any, *, default_case: int | None = None) -> str:
    text = str(value).strip()
    match = _PATIENT_UID_RE.fullmatch(text)
    if not match:
        raise ValueError(f"Invalid patient UID: {value!r}")

    case = match.group("case")
    if case is None:
        if default_case is None:
            raise ValueError(f"Patient UID has no case suffix: {value!r}")
        case = str(default_case)
    return f"{int(match.group('id'))}-{int(case)}"


def article_id_from_patient_uid(value: Any) -> str:
    normalized = normalize_patient_uid(value)
    return normalized.split("-", 1)[0]


def pmc_sort_key(value: str) -> tuple[int, str]:
    normalized = normalize_pmc_id(value)
    return int(normalized), normalized


def patient_uid_sort_key(value: str) -> tuple[int, int, str]:
    normalized = normalize_patient_uid(value)
    article_id, case = normalized.split("-", 1)
    return int(article_id), int(case), normalized


def figure_path_sort_key(value: str) -> tuple[int, str]:
    figure_number = Path(value).stem.rsplit("_", 1)[-1]
    return (int(figure_number) if figure_number.isdigit() else 0), value


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_replace_bytes(data: bytes, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def save_bytes(data: bytes, path: str | Path) -> None:
    _atomic_replace_bytes(data, path)


def save_text(text: str, path: str | Path) -> None:
    _atomic_replace_bytes(text.encode("utf-8"), path)


def save_json(value: Any, path: str | Path) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    save_text(payload + "\n", path)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        df.to_csv(temporary_path, index=False, lineterminator="\n")
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def require_columns(columns: Iterable[str], required: Iterable[str], *, label: str) -> None:
    missing = set(required) - set(columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def relative_posix(path: str | Path, base_dir: str | Path) -> str:
    return Path(path).resolve().relative_to(Path(base_dir).resolve()).as_posix()
