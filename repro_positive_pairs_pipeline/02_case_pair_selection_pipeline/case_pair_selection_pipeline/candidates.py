from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import UPSTREAM_ASSETS, UPSTREAM_MERGED
from .io_utils import sha256_file


SCHEMA_VERSION = 1
REQUIRED_COLUMNS = {
    "image_path",
    "caption_path",
    "patient_uid",
    "pmc_id",
    "unique_articles_sim_patients",
}
_PATIENT_UID_RE = re.compile(
    r"^(?:PMC)?(?P<article>\d+)(?:[-_](?P<case>\d+))$",
    flags=re.IGNORECASE,
)
_PMC_ID_RE = re.compile(r"^(?:PMC)?(?P<article>\d+)$", flags=re.IGNORECASE)


def _normalize_patient_uid(value: Any) -> str:
    match = _PATIENT_UID_RE.fullmatch(str(value).strip())
    if not match:
        raise ValueError(f"Invalid patient_uid: {value!r}")
    return f"{int(match.group('article'))}-{int(match.group('case'))}"


def _normalize_pmc_id(value: Any) -> str:
    match = _PMC_ID_RE.fullmatch(str(value).strip())
    if not match:
        raise ValueError(f"Invalid pmc_id: {value!r}")
    return str(int(match.group("article")))


def _uid_sort_key(uid: str) -> tuple[int, int, str]:
    article, case = _normalize_patient_uid(uid).split("-", 1)
    return int(article), int(case), uid


def _pair_key(case_a_uid: str, case_b_uid: str) -> str:
    payload = f"{case_a_uid}\0{case_b_uid}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_blacklist(value: str | Path | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        source = Path(value)
        with source.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, list):
            raise ValueError(f"Diagram blacklist must be a JSON list: {source}")
        return [str(item) for item in loaded]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    raise TypeError("diagram_blacklist must be a JSON path, an iterable, or None")


def _resolve_run_path(
    raw_path: Any,
    run_root: Path,
    *,
    label: str,
    suffix: str | None = None,
    require_file: bool = True,
) -> tuple[Path, str]:
    """Resolve a recorded path without permitting traversal or symlink escape."""
    value = str(raw_path).strip().replace("\\", "/")
    if not value:
        raise ValueError(f"{label} is empty")

    supplied = Path(value)
    if supplied.is_absolute():
        resolved = supplied.resolve()
    else:
        if ".." in supplied.parts:
            raise ValueError(f"{label} contains parent traversal: {value!r}")
        resolved = (run_root / supplied).resolve()

    resolved_root = run_root.resolve()
    resolved_data = (resolved_root / UPSTREAM_ASSETS).resolve()
    try:
        relative = resolved.relative_to(resolved_root)
        resolved.relative_to(resolved_data)
    except ValueError as exc:
        raise ValueError(
            f"{label} must resolve inside the retrieval run's unified asset tree "
            f"({UPSTREAM_ASSETS}/): {value!r}"
        ) from exc

    if suffix is not None and resolved.suffix.lower() != suffix:
        raise ValueError(f"{label} must have suffix {suffix!r}: {value!r}")
    if require_file and not resolved.is_file():
        raise ValueError(f"{label} is missing or is not a file: {relative.as_posix()}")
    return resolved, relative.as_posix()


def _parse_similarity_list(raw: Any) -> list[str]:
    text = str(raw).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "unique_articles_sim_patients must contain a JSON array"
        ) from exc
    if not isinstance(parsed, list):
        raise ValueError("unique_articles_sim_patients must contain a JSON array")
    normalized = {_normalize_patient_uid(item) for item in parsed}
    return sorted(normalized, key=_uid_sort_key)


def build_candidates(
    retrieval_run: str | Path,
    diagram_blacklist: str | Path | Iterable[str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build deterministic case and undirected-pair records from a Level-1 run.

    The Level-1 CSV is treated as a manifest, not as a source of trusted paths:
    every referenced file is resolved and checked beneath the unified asset
    tree before any record is returned.
    """
    run_root = Path(retrieval_run).resolve()
    if not run_root.is_dir():
        raise ValueError(f"Retrieval run is not a directory: {retrieval_run}")
    if not (run_root / UPSTREAM_ASSETS).is_dir():
        raise ValueError(
            f"Retrieval run has no unified {UPSTREAM_ASSETS}/ asset directory: {run_root}"
        )

    merged_path = run_root / UPSTREAM_MERGED
    if not merged_path.is_file():
        raise ValueError(f"Retrieval run has no {UPSTREAM_MERGED}: {run_root}")
    frame = pd.read_csv(merged_path, dtype=str, keep_default_na=False)
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{UPSTREAM_MERGED} is missing required columns: {sorted(missing_columns)}"
        )

    raw_blacklist = _load_blacklist(diagram_blacklist)
    blacklist: set[str] = set()
    for entry in raw_blacklist:
        _, relative = _resolve_run_path(
            entry,
            run_root,
            label="diagram blacklist path",
            suffix=".jpg",
            require_file=True,
        )
        blacklist.add(relative)

    seen_image_paths: set[str] = set()
    seen_caption_paths: set[str] = set()
    seen_caption_ids: dict[str, str] = {}
    rows_by_uid: dict[str, list[dict[str, Any]]] = {}
    similarities_by_uid: dict[str, list[str]] = {}
    all_manifest_images: set[str] = set()
    selected_rows = 0

    for row_number, row in enumerate(frame.to_dict(orient="records"), start=2):
        uid = _normalize_patient_uid(row["patient_uid"])
        pmc_id = _normalize_pmc_id(row["pmc_id"])
        article_id, case_number = uid.split("-", 1)
        if pmc_id != article_id:
            raise ValueError(
                f"{UPSTREAM_MERGED} row {row_number}: pmc_id does not match patient_uid"
            )

        expected_directory = (
            run_root
            / UPSTREAM_ASSETS
            / f"PMC{article_id}"
            / f"{article_id}_{case_number}"
        ).resolve()
        image, image_relative = _resolve_run_path(
            row["image_path"],
            run_root,
            label=f"row {row_number} image_path",
            suffix=".jpg",
        )
        caption, caption_relative = _resolve_run_path(
            row["caption_path"],
            run_root,
            label=f"row {row_number} caption_path",
            suffix=".txt",
        )
        # Pipeline 01 retains the source-article URL in ``article_path``.  The
        # canonical local NXML is materialized in every case directory, so its
        # location is derived from the already validated patient UID instead
        # of interpreting that provenance URL as a filesystem path.
        nxml, nxml_relative = _resolve_run_path(
            expected_directory / "article.nxml",
            run_root,
            label=f"row {row_number} canonical article.nxml",
            suffix=".nxml",
        )
        if image.parent != expected_directory or caption.parent != expected_directory:
            raise ValueError(
                f"{UPSTREAM_MERGED} row {row_number}: image/caption is outside its case directory"
            )
        if image.stem != caption.stem:
            raise ValueError(
                f"{UPSTREAM_MERGED} row {row_number}: image and caption stems differ"
            )
        if image_relative in seen_image_paths:
            raise ValueError(
                f"Duplicate image_path in {UPSTREAM_MERGED}: {image_relative}"
            )
        if caption_relative in seen_caption_paths:
            raise ValueError(
                f"Duplicate caption_path in {UPSTREAM_MERGED}: {caption_relative}"
            )
        seen_image_paths.add(image_relative)
        seen_caption_paths.add(caption_relative)
        all_manifest_images.add(image_relative)

        caption_id = image.stem
        if caption_id in seen_caption_ids:
            raise ValueError(
                f"Duplicate caption ID {caption_id!r} in {image_relative} and "
                f"{seen_caption_ids[caption_id]}"
            )
        seen_caption_ids[caption_id] = image_relative

        similarities = _parse_similarity_list(row["unique_articles_sim_patients"])
        previous_similarities = similarities_by_uid.setdefault(uid, similarities)
        if previous_similarities != similarities:
            raise ValueError(
                f"Inconsistent unique_articles_sim_patients values for {uid}"
            )

        if image_relative in blacklist:
            continue
        try:
            caption_text = caption.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(f"Caption is not valid UTF-8: {caption_relative}") from exc
        if not caption_text:
            raise ValueError(f"Caption is empty: {caption_relative}")

        rows_by_uid.setdefault(uid, []).append(
            {
                "caption_id": caption_id,
                "caption": caption_text,
                "caption_path": caption_relative,
                "caption_sha256": sha256_file(caption),
                "image_path": image_relative,
                "image_sha256": sha256_file(image),
                "nxml_path": nxml_relative,
                "nxml_file": nxml,
            }
        )
        selected_rows += 1

    unknown_blacklist = sorted(blacklist - all_manifest_images)
    if unknown_blacklist:
        raise ValueError(
            f"Diagram blacklist contains images absent from {UPSTREAM_MERGED}: "
            f"{unknown_blacklist[:20]}"
        )

    cases: list[dict[str, Any]] = []
    for uid in sorted(rows_by_uid, key=_uid_sort_key):
        rows = sorted(
            rows_by_uid[uid],
            key=lambda item: (item["caption_id"], item["image_path"]),
        )
        nxml_paths = {str(item["nxml_path"]) for item in rows}
        if len(nxml_paths) != 1:
            raise ValueError(f"Case {uid} refers to more than one NXML file")
        nxml_files = {Path(item["nxml_file"]) for item in rows}
        if len(nxml_files) != 1:
            raise ValueError(f"Case {uid} resolves to more than one NXML file")
        nxml_file = next(iter(nxml_files))
        captions = [
            {key: value for key, value in item.items() if key not in {"nxml_path", "nxml_file"}}
            for item in rows
        ]
        cases.append(
            {
                "schema_version": SCHEMA_VERSION,
                "patient_uid": uid,
                "pmc_id": uid.split("-", 1)[0],
                "nxml_path": next(iter(nxml_paths)),
                "nxml_sha256": sha256_file(nxml_file),
                "captions": captions,
            }
        )

    available_uids = {case["patient_uid"] for case in cases}
    pair_values: set[tuple[str, str]] = set()
    unavailable_targets: set[str] = set()
    for source_uid in sorted(available_uids, key=_uid_sort_key):
        for target_uid in similarities_by_uid.get(source_uid, []):
            if target_uid == source_uid:
                continue
            if target_uid not in available_uids:
                unavailable_targets.add(target_uid)
                continue
            ordered = tuple(sorted((source_uid, target_uid), key=_uid_sort_key))
            pair_values.add((ordered[0], ordered[1]))

    ordered_pairs = sorted(
        pair_values,
        key=lambda pair: (_uid_sort_key(pair[0]), _uid_sort_key(pair[1])),
    )
    pairs = [
        {
            "schema_version": SCHEMA_VERSION,
            "pair_index": pair_index,
            "pair_key": _pair_key(case_a_uid, case_b_uid),
            "case_a_uid": case_a_uid,
            "case_b_uid": case_b_uid,
        }
        for pair_index, (case_a_uid, case_b_uid) in enumerate(ordered_pairs)
    ]

    stats = {
        "schema_version": SCHEMA_VERSION,
        "merged_csv_sha256": sha256_file(merged_path),
        "rows_input": int(len(frame)),
        "diagram_blacklist_entries": len(blacklist),
        "rows_excluded_as_diagrams": int(len(frame) - selected_rows),
        "rows_output": selected_rows,
        "cases_output": len(cases),
        "pairs_output": len(pairs),
        "similarity_targets_without_selected_cases": sorted(
            unavailable_targets,
            key=_uid_sort_key,
        ),
    }
    return cases, pairs, stats
