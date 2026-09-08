from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import PIPELINE_VERSION, artifact_path
from .io_utils import (
    normalize_patient_uid,
    patient_uid_sort_key,
    require_columns,
    save_dataframe,
    save_json,
    sha256_file,
)
from .patients import REQUIRED_COLUMNS, parse_similarity_mapping


def normalize_input_csvs(
    input_csvs: str | Path | Sequence[str | Path],
) -> list[Path]:
    """Return the single canonical patient source accepted by Pipeline 01."""
    if isinstance(input_csvs, (str, Path)):
        paths = [Path(input_csvs)]
    else:
        paths = [Path(path) for path in input_csvs]
    if len(paths) != 1:
        raise ValueError(
            "Pipeline 01 requires exactly one canonical unified --input-csv"
        )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Patient source CSV does not exist: {path}")
    return paths


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def materialize_unified_patient_input(
    input_csvs: str | Path | Sequence[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Normalize one canonical patient CSV into one deterministic source table.

    Duplicate normalized patient identifiers within the source are resolved
    deterministically: the later physical row supplies metadata, while
    similarity mappings are unioned and later scores win on repeated targets.
    """
    source_path = normalize_input_csvs(input_csvs)[0]
    output_path = artifact_path(output_dir, "unified_patients_csv")
    provenance_path = artifact_path(output_dir, "unification_provenance_csv")
    report_path = artifact_path(output_dir, "unification_report_json")

    frame = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    require_columns(frame.columns, REQUIRED_COLUMNS, label=str(source_path))
    output_columns = frame.columns.tolist()
    metadata_columns = [
        column for column in output_columns if column not in REQUIRED_COLUMNS
    ]
    metadata_by_uid: dict[str, dict[str, str]] = {}
    similarities_by_uid: dict[str, dict[str, Any]] = defaultdict(dict)
    provenance_by_uid: dict[str, list[int]] = defaultdict(list)
    parse_status_counts: Counter[str] = Counter()
    metadata_conflicts_by_column: Counter[str] = Counter()
    metadata_conflict_uids: set[str] = set()
    similarity_score_conflicts = 0
    similarity_overlap_count = 0
    input_similarity_links = 0

    for csv_row, values in enumerate(frame.itertuples(index=False, name=None), start=2):
        row = dict(zip(output_columns, values))
        try:
            uid = normalize_patient_uid(row["patient_uid"])
        except ValueError as exc:
            raise ValueError(
                f"Invalid patient_uid in {source_path} at CSV row "
                f"{csv_row}: {row['patient_uid']!r}"
            ) from exc
        provenance_by_uid[uid].append(csv_row)
        metadata = {column: row[column] for column in metadata_columns}
        previous_metadata = metadata_by_uid.get(uid, {})
        for column, value in metadata.items():
            if column in previous_metadata and previous_metadata[column] != value:
                metadata_conflicts_by_column[column] += 1
                metadata_conflict_uids.add(uid)
        metadata_by_uid[uid] = metadata

        mapping, status = parse_similarity_mapping(row["similar_patients"])
        parse_status_counts[status] += 1
        input_similarity_links += len(mapping)
        merged_mapping = similarities_by_uid[uid]
        for target_uid, score in mapping.items():
            if target_uid in merged_mapping:
                similarity_overlap_count += 1
                if _canonical_json(merged_mapping[target_uid]) != _canonical_json(score):
                    similarity_score_conflicts += 1
            merged_mapping[target_uid] = score

    ordered_uids = sorted(metadata_by_uid, key=patient_uid_sort_key)
    unified_rows: list[dict[str, str]] = []
    provenance_rows: list[dict[str, str | int]] = []
    for uid in ordered_uids:
        mapping = similarities_by_uid[uid]
        unified_rows.append({
            **metadata_by_uid[uid],
            "patient_uid": uid,
            "similar_patients": _canonical_json(mapping),
        })
        occurrences = provenance_by_uid[uid]
        provenance_rows.append({
            "patient_uid": uid,
            "source_indices": "[0]",
            "source_csv_rows": _canonical_json([
                {"source_index": 0, "csv_row": csv_row}
                for csv_row in occurrences
            ]),
            "highest_priority_row_source_index": 0,
            "highest_priority_csv_row": occurrences[-1],
            "metadata_column_source_indices": _canonical_json({
                column: 0 for column in metadata_columns
            }),
            "similarity_target_source_indices": _canonical_json({
                target_uid: 0 for target_uid in mapping
            }),
        })

    unified_df = pd.DataFrame(unified_rows, columns=output_columns)
    provenance_df = pd.DataFrame(
        provenance_rows,
        columns=[
            "patient_uid",
            "source_indices",
            "source_csv_rows",
            "highest_priority_row_source_index",
            "highest_priority_csv_row",
            "metadata_column_source_indices",
            "similarity_target_source_indices",
        ],
    )
    save_dataframe(unified_df, output_path)
    save_dataframe(provenance_df, provenance_path)

    duplicate_uid_count = sum(len(rows) > 1 for rows in provenance_by_uid.values())
    parse_counts = dict(sorted(parse_status_counts.items()))
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "stage": "00_materialize_unified_patient_source",
        "dataset_mode": "unified",
        "resolution_policy": {
            "source_order": "one canonical source",
            "metadata": (
                "the later row wins for columns present in its source, including "
                "empty values"
            ),
            "similarity_targets": (
                "union across rows; the later row's score wins for a repeated target"
            ),
            "within_source_duplicates": "later CSV row has higher priority",
        },
        "sources": [{
            "priority": 0,
            "name": source_path.name,
            "sha256": sha256_file(source_path),
            "rows": int(len(frame)),
            "unique_normalized_patient_uids": len(ordered_uids),
        }],
        "source_file_count": 1,
        "input_rows": int(len(frame)),
        "output_rows": int(len(unified_df)),
        "rows_collapsed": int(len(frame) - len(unified_df)),
        "duplicate_patient_uid_count": duplicate_uid_count,
        "cross_source_duplicate_patient_uid_count": 0,
        "within_source_duplicate_patient_uid_count": duplicate_uid_count,
        "metadata_conflict_event_count": int(sum(metadata_conflicts_by_column.values())),
        "metadata_conflict_patient_uid_count": len(metadata_conflict_uids),
        "metadata_conflicts_by_column": dict(sorted(metadata_conflicts_by_column.items())),
        "similarity_links_input": input_similarity_links,
        "similarity_target_overlap_count": similarity_overlap_count,
        "similarity_score_conflict_count": similarity_score_conflicts,
        "similarity_links_output": sum(len(value) for value in similarities_by_uid.values()),
        "parse_status_counts": parse_counts,
        "parse_status_counts_by_source": [parse_counts],
        "outputs": {
            output_path.name: sha256_file(output_path),
            provenance_path.name: sha256_file(provenance_path),
        },
    }
    save_json(report, report_path)
    return report
