from __future__ import annotations

import ast
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import PIPELINE_VERSION, artifact_path
from .io_utils import (
    article_id_from_patient_uid,
    normalize_patient_uid,
    patient_uid_sort_key,
    require_columns,
    save_dataframe,
    save_json,
    sha256_file,
)


REQUIRED_COLUMNS = {"patient_uid", "similar_patients"}


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def parse_similarity_mapping(raw: Any) -> tuple[dict[str, Any], str]:
    """Parse one similarity cell and return (canonical mapping, parse status)."""
    if isinstance(raw, Mapping):
        parsed: Any = dict(raw)
    elif raw is None:
        return {}, "missing"
    else:
        try:
            missing = pd.isna(raw)
        except Exception:
            missing = False
        try:
            if bool(missing):
                return {}, "missing"
        except (TypeError, ValueError):
            pass

        text = str(raw).strip()
        if not text:
            return {}, "missing"

        parsed = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                break
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue

        if parsed is None:
            return {}, "malformed"

    if not isinstance(parsed, Mapping):
        return {}, "non_mapping"

    normalized: dict[str, Any] = {}
    invalid_keys = 0
    for raw_uid, score in parsed.items():
        if not isinstance(raw_uid, str):
            invalid_keys += 1
            continue
        try:
            uid = normalize_patient_uid(raw_uid)
        except ValueError:
            invalid_keys += 1
            continue
        normalized[uid] = _json_safe(score)

    ordered = {
        uid: normalized[uid]
        for uid in sorted(normalized, key=patient_uid_sort_key)
    }
    if invalid_keys:
        return ordered, "invalid_keys"
    return ordered, "ok"


def prepare_patient_cohort(
    input_csv: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """
    Build the single-patient article cohort used by the retrieval stages.

    Article multiplicity is calculated across the complete canonical patient
    table before similarity mappings are filtered.
    """
    input_path = Path(input_csv)
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    require_columns(df.columns, REQUIRED_COLUMNS, label=str(input_path))

    working = df.copy()
    working["patient_uid"] = working["patient_uid"].map(normalize_patient_uid)
    duplicate_patient_uids = sorted(
        working.loc[
            working["patient_uid"].duplicated(keep=False), "patient_uid"
        ].unique(),
        key=patient_uid_sort_key,
    )
    if duplicate_patient_uids:
        raise ValueError(
            "Input cohort contains duplicate patient_uid values after "
            "normalization. Preview: "
            f"{duplicate_patient_uids[:20]}"
        )

    working["pmc_id_clean"] = working["patient_uid"].map(
        article_id_from_patient_uid
    )
    working["pmc_id"] = "PMC" + working["pmc_id_clean"]

    parsed_results = working["similar_patients"].map(parse_similarity_mapping)
    working["_similarity_mapping"] = parsed_results.map(lambda item: item[0])
    working["_parse_status"] = parsed_results.map(lambda item: item[1])

    eligible = working.loc[working["_similarity_mapping"].map(bool)].copy()
    article_counts = working.groupby("pmc_id_clean")["patient_uid"].nunique()
    unique_article_ids = set(article_counts.loc[article_counts == 1].index)

    unique_rows = working.loc[
        working["pmc_id_clean"].isin(unique_article_ids)
    ].copy()
    unique_patient_uids = set(unique_rows["patient_uid"])

    unique_rows["_filtered_similarity_mapping"] = [
        {
            uid: score
            for uid, score in mapping.items()
            if uid in unique_patient_uids and uid != source_uid
        }
        for source_uid, mapping in zip(
            unique_rows["patient_uid"],
            unique_rows["_similarity_mapping"],
        )
    ]
    relationship_sources = unique_rows.loc[
        unique_rows["_filtered_similarity_mapping"].map(bool)
    ].copy()
    relationship_source_uids = set(relationship_sources["patient_uid"])

    referenced_target_uids = {
        target_uid
        for mapping in relationship_sources["_filtered_similarity_mapping"]
        for target_uid in mapping
    }
    included_uids = relationship_source_uids | referenced_target_uids
    final_df = unique_rows.loc[
        unique_rows["patient_uid"].isin(included_uids)
    ].copy()

    final_df["similar_patients"] = final_df[
        "_filtered_similarity_mapping"
    ].map(
        lambda mapping: json.dumps(
            mapping,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    final_df["similar_patients_list"] = final_df[
        "_filtered_similarity_mapping"
    ].map(
        lambda mapping: json.dumps(
            list(mapping),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    final_df["unique_articles_sim_patients"] = final_df[
        "similar_patients_list"
    ]

    unsupported_cases = sorted(
        {
            uid
            for uid in final_df["patient_uid"]
            if uid.rsplit("-", 1)[1] != "1"
        },
        key=patient_uid_sort_key,
    )
    if unsupported_cases:
        raise ValueError(
            "The source extraction contract supports only the single-case "
            "suffix '-1'. Unsupported selected UIDs: "
            f"{unsupported_cases[:20]}"
        )

    internal_columns = [
        "_similarity_mapping",
        "_filtered_similarity_mapping",
        "_parse_status",
    ]
    final_df = final_df.drop(columns=internal_columns)
    final_df["_pmc_sort"] = final_df["pmc_id_clean"].map(int)
    final_df["_patient_sort"] = final_df["patient_uid"].map(
        patient_uid_sort_key
    )
    final_df = (
        final_df.sort_values(
            by=["_pmc_sort", "_patient_sort"],
            kind="mergesort",
        )
        .drop(columns=["_pmc_sort", "_patient_sort"])
        .reset_index(drop=True)
    )

    article_ids = sorted(
        final_df["pmc_id_clean"].astype(str).unique().tolist(),
        key=lambda value: (int(value), value),
    )

    patients_path = artifact_path(output_dir, "patients_csv")
    article_ids_path = artifact_path(output_dir, "article_ids_json")
    stats_path = artifact_path(output_dir, "patient_stats_json")

    save_dataframe(final_df, patients_path)
    save_json(article_ids, article_ids_path)

    parse_counts = working["_parse_status"].value_counts().to_dict()
    stats = {
        "pipeline_version": PIPELINE_VERSION,
        "stage": "01_prepare_articles",
        "configuration": {"uniqueness_scope": "all_rows"},
        "input": {
            "name": input_path.name,
            "sha256": sha256_file(input_path),
            "rows": int(len(working)),
        },
        "parse_status_counts": {
            str(key): int(value) for key, value in sorted(parse_counts.items())
        },
        "rows_with_nonempty_similarity": int(len(eligible)),
        "single_patient_article_count": int(len(unique_article_ids)),
        "rows_before_final_relationship_filter": int(len(unique_rows)),
        "relationship_source_rows_output": int(len(relationship_source_uids)),
        "referenced_target_only_rows_output": int(
            len(set(final_df["patient_uid"]) - relationship_source_uids)
        ),
        "rows_output": int(len(final_df)),
        "article_ids_output": int(len(article_ids)),
        "similarity_links_output": int(
            final_df["unique_articles_sim_patients"]
            .map(lambda value: len(json.loads(value)))
            .sum()
        ),
        "outputs": {
            patients_path.name: sha256_file(patients_path),
            article_ids_path.name: sha256_file(article_ids_path),
        },
    }
    save_json(stats, stats_path)
    return stats
