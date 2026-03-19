from __future__ import annotations

import ast
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .io_utils import load_json


REQUIRED_DATASET_COLUMNS = {
    "patient_uid",
    "unique_articles_sim_patients",
    "caption_path",
    "image_path",
}


def caption_id_from_path(path: str) -> str:
    return Path(str(path)).stem


def normalize_relative_to_data2(path: str) -> str:
    value = str(path or "").replace("\\", "/").strip()
    marker = "data2/"
    if marker in value:
        value = value.split(marker, 1)[1]
    return value.lstrip("/")


def load_and_filter_rows(dataset_csv: str | Path, diagram_paths_json: str | Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv)
    missing_columns = REQUIRED_DATASET_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{dataset_csv} is missing required columns: {sorted(missing_columns)}"
        )

    df = df.copy()
    df["patient_uid"] = df["patient_uid"].astype(str).str.strip()
    df["caption_id"] = df["caption_path"].map(caption_id_from_path)
    df["_image_path_norm"] = df["image_path"].map(normalize_relative_to_data2)

    diagram_paths = load_json(diagram_paths_json)
    diagram_rel = {normalize_relative_to_data2(path) for path in diagram_paths}

    filtered = df.loc[~df["_image_path_norm"].isin(diagram_rel)].copy()
    filtered = filtered.drop(columns=["_image_path_norm"])
    return filtered


def parse_similar_patient_list(raw: Any) -> list[str]:
    if raw is None or pd.isna(raw):
        return []

    if isinstance(raw, (list, tuple, set)):
        return [str(item).strip() for item in raw if str(item).strip()]

    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(text)
        except Exception:
            continue
        if isinstance(obj, (list, tuple, set)):
            return [str(item).strip() for item in obj if str(item).strip()]

    text = text.strip("[]")
    text = text.replace("'", "").replace('"', "")
    return [item.strip() for item in text.split(",") if item.strip()]


def create_patient_uid_pairs(df: pd.DataFrame) -> list[list[str]]:
    matching_uids: list[list[str]] = []
    checked_pairs: set[tuple[str, str]] = set()
    uids_in_df = set(df["patient_uid"].astype(str))

    for row in df.itertuples(index=False):
        patient_uid = str(row.patient_uid).strip()
        sim_patients_list = parse_similar_patient_list(row.unique_articles_sim_patients)

        for similar_uid in sim_patients_list:
            similar_uid = str(similar_uid).strip()
            if not similar_uid or similar_uid == patient_uid:
                continue
            if similar_uid not in uids_in_df:
                continue

            pair_key = tuple(sorted((patient_uid, similar_uid)))
            if pair_key in checked_pairs:
                continue

            matching_uids.append([patient_uid, similar_uid])
            checked_pairs.add(pair_key)

    return matching_uids


def build_patient_caption_map(df: pd.DataFrame) -> dict[str, set[str]]:
    grouped = df.groupby("patient_uid")["caption_id"].apply(set)
    return grouped.to_dict()


def strip_dash_suffix(uid: str) -> str:
    return str(uid).split("-", 1)[0]


def patient_article_dir(data2_base: str | Path, patient_uid: str) -> Path:
    pmc_id = strip_dash_suffix(patient_uid)
    return Path(data2_base) / f"PMC{pmc_id}" / f"{pmc_id}_1"


@lru_cache(maxsize=None)
def _list_patient_files_cached(data2_base_str: str, patient_uid: str) -> tuple[str | None, dict[str, str], dict[str, str]]:
    article_dir = patient_article_dir(data2_base_str, patient_uid)
    if not article_dir.is_dir():
        return None, {}, {}

    txt_files = {path.stem: str(path) for path in article_dir.glob("*.txt")}
    jpg_files = {path.stem: str(path) for path in article_dir.glob("*.jpg")}
    return str(article_dir), txt_files, jpg_files


@lru_cache(maxsize=None)
def _read_text_cached(path_str: str) -> str:
    return Path(path_str).read_text(encoding="utf-8", errors="ignore").strip()


def build_pair_caption_artifacts(
    patient_uid_pairs: list[list[str]],
    data2_base: str | Path,
    patient_uid_to_captions: Mapping[str, set[str]],
) -> tuple[list[Any], list[Any], list[Any], dict[str, Any]]:
    text_path_pairs: list[Any] = []
    image_path_pairs: list[Any] = []
    caption_dict_pairs: list[Any] = []

    missing_patient_dirs: list[dict[str, Any]] = []
    pairs_with_missing_dirs: list[int] = []
    pairs_with_empty_caption_groups: list[int] = []

    data2_base_str = str(data2_base)

    for pair_index, pair in enumerate(patient_uid_pairs):
        pair_text_groups: list[list[str]] = []
        pair_image_groups: list[list[str]] = []
        pair_caption_groups: list[dict[str, str]] = []
        pair_missing_dir = False

        for patient_uid in pair:
            article_dir, txt_files, jpg_files = _list_patient_files_cached(data2_base_str, patient_uid)
            if article_dir is None:
                pair_missing_dir = True
                missing_patient_dirs.append(
                    {
                        "pair_index": pair_index,
                        "patient_uid": patient_uid,
                        "article_dir": str(patient_article_dir(data2_base_str, patient_uid)),
                    }
                )
                break

            valid_caption_ids = patient_uid_to_captions.get(patient_uid, set())
            common_keys = sorted(set(txt_files) & set(jpg_files) & set(valid_caption_ids))

            pair_text_groups.append([txt_files[key] for key in common_keys])
            pair_image_groups.append([jpg_files[key] for key in common_keys])
            pair_caption_groups.append(
                {
                    key: _read_text_cached(txt_files[key])
                    for key in common_keys
                }
            )

        if pair_missing_dir or len(pair_caption_groups) != 2:
            text_path_pairs.append([])
            image_path_pairs.append([])
            caption_dict_pairs.append([])
            pairs_with_missing_dirs.append(pair_index)
            continue

        if any(len(group) == 0 for group in pair_caption_groups):
            pairs_with_empty_caption_groups.append(pair_index)

        text_path_pairs.append(pair_text_groups)
        image_path_pairs.append(pair_image_groups)
        caption_dict_pairs.append(pair_caption_groups)

    stats = {
        "missing_patient_dir_count": len(missing_patient_dirs),
        "pairs_with_missing_dirs_count": len(pairs_with_missing_dirs),
        "pairs_with_empty_caption_groups_count": len(pairs_with_empty_caption_groups),
        "pairs_with_missing_dirs": pairs_with_missing_dirs,
        "pairs_with_empty_caption_groups": pairs_with_empty_caption_groups,
        "missing_patient_dirs_preview": missing_patient_dirs[:100],
    }
    return text_path_pairs, image_path_pairs, caption_dict_pairs, stats


@lru_cache(maxsize=None)
def _pmc_id_to_nxml_path_cached(data2_base_str: str, pmc_id: str) -> str | None:
    article_dir = Path(data2_base_str) / f"PMC{pmc_id}" / f"{pmc_id}_1"
    if not article_dir.is_dir():
        return None

    candidates = list(article_dir.glob("*.nxml"))
    if not candidates:
        return None

    if len(candidates) == 1:
        return str(candidates[0])

    candidates.sort(key=lambda path: path.stat().st_size, reverse=True)
    return str(candidates[0])


def build_xml_pairs(
    patient_uid_pairs: list[list[str]],
    data2_base: str | Path,
) -> tuple[list[list[str | None]], list[int], list[dict[str, Any]]]:
    xml_pairs: list[list[str | None]] = []
    missing: list[dict[str, Any]] = []

    data2_base_str = str(data2_base)

    for pair_index, (uid1, uid2) in enumerate(patient_uid_pairs):
        pmc1 = strip_dash_suffix(uid1)
        pmc2 = strip_dash_suffix(uid2)

        path1 = _pmc_id_to_nxml_path_cached(data2_base_str, pmc1)
        path2 = _pmc_id_to_nxml_path_cached(data2_base_str, pmc2)
        xml_pairs.append([path1, path2])

        if path1 is None or path2 is None:
            missing.append(
                {
                    "pair_index": pair_index,
                    "uid1": uid1,
                    "uid2": uid2,
                    "xml_path_1": path1,
                    "xml_path_2": path2,
                }
            )

    valid_indices = [
        idx for idx, (path1, path2) in enumerate(xml_pairs)
        if path1 is not None and path2 is not None
    ]
    return xml_pairs, valid_indices, missing
