from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from .artifacts import PIPELINE_VERSION, artifact_path
from .io_utils import (
    figure_path_sort_key,
    normalize_patient_uid,
    normalize_pmc_id,
    patient_uid_sort_key,
    require_columns,
    save_dataframe,
    save_json,
    sha256_file,
)


IMAGE_COLUMNS = {
    "image_path",
    "caption_path",
    "patient_uid",
    "pmc_id",
    "article_path",
}
PATIENT_COLUMNS = {"patient_uid", "unique_articles_sim_patients"}
OUTPUT_COLUMNS = [
    "image_path",
    "caption_path",
    "patient_uid",
    "pmc_id",
    "article_path",
    "unique_articles_sim_patients",
]


def parse_similarity_list(raw: Any) -> list[str]:
    if isinstance(raw, (list, tuple, set)):
        values = list(raw)
    elif raw is None:
        return []
    else:
        try:
            if bool(pd.isna(raw)):
                return []
        except (TypeError, ValueError):
            pass

        text = str(raw).strip()
        if not text:
            return []
        parsed = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                break
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
        if not isinstance(parsed, (list, tuple, set)):
            raise ValueError(f"Invalid similar-patient list: {raw!r}")
        values = list(parsed)

    normalized = {normalize_patient_uid(value) for value in values}
    return sorted(normalized, key=patient_uid_sort_key)


def _validate_local_assets(
    df: pd.DataFrame,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], int]:
    problems: list[dict[str, Any]] = []
    nxml_checked: set[str] = set()

    for row_index, row in enumerate(df.itertuples(index=False)):
        pmc_id = normalize_pmc_id(row.pmc_id)
        uid = normalize_patient_uid(row.patient_uid)
        article_id, case_number = uid.split("-", 1)
        expected_folder = (
            output_dir
            / "new_data"
            / f"PMC{article_id}"
            / f"{article_id}_{case_number}"
        ).resolve()

        image_path = Path(row.image_path)
        caption_path = Path(row.caption_path)
        resolved_image = (
            image_path if image_path.is_absolute() else output_dir / image_path
        ).resolve()
        resolved_caption = (
            caption_path if caption_path.is_absolute() else output_dir / caption_path
        ).resolve()

        row_problems: list[str] = []
        if pmc_id != article_id:
            row_problems.append("pmc_id does not match patient_uid")
        if resolved_image.suffix.lower() != ".jpg":
            row_problems.append("image is not canonical .jpg")
        if resolved_caption.suffix.lower() != ".txt":
            row_problems.append("caption is not .txt")
        if resolved_image.stem != resolved_caption.stem:
            row_problems.append("image/caption stems differ")
        if resolved_image.parent != expected_folder:
            row_problems.append("image is outside expected patient folder")
        if resolved_caption.parent != expected_folder:
            row_problems.append("caption is outside expected patient folder")
        if not resolved_image.is_file():
            row_problems.append("image file is missing")
        else:
            try:
                with Image.open(resolved_image) as image:
                    image.load()
                    if image.format != "JPEG" or image.mode != "RGB":
                        row_problems.append("image is not a decoded RGB JPEG")
            except Exception as error:
                row_problems.append(
                    "image cannot be decoded: "
                    f"{type(error).__name__}: {error}"
                )
        if not resolved_caption.is_file():
            row_problems.append("caption file is missing")
        else:
            try:
                caption_text = resolved_caption.read_text(encoding="utf-8")
                if not caption_text.strip():
                    row_problems.append("caption file is empty")
            except Exception as error:
                row_problems.append(
                    "caption is not valid UTF-8 text: "
                    f"{type(error).__name__}: {error}"
                )

        if row_problems:
            problems.append(
                {
                    "row_index": row_index,
                    "patient_uid": uid,
                    "image_path": str(row.image_path),
                    "problems": row_problems,
                }
            )

        if uid not in nxml_checked:
            nxml_checked.add(uid)
            nxml_path = expected_folder / "article.nxml"
            if not nxml_path.is_file():
                problems.append(
                    {
                        "patient_uid": uid,
                        "problems": ["patient folder has no canonical article.nxml"],
                    }
                )
            else:
                try:
                    ET.parse(nxml_path)
                except Exception as error:
                    problems.append(
                        {
                            "patient_uid": uid,
                            "problems": [
                                "article.nxml cannot be parsed: "
                                f"{type(error).__name__}: {error}"
                            ],
                        }
                    )

    return problems, len(nxml_checked)


def build_merged_dataset(
    images_csv: str | Path,
    patients_csv: str | Path,
    output_dir: str | Path,
    *,
    strict_files: bool = True,
) -> dict[str, Any]:
    images_path = Path(images_csv)
    patients_path = Path(patients_csv)
    run_dir = Path(output_dir)

    images = pd.read_csv(images_path, dtype={"patient_uid": str, "pmc_id": str})
    patients = pd.read_csv(
        patients_path,
        dtype={"patient_uid": str, "pmc_id_clean": str, "pmc_id": str},
    )
    require_columns(images.columns, IMAGE_COLUMNS, label=str(images_path))
    require_columns(patients.columns, PATIENT_COLUMNS, label=str(patients_path))

    images = images.copy()
    patients = patients.copy()
    images["patient_uid"] = images["patient_uid"].map(normalize_patient_uid)
    images["pmc_id"] = images["pmc_id"].map(normalize_pmc_id)
    patients["patient_uid"] = patients["patient_uid"].map(normalize_patient_uid)
    patients["unique_articles_sim_patients"] = patients[
        "unique_articles_sim_patients"
    ].map(
        lambda value: json.dumps(
            parse_similarity_list(value),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )

    duplicate_patient_uids = sorted(
        patients.loc[
            patients["patient_uid"].duplicated(keep=False), "patient_uid"
        ].unique(),
        key=patient_uid_sort_key,
    )
    if duplicate_patient_uids:
        preview = duplicate_patient_uids[:20]
        raise ValueError(
            "Patient cohort contains duplicate patient_uid values; refusing a "
            f"many-to-many merge. Preview: {preview}"
        )

    for column in ("image_path", "caption_path"):
        duplicates = images.loc[images[column].duplicated(keep=False), column]
        if not duplicates.empty:
            raise ValueError(
                f"Image manifest contains duplicate {column} values. "
                f"Preview: {duplicates.astype(str).head(20).tolist()}"
            )

    image_uids = set(images["patient_uid"])
    patient_uids = set(patients["patient_uid"])
    image_uids_without_patient_row = sorted(
        image_uids - patient_uids,
        key=patient_uid_sort_key,
    )
    patient_uids_without_images = sorted(
        patient_uids - image_uids,
        key=patient_uid_sort_key,
    )

    merged = images.merge(
        patients[["patient_uid", "unique_articles_sim_patients"]],
        on="patient_uid",
        how="inner",
        validate="many_to_one",
        sort=False,
    )
    merged = merged[OUTPUT_COLUMNS].copy()
    merged["_pmc_sort"] = merged["pmc_id"].map(int)
    merged["_figure_sort"] = merged["image_path"].map(figure_path_sort_key)
    merged = merged.sort_values(
        by=["_pmc_sort", "patient_uid", "_figure_sort"],
        kind="mergesort",
    ).drop(columns=["_pmc_sort", "_figure_sort"]).reset_index(drop=True)

    file_problems, patients_checked_for_nxml = _validate_local_assets(
        merged,
        run_dir,
    )

    merged_uids = set(merged["patient_uid"])
    missing_similarity_targets: set[str] = set()
    candidate_pairs: set[tuple[str, str]] = set()
    patient_similarity = merged.drop_duplicates("patient_uid")[[
        "patient_uid",
        "unique_articles_sim_patients",
    ]]
    for row in patient_similarity.itertuples(index=False):
        source_uid = row.patient_uid
        for target_uid in parse_similarity_list(row.unique_articles_sim_patients):
            if target_uid not in merged_uids:
                missing_similarity_targets.add(target_uid)
                continue
            if target_uid != source_uid:
                candidate_pairs.add(tuple(sorted((source_uid, target_uid))))

    report_path = artifact_path(run_dir, "handoff_report_json")
    base_report = {
        "pipeline_version": PIPELINE_VERSION,
        "stage": "04_merge_dataset",
        "configuration": {"strict_files": bool(strict_files)},
        "inputs": {
            images_path.name: sha256_file(images_path),
            patients_path.name: sha256_file(patients_path),
        },
        "image_rows_input": int(len(images)),
        "patient_rows_input": int(len(patients)),
        "rows_output": int(len(merged)),
        "patients_output": int(len(merged_uids)),
        "candidate_patient_pairs": len(candidate_pairs),
        "patients_checked_for_nxml": patients_checked_for_nxml,
        "image_uids_without_patient_row": image_uids_without_patient_row,
        "patient_uids_without_images": patient_uids_without_images,
        "similarity_targets_without_images": sorted(
            missing_similarity_targets,
            key=patient_uid_sort_key,
        ),
        "file_problem_count": len(file_problems),
        "file_problems_preview": file_problems[:100],
    }

    if strict_files and file_problems:
        save_json(base_report, report_path)
        raise ValueError(
            f"Handoff validation found {len(file_problems)} local asset problems; "
            f"see {report_path}"
        )

    merged_path = artifact_path(run_dir, "merged_csv")
    image_paths_path = artifact_path(run_dir, "image_paths_json")
    save_dataframe(merged, merged_path)

    relative_image_paths = merged["image_path"].astype(str).tolist()
    save_json(relative_image_paths, image_paths_path)

    report = {
        **base_report,
        "outputs": {
            merged_path.name: sha256_file(merged_path),
            image_paths_path.name: sha256_file(image_paths_path),
        },
    }
    save_json(report, report_path)
    return report
