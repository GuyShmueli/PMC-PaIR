from __future__ import annotations

from pathlib import Path

ARTIFACTS = {
    # Stage 1: dataset preparation / candidate construction
    "filtered_rows_csv": "01_filtered_rows.csv",
    "pair_patient_uids_json": "01_pair_patient_uids.json",
    "pair_text_paths_json": "01_pair_text_paths.json",
    "pair_image_paths_json": "01_pair_image_paths.json",
    "pair_caption_dicts_json": "01_pair_caption_dicts.json",
    "pair_xml_paths_json": "01_pair_xml_paths.json",
    "pair_xml_valid_indices_json": "01_pair_xml_valid_indices.json",
    "pair_xml_missing_json": "01_pair_xml_missing.json",
    "stage1_stats_json": "01_stage1_stats.json",

    # Stage 2: case summaries
    "case_summary_pairs_json": "02_case_summary_pairs.json",
    "summary_valid_indices_json": "02_summary_valid_indices.json",
    "summary_invalid_indices_json": "02_summary_invalid_indices.json",
    "stage2_stats_json": "02_stage2_stats.json",

    # Stage 3: citation matching
    "citation_records_json": "03_citation_records.json",
    "citation_valid_indices_json": "03_citation_valid_indices.json",
    "common_indices_json": "03_common_indices.json",
    "stage3_stats_json": "03_stage3_stats.json",

    # Stage 4: citation reasoning batch
    "citation_request_jsonl": "batch/04_citation_reasoning_requests.jsonl",
    "citation_request_indices_json": "batch/04_citation_reasoning_request_indices.json",
    "citation_batch_meta_json": "batch/04_citation_reasoning_batch.json",
    "citation_batch_raw_jsonl": "batch/04_citation_reasoning_raw_responses.jsonl",
    "citation_batch_errors_jsonl": "batch/04_citation_reasoning_errors.jsonl",
    "citation_texts_json": "batch/04_citation_reasoning_texts.json",

    # Stage 5: question generation batch
    "question_request_jsonl": "batch/05_question_requests.jsonl",
    "question_request_indices_json": "batch/05_question_request_indices.json",
    "question_batch_meta_json": "batch/05_question_batch.json",
    "question_batch_raw_jsonl": "batch/05_question_raw_responses.jsonl",
    "question_batch_errors_jsonl": "batch/05_question_errors.jsonl",
    "question_texts_json": "batch/05_question_texts.json",

    # Stage 6: answer generation batch
    "answer_request_jsonl": "batch/06_answer_requests.jsonl",
    "answer_request_indices_json": "batch/06_answer_request_indices.json",
    "answer_batch_meta_json": "batch/06_answer_batch.json",
    "answer_batch_raw_jsonl": "batch/06_answer_raw_responses.jsonl",
    "answer_batch_errors_jsonl": "batch/06_answer_errors.jsonl",
    "answer_texts_json": "batch/06_answer_texts.json",

    # Stage 7: positive-pair classification batch
    "positive_request_jsonl": "batch/07_positive_requests.jsonl",
    "positive_request_indices_json": "batch/07_positive_request_indices.json",
    "positive_batch_meta_json": "batch/07_positive_batch.json",
    "positive_batch_raw_jsonl": "batch/07_positive_raw_responses.jsonl",
    "positive_batch_errors_jsonl": "batch/07_positive_errors.jsonl",
    "positive_texts_json": "batch/07_positive_texts.json",

    # Stage 8: final cleaned outputs
    "labeled_positive_pairs_json": "08_labeled_positive_pairs.json",
    "positive_pairs_clean_json": "08_positive_pairs_clean.json",
    "text_pairs_json": "08_text_pairs.json",
    "positive_pairs_by_bucket_json": "08_positive_pairs_by_bucket.json",
    "radiology_labeled_positive_pairs_json": "08_radiology_labeled_positive_pairs.json",
    "radiology_pairs_clean_json": "08_radiology_pairs_clean.json",
    "radiology_text_pairs_json": "08_radiology_text_pairs.json",
    "bad_rows_json": "08_bad_rows.json",
    "final_stats_json": "08_final_stats.json",
}


BATCH_STAGE_TO_KEYS = {
    "citation": {
        "requests": "citation_request_jsonl",
        "request_indices": "citation_request_indices_json",
        "batch_meta": "citation_batch_meta_json",
        "raw_responses": "citation_batch_raw_jsonl",
        "error_responses": "citation_batch_errors_jsonl",
        "texts": "citation_texts_json",
    },
    "question": {
        "requests": "question_request_jsonl",
        "request_indices": "question_request_indices_json",
        "batch_meta": "question_batch_meta_json",
        "raw_responses": "question_batch_raw_jsonl",
        "error_responses": "question_batch_errors_jsonl",
        "texts": "question_texts_json",
    },
    "answer": {
        "requests": "answer_request_jsonl",
        "request_indices": "answer_request_indices_json",
        "batch_meta": "answer_batch_meta_json",
        "raw_responses": "answer_batch_raw_jsonl",
        "error_responses": "answer_batch_errors_jsonl",
        "texts": "answer_texts_json",
    },
    "positive": {
        "requests": "positive_request_jsonl",
        "request_indices": "positive_request_indices_json",
        "batch_meta": "positive_batch_meta_json",
        "raw_responses": "positive_batch_raw_jsonl",
        "error_responses": "positive_batch_errors_jsonl",
        "texts": "positive_texts_json",
    },
}


def artifact_path(output_dir: str | Path, key: str, *, create_parent: bool = True) -> Path:
    if key not in ARTIFACTS:
        raise KeyError(f"Unknown artifact key: {key}")

    path = Path(output_dir) / ARTIFACTS[key]
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def batch_stage_path(output_dir: str | Path, stage: str, kind: str, *, create_parent: bool = True) -> Path:
    if stage not in BATCH_STAGE_TO_KEYS:
        raise KeyError(f"Unknown batch stage: {stage}")
    if kind not in BATCH_STAGE_TO_KEYS[stage]:
        raise KeyError(f"Unknown batch stage artifact kind: {kind}")
    return artifact_path(output_dir, BATCH_STAGE_TO_KEYS[stage][kind], create_parent=create_parent)
