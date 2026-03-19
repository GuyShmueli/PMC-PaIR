from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from typing import Any

from .io_utils import normalize_text_pair_item, strip_code_fences


_PAIR_ID_CLEAN_RE = re.compile(r"[^0-9_,]")
_ID_RE = re.compile(r"\d+_\d+_\d+")
_OBJECT_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.S)


def normalize_pair_id(pair_id_val: Any) -> list[str] | None:
    if pair_id_val is None:
        return None

    if isinstance(pair_id_val, (list, tuple)):
        text = ",".join(str(item) for item in pair_id_val)
    else:
        text = str(pair_id_val)

    cleaned = _PAIR_ID_CLEAN_RE.sub("", text)
    parts = [part for part in cleaned.split(",") if part]
    return parts if parts else None


def canonical_pair_key(pair_id: list[str]) -> tuple[str, str]:
    a, b = str(pair_id[0]).strip(), str(pair_id[1]).strip()
    return tuple(sorted((a, b)))


def build_short_caption_from_record(obj: dict[str, Any]) -> list[str]:
    diagnosis = str(obj.get("diagnosis", "")).strip()
    anatomy = str(obj.get("anatomy", "")).strip()
    modality = str(obj.get("modality", "")).strip()

    parts = []
    if diagnosis:
        parts.append(diagnosis)
    if anatomy:
        parts.append(f"in the {anatomy}")
    if modality:
        parts.append(f"on {modality}")

    caption = " ".join(parts).strip()
    if caption and not caption.endswith("."):
        caption += "."
    return [caption] if caption else []


def modality_bucket(modality: str, *, legacy_order: bool = False) -> str:
    value = (modality or "").strip().lower()

    radiology_keywords = [
        "ct",
        "computed tomography",
        "mri",
        "magnetic resonance",
        "radiograph",
        "x-ray",
        "xray",
        "ultrasound",
        "sonograph",
        "sonography",
        "pet",
        "spect",
        "angiograph",
        "angiography",
        "fluoroscopy",
        "mammograph",
        "mammography",
        "tomograph",
        "tomography",
    ]
    pathology_keywords = [
        "histolog",
        "patholog",
        "microscop",
        "immunohistochem",
        "ihc",
        "hematoxylin",
        "eosin",
        "h&e",
        "stain",
        "biopsy",
        "slide",
    ]
    ophtho_keywords = [
        "fundus",
        "oct",
        "optical coherence",
        "slit lamp",
        "fluorescein",
        "indocyanine",
        "retinal",
        "ophthalm",
    ]
    endoscopy_keywords = [
        "endoscop",
        "colonoscopy",
        "gastroscopy",
        "bronchoscopy",
        "laparoscopy",
        "hysteroscopy",
        "arthroscopy",
    ]
    cardiology_keywords = [
        "ecg",
        "ekg",
        "electrocardi",
        "echocardi",
        "echo",
    ]

    if legacy_order:
        if any(keyword in value for keyword in radiology_keywords):
            return "radiology"
        if any(keyword in value for keyword in pathology_keywords):
            return "pathology_microscopy"
        if any(keyword in value for keyword in ophtho_keywords):
            return "ophthalmology"
        if any(keyword in value for keyword in endoscopy_keywords):
            return "endoscopy"
        if any(keyword in value for keyword in cardiology_keywords):
            return "cardiology"
        return "other"

    if any(keyword in value for keyword in pathology_keywords):
        return "pathology_microscopy"
    if any(keyword in value for keyword in ophtho_keywords):
        return "ophthalmology"
    if any(keyword in value for keyword in endoscopy_keywords):
        return "endoscopy"
    if any(keyword in value for keyword in cardiology_keywords):
        return "cardiology"
    if any(keyword in value for keyword in radiology_keywords):
        return "radiology"
    return "other"


def parse_jsonish_obj(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str):
        return None

    text = strip_code_fences(raw)
    if not text:
        return []

    candidates = [text]

    first_arr = text.find("[")
    last_arr = text.rfind("]")
    if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        candidates.append(text[first_arr:last_arr + 1])

    first_obj = text.find("{")
    last_obj = text.rfind("}")
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        candidates.append(text[first_obj:last_obj + 1])

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)

        try:
            return json.loads(candidate)
        except Exception:
            pass

        try:
            return ast.literal_eval(candidate)
        except Exception:
            pass

    return None


def flatten_dict_records(obj: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def recurse(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            if value != {}:
                records.append(value)
            return
        if isinstance(value, list):
            for item in value:
                recurse(item)

    recurse(obj)
    return records


def parse_jsonish_to_records(raw: Any) -> list[dict[str, Any]] | None:
    obj = parse_jsonish_obj(raw)
    if obj is None:
        return None
    return flatten_dict_records(obj)


def _extract_field_from_text(text: str, field: str) -> str | None:
    patterns = [
        rf'"{re.escape(field)}"\s*:\s*"([^"]*)"',
        rf"'{re.escape(field)}'\s*:\s*'([^']*)'",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.S)
        if match:
            return match.group(1).strip()
    return None


def salvage_records_from_raw_text(raw: Any) -> list[dict[str, Any]]:
    text = strip_code_fences(str(raw))
    if not text:
        return []

    candidate_blocks = _OBJECT_BLOCK_RE.findall(text)
    if not candidate_blocks:
        candidate_blocks = [text]

    recovered: list[dict[str, Any]] = []
    seen = set()

    for block in candidate_blocks:
        obj = parse_jsonish_obj(block)
        if isinstance(obj, dict) and obj:
            pair_id = normalize_pair_id(obj.get("pair_id"))
            if pair_id and len(pair_id) == 2:
                obj2 = dict(obj)
                obj2["pair_id"] = pair_id
                key = tuple(pair_id)
                if key not in seen:
                    seen.add(key)
                    recovered.append(obj2)
                continue

        pair_val = _extract_field_from_text(block, "pair_id")
        pair_id = normalize_pair_id(pair_val)

        if not pair_id or len(pair_id) != 2:
            ids = _ID_RE.findall(block)
            if len(ids) >= 2:
                pair_id = [ids[0], ids[1]]

        if not pair_id or len(pair_id) != 2:
            continue

        rec = {
            "pair_id": pair_id,
            "modality": _extract_field_from_text(block, "modality") or "",
            "anatomy": _extract_field_from_text(block, "anatomy") or "",
            "diagnosis": _extract_field_from_text(block, "diagnosis") or "",
            "reasoning": _extract_field_from_text(block, "reasoning") or "",
        }

        key = tuple(pair_id)
        if key not in seen:
            seen.add(key)
            recovered.append(rec)

    return recovered


def _validate_inter_case_pair(
    pair_id: list[str],
    case_a_ids: set[str],
    case_b_ids: set[str],
) -> list[str] | None:
    if len(pair_id) != 2:
        return None

    left, right = pair_id[0], pair_id[1]
    if left in case_a_ids and right in case_b_ids:
        return [left, right]
    if left in case_b_ids and right in case_a_ids:
        return [right, left]
    return None


def process_positive_responses(
    response_texts: list[Any],
    request_indices: list[int],
    pair_caption_dicts: list[Any],
    *,
    legacy_modality_bucket: bool = False,
) -> dict[str, Any]:
    if len(response_texts) != len(request_indices):
        raise ValueError(
            f"Length mismatch: {len(response_texts)=} vs {len(request_indices)=}"
        )

    labeled_all: list[dict[str, Any]] = []
    pairs_clean_all: list[list[str]] = []
    text_pairs_all: list[list[str]] = []
    bad_rows: list[dict[str, Any]] = []

    by_bucket = defaultdict(lambda: {"labeled": [], "pairs_clean": [], "text_pairs": []})
    seen_pairs: set[tuple[str, str]] = set()

    empty_outputs = 0
    parse_failures = 0
    regex_salvaged_outputs = 0
    invalid_pair_ids = 0
    duplicate_pairs = 0
    multi_record_outputs = 0

    for request_position, (raw, original_pair_index) in enumerate(zip(response_texts, request_indices)):
        pair_caption_group = pair_caption_dicts[original_pair_index]

        if not isinstance(pair_caption_group, list) or len(pair_caption_group) != 2:
            bad_rows.append(
                {
                    "request_position": request_position,
                    "pair_index": original_pair_index,
                    "reason": "missing_caption_group",
                }
            )
            continue

        case_a_captions = pair_caption_group[0] or {}
        case_b_captions = pair_caption_group[1] or {}
        case_a_ids = set(case_a_captions.keys())
        case_b_ids = set(case_b_captions.keys())

        records = parse_jsonish_to_records(raw)
        source_type = "parsed"

        if records is None or len(records) == 0:
            salvaged = salvage_records_from_raw_text(raw)
            if salvaged:
                records = salvaged
                source_type = "regex_salvaged"
                regex_salvaged_outputs += 1

        if records is None:
            parse_failures += 1
            bad_rows.append(
                {
                    "request_position": request_position,
                    "pair_index": original_pair_index,
                    "reason": "unparseable",
                    "raw_preview": str(raw)[:500],
                }
            )
            continue

        if len(records) == 0:
            empty_outputs += 1
            continue

        if len(records) > 1:
            multi_record_outputs += 1

        for subidx, obj in enumerate(records):
            pair_id = normalize_pair_id(obj.get("pair_id"))
            if not pair_id or len(pair_id) != 2:
                invalid_pair_ids += 1
                bad_rows.append(
                    {
                        "request_position": request_position,
                        "pair_index": original_pair_index,
                        "subidx": subidx,
                        "reason": "bad_pair_id",
                        "pair_id": obj.get("pair_id"),
                        "raw_preview": str(raw)[:500],
                        "source_type": source_type,
                    }
                )
                continue

            validated_pair = _validate_inter_case_pair(pair_id, case_a_ids, case_b_ids)
            if validated_pair is None:
                invalid_pair_ids += 1
                bad_rows.append(
                    {
                        "request_position": request_position,
                        "pair_index": original_pair_index,
                        "subidx": subidx,
                        "reason": "pair_id_not_in_case_caption_dicts",
                        "pair_id": pair_id,
                        "allowed_case_a_ids": sorted(case_a_ids),
                        "allowed_case_b_ids": sorted(case_b_ids),
                        "source_type": source_type,
                    }
                )
                continue

            canonical_key = canonical_pair_key(validated_pair)
            if canonical_key in seen_pairs:
                duplicate_pairs += 1
                continue

            seen_pairs.add(canonical_key)

            record = dict(obj)
            record["pair_id"] = list(canonical_key)
            record["source_pair_index"] = original_pair_index
            record["request_position"] = request_position
            record["source_type"] = source_type

            bucket = modality_bucket(
                str(record.get("modality", "")),
                legacy_order=legacy_modality_bucket,
            )
            record["modality_bucket"] = bucket

            caption = build_short_caption_from_record(record)

            labeled_all.append(record)
            pairs_clean_all.append(list(canonical_key))
            text_pairs_all.append(caption)

            by_bucket[bucket]["labeled"].append(record)
            by_bucket[bucket]["pairs_clean"].append(list(canonical_key))
            by_bucket[bucket]["text_pairs"].append(caption)

    stats = {
        "requests_processed": len(response_texts),
        "labeled_pairs_kept": len(labeled_all),
        "empty_outputs": empty_outputs,
        "parse_failures": parse_failures,
        "regex_salvaged_outputs": regex_salvaged_outputs,
        "invalid_pair_ids": invalid_pair_ids,
        "duplicate_pairs_removed": duplicate_pairs,
        "multi_record_outputs": multi_record_outputs,
        "bucket_sizes": {
            bucket: len(payload["pairs_clean"])
            for bucket, payload in by_bucket.items()
        },
        "radiology_pairs_kept": len(by_bucket.get("radiology", {}).get("pairs_clean", [])),
    }

    return {
        "labeled_all": labeled_all,
        "pairs_clean_all": pairs_clean_all,
        "text_pairs_all": [normalize_text_pair_item(item) for item in text_pairs_all],
        "by_bucket": dict(by_bucket),
        "bad_rows": bad_rows,
        "stats": stats,
    }
