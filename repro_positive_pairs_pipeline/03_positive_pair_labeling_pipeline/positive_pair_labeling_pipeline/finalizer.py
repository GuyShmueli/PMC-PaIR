"""Strict, deterministic finalization of positive caption-pair responses."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from .errors import ResponseValidationError


_OUTPUT_FIELDS = {"pair_id", "modality", "anatomy", "diagnosis"}
_CODE_FENCE_RE = re.compile(
    r"\A\s*```(?:json)?[^\S\r\n]*\r?\n(?P<body>.*?)\r?\n?```[^\S\r\n]*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


class _DuplicateJsonKey(ValueError):
    """Internal marker for duplicate keys in a JSON object."""


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _parse_response_array(text: Any, *, pair_index: int) -> list[dict[str, Any]]:
    if not isinstance(text, str):
        raise ResponseValidationError(
            f"Response for pair_index {pair_index} has a non-string text field"
        )

    payload = text.strip()
    fence_match = _CODE_FENCE_RE.fullmatch(payload)
    if fence_match is not None:
        payload = fence_match.group("body").strip()

    try:
        parsed = json.loads(
            payload,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, _DuplicateJsonKey, ValueError) as exc:
        detail = str(exc)
        raise ResponseValidationError(
            f"Response for pair_index {pair_index} is not a valid JSON array: {detail}"
        ) from exc

    if not isinstance(parsed, list):
        raise ResponseValidationError(
            f"Response for pair_index {pair_index} must be a JSON array, "
            f"not {type(parsed).__name__}"
        )

    records: list[dict[str, Any]] = []
    for item_index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ResponseValidationError(
                f"Response for pair_index {pair_index}, item {item_index}, "
                "must be a JSON object"
            )

        actual_fields = set(item)
        if actual_fields != _OUTPUT_FIELDS:
            missing = sorted(_OUTPUT_FIELDS - actual_fields)
            extra = sorted(actual_fields - _OUTPUT_FIELDS)
            raise ResponseValidationError(
                f"Response for pair_index {pair_index}, item {item_index}, has "
                f"incorrect fields (missing={missing}, extra={extra})"
            )

        pair_id = item["pair_id"]
        if (
            not isinstance(pair_id, list)
            or len(pair_id) != 2
            or any(not isinstance(value, str) or not value for value in pair_id)
        ):
            raise ResponseValidationError(
                f"Response for pair_index {pair_index}, item {item_index}, "
                "must contain pair_id as exactly two non-empty strings"
            )
        if any(value != value.strip() for value in pair_id):
            raise ResponseValidationError(
                f"Response for pair_index {pair_index}, item {item_index}, "
                "contains a pair_id with surrounding whitespace"
            )

        normalized = {"pair_id": list(pair_id)}
        for field in ("modality", "anatomy", "diagnosis"):
            value = item[field]
            if not isinstance(value, str) or not value.strip():
                raise ResponseValidationError(
                    f"Response for pair_index {pair_index}, item {item_index}, "
                    f"has an empty or non-string {field!r} field"
                )
            normalized[field] = value.strip()
        records.append(normalized)

    return records


def _caption_ids(case: Any, *, case_uid: str) -> set[str]:
    if not isinstance(case, Mapping):
        raise ResponseValidationError(f"Case {case_uid!r} is not an object")
    captions = case.get("captions")
    if not isinstance(captions, list):
        raise ResponseValidationError(
            f"Case {case_uid!r} must contain a captions list"
        )

    result: set[str] = set()
    for caption_index, caption in enumerate(captions):
        if not isinstance(caption, Mapping):
            raise ResponseValidationError(
                f"Case {case_uid!r}, caption {caption_index}, is not an object"
            )
        caption_id = caption.get("caption_id")
        if not isinstance(caption_id, str) or not caption_id:
            raise ResponseValidationError(
                f"Case {case_uid!r}, caption {caption_index}, has no valid caption_id"
            )
        if caption_id != caption_id.strip():
            raise ResponseValidationError(
                f"Case {case_uid!r}, caption {caption_index}, has a caption_id "
                "with surrounding whitespace"
            )
        if caption_id in result:
            raise ResponseValidationError(
                f"Case {case_uid!r} contains duplicate caption_id {caption_id!r}"
            )
        result.add(caption_id)
    return result


def _orient_inter_case_pair(
    pair_id: list[str],
    *,
    case_a_ids: set[str],
    case_b_ids: set[str],
    pair_index: int,
    item_index: int,
) -> tuple[str, str]:
    left, right = pair_id
    forward = left in case_a_ids and right in case_b_ids
    reverse = left in case_b_ids and right in case_a_ids

    if forward == reverse:
        if forward:
            reason = "is ambiguous because caption IDs overlap between the cases"
        else:
            reason = "does not contain one caption ID from each case"
        raise ResponseValidationError(
            f"Response for pair_index {pair_index}, item {item_index}, pair_id "
            f"{pair_id!r} {reason}"
        )

    return (left, right) if forward else (right, left)


_BUCKET_PATTERNS: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    (
        "pathology_microscopy",
        tuple(
            re.compile(pattern, flags=re.IGNORECASE)
            for pattern in (
                r"\bhistolog(?:y|ic|ical)?\b",
                r"\bpatholog(?:y|ic|ical)?\b",
                r"\bmicroscop(?:y|ic|ical)?\b",
                r"\bimmunohistochem(?:istry|ical|ically)?\b",
                r"\bihc\b",
                r"\bhematoxylin\b",
                r"\beosin\b",
                r"\bh\s*&\s*e\b",
                r"\bstain(?:ed|ing|s)?\b",
                r"\bbiops(?:y|ies)\b",
                r"\bslides?\b",
            )
        ),
    ),
    (
        "ophthalmology",
        tuple(
            re.compile(pattern, flags=re.IGNORECASE)
            for pattern in (
                r"\bfundus\b",
                r"\boct\b",
                r"\boptical\s+coherence(?:\s+tomography)?\b",
                r"\bslit[ -]lamp\b",
                r"\bfluorescein\b",
                r"\bindocyanine\b",
                r"\bretin(?:a|al)\b",
                r"\bophthalm(?:ic|ology|ological)?\b",
            )
        ),
    ),
    (
        "endoscopy",
        tuple(
            re.compile(pattern, flags=re.IGNORECASE)
            for pattern in (
                r"\bendoscop(?:e|y|ic)\b",
                r"\bcolonoscopy\b",
                r"\bgastroscopy\b",
                r"\bbronchoscopy\b",
                r"\blaparoscopy\b",
                r"\bhysteroscopy\b",
                r"\barthroscopy\b",
            )
        ),
    ),
    (
        "cardiology",
        tuple(
            re.compile(pattern, flags=re.IGNORECASE)
            for pattern in (
                r"\becg\b",
                r"\bekg\b",
                r"\belectrocardiograph(?:y|ic)?\b",
                r"\bechocardiograph(?:y|ic)?\b",
                r"\bechocardiogram\b",
                r"\bcardiac\s+echo\b",
            )
        ),
    ),
    (
        "radiology",
        tuple(
            re.compile(pattern, flags=re.IGNORECASE)
            for pattern in (
                r"\bct\b",
                r"\bcomputed\s+tomograph(?:y|ic)\b",
                r"\bmri\b",
                r"\bmagnetic\s+resonance(?:\s+imaging)?\b",
                r"\bx[ -]?rays?\b",
                r"\bradiograph(?:s|y|ic|ical)?\b",
                r"\bultrasound\b",
                r"\bultrasonograph(?:y|ic)?\b",
                r"\bsonograph(?:y|ic)?\b",
                r"\bpet\b",
                r"\bspect\b",
                r"\bangiograph(?:y|ic)?\b",
                r"\bfluoroscop(?:y|ic)?\b",
                r"\bmammograph(?:y|ic)?\b",
                r"\btomograph(?:y|ic)?\b",
            )
        ),
    ),
)


def modality_bucket(modality: str) -> str:
    """Classify a modality without matching short tokens inside other words."""

    for bucket, patterns in _BUCKET_PATTERNS:
        if any(pattern.search(modality) is not None for pattern in patterns):
            return bucket
    return "other"


def _short_caption(record: Mapping[str, Any]) -> list[str]:
    text = (
        f"{record['diagnosis']} in the {record['anatomy']} "
        f"on {record['modality']}."
    )
    return [text]


def _selected_pair_lookup(
    selected_pairs: Any,
    cases_by_uid: Any,
) -> dict[int, tuple[dict[str, Any], set[str], set[str]]]:
    if not isinstance(selected_pairs, list):
        raise ResponseValidationError("selected_pairs must be a list")
    if not isinstance(cases_by_uid, Mapping):
        raise ResponseValidationError("cases_by_uid must be an object keyed by case UID")

    lookup: dict[int, tuple[dict[str, Any], set[str], set[str]]] = {}
    seen_pair_keys: set[str] = set()
    for position, selected in enumerate(selected_pairs):
        if not isinstance(selected, dict):
            raise ResponseValidationError(
                f"selected_pairs item {position} is not an object"
            )
        missing = {
            "pair_index",
            "pair_key",
            "case_a_uid",
            "case_b_uid",
        } - set(selected)
        if missing:
            raise ResponseValidationError(
                f"selected_pairs item {position} is missing fields {sorted(missing)}"
            )

        pair_index = selected["pair_index"]
        if isinstance(pair_index, bool) or not isinstance(pair_index, int) or pair_index < 0:
            raise ResponseValidationError(
                f"selected_pairs item {position} has an invalid pair_index"
            )
        if pair_index in lookup:
            raise ResponseValidationError(
                f"selected_pairs contains duplicate pair_index {pair_index}"
            )

        pair_key = selected["pair_key"]
        if not isinstance(pair_key, str) or not pair_key.strip():
            raise ResponseValidationError(
                f"selected_pairs item {position} has an invalid pair_key"
            )
        if pair_key in seen_pair_keys:
            raise ResponseValidationError(
                f"selected_pairs contains duplicate pair_key {pair_key!r}"
            )
        seen_pair_keys.add(pair_key)

        case_a_uid = selected["case_a_uid"]
        case_b_uid = selected["case_b_uid"]
        if (
            not isinstance(case_a_uid, str)
            or not case_a_uid
            or not isinstance(case_b_uid, str)
            or not case_b_uid
            or case_a_uid == case_b_uid
        ):
            raise ResponseValidationError(
                f"selected_pairs item {position} must name two distinct case UIDs"
            )
        if case_a_uid not in cases_by_uid or case_b_uid not in cases_by_uid:
            missing_uids = sorted(
                uid for uid in (case_a_uid, case_b_uid) if uid not in cases_by_uid
            )
            raise ResponseValidationError(
                f"selected_pairs item {position} references unknown cases {missing_uids}"
            )

        case_a_ids = _caption_ids(cases_by_uid[case_a_uid], case_uid=case_a_uid)
        case_b_ids = _caption_ids(cases_by_uid[case_b_uid], case_uid=case_b_uid)
        lookup[pair_index] = (dict(selected), case_a_ids, case_b_ids)

    return lookup


def finalize_positive_responses(
    response_records: Any,
    selected_pairs: Any,
    cases_by_uid: Any,
) -> dict[str, Any]:
    """Validate, canonicalize, deduplicate, and bucket positive-pair outputs.

    The function deliberately performs no best-effort recovery. A malformed,
    missing, duplicate, or misaligned response raises ``ResponseValidationError``.
    An explicit JSON ``[]`` is the only valid representation of no matches.
    """

    selected_lookup = _selected_pair_lookup(selected_pairs, cases_by_uid)
    if not isinstance(response_records, list):
        raise ResponseValidationError("response_records must be a list")

    responses_by_index: dict[int, dict[str, Any]] = {}
    for position, response in enumerate(response_records):
        if not isinstance(response, dict):
            raise ResponseValidationError(
                f"response_records item {position} is not an object"
            )
        if "pair_index" not in response or "text" not in response:
            raise ResponseValidationError(
                f"response_records item {position} must contain pair_index and text"
            )
        pair_index = response["pair_index"]
        if isinstance(pair_index, bool) or not isinstance(pair_index, int):
            raise ResponseValidationError(
                f"response_records item {position} has an invalid pair_index"
            )
        if pair_index not in selected_lookup:
            raise ResponseValidationError(
                f"Response references unselected pair_index {pair_index}"
            )
        if pair_index in responses_by_index:
            raise ResponseValidationError(
                f"Duplicate response for pair_index {pair_index}"
            )
        responses_by_index[pair_index] = response

    missing_responses = sorted(set(selected_lookup) - set(responses_by_index))
    if missing_responses:
        raise ResponseValidationError(
            f"Missing responses for pair_index values {missing_responses}"
        )

    candidates: list[dict[str, Any]] = []
    empty_outputs = 0
    multi_record_outputs = 0
    for pair_index in sorted(responses_by_index):
        response = responses_by_index[pair_index]
        records = _parse_response_array(response["text"], pair_index=pair_index)
        if not records:
            empty_outputs += 1
            continue
        if len(records) > 1:
            multi_record_outputs += 1

        _, case_a_ids, case_b_ids = selected_lookup[pair_index]
        for item_index, record in enumerate(records):
            oriented_pair = _orient_inter_case_pair(
                record["pair_id"],
                case_a_ids=case_a_ids,
                case_b_ids=case_b_ids,
                pair_index=pair_index,
                item_index=item_index,
            )
            canonical_pair = tuple(sorted(oriented_pair))
            candidate = dict(record)
            candidate["pair_id"] = list(canonical_pair)
            candidate["source_pair_index"] = pair_index
            candidate["modality_bucket"] = modality_bucket(candidate["modality"])
            candidates.append(candidate)

    candidates.sort(
        key=lambda record: (
            record["source_pair_index"],
            record["pair_id"][0],
            record["pair_id"][1],
            record["modality"],
            record["anatomy"],
            record["diagnosis"],
        )
    )

    labeled_all: list[dict[str, Any]] = []
    pairs_clean_all: list[list[str]] = []
    text_pairs_all: list[list[str]] = []
    by_bucket_mutable: defaultdict[
        str, dict[str, list[Any]]
    ] = defaultdict(lambda: {"labeled": [], "pairs_clean": [], "text_pairs": []})
    seen_pairs: set[tuple[str, str]] = set()
    duplicate_pairs = 0

    for record in candidates:
        canonical_pair = tuple(record["pair_id"])
        if canonical_pair in seen_pairs:
            duplicate_pairs += 1
            continue
        seen_pairs.add(canonical_pair)

        clean_pair = list(canonical_pair)
        caption = _short_caption(record)
        bucket = record["modality_bucket"]
        labeled_all.append(record)
        pairs_clean_all.append(clean_pair)
        text_pairs_all.append(caption)
        by_bucket_mutable[bucket]["labeled"].append(record)
        by_bucket_mutable[bucket]["pairs_clean"].append(clean_pair)
        by_bucket_mutable[bucket]["text_pairs"].append(caption)

    by_bucket = {
        bucket: by_bucket_mutable[bucket]
        for bucket in sorted(by_bucket_mutable)
    }
    bucket_sizes = {
        bucket: len(payload["pairs_clean"])
        for bucket, payload in by_bucket.items()
    }
    stats = {
        "requests_processed": len(response_records),
        "selected_pairs": len(selected_lookup),
        "labeled_pairs_kept": len(labeled_all),
        "empty_outputs": empty_outputs,
        "parse_failures": 0,
        "invalid_pair_ids": 0,
        "duplicate_pairs_removed": duplicate_pairs,
        "multi_record_outputs": multi_record_outputs,
        "bucket_sizes": bucket_sizes,
        "radiology_pairs_kept": bucket_sizes.get("radiology", 0),
    }

    return {
        "labeled_all": labeled_all,
        "pairs_clean_all": pairs_clean_all,
        "text_pairs_all": text_pairs_all,
        "by_bucket": by_bucket,
        "bad_rows": [],
        "stats": stats,
    }


__all__ = ["finalize_positive_responses", "modality_bucket"]
