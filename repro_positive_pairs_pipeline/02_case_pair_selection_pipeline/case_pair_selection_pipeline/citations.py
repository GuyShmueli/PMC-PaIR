from __future__ import annotations

import hashlib
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from lxml import etree

from .articles import parse_nxml, resolve_nxml
from .citation_recovery import extract_citation_contexts


_PATIENT_UID_RE = re.compile(r"^(?P<article>\d+)-(?P<case>\d+)$")
_IDENTIFIER_PRECEDENCE = ("doi", "pmid", "pmcid")


def _uid_sort_key(uid: str) -> tuple[int, int, str]:
    match = _PATIENT_UID_RE.fullmatch(str(uid).strip())
    if not match:
        raise ValueError(f"Invalid canonical patient_uid: {uid!r}")
    canonical = f"{int(match.group('article'))}-{int(match.group('case'))}"
    if canonical != uid:
        raise ValueError(f"patient_uid is not canonical: {uid!r}")
    return int(match.group("article")), int(match.group("case")), uid


def _expected_pair_key(case_a_uid: str, case_b_uid: str) -> str:
    return hashlib.sha256(f"{case_a_uid}\0{case_b_uid}".encode("utf-8")).hexdigest()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _local_name(element: etree._Element) -> str:
    if not isinstance(element.tag, str):
        return ""
    return etree.QName(element).localname


def _normalize_doi(value: str | None) -> str | None:
    text = _normalize_text(value or "").casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    return text or None


def _normalize_pmid(value: str | None) -> str | None:
    text = _normalize_text(value or "")
    text = re.sub(r"^pmid\s*:\s*", "", text, flags=re.IGNORECASE)
    return str(int(text)) if text.isdigit() else (text.casefold() or None)


def _normalize_pmcid(value: str | None) -> str | None:
    text = _normalize_text(value or "")
    text = re.sub(r"^pmcid\s*:\s*", "", text, flags=re.IGNORECASE)
    match = re.fullmatch(r"(?:PMC)?(\d+)", text, flags=re.IGNORECASE)
    return f"PMC{int(match.group(1))}" if match else (text.upper() or None)


def _normalize_identifier(kind: str, value: str | None) -> str | None:
    if kind == "doi":
        return _normalize_doi(value)
    if kind == "pmid":
        return _normalize_pmid(value)
    if kind == "pmcid":
        return _normalize_pmcid(value)
    raise ValueError(f"Unknown identifier kind: {kind}")


def _normalize_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return _normalize_text(normalized.replace("_", " "))


def _title_matches(
    reference_title: str,
    cited_title: str,
    *,
    min_chars: int,
    min_tokens: int,
) -> bool:
    reference = _normalize_title(reference_title)
    cited = _normalize_title(cited_title)
    if not reference or not cited:
        return False
    if len(cited) < min_chars or len(cited.split()) < min_tokens:
        return False
    # Padding makes this a token-boundary substring match, avoiding matches
    # inside longer words while still supporting verbose mixed citations.
    return cited == reference or f" {cited} " in f" {reference} "


def _element_text(element: etree._Element) -> str:
    return _normalize_text(" ".join(element.itertext()))


def _article_identifiers(root: etree._Element) -> dict[str, str | None]:
    result: dict[str, str | None] = {kind: None for kind in _IDENTIFIER_PRECEDENCE}
    for element in root.iter():
        if _local_name(element) != "article-id":
            continue
        id_type = str(element.get("pub-id-type", "")).strip().casefold()
        kind = "pmcid" if id_type in {"pmc", "pmcid"} else id_type
        if kind not in result or result[kind] is not None:
            continue
        result[kind] = _normalize_identifier(kind, _element_text(element))
    return result


def _article_title(root: etree._Element) -> str:
    for element in root.iter():
        if _local_name(element) != "article-title":
            continue
        if any(_local_name(ancestor) == "ref" for ancestor in element.iterancestors()):
            continue
        return _element_text(element)
    return ""


def _article_abstract(root: etree._Element) -> str:
    abstracts: list[str] = []
    for element in root.iter():
        if _local_name(element) != "abstract":
            continue
        if any(_local_name(ancestor) == "ref" for ancestor in element.iterancestors()):
            continue
        text = _element_text(element)
        if text and text not in abstracts:
            abstracts.append(text)
    return " ".join(abstracts)


def _reference_title(reference: etree._Element) -> str:
    for element in reference.iter():
        if _local_name(element) == "article-title":
            return _element_text(element)
    for preferred_name in ("mixed-citation", "comment"):
        for element in reference.iter():
            if _local_name(element) == preferred_name:
                return _element_text(element)
    return ""


def _references(root: etree._Element) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for reference in root.iter():
        if _local_name(reference) != "ref":
            continue
        ref_id = str(reference.get("id", "")).strip()
        if not ref_id:
            continue
        if ref_id in seen_ids:
            raise ValueError(f"NXML contains duplicate reference ID: {ref_id}")
        seen_ids.add(ref_id)

        identifiers: dict[str, str | None] = {
            kind: None for kind in _IDENTIFIER_PRECEDENCE
        }
        for element in reference.iter():
            if _local_name(element) != "pub-id":
                continue
            id_type = str(element.get("pub-id-type", "")).strip().casefold()
            kind = "pmcid" if id_type in {"pmc", "pmcid"} else id_type
            if kind not in identifiers or identifiers[kind] is not None:
                continue
            identifiers[kind] = _normalize_identifier(kind, _element_text(element))
        records.append(
            {
                "ref_id": ref_id,
                **identifiers,
                "title": _reference_title(reference),
            }
        )
    return records


def _citation_paragraphs(root: etree._Element) -> dict[str, list[str]]:
    return extract_citation_contexts(root)


def _parse_case_worker(task: tuple[str, dict[str, Any], str]) -> tuple[str, dict[str, Any]]:
    uid, case, run_root = task
    try:
        if case.get("schema_version") != 1:
            raise ValueError("case schema_version must be 1")
        _uid_sort_key(uid)
        if str(case.get("patient_uid", "")) != uid:
            raise ValueError("cases_by_uid key does not match case patient_uid")
        nxml = resolve_nxml(case, run_root)
        root = parse_nxml(nxml)
        identifiers = _article_identifiers(root)
        expected_pmcid = _normalize_pmcid(str(case.get("pmc_id", "")))
        if expected_pmcid != f"PMC{uid.split('-', 1)[0]}":
            raise ValueError("case pmc_id does not match patient_uid")
        if identifiers["pmcid"] and identifiers["pmcid"] != expected_pmcid:
            raise ValueError("NXML PMCID does not match the case record")
        return uid, {
            "status": "valid",
            **identifiers,
            "title": _article_title(root),
            "abstract": _article_abstract(root),
            "references": _references(root),
            "paragraphs": _citation_paragraphs(root),
        }
    except etree.XMLSyntaxError:
        return uid, {"status": "error", "error": "XMLSyntaxError: malformed NXML"}
    except OSError:
        return uid, {"status": "error", "error": "OSError: unable to read NXML"}
    except ValueError as exc:
        return uid, {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _has_identifier_conflict(reference: dict[str, Any], cited: dict[str, Any]) -> bool:
    for kind in _IDENTIFIER_PRECEDENCE:
        reference_id = reference.get(kind)
        cited_id = cited.get(kind)
        if reference_id and cited_id and reference_id != cited_id:
            return True
    return False


def _match_direction(
    citing: dict[str, Any],
    cited: dict[str, Any],
    *,
    title_fallback: bool,
    title_min_chars: int,
    title_min_tokens: int,
) -> tuple[dict[str, Any] | None, str | None]:
    references = citing["references"]
    for kind in _IDENTIFIER_PRECEDENCE:
        cited_id = cited.get(kind)
        if not cited_id:
            continue
        matches = [
            reference
            for reference in references
            if reference.get(kind) == cited_id
            and not _has_identifier_conflict(reference, cited)
        ]
        if len(matches) == 1:
            return matches[0], kind
        if len(matches) > 1:
            raise ValueError(f"multiple references match the cited article by {kind}")

    if not title_fallback:
        return None, None
    title_matches = [
        reference
        for reference in references
        if not _has_identifier_conflict(reference, cited)
        and _title_matches(
            str(reference.get("title", "")),
            str(cited.get("title", "")),
            min_chars=title_min_chars,
            min_tokens=title_min_tokens,
        )
    ]
    if len(title_matches) == 1:
        return title_matches[0], "title_substring"
    if len(title_matches) > 1:
        raise ValueError("multiple references match the cited article by title")
    return None, None


def _base_record(pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_index": pair["pair_index"],
        "pair_key": pair["pair_key"],
        "case_a_uid": pair["case_a_uid"],
        "case_b_uid": pair["case_b_uid"],
    }


def _matched_record(
    pair: dict[str, Any],
    *,
    direction: str,
    citing_uid: str,
    cited_uid: str,
    citing: dict[str, Any],
    cited: dict[str, Any],
    reference: dict[str, Any],
    match_type: str,
) -> dict[str, Any]:
    return {
        **_base_record(pair),
        "status": "matched",
        "direction": direction,
        "citing_uid": citing_uid,
        "cited_uid": cited_uid,
        "match_type": match_type,
        "ref_id": reference["ref_id"],
        "citing_title": citing["title"],
        "citing_abstract": citing["abstract"],
        "citation_paragraphs": citing["paragraphs"].get(reference["ref_id"], []),
        "cited_title": cited["title"],
        "cited_abstract": cited["abstract"],
    }


def _validate_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    seen_keys: set[str] = set()
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("pairs must contain only pair objects")
        if pair.get("schema_version") != 1:
            raise ValueError("pair schema_version must be 1")
        pair_index = pair.get("pair_index")
        if isinstance(pair_index, bool) or not isinstance(pair_index, int) or pair_index < 0:
            raise ValueError("Every pair_index must be a nonnegative integer")
        case_a_uid = str(pair.get("case_a_uid", ""))
        case_b_uid = str(pair.get("case_b_uid", ""))
        if _uid_sort_key(case_a_uid) >= _uid_sort_key(case_b_uid):
            raise ValueError("Pair UIDs must be distinct and in canonical order")
        pair_key = str(pair.get("pair_key", ""))
        if pair_key != _expected_pair_key(case_a_uid, case_b_uid):
            raise ValueError(f"Pair {pair_index} has an invalid pair_key")
        if pair_index in seen_indices or pair_key in seen_keys:
            raise ValueError("pairs contains a duplicate pair_index or pair_key")
        seen_indices.add(pair_index)
        seen_keys.add(pair_key)
        ordered.append(pair)
    return sorted(ordered, key=lambda pair: pair["pair_index"])


def match_pair_citations(
    pairs: list[dict[str, Any]],
    cases_by_uid: dict[str, dict[str, Any]],
    retrieval_run: str | Path,
    *,
    title_fallback: bool = False,
    title_min_chars: int = 24,
    title_min_tokens: int = 4,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Match citations in either direction and return one status per pair.

    Direction precedence is canonical A-to-B before B-to-A.  Within a
    direction, DOI precedes PMID, which precedes PMCID; guarded title matching
    is attempted only when no identifier matches.
    """
    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError("workers must be at least 1")
    title_min_chars = int(title_min_chars)
    title_min_tokens = int(title_min_tokens)
    if title_min_chars < 0 or title_min_tokens < 0:
        raise ValueError("Title thresholds cannot be negative")
    if title_fallback and (title_min_chars < 1 or title_min_tokens < 1):
        raise ValueError("Safe title fallback requires nonzero thresholds")

    run_root = Path(retrieval_run).resolve()
    if not run_root.is_dir():
        raise ValueError(f"Retrieval run is not a directory: {retrieval_run}")
    if not isinstance(pairs, list):
        raise ValueError("pairs must be a list")
    if not isinstance(cases_by_uid, dict):
        raise ValueError("cases_by_uid must be a mapping")
    ordered_pairs = _validate_pairs(pairs)

    required_uids = sorted(
        {
            str(pair[field])
            for pair in ordered_pairs
            for field in ("case_a_uid", "case_b_uid")
        },
        key=_uid_sort_key,
    )
    tasks: list[tuple[str, dict[str, Any], str]] = []
    missing_uids: set[str] = set()
    for uid in required_uids:
        case = cases_by_uid.get(uid)
        if case is None:
            missing_uids.add(uid)
            continue
        if not isinstance(case, dict):
            raise ValueError(f"cases_by_uid[{uid!r}] is not a case object")
        tasks.append((uid, case, str(run_root)))

    if worker_count == 1 or len(tasks) < 2:
        parsed = dict(_parse_case_worker(task) for task in tasks)
    else:
        chunksize = max(1, len(tasks) // (worker_count * 4))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            parsed = dict(executor.map(_parse_case_worker, tasks, chunksize=chunksize))

    records: list[dict[str, Any]] = []
    for pair in ordered_pairs:
        uid_a = pair["case_a_uid"]
        uid_b = pair["case_b_uid"]
        if uid_a in missing_uids or uid_b in missing_uids:
            records.append(
                {
                    **_base_record(pair),
                    "status": "error",
                    "error": "cases_by_uid is missing one or both pair members",
                }
            )
            continue
        doc_a = parsed[uid_a]
        doc_b = parsed[uid_b]
        if doc_a["status"] != "valid" or doc_b["status"] != "valid":
            errors = [
                f"{uid}: {doc['error']}"
                for uid, doc in ((uid_a, doc_a), (uid_b, doc_b))
                if doc["status"] != "valid"
            ]
            records.append(
                {
                    **_base_record(pair),
                    "status": "error",
                    "error": "; ".join(errors),
                }
            )
            continue

        errors: list[str] = []
        for direction, citing_uid, cited_uid, citing, cited in (
            ("case_a_cites_case_b", uid_a, uid_b, doc_a, doc_b),
            ("case_b_cites_case_a", uid_b, uid_a, doc_b, doc_a),
        ):
            try:
                reference, match_type = _match_direction(
                    citing,
                    cited,
                    title_fallback=bool(title_fallback),
                    title_min_chars=title_min_chars,
                    title_min_tokens=title_min_tokens,
                )
            except ValueError as exc:
                errors.append(f"{direction}: {exc}")
                continue
            if reference is not None and match_type is not None:
                records.append(
                    _matched_record(
                        pair,
                        direction=direction,
                        citing_uid=citing_uid,
                        cited_uid=cited_uid,
                        citing=citing,
                        cited=cited,
                        reference=reference,
                        match_type=match_type,
                    )
                )
                break
        else:
            if errors:
                records.append(
                    {
                        **_base_record(pair),
                        "status": "error",
                        "error": "ValueError: " + "; ".join(errors),
                    }
                )
            else:
                records.append({**_base_record(pair), "status": "no_match"})
    return records
