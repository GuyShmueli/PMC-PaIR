from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from lxml import etree

from .articles import parse_nxml, resolve_nxml


_PATIENT_UID_RE = re.compile(r"^(?P<article>\d+)-(?P<case>\d+)$")
_CASE_SECTION_PHRASES = (
    "case presentation",
    "case report",
    "case description",
    "case summary",
    "patient presentation",
    "patient report",
    "patient description",
    "patient summary",
)


def _uid_sort_key(uid: str) -> tuple[int, int, str]:
    match = _PATIENT_UID_RE.fullmatch(str(uid).strip())
    if not match:
        raise ValueError(f"Invalid canonical patient_uid: {uid!r}")
    normalized = f"{int(match.group('article'))}-{int(match.group('case'))}"
    if normalized != uid:
        raise ValueError(f"patient_uid is not canonical: {uid!r}")
    return int(match.group("article")), int(match.group("case")), uid


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _local_name(element: etree._Element) -> str:
    if not isinstance(element.tag, str):
        return ""
    return etree.QName(element).localname


def _section_title(section: etree._Element) -> str:
    for child in section:
        if _local_name(child) == "title":
            return _normalize_text(" ".join(child.itertext())).casefold()
    return ""


def _is_case_section(title: str) -> bool:
    if not title:
        return False
    if title in {"case", "patient"}:
        return True
    if any(phrase in title for phrase in _CASE_SECTION_PHRASES):
        return True
    # Case reports commonly use titles such as "Case 1" or "Clinical case".
    return re.search(r"\bcases?\b", title) is not None


def _extract_summary(nxml: Path) -> str:
    root = parse_nxml(nxml)

    paragraphs: list[str] = []
    seen: set[str] = set()
    for section in root.iter():
        if _local_name(section) != "sec" or not _is_case_section(_section_title(section)):
            continue
        for paragraph in section.iter():
            if _local_name(paragraph) != "p":
                continue
            text = _normalize_text(" ".join(paragraph.itertext()))
            if text and text not in seen:
                paragraphs.append(text)
                seen.add(text)
    return "\n\n".join(paragraphs)


def _summary_worker(task: tuple[dict[str, Any], str]) -> dict[str, Any]:
    case, run_root = task
    uid = str(case.get("patient_uid", "")).strip()
    base = {"patient_uid": uid, "summary": ""}
    try:
        if case.get("schema_version") != 1:
            raise ValueError("case schema_version must be 1")
        _uid_sort_key(uid)
        nxml = resolve_nxml(case, run_root)
        summary = _extract_summary(nxml)
    except etree.XMLSyntaxError:
        return {**base, "status": "error", "error": "XMLSyntaxError: malformed NXML"}
    except OSError:
        return {**base, "status": "error", "error": "OSError: unable to read NXML"}
    except ValueError as exc:
        return {
            **base,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not summary:
        return {**base, "status": "invalid"}
    return {"patient_uid": uid, "status": "valid", "summary": summary}


def extract_case_summaries(
    cases: list[dict[str, Any]],
    retrieval_run: str | Path,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Extract one strict, namespace-independent case summary per case."""
    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError("workers must be at least 1")
    run_root = Path(retrieval_run).resolve()
    if not run_root.is_dir():
        raise ValueError(f"Retrieval run is not a directory: {retrieval_run}")

    if not isinstance(cases, list) or any(not isinstance(case, dict) for case in cases):
        raise ValueError("cases must be a list of case objects")
    ordered_cases = sorted(
        cases,
        key=lambda case: _uid_sort_key(str(case.get("patient_uid", ""))),
    )
    uids = [str(case["patient_uid"]) for case in ordered_cases]
    if len(uids) != len(set(uids)):
        raise ValueError("cases contains duplicate patient_uid values")

    tasks = [(case, str(run_root)) for case in ordered_cases]
    if worker_count == 1 or len(tasks) < 2:
        return [_summary_worker(task) for task in tasks]

    chunksize = max(1, len(tasks) // (worker_count * 4))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(_summary_worker, tasks, chunksize=chunksize))
