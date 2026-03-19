from __future__ import annotations

import re
from multiprocessing import Pool
from pathlib import Path

from lxml import etree


CASE_SUMMARY_XPATH = (
    "//sec["
    "  contains(translate(title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'case presentation') "
    "or contains(translate(title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'case report') "
    "or contains(translate(title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'case description') "
    "or contains(translate(title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'case summary') "
    "or contains(translate(title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'patient presentation') "
    "or contains(translate(title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'patient report') "
    "or contains(translate(title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'patient description') "
    "or contains(translate(title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'patient summary') "
    "or contains(translate(title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'case') "
    "or (normalize-space(translate(title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')) = 'patient') "
    "]"
)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_case_summary_from_nxml(file_path: str | None) -> str:
    if not file_path:
        return ""

    path = Path(file_path)
    if not path.is_file():
        return ""

    parser = etree.XMLParser(recover=True, huge_tree=True)
    try:
        tree = etree.parse(str(path), parser)
    except Exception:
        return ""

    try:
        sections = tree.xpath(CASE_SUMMARY_XPATH)
    except Exception:
        return ""

    seen = set()
    texts: list[str] = []

    for sec in sections:
        paragraph_text = " ".join(sec.xpath(".//p//text()"))
        paragraph_text = _normalize_whitespace(paragraph_text)
        if paragraph_text and paragraph_text not in seen:
            texts.append(paragraph_text)
            seen.add(paragraph_text)

    return "\n\n".join(texts)


def summarize_xml_pair(xml_pair: list[str | None]) -> list[str]:
    if not isinstance(xml_pair, list) or len(xml_pair) != 2:
        return ["", ""]
    return [
        extract_case_summary_from_nxml(xml_pair[0]),
        extract_case_summary_from_nxml(xml_pair[1]),
    ]


def extract_case_summary_pairs(
    xml_pairs: list[list[str | None]],
    *,
    workers: int | None = None,
) -> list[list[str]]:
    if workers == 1:
        return [summarize_xml_pair(pair) for pair in xml_pairs]

    with Pool(processes=workers) as pool:
        return pool.map(summarize_xml_pair, xml_pairs)


def find_valid_case_summary_indices(case_summary_pairs: list[list[str]]) -> tuple[list[int], list[int]]:
    valid = [
        idx for idx, pair in enumerate(case_summary_pairs)
        if isinstance(pair, list)
        and len(pair) == 2
        and all(_normalize_whitespace(text) for text in pair)
    ]
    invalid = [idx for idx in range(len(case_summary_pairs)) if idx not in set(valid)]
    return valid, invalid
