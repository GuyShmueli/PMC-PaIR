from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from lxml import etree

from .artifacts import UPSTREAM_ASSETS
from .io_utils import sha256_file


def resolve_nxml(case: dict[str, Any], run_root: str) -> Path:
    relative = str(case.get("nxml_path", "")).strip().replace("\\", "/")
    supplied = Path(relative)
    if not relative or supplied.is_absolute() or ".." in supplied.parts:
        raise ValueError("nxml_path must be a run-relative path")

    root = Path(run_root).resolve()
    nxml = (root / supplied).resolve()
    try:
        nxml.relative_to(root)
        nxml.relative_to((root / UPSTREAM_ASSETS).resolve())
    except ValueError as exc:
        raise ValueError(
            f"nxml_path escapes the retrieval run's unified {UPSTREAM_ASSETS}/ tree"
        ) from exc
    if nxml.suffix.lower() != ".nxml":
        raise ValueError("nxml_path does not identify an .nxml file")
    if not nxml.is_file():
        raise FileNotFoundError("NXML file is missing")

    expected_hash = str(case.get("nxml_sha256", "")).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("nxml_sha256 is missing or malformed")
    if sha256_file(nxml) != expected_hash:
        raise ValueError("NXML file does not match nxml_sha256")
    return nxml


def parse_nxml(path: Path) -> etree._Element:
    parser = etree.XMLParser(
        recover=False,
        resolve_entities=False,
        load_dtd=False,
        no_network=True,
        huge_tree=False,
    )
    return etree.parse(str(path), parser).getroot()
