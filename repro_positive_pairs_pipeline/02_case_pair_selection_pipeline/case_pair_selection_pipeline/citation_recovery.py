from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from lxml import etree


RECOVERY_VERSION = "1.0.0"
_BIBLIOGRAPHIC_XREF_TYPES = {"bibr", "ref"}
_SEMANTIC_CONTAINERS = {
    "p",
    "caption",
    "list-item",
    "fn",
    "disp-quote",
    "statement",
    "boxed-text",
    "verse-group",
    "speech",
    "title",
}
_CITATION_HEADER_RE = re.compile(r"\b(?:reference|references|citation|citations)\b", re.I)
_NUMBER_HEADER_RE = re.compile(r"\b(?:number|numbers|no\.?|ref\.?|reference|citation)\b", re.I)
_PMC_PATH_RE = re.compile(r"(?:^|/)PMC(?P<number>\d+)(?:/|$)", re.I)
_DASH_ONLY_RE = re.compile(r"\s*[-\u2010\u2011\u2012\u2013\u2014\u2212]\s*")
_CITATION_EXPRESSION_RE = re.compile(
    r"[\d\s,;:.\-\u2010\u2011\u2012\u2013\u2014\u2212\[\](){}]+"
)
_DASH_BETWEEN_RE = re.compile(r"\s*[-\u2010\u2011\u2012\u2013\u2014\u2212]\s*")
_LIST_BETWEEN_RE = re.compile(r"\s*[,;]\s*")
_NUMERIC_LABEL_RE = re.compile(r"\s*[\[({]?\s*(\d+)\s*[.\])}]?\s*")
_YEAR_RE = re.compile(r"^(?:18|19|20|21)\d{2}$")

_METHOD_PRIORITY = {
    "direct_xref_table": 0,
    "direct_xref_semantic": 1,
    "numeric_range_endpoint_xrefs": 2,
    "numeric_range_single_xref_interior": 3,
    "numeric_range_single_xref_endpoint": 4,
    "numeric_list_single_xref": 5,
    "structured_table_unlinked": 6,
}


@dataclass(frozen=True)
class CitationExpression:
    numbers: tuple[int, ...]
    ranges: tuple[tuple[int, int], ...]
    has_list_separator: bool


@dataclass(frozen=True)
class ReferenceMetadata:
    ref_id: str
    number: int | None
    surname: str
    year: str
    title: str


@dataclass(frozen=True)
class ContextEvidence:
    method: str
    text: str
    container_type: str
    xpath: str
    document_order: int
    details: dict[str, Any]


@dataclass(frozen=True)
class RecoveryResult:
    status: str
    method: str | None
    contexts: tuple[str, ...]
    evidence: tuple[ContextEvidence, ...]

    @property
    def extracted_citation(self) -> str:
        return "\n\n".join(self.contexts)


def _local_name(element: etree._Element) -> str:
    if not isinstance(element.tag, str):
        return ""
    return etree.QName(element).localname


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _element_text(element: etree._Element) -> str:
    return _normalize_text(" ".join(element.itertext()))


def _direct_cells(row: etree._Element) -> list[etree._Element]:
    return [child for child in row if _local_name(child) in {"td", "th"}]


def _nearest_ancestor(
    element: etree._Element,
    names: set[str],
    *,
    include_self: bool = False,
) -> etree._Element | None:
    current: etree._Element | None = element if include_self else element.getparent()
    while current is not None:
        if _local_name(current) in names:
            return current
        current = current.getparent()
    return None


def _inside_bibliography(element: etree._Element) -> bool:
    return any(_local_name(ancestor) in {"ref", "ref-list"} for ancestor in element.iterancestors())


def _is_bibliographic_xref(element: etree._Element) -> bool:
    return (
        _local_name(element) == "xref"
        and str(element.get("ref-type", "")).strip().casefold()
        in _BIBLIOGRAPHIC_XREF_TYPES
        and not _inside_bibliography(element)
    )


def _semantic_container(element: etree._Element) -> etree._Element | None:
    # A row is more meaningful than a distant paragraph when publishers embed
    # table markup inside <p>.
    row = _nearest_ancestor(element, {"tr"})
    if row is not None:
        return row

    current: etree._Element | None = element
    while current is not None:
        name = _local_name(current)
        if name in _SEMANTIC_CONTAINERS:
            if name == "title":
                caption = _nearest_ancestor(current, {"caption"})
                if caption is not None:
                    return caption
            return current
        if name in {"ref", "ref-list"}:
            return None
        current = current.getparent()
    return None


def _nearest_table(element: etree._Element) -> etree._Element | None:
    return _nearest_ancestor(element, {"table"}, include_self=True)


def _belongs_to_table(element: etree._Element, table: etree._Element) -> bool:
    return _nearest_table(element) is table


def _table_rows(table: etree._Element) -> list[etree._Element]:
    return [
        element
        for element in table.iter()
        if _local_name(element) == "tr" and _belongs_to_table(element, table)
    ]


def _table_caption(table: etree._Element) -> str:
    table_wrap = _nearest_ancestor(table, {"table-wrap"})
    search_root = table_wrap if table_wrap is not None else table
    for element in search_root.iter():
        if _local_name(element) != "caption":
            continue
        nested_table = _nearest_table(element)
        if nested_table is not None and nested_table is not table:
            continue
        text = _element_text(element)
        if text:
            return text
    return ""


def _row_text(row: etree._Element) -> str:
    cells = [_element_text(cell) for cell in _direct_cells(row)]
    cells = [text for text in cells if text]
    return " | ".join(cells) if cells else _element_text(row)


def _table_header_rows(table: etree._Element, target: etree._Element) -> list[str]:
    headers: list[str] = []
    for row in _table_rows(table):
        if row is target:
            break
        in_thead = _nearest_ancestor(row, {"thead"}) is not None
        if not in_thead and not any(_local_name(cell) == "th" for cell in _direct_cells(row)):
            continue
        text = _row_text(row)
        if text and text not in headers:
            headers.append(text)
    return headers


def _format_table_row(row: etree._Element) -> str:
    table = _nearest_table(row)
    parts: list[str] = []
    if table is not None:
        caption = _table_caption(table)
        if caption:
            parts.append(f"Table: {caption}")
        headers = _table_header_rows(table, row)
        if headers:
            parts.append("Headers: " + " || ".join(headers))
    text = _row_text(row)
    if text:
        parts.append(f"Row: {text}")
    return "\n".join(parts)


def _format_context(element: etree._Element) -> str:
    if _local_name(element) == "tr":
        return _format_table_row(element)
    return _element_text(element)


def _numeric_label(value: str) -> int | None:
    match = _NUMERIC_LABEL_RE.fullmatch(value or "")
    return int(match.group(1)) if match else None


def _first_descendant_text(element: etree._Element, name: str) -> str:
    for descendant in element.iter():
        if _local_name(descendant) == name:
            text = _element_text(descendant)
            if text:
                return text
    return ""


def _reference_metadata(root: etree._Element) -> dict[str, ReferenceMetadata]:
    result: dict[str, ReferenceMetadata] = {}
    for element in root.iter():
        if _local_name(element) != "ref":
            continue
        ref_id = str(element.get("id", "")).strip()
        if not ref_id:
            continue
        if ref_id in result:
            raise ValueError(f"Duplicate reference ID: {ref_id}")
        label = _first_descendant_text(element, "label")
        result[ref_id] = ReferenceMetadata(
            ref_id=ref_id,
            number=_numeric_label(label),
            surname=_first_descendant_text(element, "surname"),
            year=_first_descendant_text(element, "year"),
            title=_first_descendant_text(element, "article-title"),
        )
    return result


def _unique_numeric_references(
    references: dict[str, ReferenceMetadata],
) -> tuple[dict[int, str], dict[str, int]]:
    by_number: dict[int, list[str]] = defaultdict(list)
    for ref_id, reference in references.items():
        if reference.number is not None:
            by_number[reference.number].append(ref_id)
    number_to_id = {
        number: ref_ids[0]
        for number, ref_ids in by_number.items()
        if len(ref_ids) == 1
    }
    return number_to_id, {ref_id: number for number, ref_id in number_to_id.items()}


def _citation_expression(value: str) -> CitationExpression | None:
    text = _normalize_text(value)
    if not text or _CITATION_EXPRESSION_RE.fullmatch(text) is None:
        return None
    matches = list(re.finditer(r"\d+", text))
    if not matches:
        return None
    numbers = tuple(int(match.group()) for match in matches)
    ranges: list[tuple[int, int]] = []
    has_list_separator = False
    for left, right in zip(matches, matches[1:]):
        separator = text[left.end() : right.start()]
        if _DASH_BETWEEN_RE.fullmatch(separator):
            start = int(left.group())
            end = int(right.group())
            if start < end and end - start <= 1000:
                ranges.append((start, end))
        elif _LIST_BETWEEN_RE.fullmatch(separator):
            has_list_separator = True
    return CitationExpression(
        numbers=numbers,
        ranges=tuple(ranges),
        has_list_separator=has_list_separator,
    )


def _xref_rids(xref: etree._Element) -> tuple[str, ...]:
    return tuple(str(xref.get("rid", "")).split())


def _linearized_events(
    container: etree._Element,
) -> list[tuple[str, str | etree._Element]]:
    events: list[tuple[str, str | etree._Element]] = []

    def append_text(value: str | None) -> None:
        if value:
            events.append(("text", value))

    def visit(element: etree._Element) -> None:
        append_text(element.text)
        for child in element:
            if _is_bibliographic_xref(child) and _semantic_container(child) is container:
                events.append(("xref", child))
            else:
                visit(child)
            append_text(child.tail)

    visit(container)
    return events


def _ordered_unique_evidence(
    evidence: Iterable[ContextEvidence],
) -> tuple[ContextEvidence, ...]:
    ordered = sorted(
        evidence,
        key=lambda item: (
            item.document_order,
            _METHOD_PRIORITY.get(item.method, 999),
            item.xpath,
        ),
    )
    seen: set[str] = set()
    result: list[ContextEvidence] = []
    for item in ordered:
        if not item.text or item.text in seen:
            continue
        seen.add(item.text)
        result.append(item)
    return tuple(result)


def _table_grid(table: etree._Element) -> list[tuple[etree._Element, dict[int, etree._Element]]]:
    result: list[tuple[etree._Element, dict[int, etree._Element]]] = []
    active: dict[int, tuple[etree._Element, int]] = {}
    for row in _table_rows(table):
        row_map = {column: value[0] for column, value in active.items()}
        carry = {
            column: (cell, remaining - 1)
            for column, (cell, remaining) in active.items()
            if remaining > 1
        }
        column = 0
        for cell in _direct_cells(row):
            while column in row_map:
                column += 1
            try:
                colspan = max(1, int(cell.get("colspan", "1")))
                rowspan = max(1, int(cell.get("rowspan", "1")))
            except ValueError:
                colspan = rowspan = 1
            for offset in range(colspan):
                slot = column + offset
                while slot in row_map:
                    slot += 1
                row_map[slot] = cell
                if rowspan > 1:
                    carry[slot] = (cell, rowspan - 1)
            column += colspan
        result.append((row, row_map))
        active = carry
    return result


class CitationContextIndex:
    def __init__(self, root: etree._Element):
        self.root = root
        self.tree = root.getroottree()
        self.references = _reference_metadata(root)
        self.number_to_id, self.id_to_number = _unique_numeric_references(self.references)
        self.element_order = {element: index for index, element in enumerate(root.iter())}
        self._formatted_blocks: dict[etree._Element, str] = {}
        self._block_xpaths: dict[etree._Element, str] = {}
        self.direct: dict[str, list[ContextEvidence]] = defaultdict(list)
        self.endpoint_ranges: dict[str, list[ContextEvidence]] = defaultdict(list)
        self.single_range_interiors: dict[str, list[ContextEvidence]] = defaultdict(list)
        self.single_range_endpoints: dict[str, list[ContextEvidence]] = defaultdict(list)
        self.single_lists: dict[str, list[ContextEvidence]] = defaultdict(list)
        self.structured_tables: dict[str, list[ContextEvidence]] = defaultdict(list)
        self._xrefs: list[etree._Element] = []
        self._blocks: dict[etree._Element, list[etree._Element]] = defaultdict(list)
        self._index_xrefs()
        self._index_endpoint_ranges()
        self._index_single_xref_expressions()
        self._index_structured_tables()

    def _evidence(
        self,
        method: str,
        block: etree._Element,
        details: dict[str, Any],
    ) -> ContextEvidence | None:
        if block not in self._formatted_blocks:
            self._formatted_blocks[block] = _format_context(block)
        text = self._formatted_blocks[block]
        if not text:
            return None
        if block not in self._block_xpaths:
            self._block_xpaths[block] = self.tree.getpath(block)
        return ContextEvidence(
            method=method,
            text=text,
            container_type=_local_name(block),
            xpath=self._block_xpaths[block],
            document_order=self.element_order[block],
            details=details,
        )

    def _index_xrefs(self) -> None:
        for element in self.root.iter():
            if not _is_bibliographic_xref(element):
                continue
            block = _semantic_container(element)
            if block is None:
                continue
            self._xrefs.append(element)
            self._blocks[block].append(element)
            method = "direct_xref_table" if _local_name(block) == "tr" else "direct_xref_semantic"
            for ref_id in _xref_rids(element):
                evidence = self._evidence(
                    method,
                    block,
                    {"ref_id": ref_id, "xref_xpath": self.tree.getpath(element)},
                )
                if evidence is not None:
                    self.direct[ref_id].append(evidence)

    def _xref_anchor_number(self, xref: etree._Element) -> int | None:
        expression = _citation_expression(_element_text(xref))
        if expression is None or len(expression.numbers) != 1 or expression.ranges:
            return None
        number = expression.numbers[0]
        ref_id = self.number_to_id.get(number)
        if ref_id is None or ref_id not in _xref_rids(xref):
            return None
        return number

    def _index_endpoint_ranges(self) -> None:
        for block in self._blocks:
            events = _linearized_events(block)
            positions = [index for index, event in enumerate(events) if event[0] == "xref"]
            for left_position, right_position in zip(positions, positions[1:]):
                left = events[left_position][1]
                right = events[right_position][1]
                if not isinstance(left, etree._Element) or not isinstance(right, etree._Element):
                    continue
                between = "".join(
                    str(value)
                    for kind, value in events[left_position + 1 : right_position]
                    if kind == "text"
                )
                if _DASH_ONLY_RE.fullmatch(between) is None:
                    continue
                start = self._xref_anchor_number(left)
                end = self._xref_anchor_number(right)
                if start is None or end is None or start >= end or end - start > 1000:
                    continue
                for number in range(start + 1, end):
                    ref_id = self.number_to_id.get(number)
                    if ref_id is None:
                        continue
                    evidence = self._evidence(
                        "numeric_range_endpoint_xrefs",
                        block,
                        {
                            "range_start": start,
                            "range_end": end,
                            "left_xref_xpath": self.tree.getpath(left),
                            "right_xref_xpath": self.tree.getpath(right),
                        },
                    )
                    if evidence is not None:
                        self.endpoint_ranges[ref_id].append(evidence)

    def _index_single_xref_expressions(self) -> None:
        for xref in self._xrefs:
            expression = _citation_expression(_element_text(xref))
            if expression is None:
                continue
            rids = set(_xref_rids(xref))
            anchored_numbers = {
                number
                for number in expression.numbers
                if self.number_to_id.get(number) in rids
            }
            if not anchored_numbers:
                continue
            block = _semantic_container(xref)
            if block is None:
                continue

            for start, end in expression.ranges:
                if start not in self.number_to_id or end not in self.number_to_id:
                    continue
                for number in range(start + 1, end):
                    ref_id = self.number_to_id.get(number)
                    if ref_id is None or ref_id in rids:
                        continue
                    evidence = self._evidence(
                        "numeric_range_single_xref_interior",
                        block,
                        {
                            "range_start": start,
                            "range_end": end,
                            "xref_xpath": self.tree.getpath(xref),
                        },
                    )
                    if evidence is not None:
                        self.single_range_interiors[ref_id].append(evidence)

                for number in (start, end):
                    ref_id = self.number_to_id[number]
                    if ref_id in rids:
                        continue
                    evidence = self._evidence(
                        "numeric_range_single_xref_endpoint",
                        block,
                        {
                            "range_start": start,
                            "range_end": end,
                            "xref_xpath": self.tree.getpath(xref),
                        },
                    )
                    if evidence is not None:
                        self.single_range_endpoints[ref_id].append(evidence)

            if expression.has_list_separator and len(expression.numbers) > 1:
                range_endpoints = {number for pair in expression.ranges for number in pair}
                for number in set(expression.numbers) - range_endpoints:
                    ref_id = self.number_to_id.get(number)
                    if ref_id is None or ref_id in rids:
                        continue
                    evidence = self._evidence(
                        "numeric_list_single_xref",
                        block,
                        {
                            "displayed_numbers": list(expression.numbers),
                            "xref_xpath": self.tree.getpath(xref),
                        },
                    )
                    if evidence is not None:
                        self.single_lists[ref_id].append(evidence)

    def _unique_author_year_keys(self) -> set[tuple[str, str]]:
        counts: Counter[tuple[str, str]] = Counter()
        for reference in self.references.values():
            surname = _normalize_text(reference.surname).casefold()
            year = _normalize_text(reference.year)
            if surname and _YEAR_RE.fullmatch(year):
                counts[(surname, year)] += 1
        return {key for key, count in counts.items() if count == 1}

    def _index_structured_tables(self) -> None:
        unique_author_year = self._unique_author_year_keys()
        for table in (element for element in self.root.iter() if _local_name(element) == "table"):
            if _inside_bibliography(table):
                continue
            grid = _table_grid(table)
            if not grid:
                continue
            all_header_text = " ".join(
                _element_text(cell)
                for _, row_map in grid
                for cell in set(row_map.values())
                if _local_name(cell) == "th"
            )
            if _CITATION_HEADER_RE.search(all_header_text) is None:
                continue
            row_index = {row: index for index, (row, _) in enumerate(grid)}
            for row, row_map in grid:
                if any(_is_bibliographic_xref(element) for element in row.iter()):
                    continue
                for column, cell in row_map.items():
                    if cell.getparent() is not row or _local_name(cell) != "td":
                        continue
                    number = _numeric_label(_element_text(cell))
                    ref_id = self.number_to_id.get(number) if number is not None else None
                    if ref_id is None:
                        continue
                    header_texts: list[str] = []
                    for prior_row, prior_map in grid[: row_index[row]]:
                        header = prior_map.get(column)
                        if header is None or _local_name(header) != "th":
                            continue
                        text = _element_text(header)
                        if text and text not in header_texts:
                            header_texts.append(text)
                    if _CITATION_HEADER_RE.search(" ".join(header_texts)) is None and _NUMBER_HEADER_RE.search(
                        " ".join(header_texts)
                    ) is None:
                        continue
                    reference = self.references[ref_id]
                    surname = _normalize_text(reference.surname).casefold()
                    year = _normalize_text(reference.year)
                    if (surname, year) not in unique_author_year:
                        continue
                    row_casefold = _normalize_text(_row_text(row)).casefold()
                    if re.search(rf"(?<!\w){re.escape(surname)}(?!\w)", row_casefold) is None:
                        continue
                    if re.search(rf"(?<!\d){re.escape(year)}(?!\d)", row_casefold) is None:
                        continue
                    evidence = self._evidence(
                        "structured_table_unlinked",
                        row,
                        {
                            "reference_number": number,
                            "matched_surname": reference.surname,
                            "matched_year": year,
                            "column_headers": header_texts,
                        },
                    )
                    if evidence is not None:
                        self.structured_tables[ref_id].append(evidence)

    def recover(self, ref_id: str) -> RecoveryResult:
        if ref_id not in self.references:
            raise ValueError(f"Reference ID is absent from NXML: {ref_id}")

        direct = _ordered_unique_evidence(self.direct.get(ref_id, ()))
        if direct:
            method = (
                "direct_xref_table"
                if any(item.method == "direct_xref_table" for item in direct)
                else "direct_xref_semantic"
            )
            return RecoveryResult(
                status="recovered",
                method=method,
                contexts=tuple(item.text for item in direct),
                evidence=direct,
            )

        categories: Sequence[tuple[str, dict[str, list[ContextEvidence]]]] = (
            ("numeric_range_endpoint_xrefs", self.endpoint_ranges),
            ("numeric_range_single_xref_interior", self.single_range_interiors),
            ("numeric_range_single_xref_endpoint", self.single_range_endpoints),
            ("numeric_list_single_xref", self.single_lists),
        )
        combined: list[ContextEvidence] = []
        primary_method: str | None = None
        for method, mapping in categories:
            values = mapping.get(ref_id, ())
            if values and primary_method is None:
                primary_method = method
            combined.extend(values)
        range_evidence = _ordered_unique_evidence(combined)
        if range_evidence and primary_method is not None:
            return RecoveryResult(
                status="recovered",
                method=primary_method,
                contexts=tuple(item.text for item in range_evidence),
                evidence=range_evidence,
            )

        table = _ordered_unique_evidence(self.structured_tables.get(ref_id, ()))
        if table:
            return RecoveryResult(
                status="recovered",
                method="structured_table_unlinked",
                contexts=tuple(item.text for item in table),
                evidence=table,
            )

        return RecoveryResult(
            status="unrecovered",
            method=None,
            contexts=(),
            evidence=(),
        )


def recover_contexts(
    root: etree._Element,
    ref_ids: Iterable[str],
) -> dict[str, RecoveryResult]:
    index = CitationContextIndex(root)
    return {ref_id: index.recover(ref_id) for ref_id in sorted(set(ref_ids))}


def extract_citation_contexts(root: etree._Element) -> dict[str, list[str]]:
    """Return the best literal JATS contexts for every recoverable reference.

    The historical public artifact field is named ``citation_paragraphs``.
    Values returned here can additionally be header-qualified table rows when
    that is the smallest meaningful JATS container.
    """

    index = CitationContextIndex(root)
    result: dict[str, list[str]] = {}
    for ref_id in index.references:
        recovered = index.recover(ref_id)
        if recovered.status == "recovered":
            result[ref_id] = list(recovered.contexts)
    return result


def _sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sequence_digest(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for index, value in enumerate(values):
        payload = json.dumps(
            [index, value],
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _pmcid_from_path(path: str) -> str:
    matches = list(_PMC_PATH_RE.finditer(str(path).replace("\\", "/")))
    if not matches:
        raise ValueError(f"Cannot derive PMCID from path: {path}")
    return f"PMC{int(matches[-1].group('number'))}"


def _identity_from_source(record: dict[str, Any]) -> tuple[str, str]:
    return _pmcid_from_path(str(record.get("citing_file", ""))), _pmcid_from_path(
        str(record.get("cited_file", ""))
    )


def _identity_from_unified(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("citing_case", "")), str(record.get("cited_case", ""))


def _validate_input_records(
    source_records: Any,
    unified_records: Any,
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[tuple[str, str]]]:
    if not isinstance(source_records, list) or not isinstance(unified_records, list):
        raise ValueError("Both inputs must be top-level JSON lists")
    if len(source_records) != len(unified_records):
        raise ValueError("Source and unified record counts differ")

    source_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    source_order: list[tuple[str, str]] = []
    seen_pair_indices: set[int] = set()
    for source in source_records:
        if not isinstance(source, dict):
            raise ValueError("Citation source contains a non-object record")
        pair_index = source.get("pair_index")
        if isinstance(pair_index, bool) or not isinstance(pair_index, int) or pair_index < 0:
            raise ValueError("Citation source has an invalid pair_index")
        if pair_index in seen_pair_indices:
            raise ValueError(f"Duplicate source pair_index: {pair_index}")
        seen_pair_indices.add(pair_index)
        identity = _identity_from_source(source)
        if identity in source_by_identity:
            raise ValueError(f"Duplicate source PMCID pair: {identity}")
        paragraphs = source.get("citation_paragraphs")
        if not isinstance(paragraphs, list) or any(not isinstance(item, str) for item in paragraphs):
            raise ValueError(f"Malformed citation_paragraphs for pair {pair_index}")
        source_by_identity[identity] = source
        source_order.append(identity)

    unified_order: list[tuple[str, str]] = []
    for index, unified in enumerate(unified_records):
        if not isinstance(unified, dict):
            raise ValueError("Unified citations contain a non-object record")
        expected_keys = {
            "citing_case",
            "cited_case",
            "extracted_citation",
            "citation_reason",
        }
        if set(unified) != expected_keys:
            raise ValueError(f"Unified row {index} does not have the exact four-field schema")
        if any(not isinstance(unified[field], str) for field in expected_keys):
            raise ValueError(f"Unified row {index} has a non-string field")
        identity = _identity_from_unified(unified)
        unified_order.append(identity)

    if len(set(unified_order)) != len(unified_order):
        raise ValueError("Unified citations contain duplicate PMCID pairs")
    if set(unified_order) != set(source_by_identity):
        raise ValueError("Source and unified PMCID-pair populations differ")
    if unified_order != source_order:
        raise ValueError("Source and unified PMCID-pair orders differ")

    for index, identity in enumerate(unified_order):
        source = source_by_identity[identity]
        expected = "\n\n".join(source["citation_paragraphs"])
        actual = unified_records[index]["extracted_citation"]
        if expected and expected != actual:
            raise ValueError(
                f"Unified extracted_citation disagrees with source before recovery at {identity}"
            )
    return source_by_identity, unified_order


def _parse_worker(
    task: tuple[str, tuple[str, ...]],
) -> tuple[str, str, dict[str, RecoveryResult]]:
    file_path, ref_ids = task
    path = Path(file_path)
    source_hash = _sha256_file(path)
    parser = etree.XMLParser(
        recover=False,
        resolve_entities=False,
        load_dtd=False,
        no_network=True,
        huge_tree=False,
    )
    tree = etree.parse(str(path), parser)
    return file_path, source_hash, recover_contexts(tree.getroot(), ref_ids)


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_temp(value: Any, destination: Path, *, mode: int) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_only_extracted_changed(
    before: Sequence[dict[str, str]],
    after: Sequence[dict[str, str]],
) -> dict[str, Any]:
    if len(before) != len(after):
        raise ValueError("Record count changed during recovery")
    changed_indices: list[int] = []
    prior_nonempty_changed: list[int] = []
    for index, (old, new) in enumerate(zip(before, after)):
        for field in ("citing_case", "cited_case", "citation_reason"):
            if old[field] != new[field]:
                raise ValueError(f"Forbidden field changed at row {index}: {field}")
        if old["extracted_citation"] != new["extracted_citation"]:
            changed_indices.append(index)
            if old["extracted_citation"]:
                prior_nonempty_changed.append(index)
    if prior_nonempty_changed:
        raise ValueError("Previously nonempty extracted citations changed")
    return {
        "changed_rows": len(changed_indices),
        "changed_indices_sha256": _sequence_digest(changed_indices),
        "prior_nonempty_changed": 0,
        "citation_reasons_changed": 0,
        "identities_changed": 0,
    }


def recover_case_pair_citations(
    citation_records_path: str | Path,
    unified_path: str | Path,
    *,
    audit_path: str | Path | None = None,
    apply: bool = False,
    workers: int = 1,
) -> dict[str, Any]:
    source_path = Path(citation_records_path).resolve()
    target_path = Path(unified_path).resolve()
    audit_destination = Path(audit_path).resolve() if audit_path is not None else None
    if source_path == target_path:
        raise ValueError("Citation-record source and unified target must be different files")
    if audit_destination is not None and audit_destination in {source_path, target_path}:
        raise ValueError("Audit output must be different from both input files")
    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError("workers must be at least 1")
    if not source_path.is_file() or not target_path.is_file():
        raise FileNotFoundError("Citation-record or unified input is missing")

    source_hash = _sha256_file(source_path)
    target_before_hash = _sha256_file(target_path)
    target_stat = target_path.stat()
    source_records = _load_json(source_path)
    unified_records = _load_json(target_path)
    source_by_identity, identities = _validate_input_records(source_records, unified_records)

    before_reasons = [record["citation_reason"] for record in unified_records]
    before_nonempty = [
        (index, record["citing_case"], record["cited_case"], record["extracted_citation"])
        for index, record in enumerate(unified_records)
        if record["extracted_citation"]
    ]
    empty_identities = [
        identity
        for identity, record in zip(identities, unified_records)
        if not record["extracted_citation"]
    ]

    refs_by_file: dict[str, set[str]] = defaultdict(set)
    for identity in empty_identities:
        source = source_by_identity[identity]
        refs_by_file[str(source["citing_file"])].add(str(source["match_ref_id"]))
    tasks = [
        (file_path, tuple(sorted(ref_ids)))
        for file_path, ref_ids in sorted(refs_by_file.items())
    ]

    recovered_by_file: dict[str, dict[str, RecoveryResult]] = {}
    nxml_hashes: dict[str, str] = {}
    if worker_count == 1:
        parsed = map(_parse_worker, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=worker_count)
        parsed = executor.map(_parse_worker, tasks)
    try:
        for file_path, nxml_hash, recovered in parsed:
            recovered_by_file[file_path] = recovered
            nxml_hashes[file_path] = nxml_hash
    finally:
        if worker_count != 1:
            executor.shutdown(wait=True)

    updated = [dict(record) for record in unified_records]
    method_counts: Counter[str] = Counter()
    recovery_rows: list[dict[str, Any]] = []
    unrecovered_rows: list[dict[str, Any]] = []
    for row_index, identity in enumerate(identities):
        if unified_records[row_index]["extracted_citation"]:
            continue
        source = source_by_identity[identity]
        file_path = str(source["citing_file"])
        ref_id = str(source["match_ref_id"])
        result = recovered_by_file[file_path][ref_id]
        base = {
            "row_index": row_index,
            "pair_index": source["pair_index"],
            "citing_case": identity[0],
            "cited_case": identity[1],
            "source_nxml_path": file_path,
            "source_nxml_sha256": nxml_hashes[file_path],
            "ref_id": ref_id,
            "match_type": source["match_type"],
        }
        if result.status == "recovered":
            if not result.extracted_citation:
                raise ValueError(f"Recovered result is empty for {identity}")
            updated[row_index]["extracted_citation"] = result.extracted_citation
            method = str(result.method)
            method_counts[method] += 1
            recovery_rows.append(
                {
                    **base,
                    "method": method,
                    "context_count": len(result.contexts),
                    "extracted_citation_sha256": _sha256_text(result.extracted_citation),
                    "evidence": [
                        {
                            "method": item.method,
                            "container_type": item.container_type,
                            "xpath": item.xpath,
                            "details": item.details,
                        }
                        for item in result.evidence
                    ],
                }
            )
        else:
            unrecovered_rows.append({**base, "reason": "no_reliable_intext_marker_found"})

    invariant_audit = _verify_only_extracted_changed(unified_records, updated)
    after_reasons = [record["citation_reason"] for record in updated]
    after_nonempty_original = [
        (index, record["citing_case"], record["cited_case"], record["extracted_citation"])
        for index, record in enumerate(updated)
        if unified_records[index]["extracted_citation"]
    ]
    if before_reasons != after_reasons:
        raise ValueError("citation_reason vector changed")
    if before_nonempty != after_nonempty_original:
        raise ValueError("A previously nonempty extracted citation changed")

    before_empty_count = len(empty_identities)
    recovered_count = len(recovery_rows)
    after_empty_count = sum(not record["extracted_citation"] for record in updated)
    if before_empty_count - recovered_count != after_empty_count:
        raise ValueError("Empty/recovered count reconciliation failed")

    output_hash: str | None = None
    output_size: int | None = None
    if apply:
        temporary = _write_json_temp(
            updated,
            target_path,
            mode=stat.S_IMODE(target_stat.st_mode),
        )
        try:
            reloaded = _load_json(temporary)
            _verify_only_extracted_changed(unified_records, reloaded)
            if reloaded != updated:
                raise ValueError("Serialized output does not round-trip exactly")
            if _sha256_file(target_path) != target_before_hash:
                raise RuntimeError("Unified target changed concurrently before replacement")
            output_hash = _sha256_file(temporary)
            output_size = temporary.stat().st_size
            os.replace(temporary, target_path)
            _fsync_directory(target_path.parent)
        finally:
            temporary.unlink(missing_ok=True)
        if _sha256_file(target_path) != output_hash:
            raise RuntimeError("Post-replacement output hash mismatch")
        final_records = _load_json(target_path)
        _verify_only_extracted_changed(unified_records, final_records)
        if final_records != updated:
            raise RuntimeError("Post-replacement JSON differs from validated output")

    audit: dict[str, Any] = {
        "schema_version": 1,
        "recovery_version": RECOVERY_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "applied": bool(apply),
        "inputs": {
            "citation_records_path": str(source_path),
            "citation_records_sha256": source_hash,
            "unified_path": str(target_path),
            "unified_before_sha256": target_before_hash,
            "unified_before_size": target_stat.st_size,
        },
        "output": {
            "unified_after_sha256": output_hash,
            "unified_after_size": output_size,
        },
        "counts": {
            "total_rows": len(updated),
            "before_nonempty": len(updated) - before_empty_count,
            "before_empty": before_empty_count,
            "recovered": recovered_count,
            "after_nonempty": len(updated) - after_empty_count,
            "after_empty": after_empty_count,
            "recovery_methods": dict(sorted(method_counts.items())),
            "source_nxml_files_parsed": len(tasks),
            "parse_errors": 0,
        },
        "invariants": {
            **invariant_audit,
            "citation_reason_sequence_sha256_before": _sequence_digest(before_reasons),
            "citation_reason_sequence_sha256_after": _sequence_digest(after_reasons),
            "identity_sequence_sha256_before": _sequence_digest(identities),
            "identity_sequence_sha256_after": _sequence_digest(
                [_identity_from_unified(record) for record in updated]
            ),
            "prior_nonempty_sequence_sha256_before": _sequence_digest(before_nonempty),
            "prior_nonempty_sequence_sha256_after": _sequence_digest(after_nonempty_original),
        },
        "recovered_rows": recovery_rows,
        "unrecovered_rows": unrecovered_rows,
    }

    if audit_destination is not None:
        audit_destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_audit = _write_json_temp(audit, audit_destination, mode=0o600)
        try:
            if _load_json(temporary_audit) != audit:
                raise RuntimeError("Audit JSON does not round-trip exactly")
            os.replace(temporary_audit, audit_destination)
            _fsync_directory(audit_destination.parent)
        finally:
            temporary_audit.unlink(missing_ok=True)
    return audit
