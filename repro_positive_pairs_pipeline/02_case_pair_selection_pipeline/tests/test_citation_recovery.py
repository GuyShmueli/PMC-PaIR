from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lxml import etree

from case_pair_selection_pipeline.citation_recovery import (
    recover_case_pair_citations,
    recover_contexts,
)
from case_pair_selection_pipeline.citations import _citation_paragraphs


def _root(xml: str) -> etree._Element:
    return etree.fromstring(xml.encode("utf-8"))


class CitationContextRecoveryTests(unittest.TestCase):
    def test_exact_table_xref_uses_caption_headers_and_row(self) -> None:
        root = _root(
            """
            <article><body><table-wrap><caption><p>Prior cases</p></caption>
              <table><thead><tr><th>Author</th><th>Reference</th><th>Finding</th></tr></thead>
              <tbody><tr><td>Smith</td><td><xref ref-type="bibr" rid="R8">8</xref></td>
              <td>Bowel obstruction</td></tr></tbody></table>
            </table-wrap></body><back><ref-list>
              <ref id="R8"><label>8</label><element-citation><surname>Smith</surname><year>2010</year></element-citation></ref>
            </ref-list></back></article>
            """
        )

        result = recover_contexts(root, {"R8"})["R8"]

        self.assertEqual(result.status, "recovered")
        self.assertEqual(result.method, "direct_xref_table")
        self.assertEqual(
            result.contexts,
            (
                "Table: Prior cases\n"
                "Headers: Author | Reference | Finding\n"
                "Row: Smith | 8 | Bowel obstruction",
            ),
        )

    def test_two_endpoint_xrefs_recover_unique_interior_label(self) -> None:
        root = _root(
            """
            <article><body><p>Prior reports [<xref ref-type="bibr" rid="R3">3</xref>
            – <bold><xref ref-type="bibr" rid="R5">5</xref></bold>] agree.</p></body>
            <back><ref-list>
              <ref id="R3"><label>3</label></ref><ref id="R4"><label>4</label></ref>
              <ref id="R5"><label>5</label></ref>
            </ref-list></back></article>
            """
        )

        result = recover_contexts(root, {"R4"})["R4"]

        self.assertEqual(result.method, "numeric_range_endpoint_xrefs")
        self.assertIn("Prior reports", result.extracted_citation)
        self.assertEqual(_citation_paragraphs(root)["R4"], list(result.contexts))

    def test_single_xref_range_recovers_interior_and_omitted_endpoint(self) -> None:
        root = _root(
            """
            <article><body><p>Prior reports <xref ref-type="bibr" rid="R7">[7–9]</xref>.</p></body>
            <back><ref-list>
              <ref id="R7"><label>7</label></ref><ref id="R8"><label>8</label></ref>
              <ref id="R9"><label>9</label></ref>
            </ref-list></back></article>
            """
        )

        results = recover_contexts(root, {"R8", "R9"})

        self.assertEqual(results["R8"].method, "numeric_range_single_xref_interior")
        self.assertEqual(results["R9"].method, "numeric_range_single_xref_endpoint")

    def test_single_xref_list_recovers_visibly_printed_omitted_label(self) -> None:
        root = _root(
            """
            <article><body><p>Prior reports <xref ref-type="bibr" rid="R5">[5, 6, 9]</xref>.</p></body>
            <back><ref-list>
              <ref id="R5"><label>5</label></ref><ref id="R6"><label>6</label></ref>
              <ref id="R9"><label>9</label></ref>
            </ref-list></back></article>
            """
        )

        result = recover_contexts(root, {"R9"})["R9"]

        self.assertEqual(result.method, "numeric_list_single_xref")

    def test_single_xref_expands_each_visible_range_segment(self) -> None:
        root = _root(
            """
            <article><body><p>Reports <xref ref-type="bibr" rid="R7">[7, 11–14]</xref>.</p></body>
            <back><ref-list>
              <ref id="R7"><label>7</label></ref><ref id="R11"><label>11</label></ref>
              <ref id="R12"><label>12</label></ref><ref id="R13"><label>13</label></ref>
              <ref id="R14"><label>14</label></ref>
            </ref-list></back></article>
            """
        )

        result = recover_contexts(root, {"R12"})["R12"]
        self.assertEqual(result.method, "numeric_range_single_xref_interior")
        self.assertIn("11–14", result.extracted_citation)

    def test_nonunique_numeric_label_is_not_range_recovered(self) -> None:
        root = _root(
            """
            <article><body><p>Reports <xref ref-type="bibr" rid="R1">1–3</xref>.</p></body>
            <back><ref-list>
              <ref id="R1"><label>1</label></ref><ref id="R2a"><label>2</label></ref>
              <ref id="R2b"><label>2</label></ref><ref id="R3"><label>3</label></ref>
            </ref-list></back></article>
            """
        )

        result = recover_contexts(root, {"R2a"})["R2a"]

        self.assertEqual(result.status, "unrecovered")
        self.assertEqual(result.contexts, ())

    def test_ref_list_xref_is_not_treated_as_intext_evidence(self) -> None:
        root = _root(
            """
            <article><body><p>No citation marker here.</p></body><back><ref-list>
              <ref id="R2"><label>2</label><mixed-citation>
                See <xref ref-type="bibr" rid="R2">2</xref>.</mixed-citation></ref>
            </ref-list></back></article>
            """
        )

        result = recover_contexts(root, {"R2"})["R2"]

        self.assertEqual(result.status, "unrecovered")

    def test_structured_unlinked_table_requires_header_author_and_year(self) -> None:
        root = _root(
            """
            <article><body><table-wrap><caption><p>Published cases</p></caption><table>
              <thead><tr><th colspan="3">References</th><th rowspan="2">Outcome</th></tr>
              <tr><th>Year</th><th>Author</th><th>Number</th></tr></thead>
              <tbody><tr><td>2018</td><td>Chern</td><td>16</td><td>Recovered</td></tr></tbody>
            </table></table-wrap></body><back><ref-list>
              <ref id="CR16"><label>16</label><element-citation>
                <person-group><name><surname>Chern</surname></name></person-group>
                <year>2018</year><article-title>A prior case</article-title>
              </element-citation></ref>
            </ref-list></back></article>
            """
        )

        result = recover_contexts(root, {"CR16"})["CR16"]

        self.assertEqual(result.method, "structured_table_unlinked")
        self.assertIn("Row: 2018 | Chern | 16 | Recovered", result.extracted_citation)


class UnifiedMutationTests(unittest.TestCase):
    def test_apply_changes_only_empty_extracted_citation_and_writes_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            citing_dir = root / "new_data" / "PMC100" / "100_1"
            cited_dir = root / "new_data" / "PMC200" / "200_1"
            citing_dir.mkdir(parents=True)
            cited_dir.mkdir(parents=True)
            nxml = citing_dir / "article.nxml"
            nxml.write_text(
                """
                <article><body><table-wrap><table><tr><th>Reference</th><th>Finding</th></tr>
                <tr><td><xref ref-type="bibr" rid="R2">2</xref></td><td>Relevant finding</td></tr>
                </table></table-wrap></body><back><ref-list><ref id="R2"><label>2</label></ref>
                </ref-list></back></article>
                """,
                encoding="utf-8",
            )
            cited = cited_dir / "cited.nxml"
            cited.write_text("<article/>", encoding="utf-8")
            source_path = root / "03_citation_records.json"
            unified_path = root / "case_pair_citations.json"
            audit_path = root / "audit.json"
            source_path.write_text(
                json.dumps(
                    [
                        {
                            "pair_index": 17,
                            "citing_file": str(nxml),
                            "cited_file": str(cited),
                            "match_type": "pmid",
                            "match_ref_id": "R2",
                            "citing_title": "",
                            "citing_abstract": "",
                            "citation_paragraphs": [],
                            "cited_title": "",
                            "cited_abstract": "",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            reason = "Keep this reason byte-for-byte.\n\nSecond paragraph."
            unified_path.write_text(
                json.dumps(
                    [
                        {
                            "citing_case": "PMC100",
                            "cited_case": "PMC200",
                            "extracted_citation": "",
                            "citation_reason": reason,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            audit = recover_case_pair_citations(
                source_path,
                unified_path,
                audit_path=audit_path,
                apply=True,
                workers=1,
            )

            output = json.loads(unified_path.read_text(encoding="utf-8"))
            self.assertEqual(output[0]["citation_reason"], reason)
            self.assertIn("Relevant finding", output[0]["extracted_citation"])
            self.assertEqual(audit["counts"]["recovered"], 1)
            self.assertEqual(audit["invariants"]["citation_reasons_changed"], 0)
            self.assertEqual(json.loads(audit_path.read_text(encoding="utf-8")), audit)


if __name__ == "__main__":
    unittest.main()
