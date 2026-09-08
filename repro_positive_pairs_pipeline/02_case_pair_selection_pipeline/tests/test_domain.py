from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from case_pair_selection_pipeline.candidates import build_candidates
from case_pair_selection_pipeline.citations import match_pair_citations
from case_pair_selection_pipeline.pipeline import prepare_run
from case_pair_selection_pipeline.summaries import extract_case_summaries


def _article_xml(
    pmc_id: str,
    *,
    title: str,
    doi: str | None = None,
    reference: dict[str, str] | None = None,
) -> str:
    article_doi = (
        f'<article-id pub-id-type="doi">{doi}</article-id>' if doi else ""
    )
    if reference is None:
        citation_markup = ""
        reference_list = ""
    else:
        reference_doi = (
            f'<pub-id pub-id-type="doi">{reference["doi"]}</pub-id>'
            if reference.get("doi")
            else ""
        )
        citation_markup = (
            '<p>Cites the paired report <xref ref-type="bibr" rid="R1">1</xref>.</p>'
        )
        reference_list = (
            '<ref-list><ref id="R1"><element-citation>'
            f'<article-title>{reference["title"]}</article-title>'
            f"{reference_doi}"
            "</element-citation></ref></ref-list>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<article><front><article-meta>"
        f'<article-id pub-id-type="pmc">PMC{pmc_id}</article-id>'
        f"{article_doi}"
        f"<title-group><article-title>{title}</article-title></title-group>"
        f"<abstract><p>Abstract for {title}.</p></abstract>"
        "</article-meta></front><body>"
        f"<sec><title>Case presentation</title><p>Case text for {pmc_id}.</p></sec>"
        f"{citation_markup}</body><back>{reference_list}</back></article>"
    )


class DomainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.retrieval = self.root / "retrieval"
        self.retrieval.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _case_row(
        self,
        pmc_id: str,
        *,
        similar: list[str],
        nxml: str | None = None,
    ) -> dict[str, str]:
        uid = f"{pmc_id}-1"
        folder = self.retrieval / "new_data" / f"PMC{pmc_id}" / f"{pmc_id}_1"
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{pmc_id}_1_1"
        image = folder / f"{stem}.jpg"
        caption = folder / f"{stem}.txt"
        article = folder / "article.nxml"
        image.write_bytes(f"image-{pmc_id}".encode("ascii"))
        caption.write_text(f"Caption for {pmc_id}.", encoding="utf-8")
        article.write_text(
            nxml if nxml is not None else _article_xml(pmc_id, title=f"Case {pmc_id}"),
            encoding="utf-8",
        )
        return {
            "image_path": image.relative_to(self.retrieval).as_posix(),
            "caption_path": caption.relative_to(self.retrieval).as_posix(),
            "patient_uid": uid,
            "pmc_id": pmc_id,
            "article_path": article.relative_to(self.retrieval).as_posix(),
            "unique_articles_sim_patients": json.dumps(similar),
        }

    def _write_merged(self, rows: list[dict[str, str]]) -> None:
        with (self.retrieval / "04_merged.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _write_blacklist(self, value: object, name: str = "blacklist.json") -> Path:
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    @staticmethod
    def _tree_snapshot(root: Path) -> list[tuple[str, str, str | None]]:
        snapshot: list[tuple[str, str, str | None]] = []
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
            relative = path.relative_to(root).as_posix()
            if path.is_dir():
                snapshot.append((relative, "directory", None))
            elif path.is_file():
                snapshot.append(
                    (relative, "file", hashlib.sha256(path.read_bytes()).hexdigest())
                )
            else:
                snapshot.append((relative, "other", None))
        return snapshot

    def test_prepare_rejects_output_inside_upstream_without_mutation(self) -> None:
        sentinel = self.retrieval / "upstream-sentinel.txt"
        sentinel.write_text("immutable upstream", encoding="utf-8")
        config = self.root / "unused-config.toml"
        config.write_text("intentionally not parsed", encoding="utf-8")
        before = self._tree_snapshot(self.retrieval)

        nested_output = self.retrieval / "new-results" / "deep" / "run"
        destinations = (self.retrieval, nested_output)
        for destination in destinations:
            with self.subTest(destination=destination):
                with self.assertRaisesRegex(
                    ValueError, "must not equal or be nested inside"
                ):
                    prepare_run(
                        self.retrieval,
                        destination,
                        config,
                        workers=1,
                    )
                self.assertEqual(self._tree_snapshot(self.retrieval), before)

        self.assertFalse(nested_output.exists())
        self.assertFalse((self.retrieval / "new-results").exists())

    def test_candidate_paths_reject_traversal_and_out_of_tree_absolute_paths(self) -> None:
        row = self._case_row("2", similar=[])
        outside = self.root / "outside.jpg"
        outside.write_bytes(b"outside")
        blacklist = self._write_blacklist([])

        for unsafe_path in (
            "../outside.jpg",
            str(outside.resolve()),
            "outside_run/PMC2/2_1/2_1_1.jpg",
        ):
            with self.subTest(unsafe_path=unsafe_path):
                unsafe_row = dict(row)
                unsafe_row["image_path"] = unsafe_path
                self._write_merged([unsafe_row])
                with self.assertRaisesRegex(
                    ValueError, "parent traversal|inside.*new_data"
                ):
                    build_candidates(self.retrieval, blacklist)

    def test_diagram_blacklist_rejects_invalid_shape_and_unknown_image(self) -> None:
        row = self._case_row("2", similar=[])
        self._write_merged([row])

        malformed = self._write_blacklist({"not": "a list"}, "malformed.json")
        with self.assertRaisesRegex(ValueError, "JSON list"):
            build_candidates(self.retrieval, malformed)

        extra = self.retrieval / "new_data" / "PMC2" / "2_1" / "extra.jpg"
        extra.write_bytes(b"not-in-manifest")
        unknown = self._write_blacklist(
            [extra.relative_to(self.retrieval).as_posix()], "unknown.json"
        )
        with self.assertRaisesRegex(ValueError, "absent from 04_merged.csv"):
            build_candidates(self.retrieval, unknown)

    def test_candidates_use_numeric_order_and_content_derived_pair_key(self) -> None:
        rows = [
            self._case_row("10", similar=["2-1"]),
            self._case_row("2", similar=[]),
        ]
        for row in rows:
            row["article_path"] = (
                f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{row['pmc_id']}/"
            )
        self._write_merged(rows)
        cases, pairs, stats = build_candidates(self.retrieval, self._write_blacklist([]))

        self.assertEqual([case["patient_uid"] for case in cases], ["2-1", "10-1"])
        self.assertEqual(len(pairs), 1)
        expected_key = hashlib.sha256(b"2-1\0" b"10-1").hexdigest()
        self.assertEqual(
            pairs[0],
            {
                "schema_version": 1,
                "pair_index": 0,
                "pair_key": expected_key,
                "case_a_uid": "2-1",
                "case_b_uid": "10-1",
            },
        )
        self.assertEqual(stats["pairs_output"], 1)
        for case in cases:
            self.assertFalse(Path(case["nxml_path"]).is_absolute())
            for caption in case["captions"]:
                self.assertFalse(Path(caption["image_path"]).is_absolute())
                self.assertFalse(Path(caption["caption_path"]).is_absolute())

    def test_malformed_xml_is_an_explicit_summary_error(self) -> None:
        row = self._case_row("2", similar=[], nxml="<article><body><sec>")
        self._write_merged([row])
        cases, _, _ = build_candidates(self.retrieval, self._write_blacklist([]))

        summaries = extract_case_summaries(cases, self.retrieval, workers=1)

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["patient_uid"], "2-1")
        self.assertEqual(summaries[0]["status"], "error")
        self.assertEqual(summaries[0]["summary"], "")
        self.assertEqual(summaries[0]["error"], "XMLSyntaxError: malformed NXML")

    def test_exact_doi_match_reports_reverse_direction_and_paragraph(self) -> None:
        article_a = _article_xml(
            "2",
            title="Alpha paired report",
            doi="10.1000/alpha",
        )
        article_b = _article_xml(
            "10",
            title="Beta citing report",
            doi="10.1000/beta",
            reference={
                "title": "A deliberately nonmatching reference title",
                "doi": "https://doi.org/10.1000/ALPHA",
            },
        )
        rows = [
            self._case_row("10", similar=["2-1"], nxml=article_b),
            self._case_row("2", similar=[], nxml=article_a),
        ]
        self._write_merged(rows)
        cases, pairs, _ = build_candidates(self.retrieval, self._write_blacklist([]))

        records = match_pair_citations(
            pairs,
            {case["patient_uid"]: case for case in cases},
            self.retrieval,
            workers=1,
        )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["status"], "matched")
        self.assertEqual(record["direction"], "case_b_cites_case_a")
        self.assertEqual(record["match_type"], "doi")
        self.assertEqual(record["ref_id"], "R1")
        self.assertEqual(record["citing_uid"], "10-1")
        self.assertEqual(record["cited_uid"], "2-1")
        self.assertEqual(
            record["citation_paragraphs"], ["Cites the paired report 1 ."]
        )
        self.assertNotIn("citation_paragraph", record)

    def test_default_title_guard_rejects_short_title_fallback(self) -> None:
        article_a = _article_xml("2", title="Tiny case")
        article_b = _article_xml(
            "10",
            title="Unrelated citing article",
            reference={"title": "Tiny case", "doi": ""},
        )
        rows = [
            self._case_row("2", similar=[], nxml=article_a),
            self._case_row("10", similar=["2-1"], nxml=article_b),
        ]
        self._write_merged(rows)
        cases, pairs, _ = build_candidates(self.retrieval, self._write_blacklist([]))

        records = match_pair_citations(
            pairs,
            {case["patient_uid"]: case for case in cases},
            self.retrieval,
            title_fallback=True,
            workers=1,
        )

        self.assertEqual(records[0]["status"], "no_match")

    def test_conflicting_identifier_does_not_confirm_a_citation(self) -> None:
        citing = _article_xml(
            "2", title="Citing case", doi="10.1000/a",
            reference={"title": "Cited case", "doi": "10.1000/b"},
        ).replace(
            "</element-citation>",
            '<pub-id pub-id-type="pmc">PMC999</pub-id></element-citation>',
        )
        rows = [
            self._case_row("2", similar=["10-1"], nxml=citing),
            self._case_row(
                "10", similar=[],
                nxml=_article_xml("10", title="Cited case", doi="10.1000/b"),
            ),
        ]
        self._write_merged(rows)
        cases, pairs, _ = build_candidates(self.retrieval, self._write_blacklist([]))
        records = match_pair_citations(
            pairs, {case["patient_uid"]: case for case in cases}, self.retrieval,
        )
        self.assertEqual(records[0]["status"], "no_match")

    def test_ambiguous_forward_citation_allows_verified_reverse_direction(self) -> None:
        article_a = _article_xml(
            "2", title="Case A", doi="10.1000/a",
            reference={"title": "Case B", "doi": "10.1000/b"},
        ).replace(
            "</ref-list>",
            '<ref id="R2"><element-citation>'
            '<pub-id pub-id-type="doi">10.1000/b</pub-id>'
            '</element-citation></ref></ref-list>',
        )
        article_b = _article_xml(
            "10", title="Case B", doi="10.1000/b",
            reference={"title": "Case A", "doi": "10.1000/a"},
        )
        self._write_merged([
            self._case_row("2", similar=["10-1"], nxml=article_a),
            self._case_row("10", similar=[], nxml=article_b),
        ])
        cases, pairs, _ = build_candidates(self.retrieval, self._write_blacklist([]))
        records = match_pair_citations(
            pairs, {case["patient_uid"]: case for case in cases}, self.retrieval,
        )
        self.assertEqual(records[0]["status"], "matched")
        self.assertEqual(records[0]["direction"], "case_b_cites_case_a")


if __name__ == "__main__":
    unittest.main()
