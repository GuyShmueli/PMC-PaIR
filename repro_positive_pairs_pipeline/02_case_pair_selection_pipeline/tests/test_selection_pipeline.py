from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from case_pair_selection_pipeline.artifacts import (
    DIAGRAM_BLACKLIST,
    DIAGRAM_CLASSIFICATIONS,
    DIAGRAM_STATS,
    ELIGIBLE_CASES,
    ELIGIBLE_PAIRS,
    HANDOFF_MANIFEST,
    PAIR_SPECIES,
    SPECIES_CLASSIFICATIONS,
    SPECIES_STATS,
    SUMMARIES,
)
from case_pair_selection_pipeline.errors import (
    ArtifactIntegrityError,
    DiagramFilteringError,
)
from case_pair_selection_pipeline.io_utils import (
    file_metadata,
    load_json,
    load_jsonl,
    save_json,
    save_jsonl,
    sha256_file,
    sha256_tree,
)
from case_pair_selection_pipeline.pipeline import prepare_run, verify_run


def _xml(
    pmc_id: str,
    *,
    doi: str,
    reference_doi: str | None = None,
    title: str | None = None,
    abstract: str | None = None,
    journal_title: str = "Medical Cases",
) -> str:
    if reference_doi is None:
        paragraph = ""
        references = ""
    else:
        paragraph = (
            '<p>Related report <xref ref-type="bibr" rid="R1">1</xref>.</p>'
        )
        references = (
            '<ref-list><ref id="R1"><element-citation>'
            "<article-title>Referenced case report</article-title>"
            f'<pub-id pub-id-type="doi">{reference_doi}</pub-id>'
            "</element-citation></ref></ref-list>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<article><front><journal-meta><journal-title-group>"
        f"<journal-title>{journal_title}</journal-title>"
        "</journal-title-group></journal-meta><article-meta>"
        f'<article-id pub-id-type="pmc">PMC{pmc_id}</article-id>'
        f'<article-id pub-id-type="doi">{doi}</article-id>'
        f"<title-group><article-title>{title or f'Case {pmc_id}'}</article-title></title-group>"
        f"<abstract><p>{abstract or f'Abstract {pmc_id}.'}</p></abstract>"
        "</article-meta></front><body>"
        f"<sec><title>Case presentation</title><p>Description {pmc_id}.</p></sec>"
        f"{paragraph}</body><back>{references}</back></article>"
    )


class SelectionPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.retrieval = self.root / "retrieval"
        self.retrieval.mkdir()
        self.config = self.root / "selection.toml"
        self.config.write_text(
            "schema_version = 1\n"
            "[selection]\n"
            "title_fallback = false\n"
            "title_min_chars = 24\n"
            "title_min_tokens = 4\n"
            "[species_filter]\n"
            'ruleset = "human_animal_v1"\n',
            encoding="utf-8",
        )
        runtime = {
            "executable": "/test/tesseract",
            "executable_sha256": "a" * 64,
            "version": "tesseract test",
            "version_output": "tesseract test",
            "language": "eng",
            "traineddata_path": "/test/eng.traineddata",
            "traineddata_sha256": "b" * 64,
        }
        self.runtime_patcher = patch(
            "case_pair_selection_pipeline.pipeline.tesseract_runtime_metadata",
            return_value=runtime,
        )
        self.ocr_patcher = patch(
            "case_pair_selection_pipeline.diagrams._ocr_text",
            return_value="",
        )
        self.runtime_patcher.start()
        self.ocr_patcher.start()

    def tearDown(self) -> None:
        self.ocr_patcher.stop()
        self.runtime_patcher.stop()
        self.temporary.cleanup()

    def _row(
        self,
        pmc_id: str,
        *,
        similar: list[str],
        doi: str,
        reference_doi: str | None = None,
        title: str | None = None,
        abstract: str | None = None,
        journal_title: str = "Medical Cases",
    ) -> dict[str, str]:
        uid = f"{pmc_id}-1"
        folder = (
            self.retrieval
            / "new_data"
            / f"PMC{pmc_id}"
            / f"{pmc_id}_1"
        )
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{pmc_id}_1_1"
        image = folder / f"{stem}.jpg"
        caption = folder / f"{stem}.txt"
        article = folder / "article.nxml"
        Image.new("L", (8, 8), 0).save(image, format="JPEG")
        caption.write_text(f"Caption {pmc_id}.", encoding="utf-8")
        article.write_text(
            _xml(
                pmc_id,
                doi=doi,
                reference_doi=reference_doi,
                title=title,
                abstract=abstract,
                journal_title=journal_title,
            ),
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

    def _complete_upstream(self, rows: list[dict[str, str]]) -> None:
        merged = self.retrieval / "04_merged.csv"
        with merged.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        unified = self.retrieval / "00_unified_patients.csv"
        unified.write_text(
            "patient_uid\n"
            + "".join(f"{row['patient_uid']}\n" for row in rows),
            encoding="utf-8",
        )
        manifest = {
            "pipeline_version": "3.1.0",
            "status": "complete",
            "dataset_contract": {
                "schema_version": 1,
                "mode": "unified",
                "materialized_input_count": 1,
                "materialized_input": "00_unified_patients.csv",
                "merged_csv": "04_merged.csv",
                "assets_dir": "new_data",
            },
            "artifact_sha256": {
                "00_unified_patients.csv": sha256_file(unified),
                "04_merged.csv": sha256_file(merged),
                "new_data/": sha256_tree(self.retrieval / "new_data"),
            },
        }
        (self.retrieval / "run_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

    def test_prepare_emits_normalized_hash_bound_handoff_and_preserves_gap(self) -> None:
        rows = [
            self._row(
                "2",
                similar=["10-1", "20-1"],
                doi="10.1000/case2",
                reference_doi="10.1000/case20",
            ),
            self._row("10", similar=[], doi="10.1000/case10"),
            self._row("20", similar=[], doi="10.1000/case20"),
        ]
        self._complete_upstream(rows)
        output = self.root / "selection-run"

        manifest = prepare_run(
            self.retrieval,
            output,
            self.config,
            workers=1,
        )

        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(
            manifest["pipeline_id"],
            "case_pair_selection_pipeline",
        )
        self.assertEqual(manifest["pipeline_version"], "6.1.0")
        cases = load_jsonl(output / ELIGIBLE_CASES)
        pairs = load_jsonl(output / ELIGIBLE_PAIRS)
        self.assertEqual(
            [case["patient_uid"] for case in cases],
            ["2-1", "20-1"],
        )
        self.assertEqual([pair["pair_index"] for pair in pairs], [1])
        self.assertEqual(pairs[0]["citation"]["status"], "matched")
        self.assertEqual(
            pairs[0]["citation"]["direction"],
            "case_a_cites_case_b",
        )
        expected_key = hashlib.sha256(b"2-1\0" b"20-1").hexdigest()
        self.assertEqual(pairs[0]["pair_key"], expected_key)
        self.assertIn("case_description", cases[0])
        self.assertEqual(cases[0]["schema_version"], 2)
        self.assertEqual(
            cases[0]["population_scope"],
            {"ruleset": "human_animal_v1", "label": "human"},
        )

        handoff = load_json(output / HANDOFF_MANIFEST)
        self.assertEqual(handoff, manifest["handoff_contract"])
        self.assertEqual(handoff["schema_version"], 4)
        self.assertEqual(
            handoff["contracts"]["population_filter"],
            "complete_case_pair_human_animal_screen_required",
        )
        self.assertEqual(
            handoff["producer"]["pipeline_id"],
            "case_pair_selection_pipeline",
        )
        self.assertEqual(
            handoff["artifacts"]["eligible_pairs"]["records"],
            1,
        )
        self.assertEqual(
            set(handoff["artifacts"]),
            {
                "eligible_cases",
                "eligible_pairs",
                "species_classifications",
                "pair_species",
                "species_stats",
            },
        )
        self.assertEqual(
            handoff["artifacts"]["eligible_cases"]["sha256"],
            sha256_file(output / ELIGIBLE_CASES),
        )
        result = verify_run(output)
        self.assertEqual(result["eligible_cases"], 2)
        self.assertEqual(result["eligible_pairs"], 1)

    def test_prepare_computes_diagram_filter_when_blacklist_is_omitted(self) -> None:
        rows = [
            self._row(
                "2",
                similar=["20-1"],
                doi="10.1000/case2",
                reference_doi="10.1000/case20",
            ),
            self._row("20", similar=[], doi="10.1000/case20"),
        ]
        for row in rows:
            Image.new("L", (8, 8), 0).save(
                self.retrieval / row["image_path"],
                format="JPEG",
            )
        self._complete_upstream(rows)
        output = self.root / "computed-diagram-run"

        with patch(
            "case_pair_selection_pipeline.pipeline.tesseract_runtime_metadata",
            return_value={
                "executable": "/test/tesseract",
                "executable_sha256": "a" * 64,
                "version": "tesseract test",
                "version_output": "tesseract test",
                "language": "eng",
                "traineddata_path": "/test/eng.traineddata",
                "traineddata_sha256": "b" * 64,
            },
        ), patch(
            "case_pair_selection_pipeline.diagrams._ocr_text",
            return_value="",
        ):
            manifest = prepare_run(
                self.retrieval,
                output,
                self.config,
                workers=1,
            )

        self.assertEqual(
            manifest["stages"]["00_diagram_filter"]["details"]["mode"],
            "computed",
        )
        self.assertEqual(load_json(output / DIAGRAM_BLACKLIST), [])
        self.assertEqual(len(load_jsonl(output / DIAGRAM_CLASSIFICATIONS)), 2)
        self.assertEqual(load_json(output / DIAGRAM_STATS)["images_input"], 2)
        self.assertEqual(verify_run(output)["diagrams_excluded"], 0)

    def test_animal_case_removes_every_incident_pair_before_handoff(self) -> None:
        rows = [
            self._row(
                "2",
                similar=["10-1", "20-1"],
                doi="10.1000/case2",
                title="Magnetic resonance findings in a dog",
            ),
            self._row(
                "10",
                similar=["20-1"],
                doi="10.1000/case10",
                reference_doi="10.1000/case20",
            ),
            self._row("20", similar=[], doi="10.1000/case20"),
        ]
        self._complete_upstream(rows)
        output = self.root / "human-animal-screen"

        prepare_run(
            self.retrieval,
            output,
            self.config,
            workers=1,
        )

        classifications = load_jsonl(output / SPECIES_CLASSIFICATIONS)
        by_uid = {record["patient_uid"]: record for record in classifications}
        self.assertEqual(by_uid["2-1"]["final_label"], "animal")
        self.assertEqual(
            [record["patient_uid"] for record in load_jsonl(output / SUMMARIES)],
            ["10-1", "20-1"],
        )
        pair_species = load_jsonl(output / PAIR_SPECIES)
        self.assertEqual([record["pair_index"] for record in pair_species], [0, 1, 2])
        self.assertTrue(
            all(
                record["label"] == "animal"
                for record in pair_species[:2]
            )
        )
        self.assertEqual(pair_species[2]["label"], "human")
        self.assertEqual(
            pair_species[2]["endpoint_labels"],
            {"case_a": "human", "case_b": "human"},
        )
        eligible_pairs = load_jsonl(output / ELIGIBLE_PAIRS)
        self.assertEqual([record["pair_index"] for record in eligible_pairs], [2])
        self.assertEqual(
            {case["patient_uid"] for case in load_jsonl(output / ELIGIBLE_CASES)},
            {"10-1", "20-1"},
        )
        stats = load_json(output / SPECIES_STATS)
        self.assertEqual(stats["pairs_before_screen"], 3)
        self.assertEqual(stats["human_pairs"], 1)
        self.assertEqual(stats["animal_pairs"], 2)
        result = verify_run(output)
        self.assertEqual(result["human_pairs"], 1)

    def test_species_adjudication_can_override_an_automatic_label(self) -> None:
        rows = [
            self._row(
                "2",
                similar=["20-1"],
                doi="10.1000/case2",
                reference_doi="10.1000/case20",
                title="A dog presented with an abdominal mass",
            ),
            self._row("20", similar=[], doi="10.1000/case20"),
        ]
        self._complete_upstream(rows)
        automatic = self.root / "species-automatic"
        prepare_run(
            self.retrieval,
            automatic,
            self.config,
            workers=1,
        )
        self.assertEqual(load_jsonl(automatic / ELIGIBLE_PAIRS), [])

        adjudications = self.root / "species-adjudications.jsonl"
        adjudications.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "patient_uid": "2-1",
                    "label": "human",
                    "reason": "Qualified review confirmed a human case.",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        accepted = self.root / "species-adjudicated"
        manifest = prepare_run(
            self.retrieval,
            accepted,
            self.config,
            species_adjudications=adjudications,
            workers=1,
        )
        self.assertEqual(len(load_jsonl(accepted / ELIGIBLE_PAIRS)), 1)
        classification = {
            record["patient_uid"]: record
            for record in load_jsonl(accepted / SPECIES_CLASSIFICATIONS)
        }["2-1"]
        self.assertEqual(classification["automatic_label"], "animal")
        self.assertEqual(classification["final_label"], "human")
        self.assertEqual(
            classification["adjudication"],
            {"label": "human", "reason": "Qualified review confirmed a human case."},
        )
        self.assertIn(
            "species_adjudications",
            {item["label"] for item in manifest["inputs"]},
        )
        self.assertEqual(verify_run(accepted)["eligible_pairs"], 1)

    def test_species_parse_failure_is_not_published(self) -> None:
        rows = [
            self._row(
                "2",
                similar=["20-1"],
                doi="10.1000/case2",
                reference_doi="10.1000/case20",
            ),
            self._row("20", similar=[], doi="10.1000/case20"),
        ]
        (self.retrieval / rows[0]["article_path"]).write_text(
            "<article><front>", encoding="utf-8"
        )
        self._complete_upstream(rows)
        output = self.root / "species-parse-fails"

        with self.assertRaisesRegex(
            ValueError,
            "could not prepare every case",
        ):
            prepare_run(
                self.retrieval,
                output,
                self.config,
                workers=1,
            )

        self.assertFalse(output.exists())

    def test_computed_diagram_exclusion_removes_case_and_candidate_relation(self) -> None:
        rows = [
            self._row(
                "2",
                similar=["20-1"],
                doi="10.1000/case2",
                reference_doi="10.1000/case20",
            ),
            self._row("20", similar=[], doi="10.1000/case20"),
        ]
        Image.new("L", (8, 8), 0).save(
            self.retrieval / rows[0]["image_path"],
            format="JPEG",
        )
        Image.new("L", (8, 8), 255).save(
            self.retrieval / rows[1]["image_path"],
            format="JPEG",
        )
        self._complete_upstream(rows)
        output = self.root / "computed-exclusion-run"

        with patch(
            "case_pair_selection_pipeline.pipeline.tesseract_runtime_metadata",
            return_value={
                "executable": "/test/tesseract",
                "executable_sha256": "a" * 64,
                "version": "tesseract test",
                "version_output": "tesseract test",
                "language": "eng",
                "traineddata_path": "/test/eng.traineddata",
                "traineddata_sha256": "b" * 64,
            },
        ), patch(
            "case_pair_selection_pipeline.diagrams._ocr_text",
            return_value="",
        ) as ocr:
            prepare_run(
                self.retrieval,
                output,
                self.config,
                workers=1,
            )

        self.assertEqual(ocr.call_count, 1)
        self.assertEqual(
            load_json(output / DIAGRAM_BLACKLIST),
            [rows[1]["image_path"]],
        )
        self.assertEqual(load_jsonl(output / ELIGIBLE_CASES), [])
        self.assertEqual(load_jsonl(output / ELIGIBLE_PAIRS), [])
        result = verify_run(output)
        self.assertEqual(result["diagrams_excluded"], 1)
        self.assertEqual(result["eligible_pairs"], 0)

    def test_computed_filter_failure_is_not_published(self) -> None:
        rows = [
            self._row(
                "2",
                similar=["20-1"],
                doi="10.1000/case2",
                reference_doi="10.1000/case20",
            ),
            self._row("20", similar=[], doi="10.1000/case20"),
        ]
        for row in rows:
            Image.new("L", (8, 8), 0).save(
                self.retrieval / row["image_path"],
                format="JPEG",
            )
        self._complete_upstream(rows)
        output = self.root / "failed-computed-run"

        with patch(
            "case_pair_selection_pipeline.pipeline.tesseract_runtime_metadata",
            return_value={
                "executable": "/test/tesseract",
                "executable_sha256": "a" * 64,
                "version": "tesseract test",
                "version_output": "tesseract test",
                "language": "eng",
                "traineddata_path": "/test/eng.traineddata",
                "traineddata_sha256": "b" * 64,
            },
        ), patch(
            "case_pair_selection_pipeline.diagrams._ocr_text",
            side_effect=RuntimeError("timeout"),
        ):
            with self.assertRaisesRegex(DiagramFilteringError, "during ocr"):
                prepare_run(
                    self.retrieval,
                    output,
                    self.config,
                    workers=1,
                )

        self.assertFalse(output.exists())

    def test_empty_selection_is_a_valid_complete_handoff(self) -> None:
        rows = [
            self._row(
                "2",
                similar=["10-1"],
                doi="10.1000/case2",
            ),
            self._row("10", similar=[], doi="10.1000/case10"),
        ]
        self._complete_upstream(rows)
        output = self.root / "empty-selection"

        prepare_run(
            self.retrieval,
            output,
            self.config,
            workers=1,
        )

        self.assertEqual(load_jsonl(output / ELIGIBLE_CASES), [])
        self.assertEqual(load_jsonl(output / ELIGIBLE_PAIRS), [])
        result = verify_run(output)
        self.assertEqual(result["eligible_cases"], 0)
        self.assertEqual(result["eligible_pairs"], 0)

    def test_non_unified_upstream_contract_is_rejected_without_output(self) -> None:
        rows = [
            self._row(
                "2",
                similar=["20-1"],
                doi="10.1000/case2",
                reference_doi="10.1000/case20",
            ),
            self._row("20", similar=[], doi="10.1000/case20"),
        ]
        self._complete_upstream(rows)
        upstream_path = self.retrieval / "run_manifest.json"
        upstream = load_json(upstream_path)
        upstream["dataset_contract"]["mode"] = "delta"
        upstream_path.write_text(json.dumps(upstream), encoding="utf-8")
        output = self.root / "rejected-run"

        with self.assertRaisesRegex(
            ArtifactIntegrityError,
            "single unified handoff",
        ):
            prepare_run(
                self.retrieval,
                output,
                self.config,
                workers=1,
            )
        self.assertFalse(output.exists())

    def test_upstream_failures_require_explicit_recorded_acceptance(self) -> None:
        rows = [
            self._row(
                "2",
                similar=["20-1"],
                doi="10.1000/case2",
                reference_doi="10.1000/case20",
            ),
            self._row("20", similar=[], doi="10.1000/case20"),
        ]
        self._complete_upstream(rows)
        upstream_path = self.retrieval / "run_manifest.json"
        upstream = load_json(upstream_path)
        upstream["status"] = "completed_with_failures"
        upstream["failure_summary"] = {"failed_articles": 1}
        upstream_path.write_text(json.dumps(upstream), encoding="utf-8")

        rejected = self.root / "not-accepted"
        with self.assertRaisesRegex(
            ArtifactIntegrityError,
            "completed with failures",
        ):
            prepare_run(
                self.retrieval,
                rejected,
                self.config,
                workers=1,
            )
        self.assertFalse(rejected.exists())

        accepted = self.root / "accepted"
        manifest = prepare_run(
            self.retrieval,
            accepted,
            self.config,
            workers=1,
            allow_upstream_failures=True,
        )
        upstream_input = manifest["inputs"][0]
        self.assertTrue(upstream_input["upstream_failures_accepted"])
        self.assertEqual(
            upstream_input["upstream_failure_summary"],
            {"failed_articles": 1},
        )
        self.assertEqual(verify_run(accepted)["status"], "complete")

    def test_changed_upstream_tree_fails_before_output_is_published(self) -> None:
        rows = [
            self._row(
                "2",
                similar=["20-1"],
                doi="10.1000/case2",
                reference_doi="10.1000/case20",
            ),
            self._row("20", similar=[], doi="10.1000/case20"),
        ]
        self._complete_upstream(rows)
        image = self.retrieval / rows[0]["image_path"]
        image.write_bytes(b"changed-after-completion")
        output = self.root / "stale-upstream"

        with self.assertRaisesRegex(
            ArtifactIntegrityError,
            "tree changed",
        ):
            prepare_run(
                self.retrieval,
                output,
                self.config,
                workers=1,
            )
        self.assertFalse(output.exists())

    def test_handoff_mutation_fails_verification(self) -> None:
        rows = [
            self._row(
                "2",
                similar=["20-1"],
                doi="10.1000/case2",
                reference_doi="10.1000/case20",
            ),
            self._row("20", similar=[], doi="10.1000/case20"),
        ]
        self._complete_upstream(rows)
        output = self.root / "tamper-run"
        prepare_run(
            self.retrieval,
            output,
            self.config,
            workers=1,
        )
        with (output / ELIGIBLE_PAIRS).open("a", encoding="utf-8") as handle:
            handle.write("{}\n")

        with self.assertRaisesRegex(ArtifactIntegrityError, "changed"):
            verify_run(output)

    def test_species_artifact_rejects_an_extra_text_field(self) -> None:
        rows = [
            self._row(
                "2",
                similar=["20-1"],
                doi="10.1000/case2",
                reference_doi="10.1000/case20",
            ),
            self._row("20", similar=[], doi="10.1000/case20"),
        ]
        self._complete_upstream(rows)
        output = self.root / "species-schema-tamper"
        prepare_run(
            self.retrieval,
            output,
            self.config,
            workers=1,
        )

        records = load_jsonl(output / SPECIES_CLASSIFICATIONS)
        records[0]["abstract"] = "This field is not part of the compact schema."
        save_jsonl(records, output / SPECIES_CLASSIFICATIONS)
        manifest = load_json(output / "run_manifest.json")
        manifest["artifacts"][SPECIES_CLASSIFICATIONS] = file_metadata(
            output / SPECIES_CLASSIFICATIONS,
            root=output,
        )
        save_json(manifest, output / "run_manifest.json")

        with self.assertRaisesRegex(
            ArtifactIntegrityError,
            "invalid schema",
        ):
            verify_run(output)


if __name__ == "__main__":
    unittest.main()
