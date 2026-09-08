from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from positive_pair_labeling_pipeline.artifacts import (
    CASES,
    CITATIONS,
    FINAL_LABELED,
    FINAL_PAIRS,
    FINAL_STATS,
    HANDOFF_CASES,
    HANDOFF_SPECIES_CLASSIFICATIONS,
    SELECTION,
    SOURCE_ELIGIBLE_CASES,
    SOURCE_ELIGIBLE_PAIRS,
    SOURCE_HANDOFF_MANIFEST,
    SOURCE_PAIR_SPECIES,
    SOURCE_SPECIES_CLASSIFICATIONS,
    SOURCE_SPECIES_STATS,
    SUMMARIES,
    batch_stage_dir,
)
from positive_pair_labeling_pipeline.errors import (
    ArtifactIntegrityError,
    ResponseValidationError,
    StageOrderError,
)
from positive_pair_labeling_pipeline.handoff import (
    _configured_species_ruleset,
    initialize_run,
)
from positive_pair_labeling_pipeline.io_utils import (
    load_json,
    load_jsonl,
    save_json,
    save_jsonl,
    sha256_file,
)
from positive_pair_labeling_pipeline.pipeline import (
    build_model_stage,
    download_and_ingest_model_stage,
    finalize_run,
    ingest_model_responses,
    retry_model_stage,
    submit_model_stage,
    verify_run,
)


class OfflinePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = self.root / "labeling.toml"
        self.config.write_text(
            """schema_version = 1

[batch]
max_shard_bytes = 10000
max_shard_requests = 2

[models.citation]
model = "test-model-2025-01-01"
reasoning_effort = "medium"
[models.question]
model = "test-model-2025-01-01"
reasoning_effort = "low"
[models.answer]
model = "test-model-2025-01-01"
reasoning_effort = "medium"
[models.positive]
model = "test-model-2025-01-01"
reasoning_effort = "medium"
""",
            encoding="utf-8",
        )
        pair_key = hashlib.sha256(b"300-1\0" b"400-1").hexdigest()
        cases = [
            self._case("300", "A valid selected case with a CT finding."),
            self._case("400", "Another valid selected case with a CT finding."),
        ]
        pairs = [
            {
                "schema_version": 1,
                "pair_index": 1,
                "pair_key": pair_key,
                "case_a_uid": "300-1",
                "case_b_uid": "400-1",
                "citation": {
                    "status": "matched",
                    "direction": "case_a_cites_case_b",
                    "citing_uid": "300-1",
                    "cited_uid": "400-1",
                    "match_type": "pmcid",
                    "ref_id": "R1",
                    "citing_title": "Synthetic case 300",
                    "citing_abstract": "Abstract for synthetic case 300.",
                    "citation_paragraphs": ["This resembles a published case 1."],
                    "cited_title": "Synthetic case 400",
                    "cited_abstract": "Abstract for synthetic case 400.",
                },
            }
        ]
        self.selection = self._write_selection_run(
            self.root / "selection-run", cases, pairs, run_spec_id="f" * 64
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _case(pmc_id: str, description: str) -> dict:
        stem = f"{pmc_id}_1_1"
        return {
            "schema_version": 2,
            "patient_uid": f"{pmc_id}-1",
            "pmc_id": pmc_id,
            "nxml_path": f"new_data/PMC{pmc_id}/{pmc_id}_1/article.nxml",
            "nxml_sha256": hashlib.sha256(f"nxml-{pmc_id}".encode()).hexdigest(),
            "case_description": description,
            "population_scope": {
                "ruleset": "human_animal_v1",
                "label": "human",
            },
            "captions": [
                {
                    "caption_id": stem,
                    "caption": (
                        f"CT image of a pulmonary lesion in synthetic case {pmc_id}."
                    ),
                    "caption_path": (
                        f"new_data/PMC{pmc_id}/{pmc_id}_1/{stem}.txt"
                    ),
                    "caption_sha256": hashlib.sha256(
                        f"caption-{pmc_id}".encode()
                    ).hexdigest(),
                    "image_path": (
                        f"new_data/PMC{pmc_id}/{pmc_id}_1/{stem}.jpg"
                    ),
                    "image_sha256": hashlib.sha256(
                        f"image-{pmc_id}".encode()
                    ).hexdigest(),
                }
            ],
        }

    @staticmethod
    def _write_selection_run(
        root: Path,
        cases: list[dict],
        pairs: list[dict],
        *,
        run_spec_id: str,
    ) -> Path:
        root.mkdir()
        cases_path = root / SOURCE_ELIGIBLE_CASES
        pairs_path = root / SOURCE_ELIGIBLE_PAIRS
        save_jsonl(cases, cases_path)
        save_jsonl(pairs, pairs_path)
        classifications = [
            {
                "schema_version": 1,
                "patient_uid": case["patient_uid"],
                "pmc_id": case["pmc_id"],
                "nxml_sha256": case["nxml_sha256"],
                "ruleset": "human_animal_v1",
                "automatic_label": "human",
                "final_label": "human",
                "matched_rules": [],
                "title": f"Synthetic case {case['pmc_id']}",
                "journal_title": "Synthetic Journal",
                "adjudication": None,
            }
            for case in cases
        ]
        classifications_path = root / SOURCE_SPECIES_CLASSIFICATIONS
        pair_species_path = root / SOURCE_PAIR_SPECIES
        stats_path = root / SOURCE_SPECIES_STATS
        save_jsonl(classifications, classifications_path)
        pair_species = [
            {
                "schema_version": 1,
                "pair_index": pair["pair_index"],
                "pair_key": pair["pair_key"],
                "case_a_uid": pair["case_a_uid"],
                "case_b_uid": pair["case_b_uid"],
                "label": "human",
                "endpoint_labels": {
                    "case_a": "human",
                    "case_b": "human",
                },
            }
            for pair in pairs
        ]
        save_jsonl(pair_species, pair_species_path)
        final_counts = {"human": len(cases)} if cases else {}
        pair_counts = {"human": len(pairs)} if pairs else {}
        save_json(
            {
                "schema_version": 1,
                "ruleset": "human_animal_v1",
                "workers": 1,
                "cases_screened": len(cases),
                "automatic_label_counts": final_counts,
                "final_label_counts": final_counts,
                "cases_adjudicated": 0,
                "rule_counts": {},
                "pairs_before_screen": len(pairs),
                "pair_label_counts": pair_counts,
                "human_pairs": len(pairs),
                "animal_pairs": 0,
            },
            stats_path,
        )

        def artifact(path: Path, records: int) -> dict:
            return {
                "path": path.name,
                "records": records,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

        handoff = {
            "schema_version": 4,
            "producer": {
                "pipeline_id": "case_pair_selection_pipeline",
                "pipeline_version": "6.1.0",
                "run_spec_id": run_spec_id,
            },
            "artifacts": {
                "eligible_cases": artifact(cases_path, len(cases)),
                "eligible_pairs": artifact(pairs_path, len(pairs)),
                "species_classifications": artifact(
                    classifications_path, len(classifications)
                ),
                "pair_species": artifact(pair_species_path, len(pair_species)),
                "species_stats": artifact(stats_path, 1),
            },
            "contracts": {
                "case_schema_version": 2,
                "pair_schema_version": 1,
                "population_filter": "complete_case_pair_human_animal_screen_required",
                "case_order": "patient_uid_numeric_ascending",
                "pair_order": "pair_index_ascending",
                "pair_indices": "stable_non_contiguous",
                "case_orientation": "canonical_numeric_uid_order",
            },
        }
        handoff_path = root / SOURCE_HANDOFF_MANIFEST
        save_json(handoff, handoff_path)
        handoff_paths = (cases_path, pairs_path, handoff_path)
        species_paths = (
            classifications_path,
            pair_species_path,
            stats_path,
        )
        manifest_artifacts = {
            path.name: {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (*handoff_paths, *species_paths)
        }
        save_json(
            {
                "schema_version": 1,
                "pipeline_id": "case_pair_selection_pipeline",
                "pipeline_version": "6.1.0",
                "run_spec_id": run_spec_id,
                "status": "complete",
                "config": {
                    "schema_version": 1,
                    "species_filter": {
                        "ruleset": "human_animal_v1",
                    },
                },
                "workers": 1,
                "handoff_contract": handoff,
                "artifacts": manifest_artifacts,
                "stages": {
                    "02_human_animal_screen": {
                        "status": "complete",
                        "artifacts": sorted(path.name for path in species_paths),
                        "details": {
                            "ruleset": "human_animal_v1",
                            "cases_screened": len(cases),
                            "pairs_screened": len(pairs),
                            "human_pairs": len(pairs),
                            "animal_pairs": 0,
                        },
                    },
                    "03_handoff": {
                        "status": "complete",
                        "artifacts": sorted(
                            path.name for path in (*handoff_paths, *species_paths)
                        ),
                        "details": {},
                    }
                },
            },
            root / "run_manifest.json",
        )
        return root

    @staticmethod
    def _rebind_selection_artifact(root: Path, artifact_name: str) -> None:
        manifest_path = root / "run_manifest.json"
        manifest = load_json(manifest_path)
        path = root / artifact_name
        manifest["artifacts"][artifact_name] = {
            "path": artifact_name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        role_by_name = {
            SOURCE_ELIGIBLE_CASES: "eligible_cases",
            SOURCE_ELIGIBLE_PAIRS: "eligible_pairs",
            SOURCE_SPECIES_CLASSIFICATIONS: "species_classifications",
            SOURCE_PAIR_SPECIES: "pair_species",
            SOURCE_SPECIES_STATS: "species_stats",
        }
        role = role_by_name.get(artifact_name)
        if role is not None:
            handoff_path = root / SOURCE_HANDOFF_MANIFEST
            handoff = load_json(handoff_path)
            records = 1 if artifact_name == SOURCE_SPECIES_STATS else len(load_jsonl(path))
            handoff["artifacts"][role] = {
                "path": artifact_name,
                "records": records,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            save_json(handoff, handoff_path)
            manifest["handoff_contract"] = handoff
            manifest["artifacts"][SOURCE_HANDOFF_MANIFEST] = {
                "path": SOURCE_HANDOFF_MANIFEST,
                "bytes": handoff_path.stat().st_size,
                "sha256": sha256_file(handoff_path),
            }
        save_json(manifest, manifest_path)

    def _write_responses(self, run_dir: Path, stage: str, texts: dict[int, str]) -> Path:
        request_index = load_jsonl(batch_stage_dir(run_dir, stage) / "request_index.jsonl")
        rows = []
        for request in reversed(request_index):
            rows.append(
                {
                    "custom_id": request["custom_id"],
                    "error": None,
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": texts[request["pair_index"]],
                                    },
                                }
                            ]
                        },
                    },
                }
            )
        path = self.root / f"{run_dir.name}-{stage}-responses" / "output.jsonl"
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    def _complete_run(self, run_dir: Path) -> None:
        initialize_run(self.selection, run_dir, self.config)
        selection = load_json(run_dir / SELECTION)
        self.assertEqual(selection["eligible_count"], 1)
        self.assertEqual(selection["selected_count"], 1)
        self.assertEqual(selection["pairs"][0]["pair_index"], 1)

        stage_texts = {
            "citation": {1: "Reason for Citation: comparison\nPossible Similarities: lung finding\nExplanation: related cases"},
            "question": {
                1: (
                    "Level 1\n- Are both images cross-sectional studies?\n"
                    "Level 2\n- Do both cases involve pulmonary disease?\n"
                    "Level 3\n- Do both captions describe a pulmonary lesion?"
                )
            },
            "answer": {
                1: (
                    "Level 1\nYes, both are cross-sectional studies.\n"
                    "Level 2\nYes, both involve pulmonary disease.\n"
                    "Level 3\nYes, both captions describe a pulmonary lesion."
                )
            },
            "positive": {
                1: json.dumps(
                    [
                        {
                            "pair_id": ["300_1_1", "400_1_1"],
                            "modality": "CT scan",
                            "anatomy": "lung",
                            "diagnosis": "pulmonary lesion",
                        }
                    ]
                )
            },
        }
        for stage in ("citation", "question", "answer", "positive"):
            request_manifest = build_model_stage(run_dir, stage)
            self.assertEqual(request_manifest["request_count"], 1)
            response_file = self._write_responses(run_dir, stage, stage_texts[stage])
            records, diagnostics = ingest_model_responses(run_dir, stage, [response_file.parent])
            self.assertEqual([record["pair_index"] for record in records], [1])
            self.assertEqual(diagnostics["validation"], "complete")

        stats = finalize_run(run_dir)
        self.assertEqual(stats["selected_pairs"], 1)
        self.assertEqual(stats["labeled_pairs_kept"], 1)
        self.assertEqual(load_json(run_dir / FINAL_PAIRS), [["300_1_1", "400_1_1"]])
        self.assertEqual(
            load_jsonl(run_dir / "labeled_positive_pairs.jsonl"),
            [{
                "pair_id": ["300_1_1", "400_1_1"],
                "modality": "CT scan",
                "anatomy": "lung",
                "diagnosis": "pulmonary lesion",
                "modality_bucket": "radiology",
                "source_pair_index": 1,
            }],
        )
        self.assertFalse((run_dir / "08_labeled_positive_pairs.jsonl").exists())
        verification = verify_run(run_dir)
        self.assertEqual(verification["status"], "complete")

    def test_full_offline_run_is_aligned_and_reproducible(self) -> None:
        first = self.root / "run-a"
        second = self.root / "run-b"
        self._complete_run(first)
        self._complete_run(second)

        reproducible_files = [
            CASES,
            SUMMARIES,
            CITATIONS,
            SELECTION,
            FINAL_LABELED,
            FINAL_PAIRS,
            FINAL_STATS,
            "run_manifest.json",
        ]
        for relative in reproducible_files:
            with self.subTest(relative=relative):
                self.assertEqual(sha256_file(first / relative), sha256_file(second / relative))
        for stage in ("citation", "question", "answer", "positive"):
            for relative in (
                "request_manifest.json",
                "request_index.jsonl",
                "responses.jsonl",
                "response_manifest.json",
            ):
                with self.subTest(stage=stage, relative=relative):
                    self.assertEqual(
                        sha256_file(batch_stage_dir(first, stage) / relative),
                        sha256_file(batch_stage_dir(second, stage) / relative),
                    )

    def test_population_scope_is_preserved_only_in_the_immutable_handoff(self) -> None:
        run_dir = self.root / "compatibility-view-run"
        initialize_run(self.selection, run_dir, self.config)

        imported_cases = load_jsonl(run_dir / HANDOFF_CASES)
        compatibility_cases = load_jsonl(run_dir / CASES)
        self.assertEqual(imported_cases[0]["schema_version"], 2)
        self.assertEqual(
            imported_cases[0]["population_scope"],
            {"ruleset": "human_animal_v1", "label": "human"},
        )
        self.assertEqual(compatibility_cases[0]["schema_version"], 1)
        self.assertNotIn("population_scope", compatibility_cases[0])
        imported_classifications = load_jsonl(
            run_dir / HANDOFF_SPECIES_CLASSIFICATIONS
        )
        self.assertEqual(len(imported_classifications), 2)
        self.assertTrue(
            all(
                record["final_label"] == "human"
                for record in imported_classifications
            )
        )

    def test_missing_species_stage_is_rejected_atomically(self) -> None:
        manifest_path = self.selection / "run_manifest.json"
        manifest = load_json(manifest_path)
        del manifest["stages"]["02_human_animal_screen"]
        save_json(manifest, manifest_path)
        destination = self.root / "missing-species-stage-must-not-exist"
        with self.assertRaisesRegex(
            ArtifactIntegrityError,
            "02_human_animal_screen",
        ):
            initialize_run(self.selection, destination, self.config)
        self.assertFalse(destination.exists())

    def test_unsupported_species_ruleset_is_rejected(self) -> None:
        manifest = {
            "config": {
                "species_filter": {
                    "ruleset": "no_filter_v0",
                }
            }
        }
        with self.assertRaisesRegex(
            ArtifactIntegrityError,
            "Unsupported species-filter ruleset",
        ):
            _configured_species_ruleset(manifest)

    def test_animal_species_classification_is_rejected_if_scope_says_human(
        self,
    ) -> None:
        path = self.selection / SOURCE_SPECIES_CLASSIFICATIONS
        classifications = load_jsonl(path)
        classifications[1]["automatic_label"] = "animal"
        classifications[1]["final_label"] = "animal"
        classifications[1]["matched_rules"] = ["title:cat_subject"]
        save_jsonl(classifications, path)
        self._rebind_selection_artifact(
            self.selection, SOURCE_SPECIES_CLASSIFICATIONS
        )
        destination = self.root / "nonretained-classification-must-not-exist"
        with self.assertRaisesRegex(
            ArtifactIntegrityError,
            "inconsistent endpoint labels",
        ):
            initialize_run(self.selection, destination, self.config)
        self.assertFalse(destination.exists())

    def test_mixed_species_rulesets_are_rejected(self) -> None:
        path = self.selection / SOURCE_SPECIES_CLASSIFICATIONS
        classifications = load_jsonl(path)
        classifications[1]["ruleset"] = "different_ruleset_v0"
        save_jsonl(classifications, path)
        self._rebind_selection_artifact(
            self.selection, SOURCE_SPECIES_CLASSIFICATIONS
        )
        destination = self.root / "mixed-rulesets-must-not-exist"
        with self.assertRaisesRegex(
            ArtifactIntegrityError,
            "uses a different ruleset",
        ):
            initialize_run(self.selection, destination, self.config)
        self.assertFalse(destination.exists())

    def test_eligible_pair_with_animal_label_is_rejected(self) -> None:
        classifications_path = self.selection / SOURCE_SPECIES_CLASSIFICATIONS
        classifications = load_jsonl(classifications_path)
        classifications[1]["automatic_label"] = "animal"
        classifications[1]["final_label"] = "animal"
        classifications[1]["matched_rules"] = ["title:cat_subject"]
        save_jsonl(classifications, classifications_path)
        self._rebind_selection_artifact(
            self.selection, SOURCE_SPECIES_CLASSIFICATIONS
        )

        pair = load_jsonl(self.selection / SOURCE_ELIGIBLE_PAIRS)[0]
        pair_species_path = self.selection / SOURCE_PAIR_SPECIES
        save_jsonl(
            [
                {
                    "schema_version": 1,
                    "pair_index": pair["pair_index"],
                    "pair_key": pair["pair_key"],
                    "case_a_uid": pair["case_a_uid"],
                    "case_b_uid": pair["case_b_uid"],
                    "label": "animal",
                    "endpoint_labels": {
                        "case_a": "human",
                        "case_b": "animal",
                    },
                }
            ],
            pair_species_path,
        )
        self._rebind_selection_artifact(
            self.selection, SOURCE_PAIR_SPECIES
        )
        destination = self.root / "excluded-pair-must-not-exist"
        with self.assertRaisesRegex(
            ArtifactIntegrityError,
            "does not map exactly to a human pair-species row",
        ):
            initialize_run(self.selection, destination, self.config)
        self.assertFalse(destination.exists())

    def test_pair_species_table_must_cover_every_eligible_pair(self) -> None:
        pair_species_path = self.selection / SOURCE_PAIR_SPECIES
        save_jsonl([], pair_species_path)
        self._rebind_selection_artifact(
            self.selection, SOURCE_PAIR_SPECIES
        )
        destination = self.root / "missing-pair-decision-must-not-exist"
        with self.assertRaisesRegex(
            ArtifactIntegrityError,
            "has no pair-species row",
        ):
            initialize_run(self.selection, destination, self.config)
        self.assertFalse(destination.exists())

    def test_pair_species_schema_rejects_additional_fields(self) -> None:
        pair_species_path = self.selection / SOURCE_PAIR_SPECIES
        pair_species = load_jsonl(pair_species_path)
        pair_species[0]["unexpected_field"] = "unexpected"
        save_jsonl(pair_species, pair_species_path)
        self._rebind_selection_artifact(
            self.selection, SOURCE_PAIR_SPECIES
        )
        destination = self.root / "extra-pair-decision-field-must-not-exist"
        with self.assertRaisesRegex(
            ArtifactIntegrityError,
            "incorrect fields",
        ):
            initialize_run(self.selection, destination, self.config)
        self.assertFalse(destination.exists())

    def test_pair_species_statistics_must_cover_complete_partition(self) -> None:
        stats_path = self.selection / SOURCE_SPECIES_STATS
        stats = load_json(stats_path)
        stats["pairs_before_screen"] += 1
        save_json(stats, stats_path)
        self._rebind_selection_artifact(self.selection, SOURCE_SPECIES_STATS)
        destination = self.root / "incomplete-pair-stats-must-not-exist"
        with self.assertRaisesRegex(
            ArtifactIntegrityError,
            "complete binary screen",
        ):
            initialize_run(self.selection, destination, self.config)
        self.assertFalse(destination.exists())

    def test_animal_population_case_is_rejected_before_publication(self) -> None:
        cases = [
            self._case("300", "A valid selected case with a CT finding."),
            self._case("400", "Another valid selected case with a CT finding."),
        ]
        cases[1]["population_scope"]["label"] = "animal"
        pair_key = hashlib.sha256(b"300-1\0" b"400-1").hexdigest()
        pairs = [
            {
                "schema_version": 1,
                "pair_index": 1,
                "pair_key": pair_key,
                "case_a_uid": "300-1",
                "case_b_uid": "400-1",
                "citation": {
                    "status": "matched",
                    "direction": "case_a_cites_case_b",
                    "citing_uid": "300-1",
                    "cited_uid": "400-1",
                    "match_type": "pmcid",
                    "ref_id": "R1",
                    "citing_title": "Synthetic case 300",
                    "citing_abstract": "Abstract for synthetic case 300.",
                    "citation_paragraphs": ["This resembles a published case 1."],
                    "cited_title": "Synthetic case 400",
                    "cited_abstract": "Abstract for synthetic case 400.",
                },
            }
        ]
        invalid_selection = self._write_selection_run(
            self.root / "nonhuman-selection",
            cases,
            pairs,
            run_spec_id="d" * 64,
        )
        destination = self.root / "nonhuman-must-not-exist"
        with self.assertRaisesRegex(
            ArtifactIntegrityError,
            "population_scope.label must be 'human'",
        ):
            initialize_run(invalid_selection, destination, self.config)
        self.assertFalse(destination.exists())


    def test_empty_selection_finalizes_without_model_requests(self) -> None:
        empty_selection = self._write_selection_run(
            self.root / "empty-selection", [], [], run_spec_id="e" * 64
        )
        run_dir = self.root / "empty-run"
        initialize_run(empty_selection, run_dir, self.config)
        self.assertEqual(load_json(run_dir / SELECTION)["selected_count"], 0)
        stats = finalize_run(run_dir)
        self.assertEqual(stats["selected_pairs"], 0)
        self.assertEqual(load_json(run_dir / FINAL_PAIRS), [])
        self.assertEqual(verify_run(run_dir)["status"], "complete")

    def test_tampered_handoff_fails_before_output_is_published(self) -> None:
        cases_path = self.selection / SOURCE_ELIGIBLE_CASES
        cases_path.write_bytes(cases_path.read_bytes() + b" ")
        destination = self.root / "must-not-exist"
        with self.assertRaisesRegex(ArtifactIntegrityError, "bind|byte count|SHA-256"):
            initialize_run(self.selection, destination, self.config)
        self.assertFalse(destination.exists())

    def test_singular_citation_paragraph_field_is_rejected(self) -> None:
        pairs_path = self.selection / SOURCE_ELIGIBLE_PAIRS
        pairs = load_jsonl(pairs_path)
        citation = pairs[0]["citation"]
        citation["citation_paragraph"] = citation.pop("citation_paragraphs")
        save_jsonl(pairs, pairs_path)
        self._rebind_selection_artifact(self.selection, SOURCE_ELIGIBLE_PAIRS)

        destination = self.root / "singular-citation-field-must-not-exist"
        with self.assertRaisesRegex(
            ArtifactIntegrityError,
            "citation.*incorrect fields",
        ):
            initialize_run(self.selection, destination, self.config)
        self.assertFalse(destination.exists())

    def test_pipeline03_never_mutates_pipeline02_run(self) -> None:
        before = {
            path.name: sha256_file(path)
            for path in sorted(self.selection.iterdir())
            if path.is_file()
        }
        self._complete_run(self.root / "immutability-run")
        after = {
            path.name: sha256_file(path)
            for path in sorted(self.selection.iterdir())
            if path.is_file()
        }
        self.assertEqual(after, before)

    def test_invalid_positive_attempt_is_not_locked_and_can_be_corrected(self) -> None:
        run_dir = self.root / "corrected-run"
        initialize_run(self.selection, run_dir, self.config)
        valid_intermediate = {
            "citation": (
                "Reason for Citation: comparison\n"
                "Possible Similarities: lung finding\n"
                "Explanation: related cases"
            ),
            "question": (
                "Level 1\n- Are both images cross-sectional studies?\n"
                "Level 2\n- Do both cases involve pulmonary disease?\n"
                "Level 3\n- Do both captions describe a pulmonary lesion?"
            ),
            "answer": (
                "Level 1\nYes, both are cross-sectional studies.\n"
                "Level 2\nYes, both involve pulmonary disease.\n"
                "Level 3\nYes, both captions describe a pulmonary lesion."
            ),
        }
        for stage in ("citation", "question", "answer"):
            build_model_stage(run_dir, stage)
            raw = self._write_responses(run_dir, stage, {1: valid_intermediate[stage]})
            ingest_model_responses(run_dir, stage, raw)

        build_model_stage(run_dir, "positive")
        malformed = self._write_responses(run_dir, "positive", {1: "not JSON"})
        with self.assertRaises(ResponseValidationError):
            ingest_model_responses(run_dir, "positive", malformed)
        manifest = load_json(run_dir / "run_manifest.json")
        self.assertNotIn("07_positive_responses", manifest["stages"])

        corrected_text = json.dumps(
            [
                {
                    "pair_id": ["300_1_1", "400_1_1"],
                    "modality": "CT scan",
                    "anatomy": "lung",
                    "diagnosis": "pulmonary lesion",
                }
            ]
        )
        corrected = self._write_responses(
            run_dir, "positive", {1: corrected_text}
        )
        records, _ = ingest_model_responses(run_dir, "positive", corrected)
        self.assertEqual([record["pair_index"] for record in records], [1])
        self.assertEqual(finalize_run(run_dir)["labeled_pairs_kept"], 1)

    def test_managed_batch_receipts_are_verified_and_block_resubmission(self) -> None:
        run_dir = self.root / "managed-run"
        initialize_run(self.selection, run_dir, self.config)
        request_manifest = build_model_stage(run_dir, "citation")
        request_index = load_jsonl(
            batch_stage_dir(run_dir, "citation") / "request_index.jsonl"
        )
        citation_text = (
            "Reason for Citation: comparison\n"
            "Possible Similarities: lung finding\n"
            "Explanation: related cases"
        )
        output_payload = (
            json.dumps(
                {
                    "custom_id": request_index[0]["custom_id"],
                    "error": None,
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "index": 0,
                                    "message": {
                                        "content": citation_text,
                                        "role": "assistant",
                                    },
                                }
                            ]
                        },
                    },
                },
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")

        class Files:
            def __init__(self):
                self.create_count = 0

            def create(self, *, file, purpose):
                self.create_count += 1
                self.uploaded = file.read()
                self.purpose = purpose
                return SimpleNamespace(id="file_managed_input")

            def content(self, file_id):
                self.requested_file_id = file_id
                return SimpleNamespace(content=output_payload)

        class Batches:
            def __init__(self):
                self.create_count = 0

            def create(self, **kwargs):
                self.create_count += 1
                self.kwargs = kwargs
                return SimpleNamespace(
                    endpoint=kwargs["endpoint"],
                    id="batch_managed",
                    input_file_id=kwargs["input_file_id"],
                    status="validating",
                )

            def retrieve(self, batch_id):
                return SimpleNamespace(
                    endpoint="/v1/chat/completions",
                    error_file_id=None,
                    id=batch_id,
                    input_file_id="file_managed_input",
                    output_file_id="file_managed_output",
                    status="completed",
                )

        client = SimpleNamespace(files=Files(), batches=Batches())
        submissions = submit_model_stage(
            run_dir,
            "citation",
            client=client,
            acknowledge_external_api=True,
        )
        self.assertEqual(len(submissions), request_manifest["shard_count"])

        # Simulate interruption after the immutable submission receipt was
        # saved but before its stage was committed to the run manifest.
        manifest_path = run_dir / "run_manifest.json"
        submission_relative = (
            "batch/citation/attempts/attempt-0001/submission_receipt.json"
        )
        interrupted = load_json(manifest_path)
        interrupted["stages"].pop("04_citation_submissions")
        interrupted["artifacts"].pop(submission_relative)
        save_json(interrupted, manifest_path)
        recovered_submissions = submit_model_stage(
            run_dir,
            "citation",
            client=client,
            acknowledge_external_api=True,
        )
        self.assertEqual(recovered_submissions, submissions)
        self.assertEqual(client.files.create_count, 1)
        self.assertEqual(client.batches.create_count, 1)

        records, _ = download_and_ingest_model_stage(
            run_dir,
            "citation",
            client=client,
            acknowledge_external_api=True,
        )
        self.assertEqual(records[0]["text"], citation_text)

        # Simulate interruption after canonical managed responses were
        # recorded but before accepted_attempt.json was created.
        stage_dir = batch_stage_dir(run_dir, "citation")
        accepted_path = stage_dir / "accepted_attempt.json"
        accepted_relative = accepted_path.relative_to(run_dir).as_posix()
        interrupted = load_json(manifest_path)
        response_record = interrupted["stages"]["04_citation_responses"]
        response_record["artifacts"].remove(accepted_relative)
        response_record["details"].pop("managed_attempt")
        interrupted["artifacts"].pop(accepted_relative)
        save_json(interrupted, manifest_path)
        accepted_path.unlink()
        with self.assertRaisesRegex(
            ArtifactIntegrityError, "no accepted-attempt receipt"
        ):
            verify_run(run_dir)
        download_and_ingest_model_stage(
            run_dir,
            "citation",
            client=client,
            acknowledge_external_api=True,
        )

        # Simulate interruption after accepted_attempt.json was created but
        # before it was bound to the response stage.
        interrupted = load_json(manifest_path)
        response_record = interrupted["stages"]["04_citation_responses"]
        response_record["artifacts"].remove(accepted_relative)
        response_record["details"].pop("managed_attempt")
        interrupted["artifacts"].pop(accepted_relative)
        save_json(interrupted, manifest_path)
        with self.assertRaisesRegex(ArtifactIntegrityError, "not bound"):
            verify_run(run_dir)
        download_and_ingest_model_stage(
            run_dir,
            "citation",
            client=client,
            acknowledge_external_api=True,
        )

        verification = verify_run(run_dir)
        self.assertEqual(verification["completed_model_stages"], ["citation"])
        manifest = load_json(manifest_path)
        self.assertIn("04_citation_submissions", manifest["stages"])
        self.assertIn(
            "batch/citation/attempts/attempt-0001/managed_downloads.json",
            manifest["artifacts"],
        )
        self.assertIn("batch/citation/accepted_attempt.json", manifest["artifacts"])
        with self.assertRaisesRegex(StageOrderError, "already complete"):
            submit_model_stage(
                run_dir,
                "citation",
                client=client,
                acknowledge_external_api=True,
            )

    def test_managed_retry_preserves_error_only_attempt_and_binds_success(self) -> None:
        run_dir = self.root / "managed-retry-run"
        initialize_run(self.selection, run_dir, self.config)
        build_model_stage(run_dir, "citation")
        request_index = load_jsonl(
            batch_stage_dir(run_dir, "citation") / "request_index.jsonl"
        )
        custom_id = request_index[0]["custom_id"]

        def response_payload(text: str) -> bytes:
            return (
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "error": None,
                        "response": {
                            "status_code": 200,
                            "body": {
                                "choices": [
                                    {
                                        "finish_reason": "stop",
                                        "index": 0,
                                        "message": {
                                            "content": text,
                                            "role": "assistant",
                                        },
                                    }
                                ]
                            },
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")

        error_payload = (json.dumps({
            "custom_id": custom_id,
            "error": {"code": "server_error", "message": "Request failed"},
            "response": None,
        }) + "\n").encode("utf-8")
        valid_text = (
            "Reason for Citation: comparison\n"
            "Possible Similarities: lung finding\n"
            "Explanation: related cases"
        )
        valid_payload = response_payload(valid_text)

        class Files:
            def __init__(self):
                self.upload_count = 0

            def create(self, *, file, purpose):
                self.upload_count += 1
                self.uploaded = file.read()
                self.purpose = purpose
                return SimpleNamespace(id=f"file_input_{self.upload_count}")

            def content(self, file_id):
                if file_id == "file_errors_1":
                    return SimpleNamespace(content=error_payload)
                if file_id == "file_output_2":
                    return SimpleNamespace(content=valid_payload)
                raise AssertionError(f"Unexpected output file ID: {file_id}")

        class Batches:
            def __init__(self):
                self.create_count = 0
                self.input_by_batch = {}

            def create(self, **kwargs):
                self.create_count += 1
                batch_id = f"batch_attempt_{self.create_count}"
                self.input_by_batch[batch_id] = kwargs["input_file_id"]
                return SimpleNamespace(
                    endpoint=kwargs["endpoint"],
                    id=batch_id,
                    input_file_id=kwargs["input_file_id"],
                    status="validating",
                )

            def retrieve(self, batch_id):
                number = int(batch_id.rsplit("_", 1)[1])
                return SimpleNamespace(
                    endpoint="/v1/chat/completions",
                    error_file_id="file_errors_1" if number == 1 else None,
                    id=batch_id,
                    input_file_id=self.input_by_batch[batch_id],
                    output_file_id=None if number == 1 else f"file_output_{number}",
                    status="completed",
                )

        client = SimpleNamespace(files=Files(), batches=Batches())
        first_submission = submit_model_stage(
            run_dir,
            "citation",
            client=client,
            acknowledge_external_api=True,
        )
        with self.assertRaises(ResponseValidationError):
            download_and_ingest_model_stage(
                run_dir,
                "citation",
                client=client,
                acknowledge_external_api=True,
            )

        first_attempt = (
            batch_stage_dir(run_dir, "citation")
            / "attempts"
            / "attempt-0001"
        )
        self.assertTrue((first_attempt / "submission_receipt.json").is_file())
        self.assertTrue((first_attempt / "managed_downloads.json").is_file())
        self.assertTrue((first_attempt / "validation_failure.json").is_file())

        second_submission = retry_model_stage(
            run_dir,
            "citation",
            attempt=2,
            client=client,
            acknowledge_external_api=True,
        )
        records, _ = download_and_ingest_model_stage(
            run_dir,
            "citation",
            attempt=2,
            client=client,
            acknowledge_external_api=True,
        )
        self.assertEqual(records[0]["text"], valid_text)
        self.assertNotEqual(
            first_submission[0]["batch_id"], second_submission[0]["batch_id"]
        )

        second_attempt = (
            batch_stage_dir(run_dir, "citation")
            / "attempts"
            / "attempt-0002"
        )
        self.assertTrue((second_attempt / "submission_receipt.json").is_file())
        self.assertTrue((second_attempt / "managed_downloads.json").is_file())
        self.assertFalse((second_attempt / "validation_failure.json").exists())
        accepted = load_json(
            batch_stage_dir(run_dir, "citation") / "accepted_attempt.json"
        )
        self.assertEqual(accepted["attempt"], 2)

        verification = verify_run(run_dir)
        self.assertEqual(verification["completed_model_stages"], ["citation"])
        manifest = load_json(run_dir / "run_manifest.json")
        attempt_details = manifest["stages"]["04_citation_submissions"]["details"]
        self.assertEqual(
            [item["attempt"] for item in attempt_details["attempts"]], [1, 2]
        )
        for attempt_number in (1, 2):
            self.assertIn(
                "batch/citation/attempts/"
                f"attempt-{attempt_number:04d}/submission_receipt.json",
                manifest["artifacts"],
            )

        with self.assertRaisesRegex(StageOrderError, "already complete"):
            submit_model_stage(
                run_dir,
                "citation",
                client=client,
                acknowledge_external_api=True,
            )
        with self.assertRaisesRegex(StageOrderError, "already complete"):
            retry_model_stage(
                run_dir,
                "citation",
                attempt=3,
                client=client,
                acknowledge_external_api=True,
            )
        self.assertEqual(client.files.upload_count, 2)
        self.assertEqual(client.batches.create_count, 2)


if __name__ == "__main__":
    unittest.main()
