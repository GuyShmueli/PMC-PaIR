from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from case_pair_selection_pipeline.io_utils import sha256_file
from case_pair_selection_pipeline.species import (
    HUMAN_ANIMAL_RULESET,
    classify_cases,
    classify_metadata,
    human_animal_v1,
    matched_rule_vocabulary,
)


def _metadata(
    *, title: str = "", abstract: str = "", journal_title: str = ""
) -> dict[str, str]:
    return {
        "title": title,
        "abstract": abstract,
        "journal_title": journal_title,
    }


def _article_xml(
    pmc_id: str,
    *,
    title: str,
    abstract: str,
    journal_title: str,
) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<article><front>"
        "<journal-meta><journal-title-group>"
        f"<journal-title>{journal_title}</journal-title>"
        "</journal-title-group></journal-meta>"
        "<article-meta>"
        f'<article-id pub-id-type="pmc">PMC{pmc_id}</article-id>'
        f"<title-group><article-title>{title}</article-title></title-group>"
        f"<abstract><p>{abstract}</p></abstract>"
        "</article-meta></front><body/></article>"
    )


class MetadataClassifierTests(unittest.TestCase):
    def test_explicit_nonhuman_subjects_are_animal(self) -> None:
        examples = (
            "A dog presented with lymphoma",
            "Case report of a horse with an orbital mass",
            "Computed tomography in a rabbit patient",
            "A parrot was referred for dyspnea",
            "A macaque underwent magnetic resonance imaging",
            "A dog presented with lymphoma confirmed by FISH analysis",
            "A 7-year-old male dog presented with lymphoma",
        )
        for title in examples:
            with self.subTest(title=title):
                label, reasons = human_animal_v1(_metadata(title=title))
                self.assertEqual(label, "animal")
                self.assertTrue(reasons)

    def test_experimental_animal_studies_are_animal(self) -> None:
        examples = (
            "Magnetic resonance imaging in an animal model",
            "Imaging in experimental rats",
            "A canine animal model of arthritis",
            "Mice were injected with contrast material",
        )
        for title in examples:
            with self.subTest(title=title):
                label, reasons = human_animal_v1(_metadata(title=title))
                self.assertEqual(label, "animal")
                self.assertTrue(any(":model:" in reason for reason in reasons))

    def test_human_exposure_zoonosis_and_ambiguous_terms_are_human(self) -> None:
        examples = (
            "A human patient developed infection after a dog bite",
            "Zoonotic transmission from cats to a woman",
            "A boy with cat-scratch disease",
            "CAT scan of an abdominal mass",
            "Impacted canine tooth in a patient",
            "Rabbit antibody staining of a biopsy",
            "FISH analysis in a patient with leukemia",
            "Dog-ear deformity after surgery",
        )
        for title in examples:
            with self.subTest(title=title):
                result = classify_metadata(_metadata(title=title))
                self.assertEqual(result["automatic_label"], "human")
                self.assertEqual(result["final_label"], "human")
        result = classify_metadata(_metadata(
            title="An unusual abdominal case",
            abstract="A 50-year-old man presented with pain. Previous mouse model "
                     "studies examined this disease.",
        ))
        self.assertEqual(result["final_label"], "human")

    def test_no_animal_evidence_is_human(self) -> None:
        result = classify_metadata(
            _metadata(
                title="Human abdominal imaging",
                abstract="A patient presented with abdominal pain.",
            )
        )
        self.assertEqual(
            result,
            {
                "ruleset": HUMAN_ANIMAL_RULESET,
                "automatic_label": "human",
                "final_label": "human",
                "matched_rules": [],
            },
        )

    def test_journal_does_not_change_title_and_abstract_classification(self) -> None:
        for title in ("Canine lymphoma", "A dog presented with lymphoma"):
            expected = human_animal_v1(_metadata(title=title))
            actual = human_animal_v1(
                _metadata(
                    title=title,
                    journal_title="Journal of Veterinary Oncology",
                )
            )
            self.assertEqual(actual, expected)

    def test_only_one_versioned_ruleset_is_supported(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown species ruleset"):
            classify_metadata({}, "unknown")
        result = classify_metadata(
            _metadata(title="A dog presented with lymphoma"),
            HUMAN_ANIMAL_RULESET,
        )
        self.assertTrue(
            set(result["matched_rules"])
            <= matched_rule_vocabulary(HUMAN_ANIMAL_RULESET)
        )


class CaseClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.retrieval = Path(self.temporary.name) / "retrieval"
        self.retrieval.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _case(
        self,
        pmc_id: str,
        *,
        title: str,
        abstract: str = "A clinical report.",
        journal_title: str = "Medical Cases",
        xml: str | None = None,
    ) -> dict[str, object]:
        uid = f"{pmc_id}-1"
        directory = (
            self.retrieval / "new_data" / f"PMC{pmc_id}" / f"{pmc_id}_1"
        )
        directory.mkdir(parents=True, exist_ok=True)
        nxml = directory / "article.nxml"
        nxml.write_text(
            xml
            if xml is not None
            else _article_xml(
                pmc_id,
                title=title,
                abstract=abstract,
                journal_title=journal_title,
            ),
            encoding="utf-8",
        )
        return {
            "schema_version": 1,
            "patient_uid": uid,
            "pmc_id": pmc_id,
            "nxml_path": nxml.relative_to(self.retrieval).as_posix(),
            "nxml_sha256": sha256_file(nxml),
            "captions": [],
        }

    def test_classify_cases_is_deterministic_hash_bound_and_binary(self) -> None:
        cases = [
            self._case("10", title="A human abdominal case"),
            self._case("2", title="A dog presented with lymphoma"),
        ]
        serial = classify_cases(cases, self.retrieval, workers=1)
        parallel = classify_cases(cases, self.retrieval, workers=2)
        self.assertEqual(serial, parallel)
        self.assertEqual([record["patient_uid"] for record in serial], ["2-1", "10-1"])
        self.assertEqual(
            [record["automatic_label"] for record in serial],
            ["animal", "human"],
        )
        expected_fields = {
            "schema_version",
            "patient_uid",
            "pmc_id",
            "nxml_sha256",
            "ruleset",
            "automatic_label",
            "final_label",
            "matched_rules",
            "title",
            "journal_title",
            "adjudication",
        }
        self.assertTrue(all(set(record) == expected_fields for record in serial))
        self.assertTrue(all("abstract" not in record for record in serial))

    def test_source_or_parse_failure_aborts_without_an_error_label(self) -> None:
        malformed = self._case("2", title="unused", xml="<article><front>")
        with self.assertRaisesRegex(ValueError, "could not prepare every case"):
            classify_cases([malformed], self.retrieval, workers=1)

        valid = self._case("10", title="Human case")
        valid["nxml_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "could not prepare every case"):
            classify_cases([valid], self.retrieval, workers=1)

    def test_invalid_workers_duplicates_and_path_escape_are_rejected(self) -> None:
        case = self._case("2", title="Human case")
        with self.assertRaisesRegex(ValueError, "workers must be at least 1"):
            classify_cases([case], self.retrieval, workers=0)
        with self.assertRaisesRegex(ValueError, "duplicate patient_uid"):
            classify_cases([case, dict(case)], self.retrieval, workers=1)
        escaped = dict(case)
        escaped["nxml_path"] = "../outside.nxml"
        with self.assertRaisesRegex(ValueError, "could not prepare every case"):
            classify_cases([escaped], self.retrieval, workers=1)


if __name__ == "__main__":
    unittest.main()
