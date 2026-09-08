from __future__ import annotations

import json
import unittest

from positive_pair_labeling_pipeline.errors import ResponseValidationError
from positive_pair_labeling_pipeline.finalizer import (
    finalize_positive_responses,
    modality_bucket,
)


class FinalizePositiveResponsesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selected_pairs = [
            {
                "pair_index": 7,
                "pair_key": "case-a::case-b",
                "case_a_uid": "case-a",
                "case_b_uid": "case-b",
            }
        ]
        self.cases_by_uid = {
            "case-a": {
                "captions": [
                    {"caption_id": "a1", "caption": "First Case A caption."},
                    {"caption_id": "a2", "caption": "Second Case A caption."},
                ]
            },
            "case-b": {
                "captions": [
                    {"caption_id": "b1", "caption": "First Case B caption."},
                    {"caption_id": "b2", "caption": "Second Case B caption."},
                ]
            },
        }

    def response(self, text: str) -> list[dict[str, object]]:
        return [{"pair_index": 7, "custom_id": "positive-7", "text": text}]

    @staticmethod
    def positive_item(
        pair_id: list[str],
        *,
        reasoning: str = "The captions describe the same finding.",
        modality: str = "CT scan",
    ) -> dict[str, object]:
        return {
            "pair_id": pair_id,
            "reasoning": reasoning,
            "modality": modality,
            "anatomy": "chest",
            "diagnosis": "pulmonary lesion",
        }

    def finalize(self, text: str) -> dict[str, object]:
        return finalize_positive_responses(
            self.response(text),
            self.selected_pairs,
            self.cases_by_uid,
        )

    def test_empty_json_array_is_a_valid_negative_response(self) -> None:
        result = self.finalize("[]")

        self.assertEqual(result["labeled_all"], [])
        self.assertEqual(result["pairs_clean_all"], [])
        self.assertEqual(result["text_pairs_all"], [])
        self.assertEqual(result["by_bucket"], {})
        self.assertEqual(result["bad_rows"], [])
        self.assertEqual(result["stats"]["empty_outputs"], 1)

    def test_valid_reversed_inter_case_pair_is_accepted_and_canonicalized(self) -> None:
        item = self.positive_item(["b1", "a1"])
        text = "```json\n" + json.dumps([item]) + "\n```"

        result = self.finalize(text)

        self.assertEqual(result["pairs_clean_all"], [["a1", "b1"]])
        self.assertEqual(result["labeled_all"][0]["pair_id"], ["a1", "b1"])
        self.assertEqual(result["labeled_all"][0]["source_pair_index"], 7)
        self.assertEqual(result["labeled_all"][0]["modality_bucket"], "radiology")
        self.assertEqual(result["bad_rows"], [])

    def test_missing_response_is_rejected(self) -> None:
        with self.assertRaisesRegex(ResponseValidationError, "Missing responses"):
            finalize_positive_responses(
                [],
                self.selected_pairs,
                self.cases_by_uid,
            )

    def test_malformed_json_is_rejected(self) -> None:
        with self.assertRaisesRegex(ResponseValidationError, "not a valid JSON array"):
            self.finalize("not JSON")

    def test_invalid_pair_id_shapes_and_membership_are_rejected(self) -> None:
        invalid_items = {
            "wrong shape": self.positive_item(["a1"]),
            "unknown caption": self.positive_item(["a1", "unknown"]),
            "intra-case captions": self.positive_item(["a1", "a2"]),
        }

        for label, item in invalid_items.items():
            with self.subTest(label=label):
                with self.assertRaises(ResponseValidationError):
                    self.finalize(json.dumps([item]))

    def test_duplicate_json_keys_are_rejected(self) -> None:
        text = (
            '[{"pair_id":["a1","b1"],'
            '"reasoning":"same finding",'
            '"modality":"CT scan",'
            '"modality":"MRI",'
            '"anatomy":"chest",'
            '"diagnosis":"pulmonary lesion"}]'
        )

        with self.assertRaisesRegex(ResponseValidationError, "duplicate JSON key"):
            self.finalize(text)

    def test_duplicate_pairs_are_removed_deterministically(self) -> None:
        items = [
            self.positive_item(["b1", "a1"], reasoning="z reason"),
            self.positive_item(["a1", "b1"], reasoning="a reason"),
        ]

        result = self.finalize(json.dumps(items))

        self.assertEqual(result["pairs_clean_all"], [["a1", "b1"]])
        self.assertEqual(result["labeled_all"][0]["reasoning"], "a reason")
        self.assertEqual(result["stats"]["duplicate_pairs_removed"], 1)

    def test_ct_bucket_uses_word_boundaries(self) -> None:
        self.assertEqual(modality_bucket("activity"), "other")
        self.assertEqual(modality_bucket("CT scan"), "radiology")


if __name__ == "__main__":
    unittest.main()
