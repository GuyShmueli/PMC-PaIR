from __future__ import annotations

import unittest

from positive_pair_labeling_pipeline.errors import ResponseValidationError
from positive_pair_labeling_pipeline.response_schemas import validate_intermediate_response


class IntermediateResponseSchemaTests(unittest.TestCase):
    def test_valid_stage_formats(self) -> None:
        validate_intermediate_response(
            "citation",
            "Reason for Citation: comparison\n"
            "Possible Similarities: shared anatomy\n"
            "Explanation: the cited case provides context.",
        )
        validate_intermediate_response(
            "question",
            "Level 1\nIs the modality shared?\n"
            "Level 2\nIs the pathology family shared?\n"
            "Level 3\nIs the finding shared?",
        )
        validate_intermediate_response(
            "answer",
            "Level 1\nYes.\nLevel 2\nYes.\nLevel 3\nNo.",
        )

    def test_missing_labels_levels_and_refusals_are_rejected(self) -> None:
        invalid = [
            ("citation", "The cases are similar."),
            (
                "question",
                "Level 1\nIs it CT?\nLevel 3\nIs there a lesion?",
            ),
            (
                "question",
                "Level 1\nCT comparison\nLevel 2\nPathology?\nLevel 3\nFinding?",
            ),
            (
                "answer",
                "Level 1\nAs an AI language model, I cannot assist.\n"
                "Level 2\nNo.\nLevel 3\nNo.",
            ),
        ]
        for stage, text in invalid:
            with self.subTest(stage=stage, text=text):
                with self.assertRaises(ResponseValidationError):
                    validate_intermediate_response(stage, text)


if __name__ == "__main__":
    unittest.main()
