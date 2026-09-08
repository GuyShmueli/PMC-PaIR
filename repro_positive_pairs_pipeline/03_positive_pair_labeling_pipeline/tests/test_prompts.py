from __future__ import annotations

import unittest

from positive_pair_labeling_pipeline.prompts import citation_user_prompt


class CitationPromptTests(unittest.TestCase):
    def test_plural_citation_paragraphs_are_embedded_in_prompt(self) -> None:
        prompt = citation_user_prompt(
            {
                "citing_title": "Citing case",
                "citing_abstract": "Citing abstract",
                "citation_paragraphs": ["Literal recovered citation context [8]."],
                "cited_title": "Cited case",
                "cited_abstract": "Cited abstract",
            }
        )

        self.assertIn("Citation Paragraphs:\n- Literal recovered citation context [8].", prompt)
        self.assertNotIn("No explicit citation paragraph was found", prompt)


if __name__ == "__main__":
    unittest.main()
