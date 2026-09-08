from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from case_pair_selection_pipeline.config import (
    PIPELINE_ID,
    PIPELINE_VERSION,
    load_config,
)
from case_pair_selection_pipeline.errors import ConfigurationError


_BASE = """schema_version = 1
[selection]
title_fallback = false
title_min_chars = 24
title_min_tokens = 4
[species_filter]
ruleset = "human_animal_v1"
"""


class ConfigurationTests(unittest.TestCase):
    def _load(self, text: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(text, encoding="utf-8")
            return load_config(path)

    def test_valid_configuration_applies_diagram_defaults(self) -> None:
        config = self._load(_BASE)
        self.assertFalse(config.title_fallback)
        self.assertEqual(config.title_min_chars, 24)
        self.assertEqual(config.title_min_tokens, 4)
        self.assertEqual(
            config.as_dict(),
            {
                "schema_version": 1,
                "diagram_filter": {
                    "white_pixel_threshold": 250,
                    "white_ratio_threshold": 0.5,
                    "thumbnail_max_size": 384,
                    "ocr_text_length_threshold": 200,
                    "ocr_language": "eng",
                    "ocr_config": "--oem 1 --psm 11 -c tessedit_do_invert=0",
                    "ocr_timeout_seconds": 2.0,
                },
                "selection": {
                    "title_fallback": False,
                    "title_min_chars": 24,
                    "title_min_tokens": 4,
                },
                "species_filter": {
                    "ruleset": "human_animal_v1",
                },
            },
        )
        self.assertEqual(PIPELINE_ID, "case_pair_selection_pipeline")
        self.assertEqual(PIPELINE_VERSION, "6.1.0")

    def test_unknown_model_or_batch_sections_are_rejected(self) -> None:
        for extra in (
            "\n[batch]\nmax_shard_bytes = 1000\n",
            '\n[models.citation]\nmodel = "example-2025-01-01"\n',
        ):
            with self.subTest(extra=extra):
                with self.assertRaises(ConfigurationError):
                    self._load(_BASE + extra)

    def test_wrong_types_and_unsafe_title_fallback_fail(self) -> None:
        invalid = (
            _BASE.replace(
                "title_min_tokens = 4",
                "title_min_tokens = 4\ntitel_fallback = true",
            ),
            _BASE.replace(
                "title_fallback = false",
                'title_fallback = "false"',
            ),
            _BASE.replace("title_min_chars = 24", "title_min_chars = -1"),
            _BASE.replace(
                "title_fallback = false",
                "title_fallback = true",
            ).replace("title_min_chars = 24", "title_min_chars = 0"),
        )
        for text in invalid:
            with self.subTest(text=text):
                with self.assertRaises(ConfigurationError):
                    self._load(text)

    def test_invalid_diagram_filter_settings_fail(self) -> None:
        invalid_sections = (
            "[diagram_filter]\nwhite_pixel_threshold = 256\n",
            "[diagram_filter]\nwhite_ratio_threshold = nan\n",
            "[diagram_filter]\nthumbnail_max_size = 0\n",
            "[diagram_filter]\nocr_text_length_threshold = -1\n",
            "[diagram_filter]\nocr_timeout_seconds = 0\n",
            "[diagram_filter]\nunknown = true\n",
        )
        for section in invalid_sections:
            with self.subTest(section=section):
                text = _BASE.replace("[selection]", section + "[selection]")
                with self.assertRaises(ConfigurationError):
                    self._load(text)

    def test_invalid_species_filter_settings_fail(self) -> None:
        invalid = (
            _BASE.replace('ruleset = "human_animal_v1"', 'ruleset = "unknown"'),
            _BASE.replace("[species_filter]", "[species_filter]\nunknown = true"),
            _BASE.replace(
                '[species_filter]\nruleset = "human_animal_v1"\n',
                "",
            ),
        )
        for text in invalid:
            with self.subTest(text=text):
                with self.assertRaises(ConfigurationError):
                    self._load(text)

    def test_binary_ruleset_is_default_and_other_rulesets_are_rejected(self) -> None:
        without_ruleset = _BASE.replace('ruleset = "human_animal_v1"\n', "")
        self.assertEqual(
            self._load(without_ruleset).species_ruleset,
            "human_animal_v1",
        )
        for unsupported in ("other_ruleset_v1", "animal_only_v1"):
            with self.subTest(unsupported=unsupported):
                with self.assertRaises(ConfigurationError):
                    self._load(
                        _BASE.replace(
                            'ruleset = "human_animal_v1"',
                            f'ruleset = "{unsupported}"',
                        )
                    )


if __name__ == "__main__":
    unittest.main()
