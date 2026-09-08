from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from positive_pair_labeling_pipeline.config import PIPELINE_VERSION, load_config
from positive_pair_labeling_pipeline.errors import ConfigurationError


_BASE = """schema_version = 1
[batch]
max_shard_bytes = 190000000
max_shard_requests = 45000
[models.citation]
model = "model-2025-01-01"
[models.question]
model = "model-2025-01-01"
[models.answer]
model = "model-2025-01-01"
[models.positive]
model = "model-2025-01-01"
"""


class ConfigurationTests(unittest.TestCase):
    def _load(self, text: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(text, encoding="utf-8")
            return load_config(path)

    def test_valid_strict_configuration(self) -> None:
        config = self._load(_BASE)
        self.assertEqual(PIPELINE_VERSION, "6.1.0")
        self.assertEqual(config.max_shard_bytes, 190_000_000)
        self.assertEqual(config.models["positive"].model, "model-2025-01-01")

    def test_unknown_keys_wrong_types_and_invalid_dates_fail(self) -> None:
        invalid = [
            _BASE.replace("[batch]", "unexpected = true\n[batch]"),
            _BASE.replace("max_shard_requests = 45000", 'max_shard_requests = "many"'),
            _BASE.replace("190000000", "200000000"),
            _BASE.replace("model-2025-01-01", "model-2025-99-99", 1),
        ]
        for text in invalid:
            with self.subTest(text=text):
                with self.assertRaises(ConfigurationError):
                    self._load(text)


if __name__ == "__main__":
    unittest.main()
