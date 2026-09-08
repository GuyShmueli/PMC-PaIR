from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from case_pair_selection_pipeline.config import SelectionConfig
from case_pair_selection_pipeline.diagrams import (
    build_diagram_filter,
    tesseract_runtime_contract,
    white_ratio,
)
from case_pair_selection_pipeline.errors import DiagramFilteringError


def _config() -> SelectionConfig:
    return SelectionConfig(
        white_pixel_threshold=250,
        white_ratio_threshold=0.5,
        thumbnail_max_size=384,
        ocr_text_length_threshold=200,
        ocr_language="eng",
        ocr_config="--oem 1 --psm 11 -c tessedit_do_invert=0",
        ocr_timeout_seconds=2.0,
        title_fallback=False,
        title_min_chars=24,
        title_min_tokens=4,
        species_ruleset="human_animal_v1",
    )


def _runtime() -> dict[str, str]:
    return {
        "executable_sha256": "a" * 64,
        "version": "tesseract test",
        "version_output": "tesseract test",
        "language": "eng",
        "traineddata_sha256": "b" * 64,
    }


class DiagramFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.retrieval = self.root / "retrieval"
        self.case_dir = self.retrieval / "new_data" / "PMC2" / "2_1"
        self.case_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _image(self, name: str, white_pixels: int) -> Path:
        path = self.case_dir / name
        image = Image.new("L", (10, 10), 0)
        image.putdata([255] * white_pixels + [0] * (100 - white_pixels))
        # A lossless payload keeps exact boundary ratios while retaining the
        # Pipeline-01 .jpg path contract used by the manifest.
        image.save(path, format="PNG")
        return path

    def _merged(self, images: list[Path]) -> None:
        with (self.retrieval / "04_merged.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_path"])
            writer.writeheader()
            for image in images:
                writer.writerow(
                    {"image_path": image.relative_to(self.retrieval).as_posix()}
                )

    def test_white_ratio_boundary_is_strict(self) -> None:
        half = self._image("half.jpg", 50)
        above = self._image("above.jpg", 51)

        self.assertEqual(
            white_ratio(
                half,
                white_pixel_threshold=250,
                thumbnail_max_size=384,
            ),
            0.5,
        )
        self.assertEqual(
            white_ratio(
                above,
                white_pixel_threshold=250,
                thumbnail_max_size=384,
            ),
            0.51,
        )

    def test_computed_filter_short_circuits_white_and_uses_strict_ocr_rule(self) -> None:
        white = self._image("white.jpg", 51)
        ocr_positive = self._image("ocr-positive.jpg", 50)
        boundary = self._image("ocr-boundary.jpg", 50)
        self._merged([white, ocr_positive, boundary])

        def fake_ocr(path: str, *_: object, **__: object) -> str:
            if str(path).endswith("ocr-positive.jpg"):
                return "x" * 201
            if str(path).endswith("ocr-boundary.jpg"):
                return "x" * 200
            self.fail("OCR must be skipped for a white-ratio-positive image")

        with patch(
            "case_pair_selection_pipeline.diagrams.tesseract_runtime_metadata",
            return_value=_runtime(),
        ), patch(
            "case_pair_selection_pipeline.diagrams._ocr_text",
            side_effect=fake_ocr,
        ) as ocr:
            records, blacklist, stats = build_diagram_filter(
                self.retrieval,
                config=_config(),
                workers=1,
            )

        self.assertEqual(ocr.call_count, 2)
        self.assertEqual(
            [record["reason"] for record in records],
            ["white_ratio", "ocr_text_length", "retained"],
        )
        self.assertEqual(
            blacklist,
            [
                white.relative_to(self.retrieval).as_posix(),
                ocr_positive.relative_to(self.retrieval).as_posix(),
            ],
        )
        self.assertEqual(stats["diagrams_excluded"], 2)
        self.assertEqual(stats["runtime"], _runtime())

    def test_computed_filter_fails_closed_on_ocr_error(self) -> None:
        image = self._image("ocr-error.jpg", 0)
        self._merged([image])
        with patch(
            "case_pair_selection_pipeline.diagrams.tesseract_runtime_metadata",
            return_value=_runtime(),
        ), patch(
            "case_pair_selection_pipeline.diagrams._ocr_text",
            side_effect=RuntimeError("timeout"),
        ):
            with self.assertRaisesRegex(DiagramFilteringError, "during ocr"):
                build_diagram_filter(
                    self.retrieval,
                    config=_config(),
                    workers=1,
                )

    def test_worker_count_does_not_change_classifications(self) -> None:
        images = [
            self._image("white-a.jpg", 100),
            self._image("white-b.jpg", 100),
        ]
        self._merged(images)
        runtime = _runtime()
        first = build_diagram_filter(
            self.retrieval,
            config=_config(),
            workers=1,
            runtime_metadata=runtime,
        )
        second = build_diagram_filter(
            self.retrieval,
            config=_config(),
            workers=2,
            runtime_metadata=runtime,
        )

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertEqual(first[2]["reason_counts"], second[2]["reason_counts"])

    def test_runtime_contract_is_path_independent_and_hash_bound(self) -> None:
        metadata = {
            **_runtime(),
            "executable": "/private/install/bin/tesseract",
            "traineddata_path": "/private/install/share/tessdata/eng.traineddata",
        }
        self.assertEqual(tesseract_runtime_contract(metadata), _runtime())

        invalid = dict(metadata)
        invalid["traineddata_sha256"] = None
        with self.assertRaisesRegex(DiagramFilteringError, "traineddata_sha256"):
            tesseract_runtime_contract(invalid)


if __name__ == "__main__":
    unittest.main()
