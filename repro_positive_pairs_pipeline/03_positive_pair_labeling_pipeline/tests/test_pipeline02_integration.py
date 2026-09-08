from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PIPELINES_ROOT = Path(__file__).resolve().parents[2]
SELECTION_PROJECT = PIPELINES_ROOT / "02_case_pair_selection_pipeline"
try:
    import lxml  # noqa: F401
    import pandas  # noqa: F401
    from PIL import Image  # noqa: F401
except ImportError:
    SELECTION_RUNTIME_AVAILABLE = False
else:
    SELECTION_RUNTIME_AVAILABLE = True
    if str(SELECTION_PROJECT) not in sys.path:
        sys.path.insert(0, str(SELECTION_PROJECT))
    from case_pair_selection_pipeline.io_utils import (
        sha256_file as selection_sha256_file,
        sha256_tree,
    )
    from case_pair_selection_pipeline.pipeline import (
        prepare_run as prepare_selection_run,
        verify_run as verify_selection_run,
    )
from positive_pair_labeling_pipeline.artifacts import SELECTION  # noqa: E402
from positive_pair_labeling_pipeline.handoff import initialize_run  # noqa: E402
from positive_pair_labeling_pipeline.io_utils import load_json  # noqa: E402
from positive_pair_labeling_pipeline.pipeline import verify_run  # noqa: E402


def _xml(pmc_id: str, *, doi: str, reference_doi: str | None = None) -> str:
    citation = ""
    references = ""
    if reference_doi is not None:
        citation = '<p>Related report <xref ref-type="bibr" rid="R1">1</xref>.</p>'
        references = (
            '<ref-list><ref id="R1"><element-citation>'
            '<article-title>Referenced case report</article-title>'
            f'<pub-id pub-id-type="doi">{reference_doi}</pub-id>'
            '</element-citation></ref></ref-list>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<article><front><article-meta>'
        f'<article-id pub-id-type="pmc">PMC{pmc_id}</article-id>'
        f'<article-id pub-id-type="doi">{doi}</article-id>'
        f'<title-group><article-title>Case {pmc_id}</article-title></title-group>'
        f'<abstract><p>Abstract {pmc_id}.</p></abstract>'
        '</article-meta></front><body>'
        f'<sec><title>Case presentation</title><p>Description {pmc_id}.</p></sec>'
        f'{citation}</body><back>{references}</back></article>'
    )


class PipelineBoundaryIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        SELECTION_RUNTIME_AVAILABLE,
        "Pipeline 02 pandas/lxml dependencies are not installed",
    )
    def test_real_pipeline02_handoff_initializes_pipeline03_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retrieval = root / "retrieval"
            retrieval.mkdir()
            rows = []
            for pmc_id, similar, doi, reference_doi in (
                ("2", ["20-1"], "10.1000/case2", "10.1000/case20"),
                ("20", [], "10.1000/case20", None),
            ):
                folder = retrieval / "new_data" / f"PMC{pmc_id}" / f"{pmc_id}_1"
                folder.mkdir(parents=True)
                stem = f"{pmc_id}_1_1"
                image = folder / f"{stem}.jpg"
                caption = folder / f"{stem}.txt"
                Image.new("RGB", (16, 16), color=(0, 0, 0)).save(
                    image,
                    format="JPEG",
                )
                caption.write_text(f"CT caption {pmc_id}.", encoding="utf-8")
                (folder / "article.nxml").write_text(
                    _xml(pmc_id, doi=doi, reference_doi=reference_doi),
                    encoding="utf-8",
                )
                rows.append(
                    {
                        "image_path": image.relative_to(retrieval).as_posix(),
                        "caption_path": caption.relative_to(retrieval).as_posix(),
                        "patient_uid": f"{pmc_id}-1",
                        "pmc_id": pmc_id,
                        "article_path": (
                            f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmc_id}/"
                        ),
                        "unique_articles_sim_patients": json.dumps(similar),
                    }
                )
            merged = retrieval / "04_merged.csv"
            with merged.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            unified = retrieval / "00_unified_patients.csv"
            unified.write_text("patient_uid\n2-1\n20-1\n", encoding="utf-8")
            (retrieval / "run_manifest.json").write_text(
                json.dumps(
                    {
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
                            "00_unified_patients.csv": selection_sha256_file(unified),
                            "04_merged.csv": selection_sha256_file(merged),
                            "new_data/": sha256_tree(retrieval / "new_data"),
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            selection_config = root / "selection.toml"
            selection_config.write_text(
                "schema_version = 1\n[selection]\n"
                "title_fallback = false\n"
                "title_min_chars = 24\n"
                "title_min_tokens = 4\n"
                "[species_filter]\n"
                "ruleset = \"human_animal_v1\"\n",
                encoding="utf-8",
            )
            selection_run = root / "selection-run"
            synthetic_runtime = {
                "executable": "/synthetic/tesseract",
                "executable_sha256": "a" * 64,
                "version": "synthetic-tesseract-1",
                "version_output": "synthetic-tesseract-1",
                "language": "eng",
                "traineddata_path": "/synthetic/eng.traineddata",
                "traineddata_sha256": "b" * 64,
            }
            with (
                patch(
                    "case_pair_selection_pipeline.pipeline.tesseract_runtime_metadata",
                    return_value=synthetic_runtime,
                ),
                patch(
                    "case_pair_selection_pipeline.diagrams._ocr_text",
                    return_value="",
                ),
            ):
                prepare_selection_run(
                    retrieval,
                    selection_run,
                    selection_config,
                    workers=1,
                )
            self.assertEqual(verify_selection_run(selection_run)["eligible_pairs"], 1)
            before = {
                path.relative_to(selection_run).as_posix(): selection_sha256_file(path)
                for path in selection_run.rglob("*")
                if path.is_file()
            }

            labeling_config = root / "labeling.toml"
            labeling_config.write_text(
                "schema_version = 1\n[batch]\n"
                "max_shard_bytes = 10000\nmax_shard_requests = 2\n"
                "[models.citation]\nmodel = \"test-2025-01-01\"\n"
                "[models.question]\nmodel = \"test-2025-01-01\"\n"
                "[models.answer]\nmodel = \"test-2025-01-01\"\n"
                "[models.positive]\nmodel = \"test-2025-01-01\"\n",
                encoding="utf-8",
            )
            labeling_run = root / "labeling-run"
            initialize_run(selection_run, labeling_run, labeling_config)
            selection = load_json(labeling_run / SELECTION)
            self.assertEqual(selection["selected_count"], 1)
            self.assertEqual(selection["pairs"][0]["case_a_uid"], "2-1")
            self.assertEqual(verify_run(labeling_run)["selected_pairs"], 1)

            after = {
                path.relative_to(selection_run).as_posix(): selection_sha256_file(path)
                for path in selection_run.rglob("*")
                if path.is_file()
            }
            self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
