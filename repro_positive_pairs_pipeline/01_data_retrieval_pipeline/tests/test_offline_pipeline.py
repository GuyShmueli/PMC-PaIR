from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import requests
from PIL import Image

from retrieval_pipeline.extraction import extract_article_archive
from retrieval_pipeline.io_utils import sha256_tree
from retrieval_pipeline.oa import resolve_oa_packages
from retrieval_pipeline.patients import prepare_patient_cohort
from retrieval_pipeline.run import (
    run_pipeline,
    run_pipeline_from_local_sources,
)
from retrieval_pipeline.unification import materialize_unified_patient_input

from conftest import (
    LocalSources,
    PIPELINE_ROOT,
    _image_bytes,
    _tar_gz_bytes,
)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_offline_end_to_end(
    local_sources: LocalSources,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbid_network(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("offline pipeline attempted an HTTP request")

    monkeypatch.setattr(requests.sessions.Session, "request", forbid_network)
    run_dir = tmp_path / "retrieval run"

    manifest = run_pipeline_from_local_sources(
        local_sources.input_csv,
        run_dir,
        oa_responses_dir=local_sources.oa_responses_dir,
        archives_dir=local_sources.archives_dir,
        checkpoint_every=1,
    )

    assert manifest["pipeline_version"] == "3.1.0"
    assert manifest["dataset_contract"] == {
        "schema_version": 1,
        "mode": "unified",
        "materialized_input_count": 1,
        "materialized_input": "00_unified_patients.csv",
        "merged_csv": "04_merged.csv",
        "assets_dir": "new_data",
    }
    assert manifest["stage_summaries"]["00"]["source_file_count"] == 1
    assert manifest["artifact_sha256"]["00_unified_patients.csv"] == _sha256(
        run_dir / "00_unified_patients.csv"
    )
    assert manifest["artifact_sha256"]["04_merged.csv"] == _sha256(
        run_dir / "04_merged.csv"
    )
    assert manifest["artifact_sha256"]["new_data/"] == sha256_tree(
        run_dir / "new_data"
    )

    # Corrected semantics: PMC300 remains multi-patient even though one of its
    # two rows has an empty similarity mapping.
    assert _load_json(run_dir / "01_unique_article_ids.json") == ["100", "200"]
    cohort = pd.read_csv(run_dir / "01_unique_articles.csv", dtype=str)
    assert cohort["patient_uid"].tolist() == ["100-1", "200-1"]
    assert cohort["unique_articles_sim_patients"].tolist() == [
        '["200-1"]',
        '["100-1"]',
    ]

    assert manifest["stage_summaries"]["02"]["packages_resolved"] == 2
    assert manifest["stage_summaries"]["02"]["failures"] == 0
    assert manifest["stage_summaries"]["03"]["articles_completed"] == 2
    assert manifest["stage_summaries"]["03"]["figure_rows"] == 2
    assert manifest["stage_summaries"]["03"]["figure_failures"] == 1
    assert manifest["stage_summaries"]["04"]["candidate_patient_pairs"] == 1

    merged = pd.read_csv(
        run_dir / "04_merged.csv",
        dtype={"patient_uid": str, "pmc_id": str},
    )
    assert merged["patient_uid"].tolist() == ["100-1", "200-1"]
    assert merged["pmc_id"].tolist() == ["100", "200"]
    assert merged["image_path"].tolist() == [
        "new_data/PMC100/100_1/100_1_1.jpg",
        "new_data/PMC200/200_1/200_1_1.jpg",
    ]

    expected_captions = {
        "100": "Alpha case – café.",
        "200": "Beta case caption.",
    }
    for pmc_id, expected_caption in expected_captions.items():
        patient_dir = run_dir / "new_data" / f"PMC{pmc_id}" / f"{pmc_id}_1"
        image_path = patient_dir / f"{pmc_id}_1_1.jpg"
        caption_path = patient_dir / f"{pmc_id}_1_1.txt"
        assert (patient_dir / "article.nxml").is_file()
        assert caption_path.read_text(encoding="utf-8") == expected_caption
        with Image.open(image_path) as image:
            assert image.format == "JPEG"
            assert image.mode == "RGB"

    figure_failures = _load_json(run_dir / "03_figure_failures.json")
    assert figure_failures == [
        {
            "figure_index": 2,
            "pmc_id": "100",
            "reason": "missing or empty caption/graphic href",
        }
    ]
    assert not (tmp_path / "escape.jpg").exists()



@pytest.mark.parametrize(
    "script_name",
    ["run_pipeline.py", "01_prepare_articles.py"],
)
def test_cli_rejects_repeated_input_csv(
    script_name: str,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "must not be created"
    completed = subprocess.run(
        [
            sys.executable,
            str(PIPELINE_ROOT / "scripts" / script_name),
            "--input-csv",
            str(tmp_path / "first.csv"),
            "--input-csv",
            str(tmp_path / "second.csv"),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "--input-csv may be specified only once" in completed.stderr
    assert not output_dir.exists()


def test_resume_is_idempotent_and_does_not_refetch_completed_articles(
    local_sources: LocalSources,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "resume run"
    oa_calls: list[str] = []
    archive_calls: list[str] = []

    def fetch_xml(pmc_id: str) -> bytes:
        oa_calls.append(pmc_id)
        return (local_sources.oa_responses_dir / f"PMC{pmc_id}.xml").read_bytes()

    def fetch_archive(record: dict[str, str]) -> bytes:
        archive_calls.append(record["pmc_id"])
        return (local_sources.archives_dir / record["archive_name"]).read_bytes()

    first_manifest = run_pipeline(
        local_sources.input_csv,
        run_dir,
        oa_fetch_xml=fetch_xml,
        archive_fetcher=fetch_archive,
        checkpoint_every=1,
        oa_source_label="test OA cache",
        archive_source_label="test archive cache",
    )
    assert oa_calls == ["100", "200"]
    assert archive_calls == ["100", "200"]

    stable_artifacts = [
        "00_unified_patients.csv",
        "00_unification_provenance.csv",
        "00_unification_report.json",
        "01_unique_articles.csv",
        "01_unique_article_ids.json",
        "02_files_info.json",
        "02_failed_ids.json",
        "03_images_captions.csv",
        "03_article_failures.json",
        "03_figure_failures.json",
        "03_extraction_state.json",
        "04_merged.csv",
        "04_image_paths.json",
        "04_handoff_report.json",
    ]
    before = {name: _sha256(run_dir / name) for name in stable_artifacts}
    assets_before = sha256_tree(run_dir / "new_data")
    entire_run_before = sha256_tree(run_dir)

    def unexpected_oa_fetch(pmc_id: str) -> bytes:
        raise AssertionError(f"resume re-fetched OA metadata for PMC{pmc_id}")

    def unexpected_archive_fetch(record: dict[str, str]) -> bytes:
        raise AssertionError(
            f"resume re-fetched archive for PMC{record['pmc_id']}"
        )

    resumed = run_pipeline(
        local_sources.input_csv,
        run_dir,
        oa_fetch_xml=unexpected_oa_fetch,
        archive_fetcher=unexpected_archive_fetch,
        resume=True,
        checkpoint_every=1,
        oa_source_label="test OA cache",
        archive_source_label="test archive cache",
    )

    assert resumed == first_manifest
    assert {name: _sha256(run_dir / name) for name in stable_artifacts} == before
    assert sha256_tree(run_dir / "new_data") == assets_before
    assert sha256_tree(run_dir) == entire_run_before
    assert len(pd.read_csv(run_dir / "03_images_captions.csv")) == 2
    assert len(pd.read_csv(run_dir / "04_merged.csv")) == 2


def test_resume_contract_failures_are_read_only_and_precede_network(
    local_sources: LocalSources,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "immutable completed run"

    def fetch_xml(pmc_id: str) -> bytes:
        return (local_sources.oa_responses_dir / f"PMC{pmc_id}.xml").read_bytes()

    def fetch_archive(record: dict[str, str]) -> bytes:
        return (local_sources.archives_dir / record["archive_name"]).read_bytes()

    run_pipeline(
        local_sources.input_csv,
        run_dir,
        oa_fetch_xml=fetch_xml,
        archive_fetcher=fetch_archive,
        checkpoint_every=1,
        oa_source_label="test OA cache",
        archive_source_label="test archive cache",
    )
    completed_tree_sha256 = sha256_tree(run_dir)

    changed_input = tmp_path / "changed patients input.csv"
    changed_df = pd.read_csv(local_sources.input_csv)
    changed_df.loc[0, "source_note"] = "content changed after completed run"
    changed_df.to_csv(changed_input, index=False)

    oa_calls: list[str] = []
    archive_calls: list[str] = []

    def record_unexpected_oa(pmc_id: str) -> bytes:
        oa_calls.append(pmc_id)
        raise AssertionError(f"invalid resume queried OA metadata for PMC{pmc_id}")

    def record_unexpected_archive(record: dict[str, str]) -> bytes:
        archive_calls.append(record["pmc_id"])
        raise AssertionError(
            f"invalid resume fetched archive for PMC{record['pmc_id']}"
        )

    with pytest.raises(
        ValueError,
        match=r"Cannot resume: canonical --input-csv content has changed",
    ):
        run_pipeline(
            changed_input,
            run_dir,
            oa_fetch_xml=record_unexpected_oa,
            archive_fetcher=record_unexpected_archive,
            resume=True,
            checkpoint_every=1,
            oa_source_label="test OA cache",
            archive_source_label="test archive cache",
        )
    assert oa_calls == []
    assert archive_calls == []
    assert sha256_tree(run_dir) == completed_tree_sha256

    # Integrity failure is also read-only: preserve the already-corrupted tree
    # exactly, so a failed resume cannot obscure forensic evidence or partially
    # rewrite later-stage artifacts.
    article_ids_path = run_dir / "01_unique_article_ids.json"
    article_ids_path.write_text("[]\n", encoding="utf-8")
    corrupted_tree_sha256 = sha256_tree(run_dir)
    assert corrupted_tree_sha256 != completed_tree_sha256

    with pytest.raises(
        ValueError,
        match=r"Cannot resume: Stage 01 artifact integrity failed",
    ):
        run_pipeline(
            local_sources.input_csv,
            run_dir,
            oa_fetch_xml=record_unexpected_oa,
            archive_fetcher=record_unexpected_archive,
            resume=True,
            checkpoint_every=1,
            oa_source_label="test OA cache",
            archive_source_label="test archive cache",
        )
    assert oa_calls == []
    assert archive_calls == []
    assert sha256_tree(run_dir) == corrupted_tree_sha256


def test_stage_2_failure_is_retried_and_full_resume_extends_stage_3(
    local_sources: LocalSources,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "recovering run"
    first_oa_calls: list[str] = []
    first_archive_calls: list[str] = []

    def initially_flaky_oa(pmc_id: str) -> bytes:
        first_oa_calls.append(pmc_id)
        if pmc_id == "200":
            raise TimeoutError("simulated metadata timeout")
        return (local_sources.oa_responses_dir / f"PMC{pmc_id}.xml").read_bytes()

    def first_archive_fetch(record: dict[str, str]) -> bytes:
        first_archive_calls.append(record["pmc_id"])
        return (local_sources.archives_dir / record["archive_name"]).read_bytes()

    first = run_pipeline(
        local_sources.input_csv,
        run_dir,
        oa_fetch_xml=initially_flaky_oa,
        archive_fetcher=first_archive_fetch,
        checkpoint_every=1,
        oa_source_label="recoverable OA fixture",
        archive_source_label="local archive fixture",
    )
    assert first_oa_calls == ["100", "200"]
    assert first_archive_calls == ["100"]
    assert first["stage_summaries"]["02"]["packages_resolved"] == 1
    assert first["stage_summaries"]["02"]["failures"] == 1
    assert first["stage_summaries"]["03"]["articles_completed"] == 1
    assert len(pd.read_csv(run_dir / "04_merged.csv")) == 1

    resumed_oa_calls: list[str] = []
    resumed_archive_calls: list[str] = []

    def recovered_oa(pmc_id: str) -> bytes:
        resumed_oa_calls.append(pmc_id)
        return (local_sources.oa_responses_dir / f"PMC{pmc_id}.xml").read_bytes()

    def resumed_archive_fetch(record: dict[str, str]) -> bytes:
        resumed_archive_calls.append(record["pmc_id"])
        return (local_sources.archives_dir / record["archive_name"]).read_bytes()

    resumed = run_pipeline(
        local_sources.input_csv,
        run_dir,
        oa_fetch_xml=recovered_oa,
        archive_fetcher=resumed_archive_fetch,
        resume=True,
        checkpoint_every=1,
        oa_source_label="recoverable OA fixture",
        archive_source_label="local archive fixture",
    )

    # Successful metadata is not queried twice; only the formerly failed
    # article is resolved and extracted during the resumed all-stage run.
    assert resumed_oa_calls == ["200"]
    assert resumed_archive_calls == ["200"]
    assert resumed["stage_summaries"]["02"]["packages_resolved"] == 2
    assert resumed["stage_summaries"]["02"]["failures"] == 0
    assert resumed["stage_summaries"]["03"]["articles_completed"] == 2
    assert resumed["stage_summaries"]["04"]["candidate_patient_pairs"] == 1
    assert len(pd.read_csv(run_dir / "04_merged.csv")) == 2
    assert _load_json(run_dir / "02_failed_ids.json") == []


@pytest.mark.parametrize("damage", ["deleted", "corrupted"])
def test_resume_fails_closed_if_a_completed_asset_is_damaged(
    local_sources: LocalSources,
    tmp_path: Path,
    damage: str,
) -> None:
    run_dir = tmp_path / f"damaged {damage} run"

    def fetch_xml(pmc_id: str) -> bytes:
        return (local_sources.oa_responses_dir / f"PMC{pmc_id}.xml").read_bytes()

    def fetch_archive(record: dict[str, str]) -> bytes:
        return (local_sources.archives_dir / record["archive_name"]).read_bytes()

    run_pipeline(
        local_sources.input_csv,
        run_dir,
        oa_fetch_xml=fetch_xml,
        archive_fetcher=fetch_archive,
        checkpoint_every=1,
        oa_source_label="test OA cache",
        archive_source_label="test archive cache",
    )

    image_path = run_dir / "new_data" / "PMC100" / "100_1" / "100_1_1.jpg"
    if damage == "deleted":
        image_path.unlink()
    else:
        image_path.write_bytes(b"not a JPEG")

    def unexpected_oa_fetch(pmc_id: str) -> bytes:
        raise AssertionError(f"integrity check queried OA metadata for PMC{pmc_id}")

    def unexpected_archive_fetch(record: dict[str, str]) -> bytes:
        raise AssertionError(
            f"integrity check downloaded archive for PMC{record['pmc_id']}"
        )

    with pytest.raises(
        (ValueError, RuntimeError),
        match=r"(?i)(resume|completed).*(integrity|asset|file)|"
        r"(integrity|asset).*(resume|completed)",
    ):
        run_pipeline(
            local_sources.input_csv,
            run_dir,
            oa_fetch_xml=unexpected_oa_fetch,
            archive_fetcher=unexpected_archive_fetch,
            resume=True,
            checkpoint_every=1,
            oa_source_label="test OA cache",
            archive_source_label="test archive cache",
        )


def test_duplicate_patient_uid_rows_are_rejected(tmp_path: Path) -> None:
    input_csv = tmp_path / "duplicate patients.csv"
    pd.DataFrame(
        [
            {
                "patient_uid": "700-1",
                "similar_patients": '{"800-1": 0.8}',
            },
            {
                "patient_uid": "700_1",
                "similar_patients": '{"800-1": 0.8}',
            },
            {
                "patient_uid": "800-1",
                "similar_patients": '{"700-1": 0.8}',
            },
        ]
    ).to_csv(input_csv, index=False)

    with pytest.raises(ValueError, match="duplicate patient_uid"):
        prepare_patient_cohort(input_csv, tmp_path / "output")


def test_multiple_patient_source_files_are_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame(
        [
            {
                "patient_uid": "100-1",
                "similar_patients": "{}",
            }
        ]
    ).to_csv(first, index=False)
    pd.DataFrame(
        [
            {
                "patient_uid": "100-1",
                "similar_patients": "{}",
            }
        ]
    ).to_csv(second, index=False)

    with pytest.raises(ValueError, match="exactly one canonical unified"):
        materialize_unified_patient_input([first, second], tmp_path / "output")


def test_single_source_duplicates_preserve_metadata_and_relationships(tmp_path: Path) -> None:
    input_csv = tmp_path / "patients.csv"
    pd.DataFrame([
        {"patient_uid": "PMC100_1", "similar_patients": '{"200-1":1}',
         "code": "001", "note": "first"},
        {"patient_uid": "100-1", "similar_patients": '{"200-1":2,"300-1":3}',
         "code": "002", "note": "NA"},
        {"patient_uid": "200-1", "similar_patients": "{}", "code": "003", "note": ""},
        {"patient_uid": "300-1", "similar_patients": "{}", "code": "004", "note": "null"},
    ]).to_csv(input_csv, index=False)
    output_dir = tmp_path / "run"

    report = materialize_unified_patient_input(input_csv, output_dir)
    prepare_patient_cohort(output_dir / "00_unified_patients.csv", output_dir)
    cohort = pd.read_csv(output_dir / "01_unique_articles.csv", dtype=str, keep_default_na=False)
    assert cohort["patient_uid"].tolist() == ["100-1", "200-1", "300-1"]
    assert cohort["code"].tolist() == ["002", "003", "004"]
    assert cohort["note"].tolist() == ["NA", "", "null"]
    assert json.loads(cohort.loc[0, "similar_patients"]) == {"200-1": 2, "300-1": 3}
    assert report["duplicate_patient_uid_count"] == 1
    assert report["similarity_score_conflict_count"] == 1
    provenance = pd.read_csv(output_dir / "00_unification_provenance.csv")
    assert json.loads(provenance.loc[0, "source_csv_rows"]) == [
        {"source_index": 0, "csv_row": 2}, {"source_index": 0, "csv_row": 3},
    ]


def test_caption_blocks_and_figure_order_survive_handoff(
    local_sources: LocalSources, tmp_path: Path,
) -> None:
    figures = "".join(
        f'<fig><caption><title>Panel {number}.</title>'
        '<p>Car<italic>diac</italic> image.</p><p>Follow-up.</p></caption>'
        '<graphic href="image.png"/></fig>'
        for number in range(1, 12)
    )
    (local_sources.archives_dir / "PMC100.tar.gz").write_bytes(_tar_gz_bytes({
        "PMC100/article.nxml": f"<article><body>{figures}</body></article>".encode(),
        "PMC100/image.png": _image_bytes("PNG"),
    }))
    output_dir = tmp_path / "run"
    run_pipeline_from_local_sources(
        local_sources.input_csv, output_dir,
        oa_responses_dir=local_sources.oa_responses_dir,
        archives_dir=local_sources.archives_dir,
    )
    for filename in ("03_images_captions.csv", "04_merged.csv"):
        rows = pd.read_csv(output_dir / filename, dtype=str)
        article_rows = rows.loc[rows["pmc_id"] == "100"]
        assert [Path(path).stem for path in article_rows["image_path"]] == [
            f"100_1_{number}" for number in range(1, 12)
        ]
    assert (output_dir / article_rows.iloc[0]["caption_path"]).read_text() == (
        "Panel 1. Cardiac image. Follow-up."
    )


def test_default_cohort_keeps_exact_target_only_endpoint(
    tmp_path: Path,
) -> None:
    input_csv = tmp_path / "one-way similarities.csv"
    pd.DataFrame(
        [
            {
                "patient_uid": "100-1",
                "similar_patients": json.dumps(
                    {
                        "200-1": 0.9,
                        # The PMCID exists, but this exact case UID does not.
                        "200-2": 0.8,
                    }
                ),
            },
            {
                "patient_uid": "200-1",
                "similar_patients": "{}",
            },
        ]
    ).to_csv(input_csv, index=False)

    default_output = tmp_path / "default all rows"
    default_stats = prepare_patient_cohort(input_csv, default_output)
    default_cohort = pd.read_csv(
        default_output / "01_unique_articles.csv",
        dtype=str,
    )

    # The default cohort is endpoint-closed: an inbound-only exact target is
    # retained so its figures can be retrieved and the downstream pair exists.
    assert default_cohort["patient_uid"].tolist() == ["100-1", "200-1"]
    assert default_cohort["similar_patients"].tolist() == [
        '{"200-1":0.9}',
        "{}",
    ]
    assert default_cohort["unique_articles_sim_patients"].tolist() == [
        '["200-1"]',
        "[]",
    ]
    assert "200-2" not in default_cohort.to_csv(index=False)
    assert _load_json(default_output / "01_unique_article_ids.json") == [
        "100",
        "200",
    ]
    assert default_stats["relationship_source_rows_output"] == 1
    assert default_stats["referenced_target_only_rows_output"] == 1
    assert default_stats["rows_output"] == 2
    assert default_stats["article_ids_output"] == 2
    assert default_stats["similarity_links_output"] == 1

def test_nxml_with_explicit_mismatched_pmc_id_rejects_article(
    tmp_path: Path,
) -> None:
    nxml = b"""<?xml version="1.0" encoding="UTF-8"?>
<article>
  <front><article-meta>
    <article-id pub-id-type="pmc">PMC999</article-id>
  </article-meta></front>
  <sub-article><front><article-meta>
    <article-id pub-id-type="pmc">PMC700</article-id>
  </article-meta></front></sub-article>
</article>
"""
    archive = _tar_gz_bytes({"PMC700/article.nxml": nxml})
    output_dir = tmp_path / "mismatched identity"

    with pytest.raises(
        ValueError,
        match=r"NXML identity mismatch for PMC700.*999",
    ):
        extract_article_archive(
            {"pmc_id": "700"},
            archive,
            output_dir,
        )

    assert not (
        output_dir / "new_data" / "PMC700" / "700_1"
    ).exists()
    assert not list(
        (output_dir / "new_data" / "PMC700").glob(".700_1.*")
    )


def test_equally_ranked_graphic_candidates_become_figure_failure(
    tmp_path: Path,
) -> None:
    nxml = b"""<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front><article-meta>
    <article-id pub-id-type="pmcid">PMC700</article-id>
  </article-meta></front>
  <body><fig id="F1">
    <caption><p>Ambiguous image candidate.</p></caption>
    <graphic xlink:href="figure"/>
  </fig></body>
</article>
"""
    archive = _tar_gz_bytes(
        {
            "PMC700/article.nxml": nxml,
            "PMC700/figure.jpg": _image_bytes("JPEG"),
            "PMC700/figure.png": _image_bytes("PNG"),
        }
    )

    rows, failures, state = extract_article_archive(
        {"pmc_id": "700"},
        archive,
        tmp_path / "ambiguous image",
    )

    assert rows == []
    assert state["figures_in_nxml"] == 1
    assert state["figure_rows"] == 0
    assert state["figure_failures"] == 1
    assert len(failures) == 1
    assert failures[0]["pmc_id"] == "700"
    assert failures[0]["figure_index"] == 1
    assert failures[0]["graphic_hrefs"] == ["figure"]
    assert failures[0]["reason"] == "referenced image not found in archive"
    assert len(failures[0]["resolution_errors"]) == 1
    ambiguity = failures[0]["resolution_errors"][0]
    assert "ambiguous graphic href 'figure'" in ambiguity
    assert "PMC700/figure.jpg" in ambiguity
    assert "PMC700/figure.png" in ambiguity

    patient_dir = (
        tmp_path / "ambiguous image" / "new_data" / "PMC700" / "700_1"
    )
    assert (patient_dir / "article.nxml").is_file()
    assert not list(patient_dir.glob("*.jpg"))
    assert not list(patient_dir.glob("*.txt"))


def test_stage_2_resume_reconciles_success_with_stale_failure(
    local_sources: LocalSources,
    tmp_path: Path,
) -> None:
    ids_path = tmp_path / "article ids.json"
    ids_path.write_text('["100"]\n', encoding="utf-8")
    output_dir = tmp_path / "metadata checkpoint"

    def fetch_xml(pmc_id: str) -> bytes:
        return (local_sources.oa_responses_dir / f"PMC{pmc_id}.xml").read_bytes()

    initial = resolve_oa_packages(
        ids_path,
        output_dir,
        fetch_xml=fetch_xml,
        checkpoint_every=1,
        source_label="test OA cache",
    )
    assert initial["packages_resolved"] == 1
    packages_before = (output_dir / "02_files_info.json").read_bytes()

    # Simulate interruption between atomic replacements: the authoritative
    # success file is current, while the failure file still contains PMC100.
    (output_dir / "02_failed_ids.json").write_text(
        json.dumps(
            [
                {
                    "pmc_id": "100",
                    "error_type": "TimeoutError",
                    "message": "stale checkpoint failure",
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def unexpected_fetch(pmc_id: str) -> bytes:
        raise AssertionError(f"reconciliation re-fetched PMC{pmc_id}")

    resumed = resolve_oa_packages(
        ids_path,
        output_dir,
        fetch_xml=unexpected_fetch,
        resume=True,
        checkpoint_every=1,
        source_label="test OA cache",
    )

    assert resumed["packages_resolved"] == 1
    assert resumed["failures"] == 0
    assert (output_dir / "02_files_info.json").read_bytes() == packages_before
    assert _load_json(output_dir / "02_failed_ids.json") == []


def test_run_refuses_nonempty_output_directory_before_writing(
    local_sources: LocalSources,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "nonempty output"
    output_dir.mkdir()
    sentinel = output_dir / "belongs-to-user.txt"
    sentinel.write_text("preserve me\n", encoding="utf-8")

    def unexpected_oa_fetch(pmc_id: str) -> bytes:
        raise AssertionError(f"preflight queried OA metadata for PMC{pmc_id}")

    def unexpected_archive_fetch(record: dict[str, str]) -> bytes:
        raise AssertionError(
            f"preflight fetched archive for PMC{record['pmc_id']}"
        )

    with pytest.raises(FileExistsError, match="Output directory is not empty"):
        run_pipeline(
            local_sources.input_csv,
            output_dir,
            oa_fetch_xml=unexpected_oa_fetch,
            archive_fetcher=unexpected_archive_fetch,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve me\n"
    assert list(output_dir.iterdir()) == [sentinel]
