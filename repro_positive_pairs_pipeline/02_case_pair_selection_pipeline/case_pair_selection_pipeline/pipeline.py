from __future__ import annotations

import hashlib
import math
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .artifacts import (
    CASES,
    CITATIONS,
    DIAGRAM_BLACKLIST,
    DIAGRAM_CLASSIFICATIONS,
    DIAGRAM_STATS,
    ELIGIBLE_CASES,
    ELIGIBLE_PAIRS,
    HANDOFF_MANIFEST,
    HANDOFF_SCHEMA_VERSION,
    PAIRS,
    SELECTION,
    PAIR_SPECIES,
    SPECIES_CLASSIFICATIONS,
    SPECIES_STATS,
    STAGE_1_STATS,
    STAGE_2_STATS,
    STAGE_3_STATS,
    SUMMARIES,
    UPSTREAM_ASSETS,
    UPSTREAM_DATASET_CONTRACT_SCHEMA_VERSION,
    UPSTREAM_DATASET_MODE,
    UPSTREAM_MATERIALIZED_INPUT_COUNT,
    UPSTREAM_MERGED,
    UPSTREAM_PATIENTS,
    UPSTREAM_PIPELINE_VERSION,
)
from .candidates import build_candidates
from .citations import match_pair_citations
from .config import PIPELINE_ID, PIPELINE_VERSION, load_config
from .diagrams import (
    build_diagram_filter,
    tesseract_runtime_contract,
    tesseract_runtime_metadata,
)
from .errors import ArtifactIntegrityError
from .io_utils import (
    file_metadata,
    load_json,
    load_jsonl,
    save_json,
    save_jsonl,
    sha256_file,
    sha256_tree,
    verify_file_metadata,
)
from .manifest import (
    external_input_metadata,
    new_manifest,
    record_stage,
    save_manifest,
    verify_manifest,
)
from .summaries import extract_case_summaries
from .species import RULESETS, classify_cases, matched_rule_vocabulary


_CITATION_FIELDS = (
    "status",
    "direction",
    "citing_uid",
    "cited_uid",
    "match_type",
    "ref_id",
    "citing_title",
    "citing_abstract",
    "citation_paragraphs",
    "cited_title",
    "cited_abstract",
)
_HANDOFF_CONTRACTS = {
    "case_schema_version": 2,
    "pair_schema_version": 1,
    "case_order": "patient_uid_numeric_ascending",
    "pair_order": "pair_index_ascending",
    "pair_indices": "stable_non_contiguous",
    "case_orientation": "canonical_numeric_uid_order",
    "population_filter": "complete_case_pair_human_animal_screen_required",
}


def _validate_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer when supplied")
    return limit


def _validate_workers(workers: int) -> int:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    return workers


def _uid_sort_key(uid: str) -> tuple[int, int, str]:
    parts = uid.split("-")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ArtifactIntegrityError(f"Invalid canonical patient UID: {uid!r}")
    return int(parts[0]), int(parts[1]), uid


def _new_output_directory(output_dir: str | Path) -> Path:
    run_dir = Path(output_dir)
    if run_dir.exists():
        if not run_dir.is_dir():
            raise FileExistsError(f"Output path is not a directory: {run_dir}")
        if any(run_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty; choose a new run directory: {run_dir}"
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _require_output_outside_retrieval(
    retrieval_run: str | Path,
    output_dir: str | Path,
) -> None:
    retrieval_root = Path(retrieval_run).resolve()
    destination = Path(output_dir).resolve()
    try:
        destination.relative_to(retrieval_root)
    except ValueError:
        return
    raise ValueError(
        "Output directory must not equal or be nested inside the upstream "
        f"retrieval run: {destination}"
    )


def _validate_upstream_run(
    retrieval_run: Path,
    *,
    allow_upstream_failures: bool,
) -> dict[str, Any]:
    manifest_path = retrieval_run / "run_manifest.json"
    unified_patients_path = retrieval_run / UPSTREAM_PATIENTS
    merged_path = retrieval_run / UPSTREAM_MERGED
    data_path = retrieval_run / UPSTREAM_ASSETS
    if not retrieval_run.is_dir():
        raise FileNotFoundError(f"Retrieval run is missing: {retrieval_run}")
    if (
        not manifest_path.is_file()
        or not unified_patients_path.is_file()
        or not merged_path.is_file()
        or not data_path.is_dir()
    ):
        raise FileNotFoundError(
            "Retrieval run must contain one unified Pipeline-01 handoff: "
            f"run_manifest.json, {UPSTREAM_PATIENTS}, {UPSTREAM_MERGED}, and "
            f"{UPSTREAM_ASSETS}/"
        )

    upstream = load_json(manifest_path)
    if not isinstance(upstream, dict):
        raise ArtifactIntegrityError("Upstream run manifest is not an object")
    if upstream.get("pipeline_version") != UPSTREAM_PIPELINE_VERSION:
        raise ArtifactIntegrityError(
            "Unsupported upstream pipeline version: "
            f"{upstream.get('pipeline_version')!r}"
        )
    dataset_contract = upstream.get("dataset_contract")
    if not isinstance(dataset_contract, dict):
        raise ArtifactIntegrityError(
            "Upstream manifest has no unified dataset_contract; split, "
            "delta-only, and manually combined retrieval trees are not accepted"
        )
    expected_contract = {
        "schema_version": UPSTREAM_DATASET_CONTRACT_SCHEMA_VERSION,
        "mode": UPSTREAM_DATASET_MODE,
        "materialized_input_count": UPSTREAM_MATERIALIZED_INPUT_COUNT,
        "materialized_input": UPSTREAM_PATIENTS,
        "merged_csv": UPSTREAM_MERGED,
        "assets_dir": UPSTREAM_ASSETS,
    }
    mismatches = {
        key: {"expected": expected, "actual": dataset_contract.get(key)}
        for key, expected in expected_contract.items()
        if (
            dataset_contract.get(key) != expected
            or type(dataset_contract.get(key)) is not type(expected)
        )
    }
    extra_keys = sorted(set(dataset_contract) - set(expected_contract))
    if extra_keys:
        mismatches["unexpected_keys"] = {
            "expected": [],
            "actual": extra_keys,
        }
    if mismatches:
        raise ArtifactIntegrityError(
            "Upstream dataset_contract is not the supported single unified "
            f"handoff: {mismatches}"
        )

    status = upstream.get("status")
    if status not in {"complete", "completed_with_failures"}:
        raise ArtifactIntegrityError(
            f"Upstream retrieval run is not complete (status={status!r})"
        )
    if status == "completed_with_failures" and not allow_upstream_failures:
        raise ArtifactIntegrityError(
            "Upstream retrieval completed with failures; inspect its "
            "failure_summary and pass allow_upstream_failures=True only after "
            "accepting the coverage loss"
        )

    upstream_hashes = upstream.get("artifact_sha256")
    if not isinstance(upstream_hashes, dict):
        raise ArtifactIntegrityError("Upstream artifact_sha256 table is missing")
    expected_files = (
        (UPSTREAM_PATIENTS, unified_patients_path),
        (UPSTREAM_MERGED, merged_path),
    )
    for name, path in expected_files:
        expected_hash = upstream_hashes.get(name)
        if not isinstance(expected_hash, str):
            raise ArtifactIntegrityError(
                f"Upstream run manifest does not record the {name} digest"
            )
        if sha256_file(path) != expected_hash:
            raise ArtifactIntegrityError(
                f"Upstream {name} changed after completion"
            )
    expected_tree_hash = upstream_hashes.get(f"{UPSTREAM_ASSETS}/")
    if not isinstance(expected_tree_hash, str):
        raise ArtifactIntegrityError(
            f"Upstream run manifest does not record the {UPSTREAM_ASSETS}/ "
            "tree digest"
        )
    if sha256_tree(data_path) != expected_tree_hash:
        raise ArtifactIntegrityError(
            f"Upstream {UPSTREAM_ASSETS}/ tree changed after completion"
        )
    return upstream


def _unique_by(
    records: list[dict[str, Any]],
    field: str,
    *,
    label: str,
) -> dict[Any, dict[str, Any]]:
    result: dict[Any, dict[str, Any]] = {}
    for position, record in enumerate(records):
        if not isinstance(record, dict) or field not in record:
            raise ArtifactIntegrityError(
                f"{label} record {position} has no {field!r} field"
            )
        key = record[field]
        if key in result:
            raise ArtifactIntegrityError(
                f"{label} contains duplicate {field} {key!r}"
            )
        result[key] = record
    return result


_SPECIES_LABELS = {"human", "animal"}


def _load_species_adjudications(
    path: str | Path | None,
) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    records = load_jsonl(path)
    result: dict[str, dict[str, Any]] = {}
    expected = {"schema_version", "patient_uid", "label", "reason"}
    for position, record in enumerate(records):
        if set(record) != expected or record.get("schema_version") != 1:
            raise ArtifactIntegrityError(
                f"Species adjudication row {position} has an invalid schema"
            )
        uid = record.get("patient_uid")
        if not isinstance(uid, str):
            raise ArtifactIntegrityError(
                f"Species adjudication row {position} has no patient_uid"
            )
        _uid_sort_key(uid)
        if uid in result:
            raise ArtifactIntegrityError(
                f"Species adjudications contain duplicate patient_uid {uid!r}"
            )
        label = record.get("label")
        reason = record.get("reason")
        if label not in _SPECIES_LABELS:
            raise ArtifactIntegrityError(
                f"Species adjudication {uid!r} has invalid label {label!r}"
            )
        if not isinstance(reason, str) or not reason.strip() or reason != reason.strip():
            raise ArtifactIntegrityError(
                f"Species adjudication {uid!r} requires a nonempty trimmed reason"
            )
        result[uid] = dict(record)
    return result


def _apply_species_adjudications(
    classifications: list[dict[str, Any]],
    adjudications: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_uid = _unique_by(
        classifications,
        "patient_uid",
        label="species classifications",
    )
    unknown = sorted(set(adjudications) - set(by_uid), key=_uid_sort_key)
    if unknown:
        raise ArtifactIntegrityError(
            f"Species adjudications reference unknown cases: {unknown[:20]}"
        )

    resolved: list[dict[str, Any]] = []
    for uid in sorted(by_uid, key=_uid_sort_key):
        record = dict(by_uid[uid])
        automatic = record.get("automatic_label")
        if automatic not in _SPECIES_LABELS:
            raise ArtifactIntegrityError(
                f"Species classification {uid!r} has invalid automatic_label"
            )
        adjudication = adjudications.get(uid)
        if adjudication is not None:
            record["adjudication"] = {
                "label": adjudication["label"],
                "reason": adjudication["reason"],
            }
            record["final_label"] = adjudication["label"]
        else:
            record["adjudication"] = None
            record["final_label"] = automatic
        if record["final_label"] not in _SPECIES_LABELS:
            raise ArtifactIntegrityError(
                f"Species classification {uid!r} has invalid final_label"
            )
        matched_rules = record.get("matched_rules")
        if (
            not isinstance(matched_rules, list)
            or any(not isinstance(item, str) or not item for item in matched_rules)
            or matched_rules != sorted(set(matched_rules))
        ):
            raise ArtifactIntegrityError(
                f"Species classification {uid!r} has invalid matched_rules"
            )
        resolved.append(record)
    return resolved


def _screen_species_pairs(
    pairs: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    *,
    ruleset: str,
    workers: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    by_uid = _unique_by(
        classifications,
        "patient_uid",
        label="species classifications",
    )
    required_uids = {
        pair[field]
        for pair in pairs
        for field in ("case_a_uid", "case_b_uid")
    }
    missing = sorted(required_uids - set(by_uid), key=_uid_sort_key)
    if missing:
        raise ArtifactIntegrityError(
            f"Species classifications are missing pair members: {missing[:20]}"
        )

    retained: list[dict[str, Any]] = []
    pair_species: list[dict[str, Any]] = []
    pair_counts: Counter[str] = Counter()
    for pair in sorted(pairs, key=lambda item: item["pair_index"]):
        endpoints = (
            ("case_a", pair["case_a_uid"]),
            ("case_b", pair["case_b_uid"]),
        )
        endpoint_labels = {
            endpoint: str(by_uid[uid]["final_label"])
            for endpoint, uid in endpoints
        }
        label = (
            "human"
            if set(endpoint_labels.values()) == {"human"}
            else "animal"
        )
        pair_counts[label] += 1
        if label == "human":
            retained.append(dict(pair))
        pair_species.append(
            {
                "schema_version": 1,
                "pair_index": pair["pair_index"],
                "pair_key": pair["pair_key"],
                "case_a_uid": pair["case_a_uid"],
                "case_b_uid": pair["case_b_uid"],
                "label": label,
                "endpoint_labels": endpoint_labels,
            }
        )
    automatic_counts = Counter(
        str(record["automatic_label"]) for record in classifications
    )
    final_counts = Counter(
        str(record["final_label"]) for record in classifications
    )
    rule_counts = Counter(
        reason
        for record in classifications
        for reason in record.get("matched_rules", [])
    )
    stats = {
        "schema_version": 1,
        "ruleset": ruleset,
        "workers": workers,
        "cases_screened": len(classifications),
        "automatic_label_counts": dict(sorted(automatic_counts.items())),
        "final_label_counts": dict(sorted(final_counts.items())),
        "cases_adjudicated": sum(
            record.get("adjudication") is not None for record in classifications
        ),
        "rule_counts": dict(sorted(rule_counts.items())),
        "pairs_before_screen": len(pairs),
        "pair_label_counts": dict(sorted(pair_counts.items())),
        "human_pairs": len(retained),
        "animal_pairs": len(pair_species) - len(retained),
    }
    if len(pair_species) != len(pairs):
        raise AssertionError("Human/animal pair screen is incomplete")
    if stats["human_pairs"] + stats["animal_pairs"] != len(pairs):
        raise AssertionError("Human/animal pair counts do not partition all pairs")
    return retained, pair_species, stats


def _make_handoff_records(
    *,
    selected: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases_by_uid = _unique_by(cases, "patient_uid", label="cases")
    summaries_by_uid = _unique_by(summaries, "patient_uid", label="summaries")
    citations_by_index = _unique_by(
        citations,
        "pair_index",
        label="citations",
    )
    classifications_by_uid = _unique_by(
        classifications,
        "patient_uid",
        label="species classifications",
    )
    selected_uids = {
        pair[field]
        for pair in selected
        for field in ("case_a_uid", "case_b_uid")
    }

    eligible_cases: list[dict[str, Any]] = []
    for uid in sorted(selected_uids, key=_uid_sort_key):
        case = cases_by_uid.get(uid)
        summary = summaries_by_uid.get(uid)
        classification = classifications_by_uid.get(uid)
        if case is None or summary is None or classification is None:
            raise ArtifactIntegrityError(
                f"Selected case {uid!r} is absent from cases, summaries, or species classifications"
            )
        if summary.get("status") != "valid" or not summary.get("summary"):
            raise ArtifactIntegrityError(
                f"Selected case {uid!r} has no valid case description"
            )
        captions = case.get("captions")
        if not isinstance(captions, list) or not captions:
            raise ArtifactIntegrityError(
                f"Selected case {uid!r} has no caption group"
            )
        if classification.get("final_label") != "human":
            raise ArtifactIntegrityError(
                f"Selected case {uid!r} is not labeled human"
            )
        eligible_cases.append(
            {
                "schema_version": 2,
                "patient_uid": uid,
                "pmc_id": case["pmc_id"],
                "nxml_path": case["nxml_path"],
                "nxml_sha256": case["nxml_sha256"],
                "case_description": summary["summary"],
                "captions": captions,
                "population_scope": {
                    "ruleset": classification["ruleset"],
                    "label": "human",
                },
            }
        )

    eligible_pairs: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for pair in sorted(selected, key=lambda item: item["pair_index"]):
        pair_index = pair["pair_index"]
        if pair_index in seen_indices:
            raise ArtifactIntegrityError(
                f"Selection contains duplicate pair_index {pair_index}"
            )
        seen_indices.add(pair_index)
        citation = citations_by_index.get(pair_index)
        if citation is None or citation.get("status") != "matched":
            raise ArtifactIntegrityError(
                f"Selected pair {pair_index} has no matched citation evidence"
            )
        missing = [field for field in _CITATION_FIELDS if field not in citation]
        if missing:
            raise ArtifactIntegrityError(
                f"Citation {pair_index} is missing handoff fields {missing}"
            )
        eligible_pairs.append(
            {
                "schema_version": 1,
                "pair_index": pair_index,
                "pair_key": pair["pair_key"],
                "case_a_uid": pair["case_a_uid"],
                "case_b_uid": pair["case_b_uid"],
                "citation": {
                    field: citation[field] for field in _CITATION_FIELDS
                },
            }
        )
    return eligible_cases, eligible_pairs


def _artifact_contract(
    path: Path,
    *,
    root: Path,
    records: int,
) -> dict[str, Any]:
    metadata = file_metadata(path, root=root)
    return {
        "path": metadata["path"],
        "records": records,
        "bytes": metadata["bytes"],
        "sha256": metadata["sha256"],
    }


def _write_handoff(
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    selected: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    pair_species: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible_cases, eligible_pairs = _make_handoff_records(
        selected=selected,
        cases=cases,
        summaries=summaries,
        citations=citations,
        classifications=classifications,
    )
    cases_path = run_dir / ELIGIBLE_CASES
    pairs_path = run_dir / ELIGIBLE_PAIRS
    save_jsonl(eligible_cases, cases_path)
    save_jsonl(eligible_pairs, pairs_path)
    handoff = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "producer": {
            "pipeline_id": PIPELINE_ID,
            "pipeline_version": PIPELINE_VERSION,
            "run_spec_id": manifest["run_spec_id"],
        },
        "artifacts": {
            "eligible_cases": _artifact_contract(
                cases_path,
                root=run_dir,
                records=len(eligible_cases),
            ),
            "eligible_pairs": _artifact_contract(
                pairs_path,
                root=run_dir,
                records=len(eligible_pairs),
            ),
            "species_classifications": _artifact_contract(
                run_dir / SPECIES_CLASSIFICATIONS,
                root=run_dir,
                records=len(classifications),
            ),
            "pair_species": _artifact_contract(
                run_dir / PAIR_SPECIES,
                root=run_dir,
                records=len(pair_species),
            ),
            "species_stats": _artifact_contract(
                run_dir / SPECIES_STATS,
                root=run_dir,
                records=1,
            ),
        },
        "contracts": dict(_HANDOFF_CONTRACTS),
    }
    save_json(handoff, run_dir / HANDOFF_MANIFEST)
    manifest["handoff_contract"] = handoff
    manifest["status"] = "complete"
    record_stage(
        run_dir,
        manifest,
        "03_handoff",
        status="complete",
        artifact_paths=(
            cases_path,
            pairs_path,
            run_dir / SPECIES_CLASSIFICATIONS,
            run_dir / PAIR_SPECIES,
            run_dir / SPECIES_STATS,
            run_dir / HANDOFF_MANIFEST,
        ),
        details={
            "eligible_cases": len(eligible_cases),
            "eligible_pairs": len(eligible_pairs),
        },
    )
    return handoff


def _prepare_run_in_directory(
    retrieval_run: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    *,
    species_adjudications: str | Path | None = None,
    limit: int | None = None,
    workers: int = 1,
    allow_upstream_failures: bool = False,
) -> dict[str, Any]:
    limit = _validate_limit(limit)
    workers = _validate_workers(workers)
    retrieval_root = Path(retrieval_run).resolve()
    _require_output_outside_retrieval(retrieval_root, output_dir)
    config_source = Path(config_path).resolve()
    adjudications_path = (
        Path(species_adjudications).resolve()
        if species_adjudications is not None
        else None
    )
    upstream = _validate_upstream_run(
        retrieval_root,
        allow_upstream_failures=allow_upstream_failures,
    )
    if adjudications_path is not None and not adjudications_path.is_file():
        raise FileNotFoundError(
            f"Species adjudications file is missing: {adjudications_path}"
        )
    config = load_config(config_source)
    run_dir = _new_output_directory(output_dir)

    inputs = [
        {
            **external_input_metadata(
                retrieval_root / "run_manifest.json",
                label="retrieval_run_manifest",
            ),
            "upstream_status": upstream["status"],
            "upstream_failure_summary": upstream.get("failure_summary", {}),
            "upstream_failures_accepted": bool(allow_upstream_failures),
            "upstream_dataset_contract": upstream["dataset_contract"],
        },
        external_input_metadata(
            retrieval_root / UPSTREAM_PATIENTS,
            label="retrieval_unified_patients",
        ),
        external_input_metadata(
            retrieval_root / UPSTREAM_MERGED,
            label="retrieval_merged_csv",
        ),
    ]
    if adjudications_path is not None:
        inputs.append(
            external_input_metadata(
                adjudications_path,
                label="species_adjudications",
            )
        )
    inputs.append(
        external_input_metadata(
            config_source,
            label="pipeline_config_source",
        )
    )
    diagram_runtime = tesseract_runtime_metadata(config.ocr_language)
    runtime_contract = tesseract_runtime_contract(diagram_runtime)
    manifest = new_manifest(
        config=config,
        inputs=inputs,
        limit=limit,
        workers=workers,
        diagram_filter_runtime=runtime_contract,
    )
    manifest["status"] = "preparing"
    save_manifest(run_dir, manifest)

    diagram_records, normalized_blacklist, diagram_stats = build_diagram_filter(
        retrieval_root,
        config=config,
        workers=workers,
        runtime_metadata=diagram_runtime,
    )
    save_jsonl(diagram_records, run_dir / DIAGRAM_CLASSIFICATIONS)
    save_json(normalized_blacklist, run_dir / DIAGRAM_BLACKLIST)
    save_json(diagram_stats, run_dir / DIAGRAM_STATS)
    record_stage(
        run_dir,
        manifest,
        "00_diagram_filter",
        status="complete",
        artifact_paths=(
            run_dir / DIAGRAM_CLASSIFICATIONS,
            run_dir / DIAGRAM_BLACKLIST,
            run_dir / DIAGRAM_STATS,
        ),
        details={
            "mode": diagram_stats["mode"],
            "images_input": diagram_stats["images_input"],
            "diagrams_excluded": diagram_stats["diagrams_excluded"],
            "images_retained": diagram_stats["images_retained"],
        },
    )

    cases, pairs, candidate_stats = build_candidates(
        retrieval_root,
        normalized_blacklist,
    )
    if candidate_stats["diagram_blacklist_entries"] != len(normalized_blacklist):
        raise ArtifactIntegrityError(
            "Candidate construction did not apply the complete diagram blacklist"
        )
    save_jsonl(cases, run_dir / CASES)
    save_jsonl(pairs, run_dir / PAIRS)
    save_json(candidate_stats, run_dir / STAGE_1_STATS)
    record_stage(
        run_dir,
        manifest,
        "01_candidates",
        status="complete",
        artifact_paths=(
            run_dir / CASES,
            run_dir / PAIRS,
            run_dir / STAGE_1_STATS,
        ),
        details={"cases": len(cases), "pairs": len(pairs)},
    )

    classifications = classify_cases(
        cases,
        retrieval_root,
        ruleset=config.species_ruleset,
        workers=workers,
    )
    classifications = _apply_species_adjudications(
        classifications,
        _load_species_adjudications(adjudications_path),
    )
    (
        human_pairs,
        pair_species,
        species_stats,
    ) = _screen_species_pairs(
        pairs,
        classifications,
        ruleset=config.species_ruleset,
        workers=workers,
    )
    save_jsonl(classifications, run_dir / SPECIES_CLASSIFICATIONS)
    save_jsonl(pair_species, run_dir / PAIR_SPECIES)
    save_json(species_stats, run_dir / SPECIES_STATS)
    record_stage(
        run_dir,
        manifest,
        "02_human_animal_screen",
        status="complete",
        artifact_paths=(
            run_dir / SPECIES_CLASSIFICATIONS,
            run_dir / PAIR_SPECIES,
            run_dir / SPECIES_STATS,
        ),
        details={
            "ruleset": config.species_ruleset,
            "cases_screened": len(classifications),
            "pairs_screened": len(pair_species),
            "human_pairs": len(human_pairs),
            "animal_pairs": len(pair_species) - len(human_pairs),
        },
    )

    human_uids = {
        pair[field]
        for pair in human_pairs
        for field in ("case_a_uid", "case_b_uid")
    }
    human_cases = [case for case in cases if case["patient_uid"] in human_uids]
    summaries = extract_case_summaries(
        human_cases,
        retrieval_root,
        workers=workers,
    )
    summary_statuses = Counter(
        str(record.get("status")) for record in summaries
    )
    summary_stats = {
        "schema_version": 1,
        "cases_processed": len(summaries),
        "statuses": dict(sorted(summary_statuses.items())),
        "workers": workers,
    }
    save_jsonl(summaries, run_dir / SUMMARIES)
    save_json(summary_stats, run_dir / STAGE_2_STATS)
    record_stage(
        run_dir,
        manifest,
        "02_summaries",
        status="complete",
        artifact_paths=(run_dir / SUMMARIES, run_dir / STAGE_2_STATS),
        details=summary_stats,
    )

    cases_by_uid = _unique_by(cases, "patient_uid", label="cases")
    summaries_by_uid = _unique_by(summaries, "patient_uid", label="summaries")

    citations = match_pair_citations(
        human_pairs,
        cases_by_uid,
        retrieval_root,
        title_fallback=config.title_fallback,
        title_min_chars=config.title_min_chars,
        title_min_tokens=config.title_min_tokens,
        workers=workers,
    )
    citations_by_index = _unique_by(
        citations,
        "pair_index",
        label="citations",
    )
    human_pairs_by_index = _unique_by(
        human_pairs,
        "pair_index",
        label="human pairs",
    )
    if set(citations_by_index) != set(human_pairs_by_index):
        raise ArtifactIntegrityError(
            "Citation records do not align with human pairs"
        )

    pair_species_by_index = _unique_by(
        pair_species,
        "pair_index",
        label="pair species records",
    )

    eligible: list[dict[str, Any]] = []
    exclusion_reasons: Counter[str] = Counter()
    for pair in sorted(pairs, key=lambda record: record["pair_index"]):
        species_record = pair_species_by_index[pair["pair_index"]]
        if species_record["label"] != "human":
            exclusion_reasons["animal_pair"] += 1
            continue
        citation = citations_by_index[pair["pair_index"]]
        if citation.get("status") != "matched":
            exclusion_reasons[
                f"citation_{citation.get('status', 'missing')}"
            ] += 1
            continue
        invalid_members = [
            uid
            for uid in (pair["case_a_uid"], pair["case_b_uid"])
            if summaries_by_uid[uid].get("status") != "valid"
        ]
        if invalid_members:
            exclusion_reasons["invalid_case_summary"] += 1
            continue
        if any(
            not cases_by_uid[uid].get("captions")
            for uid in (pair["case_a_uid"], pair["case_b_uid"])
        ):
            exclusion_reasons["empty_caption_group"] += 1
            continue
        eligible.append(dict(pair))

    selected = eligible if limit is None else eligible[:limit]
    selection = {
        "schema_version": 1,
        "eligibility_rule": (
            "pair species label=human AND citation_status=matched "
            "AND both case summaries valid AND both caption groups nonempty"
        ),
        "limit": limit,
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "pairs": selected,
    }
    citation_statuses = Counter(
        str(record.get("status")) for record in citations
    )
    selection_stats = {
        "schema_version": 1,
        "pairs_processed": len(pairs),
        "citation_statuses": dict(sorted(citation_statuses.items())),
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "limit": limit,
        "workers": workers,
        "title_fallback": config.title_fallback,
        "title_min_chars": config.title_min_chars,
        "title_min_tokens": config.title_min_tokens,
        "species_ruleset": config.species_ruleset,
        "pairs_after_human_animal_screen": len(human_pairs),
    }
    save_jsonl(citations, run_dir / CITATIONS)
    save_json(selection, run_dir / SELECTION)
    save_json(selection_stats, run_dir / STAGE_3_STATS)
    manifest["status"] = "building_handoff"
    record_stage(
        run_dir,
        manifest,
        "03_selection",
        status="complete",
        artifact_paths=(
            run_dir / CITATIONS,
            run_dir / SELECTION,
            run_dir / STAGE_3_STATS,
        ),
        details={
            "eligible_count": len(eligible),
            "selected_count": len(selected),
        },
    )
    _write_handoff(
        run_dir,
        manifest,
        selected=selected,
        cases=cases,
        summaries=summaries,
        citations=citations,
        classifications=classifications,
        pair_species=pair_species,
    )
    return manifest


def prepare_run(
    retrieval_run: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    *,
    species_adjudications: str | Path | None = None,
    limit: int | None = None,
    workers: int = 1,
    allow_upstream_failures: bool = False,
) -> dict[str, Any]:
    """Prepare a complete selection run and publish it atomically."""

    destination = Path(output_dir)
    _require_output_outside_retrieval(retrieval_run, destination)
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(
                f"Output path is not a directory: {destination}"
            )
        if any(destination.iterdir()):
            raise FileExistsError(
                "Output directory is not empty; choose a new run directory: "
                f"{destination}"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.preparing-",
            dir=destination.parent,
        )
    )
    try:
        manifest = _prepare_run_in_directory(
            retrieval_run,
            staging,
            config_path,
            species_adjudications=species_adjudications,
            limit=limit,
            workers=workers,
            allow_upstream_failures=allow_upstream_failures,
        )
        if destination.exists():
            if any(destination.iterdir()):
                raise FileExistsError(
                    "Output directory became nonempty during preparation: "
                    f"{destination}"
                )
            destination.rmdir()
        os.replace(staging, destination)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _validate_handoff_case(case: dict[str, Any]) -> None:
    expected_fields = {
        "schema_version",
        "patient_uid",
        "pmc_id",
        "nxml_path",
        "nxml_sha256",
        "case_description",
        "captions",
        "population_scope",
    }
    if set(case) != expected_fields or case.get("schema_version") != 2:
        raise ArtifactIntegrityError("Eligible case has an invalid schema")
    uid = case.get("patient_uid")
    if not isinstance(uid, str):
        raise ArtifactIntegrityError("Eligible case has no patient_uid")
    article, _, _ = _uid_sort_key(uid)
    if case.get("pmc_id") != str(article):
        raise ArtifactIntegrityError(
            f"Eligible case {uid} has an inconsistent pmc_id"
        )
    if (
        not isinstance(case.get("case_description"), str)
        or not case["case_description"].strip()
    ):
        raise ArtifactIntegrityError(
            f"Eligible case {uid} has an empty case description"
        )
    captions = case.get("captions")
    if not isinstance(captions, list) or not captions:
        raise ArtifactIntegrityError(
            f"Eligible case {uid} has an empty caption group"
        )
    caption_ids: list[str] = []
    for caption in captions:
        if not isinstance(caption, dict):
            raise ArtifactIntegrityError(
                f"Eligible case {uid} contains a non-object caption"
            )
        required = {
            "caption_id",
            "caption",
            "caption_path",
            "caption_sha256",
            "image_path",
            "image_sha256",
        }
        if set(caption) != required:
            raise ArtifactIntegrityError(
                f"Eligible case {uid} contains an invalid caption schema"
            )
        caption_id = caption.get("caption_id")
        if not isinstance(caption_id, str) or not caption_id:
            raise ArtifactIntegrityError(
                f"Eligible case {uid} contains an invalid caption ID"
            )
        caption_ids.append(caption_id)
        if not isinstance(caption.get("caption"), str) or not caption["caption"]:
            raise ArtifactIntegrityError(
                f"Eligible case {uid} contains an empty caption"
            )
        for field in ("caption_path", "image_path"):
            path = Path(str(caption.get(field, "")))
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ArtifactIntegrityError(
                    f"Eligible case {uid} contains an unsafe {field}"
                )
    if len(caption_ids) != len(set(caption_ids)):
        raise ArtifactIntegrityError(
            f"Eligible case {uid} contains duplicate caption IDs"
        )
    population_scope = case.get("population_scope")
    if (
        not isinstance(population_scope, dict)
        or set(population_scope) != {"ruleset", "label"}
        or population_scope.get("label") != "human"
        or population_scope.get("ruleset") not in set(RULESETS)
    ):
        raise ArtifactIntegrityError(
            f"Eligible case {uid} has an invalid population_scope"
        )


def _validate_handoff_pair(
    pair: dict[str, Any],
    *,
    case_uids: set[str],
) -> None:
    expected_fields = {
        "schema_version",
        "pair_index",
        "pair_key",
        "case_a_uid",
        "case_b_uid",
        "citation",
    }
    if set(pair) != expected_fields or pair.get("schema_version") != 1:
        raise ArtifactIntegrityError("Eligible pair has an invalid schema")
    pair_index = pair.get("pair_index")
    if (
        isinstance(pair_index, bool)
        or not isinstance(pair_index, int)
        or pair_index < 0
    ):
        raise ArtifactIntegrityError("Eligible pair has an invalid pair_index")
    uid_a = pair.get("case_a_uid")
    uid_b = pair.get("case_b_uid")
    if not isinstance(uid_a, str) or not isinstance(uid_b, str):
        raise ArtifactIntegrityError(
            f"Eligible pair {pair_index} has invalid case UIDs"
        )
    if _uid_sort_key(uid_a) >= _uid_sort_key(uid_b):
        raise ArtifactIntegrityError(
            f"Eligible pair {pair_index} is not in canonical orientation"
        )
    if {uid_a, uid_b} - case_uids:
        raise ArtifactIntegrityError(
            f"Eligible pair {pair_index} references an absent case"
        )
    expected_key = hashlib.sha256(
        f"{uid_a}\0{uid_b}".encode("utf-8")
    ).hexdigest()
    if pair.get("pair_key") != expected_key:
        raise ArtifactIntegrityError(
            f"Eligible pair {pair_index} has an invalid pair_key"
        )

    citation = pair.get("citation")
    if not isinstance(citation, dict) or set(citation) != set(_CITATION_FIELDS):
        raise ArtifactIntegrityError(
            f"Eligible pair {pair_index} has an invalid citation schema"
        )
    if citation.get("status") != "matched":
        raise ArtifactIntegrityError(
            f"Eligible pair {pair_index} has no matched citation"
        )
    if {citation.get("citing_uid"), citation.get("cited_uid")} != {uid_a, uid_b}:
        raise ArtifactIntegrityError(
            f"Eligible pair {pair_index} has misaligned citation members"
        )
    if citation.get("direction") not in {
        "case_a_cites_case_b",
        "case_b_cites_case_a",
    }:
        raise ArtifactIntegrityError(
            f"Eligible pair {pair_index} has an invalid citation direction"
        )
    citation_paragraphs = citation.get("citation_paragraphs")
    if not isinstance(citation_paragraphs, list) or any(
        not isinstance(paragraph, str)
        or not paragraph
        or paragraph != paragraph.strip()
        for paragraph in citation_paragraphs
    ):
        raise ArtifactIntegrityError(
            f"Eligible pair {pair_index} has invalid citation paragraphs"
        )


def _validate_diagram_artifacts(
    run_dir: Path,
    manifest: dict[str, Any],
) -> int:
    records = load_jsonl(run_dir / DIAGRAM_CLASSIFICATIONS)
    blacklist = load_json(run_dir / DIAGRAM_BLACKLIST)
    stats = load_json(run_dir / DIAGRAM_STATS)
    candidate_stats = load_json(run_dir / STAGE_1_STATS)
    cases = load_jsonl(run_dir / CASES)
    if not isinstance(blacklist, list) or any(
        not isinstance(path, str) for path in blacklist
    ):
        raise ArtifactIntegrityError("Diagram blacklist is not a JSON string array")
    expected_stats_fields = {
        "schema_version",
        "mode",
        "images_input",
        "diagrams_excluded",
        "images_retained",
        "reason_counts",
        "workers",
        "strict_comparisons",
        "runtime",
    }
    if (
        not isinstance(stats, dict)
        or set(stats) != expected_stats_fields
        or stats.get("schema_version") != 1
    ):
        raise ArtifactIntegrityError("Diagram statistics are malformed")
    mode = stats.get("mode")
    if mode != "computed":
        raise ArtifactIntegrityError("Diagram-filter mode is invalid")
    if stats.get("workers") != manifest.get("workers"):
        raise ArtifactIntegrityError("Diagram-filter worker provenance is invalid")
    runtime = stats.get("runtime")
    if not isinstance(runtime, dict) or (
        tesseract_runtime_contract(runtime)
        != manifest.get("diagram_filter_runtime")
    ):
        raise ArtifactIntegrityError(
            "Diagram-filter runtime differs from the immutable run specification"
        )

    config = manifest.get("config", {}).get("diagram_filter", {})
    if not isinstance(config, dict):
        raise ArtifactIntegrityError("Diagram-filter configuration is missing")
    white_threshold = config.get("white_ratio_threshold")
    text_threshold = config.get("ocr_text_length_threshold")
    expected_comparisons = {
        "diagram_if_white_ratio_gt": white_threshold,
        "otherwise_diagram_if_ocr_text_length_gt": text_threshold,
        "white_pixel_if_grayscale_gte": config.get("white_pixel_threshold"),
    }
    if stats.get("strict_comparisons") != expected_comparisons:
        raise ArtifactIntegrityError(
            "Diagram statistics do not match the configured decision thresholds"
        )
    expected_fields = {
        "schema_version",
        "image_path",
        "mode",
        "white_ratio",
        "ocr_status",
        "ocr_text_length",
        "is_diagram",
        "reason",
    }
    paths: list[str] = []
    seen_paths: set[str] = set()
    excluded: list[str] = []
    reason_counts: Counter[str] = Counter()
    for position, record in enumerate(records):
        if set(record) != expected_fields or record.get("schema_version") != 1:
            raise ArtifactIntegrityError(
                f"Diagram classification {position} has an invalid schema"
            )
        path = record.get("image_path")
        if not isinstance(path, str):
            raise ArtifactIntegrityError(
                f"Diagram classification {position} has no image_path"
            )
        relative = Path(path)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] != UPSTREAM_ASSETS
            or relative.suffix.lower() != ".jpg"
        ):
            raise ArtifactIntegrityError(
                f"Diagram classification {position} has an unsafe image_path"
            )
        if path in seen_paths:
            raise ArtifactIntegrityError(
                f"Diagram classifications contain duplicate path {path!r}"
            )
        paths.append(path)
        seen_paths.add(path)
        if record.get("mode") != mode or not isinstance(
            record.get("is_diagram"), bool
        ):
            raise ArtifactIntegrityError(
                f"Diagram classification {position} has an invalid mode or decision"
            )

        reason = record.get("reason")
        if not isinstance(reason, str):
            raise ArtifactIntegrityError(
                f"Diagram classification {position} has an invalid reason"
            )
        ratio = record.get("white_ratio")
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(float(ratio))
            or not 0.0 <= float(ratio) <= 1.0
        ):
            raise ArtifactIntegrityError(
                f"Diagram classification {position} has an invalid white ratio"
            )
        if float(ratio) > float(white_threshold):
            expected = ("skipped_white_ratio", None, True, "white_ratio")
        else:
            text_length = record.get("ocr_text_length")
            if (
                isinstance(text_length, bool)
                or not isinstance(text_length, int)
                or text_length < 0
            ):
                raise ArtifactIntegrityError(
                    f"Diagram classification {position} has invalid OCR length"
                )
            text_positive = text_length > int(text_threshold)
            expected = (
                "complete",
                text_length,
                text_positive,
                "ocr_text_length" if text_positive else "retained",
            )
        actual = (
            record.get("ocr_status"),
            record.get("ocr_text_length"),
            record.get("is_diagram"),
            reason,
        )
        if actual != expected:
            raise ArtifactIntegrityError(
                f"Diagram classification {position} contradicts its measurements"
            )
        reason_counts[reason] += 1
        if record["is_diagram"]:
            excluded.append(path)

    if excluded != blacklist:
        raise ArtifactIntegrityError(
            "Diagram blacklist does not match per-image classifications"
        )
    expected_counts = {
        "images_input": len(records),
        "diagrams_excluded": len(excluded),
        "images_retained": len(records) - len(excluded),
        "reason_counts": dict(sorted(reason_counts.items())),
    }
    if any(stats.get(key) != value for key, value in expected_counts.items()):
        raise ArtifactIntegrityError(
            "Diagram statistics do not match per-image classifications"
        )
    if not isinstance(candidate_stats, dict) or (
        candidate_stats.get("diagram_blacklist_entries") != len(excluded)
        or candidate_stats.get("rows_excluded_as_diagrams") != len(excluded)
        or candidate_stats.get("rows_input") != len(records)
        or candidate_stats.get("rows_output") != len(records) - len(excluded)
    ):
        raise ArtifactIntegrityError(
            "Candidate statistics do not match the diagram-filter output"
        )
    excluded_set = set(excluded)
    retained_paths = {path for path in paths if path not in excluded_set}
    case_paths = {
        caption.get("image_path")
        for case in cases
        if isinstance(case, dict) and isinstance(case.get("captions"), list)
        for caption in case["captions"]
        if isinstance(caption, dict)
    }
    if retained_paths != case_paths:
        raise ArtifactIntegrityError(
            "Candidate cases do not contain exactly the retained diagram-filter paths"
        )
    return len(excluded)


def _validate_species_artifacts(
    run_dir: Path,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    cases = load_jsonl(run_dir / CASES)
    pairs = load_jsonl(run_dir / PAIRS)
    classifications = load_jsonl(run_dir / SPECIES_CLASSIFICATIONS)
    pair_species = load_jsonl(run_dir / PAIR_SPECIES)
    stats = load_json(run_dir / SPECIES_STATS)

    cases_by_uid = _unique_by(cases, "patient_uid", label="cases")
    classifications_by_uid = _unique_by(
        classifications,
        "patient_uid",
        label="species classifications",
    )
    if set(cases_by_uid) != set(classifications_by_uid):
        raise ArtifactIntegrityError(
            "Species classifications do not cover exactly the candidate cases"
        )
    required_fields = {
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
    configured = manifest.get("config", {}).get("species_filter", {})
    if not isinstance(configured, dict):
        raise ArtifactIntegrityError("Species-filter configuration is missing")
    ruleset = configured.get("ruleset")
    try:
        allowed_rules = matched_rule_vocabulary(str(ruleset))
    except ValueError as exc:
        raise ArtifactIntegrityError(
            "Species-filter ruleset is unsupported"
        ) from exc
    for uid in sorted(classifications_by_uid, key=_uid_sort_key):
        record = classifications_by_uid[uid]
        if set(record) != required_fields or record.get("schema_version") != 1:
            raise ArtifactIntegrityError(
                f"Species classification {uid!r} has an invalid schema"
            )
        case = cases_by_uid[uid]
        if (
            record.get("pmc_id") != case.get("pmc_id")
            or record.get("nxml_sha256") != case.get("nxml_sha256")
            or record.get("ruleset") != ruleset
            or record.get("automatic_label") not in _SPECIES_LABELS
            or record.get("final_label") not in _SPECIES_LABELS
        ):
            raise ArtifactIntegrityError(
                f"Species classification {uid!r} is inconsistent with its case or run"
            )
        for field in ("title", "journal_title"):
            if (
                not isinstance(record.get(field), str)
                or record[field] != record[field].strip()
            ):
                raise ArtifactIntegrityError(
                    f"Species classification {uid!r} has invalid {field}"
                )
        matched_rules = record.get("matched_rules")
        if (
            not isinstance(matched_rules, list)
            or any(not isinstance(item, str) or not item for item in matched_rules)
            or matched_rules != sorted(set(matched_rules))
            or any(item not in allowed_rules for item in matched_rules)
        ):
            raise ArtifactIntegrityError(
                f"Species classification {uid!r} has invalid matched rules"
            )
        if record["automatic_label"] == "animal" and not matched_rules:
            raise ArtifactIntegrityError(
                f"Animal classification {uid!r} has no supporting rule"
            )
        adjudication = record.get("adjudication")
        if adjudication is not None:
            if (
                not isinstance(adjudication, dict)
                or set(adjudication) != {"label", "reason"}
                or adjudication.get("label") not in _SPECIES_LABELS
                or record.get("final_label") != adjudication.get("label")
                or not isinstance(adjudication.get("reason"), str)
                or not adjudication["reason"]
                or adjudication["reason"] != adjudication["reason"].strip()
            ):
                raise ArtifactIntegrityError(
                    f"Species classification {uid!r} has an invalid adjudication"
                )
        elif record.get("final_label") != record.get("automatic_label"):
            raise ArtifactIntegrityError(
                f"Species classification {uid!r} changes label without adjudication"
            )

    expected_human, expected_pair_species, expected_stats = _screen_species_pairs(
        pairs,
        classifications,
        ruleset=str(ruleset),
        workers=int(manifest.get("workers", 0)),
    )
    if pair_species != expected_pair_species:
        raise ArtifactIntegrityError(
            "Pair species records do not cover the complete candidate-pair set"
        )
    if stats != expected_stats:
        raise ArtifactIntegrityError(
            "Species statistics do not match case and pair classifications"
        )
    return (
        classifications,
        expected_human,
        expected_stats["animal_pairs"],
    )


def verify_run(output_dir: str | Path) -> dict[str, Any]:
    """Verify all local artifacts and the downstream handoff contract."""

    run_dir = Path(output_dir)
    manifest = verify_manifest(run_dir)
    if manifest.get("status") != "complete":
        raise ArtifactIntegrityError(
            f"Selection run is not complete: {manifest.get('status')!r}"
        )
    for stage in (
        "00_diagram_filter",
        "01_candidates",
        "02_summaries",
        "02_human_animal_screen",
        "03_selection",
        "03_handoff",
    ):
        record = manifest.get("stages", {}).get(stage)
        if not isinstance(record, dict) or record.get("status") != "complete":
            raise ArtifactIntegrityError(f"Required stage is incomplete: {stage}")

    diagrams_excluded = _validate_diagram_artifacts(run_dir, manifest)
    classifications, human_pairs, animal_pair_count = (
        _validate_species_artifacts(run_dir, manifest)
    )
    species_stage = manifest["stages"]["02_human_animal_screen"]
    expected_species_artifacts = sorted(
        (
            SPECIES_CLASSIFICATIONS,
            PAIR_SPECIES,
            SPECIES_STATS,
        )
    )
    if species_stage.get("artifacts") != expected_species_artifacts:
        raise ArtifactIntegrityError(
            "Human/animal stage must bind exactly the three screening artifacts"
        )
    species_stats = load_json(run_dir / SPECIES_STATS)
    expected_species_details = {
        "ruleset": species_stats["ruleset"],
        "cases_screened": species_stats["cases_screened"],
        "pairs_screened": species_stats["pairs_before_screen"],
        "human_pairs": species_stats["human_pairs"],
        "animal_pairs": species_stats["animal_pairs"],
    }
    if species_stage.get("details") != expected_species_details:
        raise ArtifactIntegrityError(
            "Human/animal stage details do not match the complete pair screen"
        )
    handoff_stage = manifest["stages"]["03_handoff"]
    expected_handoff_artifacts = sorted(
        (
            ELIGIBLE_CASES,
            ELIGIBLE_PAIRS,
            SPECIES_CLASSIFICATIONS,
            PAIR_SPECIES,
            SPECIES_STATS,
            HANDOFF_MANIFEST,
        )
    )
    if handoff_stage.get("artifacts") != expected_handoff_artifacts:
        raise ArtifactIntegrityError(
            "Handoff stage does not bind all eligible and species artifacts"
        )

    handoff = load_json(run_dir / HANDOFF_MANIFEST)
    if handoff != manifest.get("handoff_contract"):
        raise ArtifactIntegrityError(
            "Handoff manifest differs from the run-manifest contract"
        )
    if handoff.get("schema_version") != HANDOFF_SCHEMA_VERSION:
        raise ArtifactIntegrityError("Unsupported handoff schema")
    expected_producer = {
        "pipeline_id": PIPELINE_ID,
        "pipeline_version": PIPELINE_VERSION,
        "run_spec_id": manifest["run_spec_id"],
    }
    if handoff.get("producer") != expected_producer:
        raise ArtifactIntegrityError("Handoff producer metadata is invalid")
    if handoff.get("contracts") != _HANDOFF_CONTRACTS:
        raise ArtifactIntegrityError("Handoff ordering/schema contract is invalid")

    handoff_artifacts = handoff.get("artifacts")
    if not isinstance(handoff_artifacts, dict) or set(handoff_artifacts) != {
        "eligible_cases",
        "eligible_pairs",
        "species_classifications",
        "pair_species",
        "species_stats",
    }:
        raise ArtifactIntegrityError("Handoff artifact table is invalid")
    path_by_key = {
        "eligible_cases": ELIGIBLE_CASES,
        "eligible_pairs": ELIGIBLE_PAIRS,
        "species_classifications": SPECIES_CLASSIFICATIONS,
        "pair_species": PAIR_SPECIES,
        "species_stats": SPECIES_STATS,
    }
    for key, expected_path in path_by_key.items():
        metadata = handoff_artifacts[key]
        if not isinstance(metadata, dict) or metadata.get("path") != expected_path:
            raise ArtifactIntegrityError(
                f"Handoff metadata is invalid for {key}"
            )
        verify_file_metadata(metadata, root=run_dir)

    cases = load_jsonl(run_dir / CASES)
    summaries = load_jsonl(run_dir / SUMMARIES)
    citations = load_jsonl(run_dir / CITATIONS)
    selection = load_json(run_dir / SELECTION)
    if (
        not isinstance(selection, dict)
        or not isinstance(selection.get("pairs"), list)
        or selection.get("selected_count") != len(selection["pairs"])
    ):
        raise ArtifactIntegrityError("Selection artifact is malformed")
    retained_indices = {
        pair["pair_index"] for pair in human_pairs
    }
    citation_indices = {citation.get("pair_index") for citation in citations}
    if citation_indices != retained_indices:
        raise ArtifactIntegrityError(
            "Citation artifacts do not cover exactly the human pairs"
        )
    selected_indices = {pair.get("pair_index") for pair in selection["pairs"]}
    if not selected_indices <= retained_indices:
        raise ArtifactIntegrityError(
            "Selection contains a pair not labeled human"
        )
    expected_cases, expected_pairs = _make_handoff_records(
        selected=selection["pairs"],
        cases=cases,
        summaries=summaries,
        citations=citations,
        classifications=classifications,
    )
    actual_cases = load_jsonl(run_dir / ELIGIBLE_CASES)
    actual_pairs = load_jsonl(run_dir / ELIGIBLE_PAIRS)
    if actual_cases != expected_cases or actual_pairs != expected_pairs:
        raise ArtifactIntegrityError(
            "Handoff records do not align with deterministic selection artifacts"
        )
    if handoff_artifacts["eligible_cases"].get("records") != len(actual_cases):
        raise ArtifactIntegrityError("Eligible-case record count is invalid")
    if handoff_artifacts["eligible_pairs"].get("records") != len(actual_pairs):
        raise ArtifactIntegrityError("Eligible-pair record count is invalid")
    expected_species_records = {
        "species_classifications": len(classifications),
        "pair_species": len(load_jsonl(run_dir / PAIR_SPECIES)),
        "species_stats": 1,
    }
    for role, expected_records in expected_species_records.items():
        if handoff_artifacts[role].get("records") != expected_records:
            raise ArtifactIntegrityError(
                f"Handoff {role} record count is invalid"
            )

    case_uids: list[str] = []
    all_caption_ids: set[str] = set()
    for case in actual_cases:
        _validate_handoff_case(case)
        uid = case["patient_uid"]
        case_uids.append(uid)
        for caption in case["captions"]:
            caption_id = caption["caption_id"]
            if caption_id in all_caption_ids:
                raise ArtifactIntegrityError(
                    f"Duplicate handoff caption ID: {caption_id}"
                )
            all_caption_ids.add(caption_id)
    if case_uids != sorted(case_uids, key=_uid_sort_key):
        raise ArtifactIntegrityError("Eligible cases are not canonically ordered")
    if len(case_uids) != len(set(case_uids)):
        raise ArtifactIntegrityError("Eligible cases contain duplicate UIDs")

    indices: list[int] = []
    referenced_uids: set[str] = set()
    for pair in actual_pairs:
        _validate_handoff_pair(pair, case_uids=set(case_uids))
        indices.append(pair["pair_index"])
        referenced_uids.update((pair["case_a_uid"], pair["case_b_uid"]))
    if indices != sorted(indices):
        raise ArtifactIntegrityError("Eligible pairs are not ordered by pair_index")
    if len(indices) != len(set(indices)):
        raise ArtifactIntegrityError("Eligible pairs contain duplicate indices")
    if referenced_uids != set(case_uids):
        raise ArtifactIntegrityError(
            "Eligible-case rows are not exactly the cases referenced by pairs"
        )

    return {
        "pipeline_id": PIPELINE_ID,
        "pipeline_version": PIPELINE_VERSION,
        "run_spec_id": manifest["run_spec_id"],
        "status": manifest["status"],
        "eligible_cases": len(actual_cases),
        "eligible_pairs": len(actual_pairs),
        "diagrams_excluded": diagrams_excluded,
        "human_pairs": len(human_pairs),
        "animal_pairs": animal_pair_count,
        "handoff_manifest_sha256": sha256_file(run_dir / HANDOFF_MANIFEST),
    }


__all__ = ["prepare_run", "verify_run"]
