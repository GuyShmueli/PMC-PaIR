from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .artifacts import (
    CASES,
    CITATIONS,
    FINAL_BUCKETS,
    FINAL_LABELED,
    FINAL_PAIRS,
    FINAL_STATS,
    FINAL_TEXT,
    HANDOFF_STAGE,
    SELECTION,
    SUMMARIES,
    batch_stage_dir,
)
from .batch_api import BatchManager
from .config import STAGES
from .errors import ArtifactIntegrityError, ResponseValidationError, StageOrderError
from .finalizer import finalize_positive_responses
from .handoff import verify_imported_population_boundary
from .io_utils import (
    canonical_json_bytes,
    iter_jsonl,
    load_json,
    load_jsonl,
    save_json,
    save_jsonl,
    sha256_bytes,
    sha256_file,
)
from .manifest import record_stage, require_stage, verify_manifest
from .prompts import (
    answer_user_prompt,
    citation_user_prompt,
    positive_user_prompt,
    question_user_prompt,
    request_body,
)
from .responses import normalize_response_files, reconcile_responses
from .response_schemas import validate_intermediate_response
from .sharding import write_request_shards


_MODEL_SEQUENCE = {stage: position for position, stage in enumerate(STAGES, start=4)}


def _request_stage(stage: str) -> str:
    return f"{_MODEL_SEQUENCE[stage]:02d}_{stage}_requests"


def _response_stage(stage: str) -> str:
    return f"{_MODEL_SEQUENCE[stage]:02d}_{stage}_responses"


def _submission_stage(stage: str) -> str:
    return f"{_MODEL_SEQUENCE[stage]:02d}_{stage}_submissions"


def _validate_attempt(attempt: int) -> int:
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("attempt must be a positive integer")
    return attempt


def _attempt_dir(run_dir: Path, stage: str, attempt: int) -> Path:
    return (
        batch_stage_dir(run_dir, stage)
        / "attempts"
        / f"attempt-{_validate_attempt(attempt):04d}"
    )


def _attempt_submission_receipt(run_dir: Path, stage: str, attempt: int) -> Path:
    return _attempt_dir(run_dir, stage, attempt) / "submission_receipt.json"


def _attempt_download_receipt(run_dir: Path, stage: str, attempt: int) -> Path:
    return _attempt_dir(run_dir, stage, attempt) / "managed_downloads.json"


def _attempt_failure_receipt(run_dir: Path, stage: str, attempt: int) -> Path:
    return _attempt_dir(run_dir, stage, attempt) / "validation_failure.json"


def _bound_managed_attempts(
    run_dir: Path,
    stage: str,
    response_record: Mapping[str, Any],
) -> list[int]:
    """Return attempts whose immutable receipts are bound to canonical responses."""

    response_artifacts = set(response_record.get("artifacts", []))
    bound: list[int] = []
    for attempt in _attempt_numbers(run_dir, stage):
        required = {
            _attempt_submission_receipt(run_dir, stage, attempt)
            .relative_to(run_dir)
            .as_posix(),
            _attempt_download_receipt(run_dir, stage, attempt)
            .relative_to(run_dir)
            .as_posix(),
        }
        if required <= response_artifacts:
            bound.append(attempt)
    return bound


def _attempt_numbers(run_dir: Path, stage: str) -> list[int]:
    root = batch_stage_dir(run_dir, stage) / "attempts"
    if not root.is_dir():
        return []
    numbers: list[int] = []
    for path in root.iterdir():
        prefix = "attempt-"
        suffix = path.name[len(prefix) :] if path.name.startswith(prefix) else ""
        if path.is_dir() and suffix.isdigit() and int(suffix) >= 1:
            numbers.append(int(suffix))
    return sorted(numbers)


def _require_attempt_generation(
    run_dir: Path,
    manifest: dict[str, Any],
    stage: str,
    attempt: int,
) -> None:
    attempt = _validate_attempt(attempt)
    if manifest.get("status") == "complete" or (
        manifest.get("stages", {}).get(_response_stage(stage), {}).get("status")
        == "complete"
    ):
        raise StageOrderError(
            f"Refusing attempt {attempt} for {stage}: its canonical responses are already complete"
        )
    numbers = _attempt_numbers(run_dir, stage)
    if any(number > attempt for number in numbers):
        raise StageOrderError(
            f"Attempt {attempt} is older than an existing managed attempt"
        )
    if attempt == 1:
        if any(number != 1 for number in numbers):
            raise StageOrderError("Managed attempt history does not start at attempt 1")
        return
    expected_prior = list(range(1, attempt))
    if [number for number in numbers if number < attempt] != expected_prior:
        raise StageOrderError(
            f"Retry attempt {attempt} requires contiguous attempts {expected_prior}"
        )
    prior_failure = _attempt_failure_receipt(run_dir, stage, attempt - 1)
    prior_download = _attempt_download_receipt(run_dir, stage, attempt - 1)
    if not prior_failure.is_file() or not prior_download.is_file():
        raise StageOrderError(
            f"Retry attempt {attempt} is allowed only after attempt {attempt - 1} "
            "was downloaded and failed response validation"
        )
    failure = load_json(prior_failure)
    if (
        failure.get("schema_version") != 1
        or failure.get("run_spec_id") != manifest.get("run_spec_id")
        or failure.get("stage") != stage
        or failure.get("attempt") != attempt - 1
        or failure.get("download_receipt_sha256") != sha256_file(prior_download)
    ):
        raise ArtifactIntegrityError(
            f"Attempt {attempt - 1} validation-failure receipt is inconsistent"
        )


def _attempt_audit_artifacts(run_dir: Path, stage: str) -> list[Path]:
    paths: list[Path] = []
    for attempt in _attempt_numbers(run_dir, stage):
        attempt_root = _attempt_dir(run_dir, stage, attempt)
        for name in (
            "submission_receipt.json",
            "managed_downloads.json",
            "validation_failure.json",
        ):
            candidate = attempt_root / name
            if candidate.is_file():
                paths.append(candidate)
        for folder in ("resolutions", "statuses"):
            receipt_dir = attempt_root / folder
            if receipt_dir.is_dir():
                paths.extend(sorted(receipt_dir.glob("*.json")))
        download_receipt = attempt_root / "managed_downloads.json"
        if download_receipt.is_file():
            description = load_json(download_receipt)
            for download in description.get("downloads", []):
                relative = download.get("download_dir")
                if not isinstance(relative, str):
                    raise ArtifactIntegrityError(
                        f"Attempt {attempt} download receipt has no download_dir"
                    )
                destination = attempt_root / relative
                manifest_path = destination / "download_manifest.json"
                if not manifest_path.is_file():
                    raise ArtifactIntegrityError(
                        f"Attempt {attempt} download manifest is missing: {manifest_path}"
                    )
                paths.append(manifest_path)
                for item in download.get("files", []):
                    name = item.get("path")
                    if not isinstance(name, str) or Path(name).name != name:
                        raise ArtifactIntegrityError(
                            f"Attempt {attempt} download file path is unsafe"
                        )
                    paths.append(destination / name)
    return sorted(set(paths))


def _record_attempt_audit(
    run_dir: Path,
    stage: str,
    *,
    status_override: str | None = None,
    run_status: str | None = None,
) -> dict[str, Any]:
    manifest = verify_manifest(run_dir)
    if run_status is not None:
        manifest["status"] = run_status
    attempts = _attempt_numbers(run_dir, stage)
    artifacts = _attempt_audit_artifacts(run_dir, stage)
    summaries = []
    all_submitted = bool(attempts)
    for attempt in attempts:
        submitted = _attempt_submission_receipt(run_dir, stage, attempt).is_file()
        all_submitted = all_submitted and submitted
        summaries.append(
            {
                "attempt": attempt,
                "downloaded": _attempt_download_receipt(
                    run_dir, stage, attempt
                ).is_file(),
                "submitted": submitted,
                "validation_failed": _attempt_failure_receipt(
                    run_dir, stage, attempt
                ).is_file(),
            }
        )
    record_stage(
        run_dir,
        manifest,
        _submission_stage(stage),
        status=status_override or ("complete" if all_submitted else "in_progress"),
        artifact_paths=artifacts,
        details={
            "attempts": summaries,
            "latest_attempt": attempts[-1] if attempts else None,
        },
    )
    return manifest


def _validate_model_stage(stage: str) -> str:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}; got {stage!r}")
    return stage
def _save_immutable_json(value: Any, path: Path) -> None:
    if path.exists():
        if not path.is_file() or load_json(path) != value:
            raise ArtifactIntegrityError(
                f"Refusing to replace non-identical immutable artifact: {path}"
            )
        return
    save_json(value, path)
def _unique_by(records: list[dict[str, Any]], field: str, *, label: str) -> dict[Any, dict[str, Any]]:
    result: dict[Any, dict[str, Any]] = {}
    for position, record in enumerate(records):
        if not isinstance(record, dict) or field not in record:
            raise ArtifactIntegrityError(
                f"{label} record {position} has no {field!r} field"
            )
        key = record[field]
        if key in result:
            raise ArtifactIntegrityError(f"{label} contains duplicate {field} {key!r}")
        result[key] = record
    return result
def _selected_pairs(run_dir: Path) -> list[dict[str, Any]]:
    selection = load_json(run_dir / SELECTION)
    if not isinstance(selection, dict) or not isinstance(selection.get("pairs"), list):
        raise ArtifactIntegrityError("Selection artifact is malformed")
    pairs = selection["pairs"]
    if selection.get("selected_count") != len(pairs):
        raise ArtifactIntegrityError("Selection count does not match its pair list")
    _unique_by(pairs, "pair_index", label="selected pairs")
    return pairs


def _responses_by_pair(run_dir: Path, stage: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for record in iter_jsonl(batch_stage_dir(run_dir, stage) / "responses.jsonl"):
        pair_index = record.get("pair_index")
        if pair_index in result:
            raise ArtifactIntegrityError(
                f"{stage} responses contains duplicate pair_index {pair_index!r}"
            )
        result[pair_index] = record
    return result


def _require_exact_selected_indices(
    selected: list[dict[str, Any]],
    records: Mapping[int, Any],
    *,
    label: str,
) -> None:
    expected = {pair["pair_index"] for pair in selected}
    actual = set(records)
    if actual != expected:
        raise ArtifactIntegrityError(
            f"{label} pair indices differ from immutable selection; "
            f"missing={sorted(expected - actual)[:20]}, unexpected={sorted(actual - expected)[:20]}"
        )


def _model_spec(manifest: dict[str, Any], stage: str) -> tuple[str, str | None]:
    raw = manifest.get("config", {}).get("models", {}).get(stage)
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), str):
        raise ArtifactIntegrityError(f"Manifest has no valid model configuration for {stage}")
    effort = raw.get("reasoning_effort")
    if effort is not None and not isinstance(effort, str):
        raise ArtifactIntegrityError(f"Manifest reasoning_effort for {stage} is invalid")
    return raw["model"], effort


def _selected_jsonl_records(
    path: Path,
    *,
    field: str,
    selected_values: set[Any],
    label: str,
) -> dict[Any, dict[str, Any]]:
    result: dict[Any, dict[str, Any]] = {}
    for record in iter_jsonl(path):
        value = record.get(field)
        if value not in selected_values:
            continue
        if value in result:
            raise ArtifactIntegrityError(
                f"{label} contains duplicate selected {field} {value!r}"
            )
        result[value] = record
    if set(result) != selected_values:
        raise ArtifactIntegrityError(
            f"{label} is missing selected {field} values: "
            f"{sorted(selected_values - set(result))[:20]}"
        )
    return result


def _iter_model_requests(
    *,
    stage: str,
    selected: list[dict[str, Any]],
    cases_by_uid: Mapping[str, dict[str, Any]],
    summaries_by_uid: Mapping[str, dict[str, Any]],
    citations_by_index: Mapping[int, dict[str, Any]],
    prior_responses: Mapping[str, Mapping[int, dict[str, Any]]],
    model: str,
    reasoning_effort: str | None,
) -> Iterator[dict[str, Any]]:
    for pair in selected:
        pair_index = pair["pair_index"]
        case_a_uid = pair["case_a_uid"]
        case_b_uid = pair["case_b_uid"]
        case_a = cases_by_uid[case_a_uid]
        case_b = cases_by_uid[case_b_uid]
        summary_a = summaries_by_uid[case_a_uid]
        summary_b = summaries_by_uid[case_b_uid]
        citation = citations_by_index[pair_index]
        if summary_a.get("status") != "valid" or summary_b.get("status") != "valid":
            raise ArtifactIntegrityError(
                f"Selected pair {pair_index} has an invalid summary"
            )
        if citation.get("status") != "matched":
            raise ArtifactIntegrityError(
                f"Selected pair {pair_index} has no citation match"
            )

        if stage == "citation":
            user_prompt = citation_user_prompt(citation)
        elif stage == "question":
            user_prompt = question_user_prompt(
                summary_a["summary"],
                summary_b["summary"],
                prior_responses["citation"][pair_index]["text"],
            )
        elif stage == "answer":
            user_prompt = answer_user_prompt(
                summary_a["summary"],
                summary_b["summary"],
                prior_responses["citation"][pair_index]["text"],
                prior_responses["question"][pair_index]["text"],
            )
        else:
            user_prompt = positive_user_prompt(
                {
                    item["caption_id"]: item["caption"]
                    for item in case_a["captions"]
                },
                {
                    item["caption_id"]: item["caption"]
                    for item in case_b["captions"]
                },
                prior_responses["question"][pair_index]["text"],
                prior_responses["answer"][pair_index]["text"],
            )
        yield {
            "pair_index": pair_index,
            "body": request_body(
                stage=stage,
                model=model,
                reasoning_effort=reasoning_effort,
                user_prompt=user_prompt,
            ),
        }


def build_model_stage(output_dir: str | Path, stage: str) -> dict[str, Any]:
    """Build deterministic request shards for exactly the immutable selection."""

    stage = _validate_model_stage(stage)
    run_dir = Path(output_dir)
    manifest = verify_manifest(run_dir)
    require_stage(manifest, HANDOFF_STAGE)
    population_boundary = verify_imported_population_boundary(run_dir)
    handoff_details = manifest["stages"][HANDOFF_STAGE].get("details")
    if not isinstance(handoff_details, dict) or any(
        handoff_details.get(field) != population_boundary[field]
        for field in (
            "ruleset",
            "cases_screened",
            "pairs_screened",
            "human_pairs",
            "animal_pairs",
        )
    ):
        raise ArtifactIntegrityError(
            "Pipeline 03 handoff details do not match the human/animal-screen snapshots"
        )
    selected = _selected_pairs(run_dir)
    if not selected:
        raise StageOrderError(
            "The immutable selection is empty; run finalize directly without API stages"
        )

    if stage != "citation":
        prior = STAGES[STAGES.index(stage) - 1]
        require_stage(manifest, _response_stage(prior))

    request_stage = _request_stage(stage)
    existing = manifest.get("stages", {}).get(request_stage)
    if isinstance(existing, dict) and existing.get("status") == "complete":
        return load_json(batch_stage_dir(run_dir, stage) / "request_manifest.json")

    selected_indices = {pair["pair_index"] for pair in selected}
    selected_uids = {
        pair[field]
        for pair in selected
        for field in ("case_a_uid", "case_b_uid")
    }
    cases_by_uid = _selected_jsonl_records(
        run_dir / CASES,
        field="patient_uid",
        selected_values=selected_uids,
        label="cases",
    )
    summaries_by_uid = _selected_jsonl_records(
        run_dir / SUMMARIES,
        field="patient_uid",
        selected_values=selected_uids,
        label="summaries",
    )
    citations_by_index = _selected_jsonl_records(
        run_dir / CITATIONS,
        field="pair_index",
        selected_values=selected_indices,
        label="citations",
    )

    prior_responses: dict[str, dict[int, dict[str, Any]]] = {}
    for prior_stage in STAGES[: STAGES.index(stage)]:
        response_map = _responses_by_pair(run_dir, prior_stage)
        _require_exact_selected_indices(
            selected, response_map, label=f"{prior_stage} responses"
        )
        prior_responses[prior_stage] = response_map

    model, reasoning_effort = _model_spec(manifest, stage)
    batch_config = manifest["config"]["batch"]
    stage_dir = batch_stage_dir(run_dir, stage)
    request_manifest = write_request_shards(
        _iter_model_requests(
            stage=stage,
            selected=selected,
            cases_by_uid=cases_by_uid,
            summaries_by_uid=summaries_by_uid,
            citations_by_index=citations_by_index,
            prior_responses=prior_responses,
            model=model,
            reasoning_effort=reasoning_effort,
        ),
        stage=stage,
        output_dir=stage_dir,
        max_shard_bytes=batch_config["max_shard_bytes"],
        max_shard_requests=batch_config["max_shard_requests"],
    )
    artifact_paths = [stage_dir / "request_manifest.json", stage_dir / "request_index.jsonl"]
    artifact_paths.extend(stage_dir / item["path"] for item in request_manifest["shards"])
    manifest["status"] = f"waiting_for_{stage}_responses"
    record_stage(
        run_dir,
        manifest,
        request_stage,
        status="complete",
        artifact_paths=artifact_paths,
        details={
            "model": model,
            "reasoning_effort": reasoning_effort,
            "request_count": request_manifest["request_count"],
            "shard_count": request_manifest["shard_count"],
        },
    )
    return request_manifest


def _snapshot_raw_responses(stage_dir: Path, response_files: str | Path | Iterable[str | Path]) -> tuple[list[Path], Path]:
    source_records = [
        (source, {"bytes": source.stat().st_size, "sha256": sha256_file(source)})
        for source in normalize_response_files(response_files)
    ]
    source_records.sort(key=lambda item: (item[1]["sha256"], item[1]["bytes"]))
    source_metadata = [metadata for _, metadata in source_records]
    bundle_id = sha256_bytes(canonical_json_bytes(source_metadata))
    raw_root = stage_dir / "raw"
    destination = raw_root / bundle_id
    manifest_path = destination / "raw_manifest.json"
    if destination.exists():
        if not manifest_path.is_file():
            raise ArtifactIntegrityError(f"Incomplete raw response snapshot: {destination}")
        recorded = load_json(manifest_path)
        if recorded.get("bundle_id") != bundle_id or recorded.get("sources") != source_metadata:
            raise ArtifactIntegrityError(f"Raw response snapshot is inconsistent: {destination}")
        copied = [destination / item["path"] for item in recorded.get("files", [])]
        for path, metadata in zip(copied, recorded.get("files", []), strict=True):
            if not path.is_file() or path.stat().st_size != metadata.get("bytes") or sha256_file(path) != metadata.get("sha256"):
                raise ArtifactIntegrityError(f"Raw response snapshot changed: {path}")
        return copied, manifest_path

    raw_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{bundle_id}.staging-", dir=raw_root))
    try:
        files: list[dict[str, Any]] = []
        for index, (source, expected) in enumerate(source_records):
            name = f"response-{index:05d}.jsonl"
            target = staging / name
            shutil.copyfile(source, target)
            actual = {"path": name, "bytes": target.stat().st_size, "sha256": sha256_file(target)}
            if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
                raise ArtifactIntegrityError(f"Response source changed while being copied: {source}")
            files.append(actual)
        raw_manifest = {
            "schema_version": 1,
            "bundle_id": bundle_id,
            "sources": source_metadata,
            "files": files,
        }
        save_json(raw_manifest, staging / "raw_manifest.json")
        os.replace(staging, destination)
        return [destination / item["path"] for item in files], destination / "raw_manifest.json"
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def ingest_model_responses(
    output_dir: str | Path,
    stage: str,
    response_files: str | Path | Iterable[str | Path],
    *,
    provenance_paths: Iterable[str | Path] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Snapshot, strictly reconcile, and record a complete offline response bundle."""

    stage = _validate_model_stage(stage)
    run_dir = Path(output_dir)
    manifest = verify_manifest(run_dir)
    require_stage(manifest, _request_stage(stage))
    response_stage = _response_stage(stage)
    existing = manifest.get("stages", {}).get(response_stage)
    if isinstance(existing, dict) and existing.get("status") == "complete":
        return (
            load_jsonl(batch_stage_dir(run_dir, stage) / "responses.jsonl"),
            load_json(batch_stage_dir(run_dir, stage) / "response_manifest.json"),
        )

    stage_dir = batch_stage_dir(run_dir, stage)
    copied_sources, raw_manifest = _snapshot_raw_responses(stage_dir, response_files)
    records, diagnostics = reconcile_responses(stage_dir, copied_sources)
    selected = _selected_pairs(run_dir)
    responses_by_pair = _unique_by(records, "pair_index", label=f"{stage} responses")
    _require_exact_selected_indices(selected, responses_by_pair, label=f"{stage} responses")
    if stage == "positive":
        selected_uids = {
            pair[field]
            for pair in selected
            for field in ("case_a_uid", "case_b_uid")
        }
        cases_by_uid = _selected_jsonl_records(
            run_dir / CASES,
            field="patient_uid",
            selected_values=selected_uids,
            label="cases",
        )
        finalize_positive_responses(records, selected, cases_by_uid)
    else:
        for record in records:
            validate_intermediate_response(stage, record["text"])

    raw_description = load_json(raw_manifest)
    raw_files = [raw_manifest.parent / item["path"] for item in raw_description["files"]]
    artifacts = [
        *raw_files,
        raw_manifest,
        stage_dir / "responses.jsonl",
        stage_dir / "response_manifest.json",
    ]
    provenance = [Path(path) for path in provenance_paths]
    for path in provenance:
        try:
            path.resolve().relative_to(run_dir.resolve())
        except ValueError as exc:
            raise ArtifactIntegrityError(
                f"Response provenance must be stored inside the run: {path}"
            ) from exc
        if not path.is_file():
            raise ArtifactIntegrityError(f"Response provenance is missing: {path}")
    artifacts.extend(provenance)
    manifest["status"] = (
        "ready_to_finalize"
        if stage == "positive"
        else f"ready_for_{STAGES[STAGES.index(stage) + 1]}"
    )
    record_stage(
        run_dir,
        manifest,
        response_stage,
        status="complete",
        artifact_paths=artifacts,
        details={
            "request_count": diagnostics["request_count"],
            "response_count": diagnostics["response_count"],
            "raw_bundle_id": raw_description["bundle_id"],
            "provenance_artifacts": len(provenance),
        },
    )
    return records, diagnostics


def finalize_run(output_dir: str | Path) -> dict[str, Any]:
    """Write strict final products; an explicit empty selection needs no API calls."""

    run_dir = Path(output_dir)
    manifest = verify_manifest(run_dir)
    require_stage(manifest, HANDOFF_STAGE)
    existing = manifest.get("stages", {}).get("08_finalize")
    if (
        manifest.get("status") == "complete"
        and isinstance(existing, dict)
        and existing.get("status") == "complete"
    ):
        verify_run(run_dir)
        return load_json(run_dir / FINAL_STATS)

    selected = _selected_pairs(run_dir)
    if selected:
        require_stage(manifest, _response_stage("positive"))
        response_records = load_jsonl(batch_stage_dir(run_dir, "positive") / "responses.jsonl")
    else:
        response_records = []
    # Validate every completed upstream request/response contract before
    # deriving terminal artifacts.
    verify_run(run_dir)
    cases = load_jsonl(run_dir / CASES)
    cases_by_uid = _unique_by(cases, "patient_uid", label="cases")
    result = finalize_positive_responses(response_records, selected, cases_by_uid)

    save_jsonl(result["labeled_all"], run_dir / FINAL_LABELED)
    save_json(result["pairs_clean_all"], run_dir / FINAL_PAIRS)
    save_json(result["text_pairs_all"], run_dir / FINAL_TEXT)
    save_json(result["by_bucket"], run_dir / FINAL_BUCKETS)
    save_json(result["stats"], run_dir / FINAL_STATS)
    _verify_final_artifacts(run_dir, selected_count=len(selected))
    manifest["status"] = "complete"
    record_stage(
        run_dir,
        manifest,
        "08_finalize",
        status="complete",
        artifact_paths=(
            run_dir / FINAL_LABELED,
            run_dir / FINAL_PAIRS,
            run_dir / FINAL_TEXT,
            run_dir / FINAL_BUCKETS,
            run_dir / FINAL_STATS,
        ),
        details=result["stats"],
    )
    verify_run(run_dir)
    return result["stats"]


def _verify_request_semantics(
    run_dir: Path,
    manifest: dict[str, Any],
    stage: str,
    expected_indices: set[int],
) -> None:
    stage_dir = batch_stage_dir(run_dir, stage)
    request_manifest = load_json(stage_dir / "request_manifest.json")
    index = load_jsonl(stage_dir / "request_index.jsonl")
    index_by_custom_id = _unique_by(index, "custom_id", label=f"{stage} request index")
    if {record.get("pair_index") for record in index} != expected_indices:
        raise ArtifactIntegrityError(
            f"{stage} request indices differ from immutable selection"
        )
    if request_manifest.get("request_count") != len(index):
        raise ArtifactIntegrityError(f"{stage} request manifest count is inconsistent")
    model = manifest["config"]["models"][stage]
    expected_prompt_hash = manifest["prompts"]["sha256"][stage]
    seen: set[str] = set()
    for shard in request_manifest.get("shards", []):
        for envelope in iter_jsonl(stage_dir / shard["path"]):
            custom_id = envelope.get("custom_id")
            if custom_id not in index_by_custom_id or custom_id in seen:
                raise ArtifactIntegrityError(
                    f"{stage} shard contains an unexpected or duplicate custom_id"
                )
            seen.add(custom_id)
            if envelope.get("method") != "POST" or envelope.get("url") != "/v1/chat/completions":
                raise ArtifactIntegrityError(f"{stage} request has an invalid API contract")
            body = envelope.get("body")
            if not isinstance(body, dict) or body.get("model") != model["model"]:
                raise ArtifactIntegrityError(f"{stage} request model differs from run config")
            allowed_body_keys = {"model", "messages"}
            expected_effort = model.get("reasoning_effort")
            if expected_effort is not None:
                allowed_body_keys.add("reasoning_effort")
            if set(body) != allowed_body_keys or body.get("reasoning_effort") != expected_effort:
                raise ArtifactIntegrityError(
                    f"{stage} request parameters differ from run config"
                )
            messages = body.get("messages")
            if not isinstance(messages, list) or len(messages) != 2:
                raise ArtifactIntegrityError(f"{stage} request messages are malformed")
            developer_message, user_message = messages
            if (
                not isinstance(developer_message, dict)
                or developer_message.get("role") != "developer"
                or not isinstance(developer_message.get("content"), str)
                or sha256_bytes(developer_message["content"].encode("utf-8"))
                != expected_prompt_hash
            ):
                raise ArtifactIntegrityError(
                    f"{stage} request prompt differs from the recorded prompt"
                )
            if (
                not isinstance(user_message, dict)
                or user_message.get("role") != "user"
                or not isinstance(user_message.get("content"), str)
                or not user_message["content"].strip()
            ):
                raise ArtifactIntegrityError(f"{stage} request user prompt is empty")
            request_core = {
                "body": body,
                "method": "POST",
                "url": "/v1/chat/completions",
            }
            request_sha256 = sha256_bytes(canonical_json_bytes(request_core))
            index_record = index_by_custom_id[custom_id]
            expected_custom_id = (
                f"{stage}-{index_record.get('ordinal'):08d}-{request_sha256[:16]}"
                if isinstance(index_record.get("ordinal"), int)
                else None
            )
            if (
                index_record.get("request_sha256") != request_sha256
                or custom_id != expected_custom_id
            ):
                raise ArtifactIntegrityError(
                    f"{stage} request index is not bound to its request body"
                )
    if seen != set(index_by_custom_id):
        raise ArtifactIntegrityError(f"{stage} shards do not cover the request index")


def _verify_final_artifacts(run_dir: Path, *, selected_count: int) -> None:
    labeled = load_jsonl(run_dir / FINAL_LABELED)
    pairs = load_json(run_dir / FINAL_PAIRS)
    text_pairs = load_json(run_dir / FINAL_TEXT)
    buckets = load_json(run_dir / FINAL_BUCKETS)
    stats = load_json(run_dir / FINAL_STATS)
    if not all(isinstance(value, list) for value in (pairs, text_pairs)) or not isinstance(
        buckets, dict
    ):
        raise ArtifactIntegrityError("Final output containers are malformed")
    if not (len(labeled) == len(pairs) == len(text_pairs)):
        raise ArtifactIntegrityError("Final labeled/pair/text outputs are misaligned")
    seen_pairs: set[tuple[str, str]] = set()
    reconstructed: dict[str, dict[str, list[Any]]] = {}
    for position, (record, pair, text_pair) in enumerate(
        zip(labeled, pairs, text_pairs, strict=True)
    ):
        if record.get("pair_id") != pair:
            raise ArtifactIntegrityError(f"Final pair {position} differs from its label")
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not isinstance(value, str) or not value for value in pair)
        ):
            raise ArtifactIntegrityError(f"Final pair {position} is malformed")
        key = (pair[0], pair[1])
        if key in seen_pairs:
            raise ArtifactIntegrityError(f"Final pair {position} is duplicated")
        seen_pairs.add(key)
        if (
            not isinstance(text_pair, list)
            or len(text_pair) != 1
            or not isinstance(text_pair[0], str)
            or not text_pair[0].strip()
        ):
            raise ArtifactIntegrityError(f"Final text pair {position} is malformed")
        bucket = record.get("modality_bucket")
        if not isinstance(bucket, str) or not bucket:
            raise ArtifactIntegrityError(f"Final label {position} has no modality bucket")
        payload = reconstructed.setdefault(
            bucket, {"labeled": [], "pairs_clean": [], "text_pairs": []}
        )
        payload["labeled"].append(record)
        payload["pairs_clean"].append(pair)
        payload["text_pairs"].append(text_pair)
    reconstructed = {key: reconstructed[key] for key in sorted(reconstructed)}
    if buckets != reconstructed:
        raise ArtifactIntegrityError("Final modality buckets are not aligned")
    expected_bucket_sizes = {
        key: len(value["pairs_clean"]) for key, value in reconstructed.items()
    }
    if (
        stats.get("selected_pairs") != selected_count
        or stats.get("requests_processed") != selected_count
        or stats.get("labeled_pairs_kept") != len(labeled)
        or stats.get("bucket_sizes") != expected_bucket_sizes
    ):
        raise ArtifactIntegrityError("Final statistics are inconsistent with outputs")


def _verify_submission_semantics(
    run_dir: Path, manifest: dict[str, Any], stage: str
) -> None:
    stage_dir = batch_stage_dir(run_dir, stage)
    request_manifest = load_json(stage_dir / "request_manifest.json")
    expected_hashes = [item["sha256"] for item in request_manifest["shards"]]
    attempts = _attempt_numbers(run_dir, stage)
    if not attempts or attempts != list(range(1, attempts[-1] + 1)):
        raise ArtifactIntegrityError(f"{stage} managed attempts are not contiguous")
    stage_record = manifest["stages"][_submission_stage(stage)]
    expected_artifacts = {
        path.relative_to(run_dir).as_posix()
        for path in _attempt_audit_artifacts(run_dir, stage)
    }
    if set(stage_record.get("artifacts", [])) != expected_artifacts:
        raise ArtifactIntegrityError(
            f"{stage} managed-attempt artifacts are not fully recorded"
        )

    summaries: list[dict[str, Any]] = []
    for attempt in attempts:
        attempt_root = _attempt_dir(run_dir, stage, attempt)
        receipt_path = _attempt_submission_receipt(run_dir, stage, attempt)
        if not receipt_path.is_file():
            raise ArtifactIntegrityError(
                f"{stage} attempt {attempt} has no immutable submission receipt"
            )
        receipt = load_json(receipt_path)
        submissions = receipt.get("submissions")
        if (
            receipt.get("schema_version") != 2
            or receipt.get("run_spec_id") != manifest.get("run_spec_id")
            or receipt.get("stage") != stage
            or receipt.get("attempt") != attempt
            or receipt.get("request_manifest_sha256")
            != sha256_file(stage_dir / "request_manifest.json")
            or not isinstance(submissions, list)
        ):
            raise ArtifactIntegrityError(
                f"{stage} attempt {attempt} submission receipt is malformed"
            )
        if [item.get("shard_sha256") for item in submissions] != expected_hashes:
            raise ArtifactIntegrityError(
                f"{stage} attempt {attempt} does not cover its request shards"
            )
        submissions_by_hash = {
            item.get("shard_sha256"): item for item in submissions
        }
        for item in submissions:
            if (
                not isinstance(item.get("batch_id"), str)
                or not isinstance(item.get("uploaded_file_id"), str)
                or item.get("endpoint") != "/v1/chat/completions"
                or item.get("completion_window") != "24h"
            ):
                raise ArtifactIntegrityError(
                    f"{stage} attempt {attempt} submission identity is incomplete"
                )
            remote_record = load_json(
                attempt_root
                / "remote"
                / "submissions"
                / f"{item['shard_sha256']}.json"
            )
            if (
                remote_record.get("batch_id") != item["batch_id"]
                or remote_record.get("uploaded_file_id")
                != item["uploaded_file_id"]
                or remote_record.get("endpoint") != item["endpoint"]
            ):
                raise ArtifactIntegrityError(
                    f"{stage} attempt {attempt} remote submission state is inconsistent"
                )

        download_path = _attempt_download_receipt(run_dir, stage, attempt)
        downloaded = download_path.is_file()
        if downloaded:
            download_receipt = load_json(download_path)
            downloads = download_receipt.get("downloads")
            if (
                download_receipt.get("schema_version") != 2
                or download_receipt.get("run_spec_id")
                != manifest.get("run_spec_id")
                or download_receipt.get("stage") != stage
                or download_receipt.get("attempt") != attempt
                or download_receipt.get("submission_receipt_sha256")
                != sha256_file(receipt_path)
                or not isinstance(downloads, list)
                or [item.get("shard_sha256") for item in downloads]
                != expected_hashes
            ):
                raise ArtifactIntegrityError(
                    f"{stage} attempt {attempt} download receipt is malformed"
                )
            for download in downloads:
                submission = submissions_by_hash.get(download.get("shard_sha256"))
                expected_dir = f"remote/downloads/{download.get('batch_id')}"
                if (
                    submission is None
                    or download.get("batch_id") != submission.get("batch_id")
                    or download.get("download_dir") != expected_dir
                ):
                    raise ArtifactIntegrityError(
                        f"{stage} attempt {attempt} download identity is inconsistent"
                    )

        failure_path = _attempt_failure_receipt(run_dir, stage, attempt)
        validation_failed = failure_path.is_file()
        if validation_failed:
            failure = load_json(failure_path)
            if (
                not downloaded
                or failure.get("schema_version") != 1
                or failure.get("run_spec_id") != manifest.get("run_spec_id")
                or failure.get("stage") != stage
                or failure.get("attempt") != attempt
                or failure.get("download_receipt_sha256")
                != sha256_file(download_path)
                or failure.get("error_type") != "ResponseValidationError"
            ):
                raise ArtifactIntegrityError(
                    f"{stage} attempt {attempt} validation-failure receipt is malformed"
                )
        if attempt < attempts[-1] and not validation_failed:
            raise ArtifactIntegrityError(
                f"{stage} attempt {attempt + 1} lacks authorization from a failed prior attempt"
            )
        summaries.append(
            {
                "attempt": attempt,
                "downloaded": downloaded,
                "submitted": True,
                "validation_failed": validation_failed,
            }
        )
    if stage_record.get("details", {}).get("attempts") != summaries:
        raise ArtifactIntegrityError(f"{stage} managed-attempt summary is inconsistent")
    if stage_record.get("details", {}).get("latest_attempt") != attempts[-1]:
        raise ArtifactIntegrityError(f"{stage} latest managed attempt is inconsistent")


def _verify_managed_download_semantics(
    run_dir: Path, manifest: dict[str, Any], stage: str
) -> None:
    stage_dir = batch_stage_dir(run_dir, stage)
    response_record = manifest["stages"][_response_stage(stage)]
    accepted_path = stage_dir / "accepted_attempt.json"
    if not accepted_path.is_file():
        if _bound_managed_attempts(run_dir, stage, response_record):
            raise ArtifactIntegrityError(
                f"{stage} managed responses have no accepted-attempt receipt"
            )
        return
    relative_accepted = accepted_path.relative_to(run_dir).as_posix()
    if relative_accepted not in response_record.get("artifacts", []):
        raise ArtifactIntegrityError(
            f"{stage} accepted managed attempt is not bound to its response stage"
        )
    accepted = load_json(accepted_path)
    attempt = accepted.get("attempt")
    if (
        accepted.get("schema_version") != 1
        or accepted.get("run_spec_id") != manifest.get("run_spec_id")
        or accepted.get("stage") != stage
        or isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or attempt < 1
        or attempt != max(_attempt_numbers(run_dir, stage), default=0)
    ):
        raise ArtifactIntegrityError(f"{stage} accepted-attempt receipt is malformed")
    submission_path = _attempt_submission_receipt(run_dir, stage, attempt)
    download_path = _attempt_download_receipt(run_dir, stage, attempt)
    if (
        accepted.get("submission_receipt_sha256") != sha256_file(submission_path)
        or accepted.get("download_receipt_sha256") != sha256_file(download_path)
        or _attempt_failure_receipt(run_dir, stage, attempt).exists()
    ):
        raise ArtifactIntegrityError(
            f"{stage} accepted attempt is not bound to its immutable receipts"
        )
    receipt = load_json(download_path)
    downloads = receipt.get("downloads")
    raw_bundle_id = response_record.get("details", {}).get("raw_bundle_id")
    if (
        accepted.get("raw_bundle_id") != raw_bundle_id
        or accepted.get("response_manifest_sha256")
        != sha256_file(stage_dir / "response_manifest.json")
        or response_record.get("details", {}).get("managed_attempt") != attempt
    ):
        raise ArtifactIntegrityError(
            f"{stage} accepted attempt is not bound to canonical responses"
        )
    raw_manifest = load_json(
        stage_dir / "raw" / str(raw_bundle_id) / "raw_manifest.json"
    )
    raw_sources = sorted(
        (
            {"bytes": item.get("bytes"), "sha256": item.get("sha256")}
            for item in raw_manifest.get("sources", [])
        ),
        key=lambda item: (str(item["sha256"]), str(item["bytes"])),
    )
    downloaded_jsonl = sorted(
        (
            {"bytes": file.get("bytes"), "sha256": file.get("sha256")}
            for download in downloads
            for file in download.get("files", [])
            if str(file.get("path", "")).endswith(".jsonl")
        ),
        key=lambda item: (str(item["sha256"]), str(item["bytes"])),
    )
    if raw_sources != downloaded_jsonl:
        raise ArtifactIntegrityError(
            f"{stage} raw response snapshot differs from managed downloads"
        )


def _commit_accepted_attempt(run_dir: Path, stage: str, attempt: int) -> None:
    """Idempotently bind one validated managed attempt to canonical responses."""

    attempt = _validate_attempt(attempt)
    manifest = verify_manifest(run_dir)
    response_record = require_stage(manifest, _response_stage(stage))
    bound_attempts = _bound_managed_attempts(run_dir, stage, response_record)
    if bound_attempts != [attempt]:
        raise ArtifactIntegrityError(
            f"{stage} canonical responses are not bound exclusively to managed "
            f"attempt {attempt}; bound={bound_attempts}"
        )
    if attempt != max(_attempt_numbers(run_dir, stage), default=0):
        raise ArtifactIntegrityError(
            f"{stage} attempt {attempt} is not the latest managed attempt"
        )

    stage_dir = batch_stage_dir(run_dir, stage)
    submission_path = _attempt_submission_receipt(run_dir, stage, attempt)
    download_path = _attempt_download_receipt(run_dir, stage, attempt)
    if _attempt_failure_receipt(run_dir, stage, attempt).exists():
        raise ArtifactIntegrityError(
            f"{stage} attempt {attempt} has a validation-failure receipt"
        )
    accepted = {
        "attempt": attempt,
        "download_receipt_sha256": sha256_file(download_path),
        "raw_bundle_id": response_record["details"]["raw_bundle_id"],
        "response_manifest_sha256": sha256_file(stage_dir / "response_manifest.json"),
        "run_spec_id": manifest["run_spec_id"],
        "schema_version": 1,
        "stage": stage,
        "submission_receipt_sha256": sha256_file(submission_path),
    }
    accepted_path = stage_dir / "accepted_attempt.json"
    _save_immutable_json(accepted, accepted_path)

    accepted_relative = accepted_path.relative_to(run_dir).as_posix()
    response_artifacts = [
        run_dir / relative
        for relative in response_record.get("artifacts", [])
        if relative != accepted_relative
    ]
    response_details = dict(response_record.get("details", {}))
    response_details["managed_attempt"] = attempt
    record_stage(
        run_dir,
        manifest,
        _response_stage(stage),
        status="complete",
        artifact_paths=(*response_artifacts, accepted_path),
        details=response_details,
    )


def verify_run(output_dir: str | Path) -> dict[str, Any]:
    """Verify recorded hashes plus cross-stage selected-pair alignment."""

    run_dir = Path(output_dir)
    manifest = verify_manifest(run_dir)
    require_stage(manifest, HANDOFF_STAGE)
    population_boundary = verify_imported_population_boundary(run_dir)
    handoff_details = manifest["stages"][HANDOFF_STAGE].get("details")
    if not isinstance(handoff_details, dict) or any(
        handoff_details.get(field) != population_boundary[field]
        for field in (
            "ruleset",
            "cases_screened",
            "pairs_screened",
            "human_pairs",
            "animal_pairs",
        )
    ):
        raise ArtifactIntegrityError(
            "Pipeline 03 handoff details do not match the human/animal-screen snapshots"
        )
    selected = _selected_pairs(run_dir)
    expected = {pair["pair_index"] for pair in selected}
    completed_request_stages: list[str] = []
    completed_model_stages: list[str] = []
    for stage in STAGES:
        request_record = manifest.get("stages", {}).get(_request_stage(stage))
        if isinstance(request_record, dict) and request_record.get("status") == "complete":
            if stage != "citation":
                prior = STAGES[STAGES.index(stage) - 1]
                require_stage(manifest, _response_stage(prior))
            _verify_request_semantics(run_dir, manifest, stage, expected)
            completed_request_stages.append(stage)
        submission_record = manifest.get("stages", {}).get(_submission_stage(stage))
        if isinstance(submission_record, dict) and submission_record.get("status") == "complete":
            require_stage(manifest, _request_stage(stage))
            _verify_submission_semantics(run_dir, manifest, stage)
        response_record = manifest.get("stages", {}).get(_response_stage(stage))
        if isinstance(response_record, dict) and response_record.get("status") == "complete":
            require_stage(manifest, _request_stage(stage))
            response_map = _responses_by_pair(run_dir, stage)
            if set(response_map) != expected:
                raise ArtifactIntegrityError(
                    f"{stage} response indices differ from immutable selection"
                )
            if stage == "positive":
                selected_uids = {
                    pair[field]
                    for pair in selected
                    for field in ("case_a_uid", "case_b_uid")
                }
                cases_by_uid = _selected_jsonl_records(
                    run_dir / CASES,
                    field="patient_uid",
                    selected_values=selected_uids,
                    label="cases",
                )
                finalize_positive_responses(
                    list(response_map.values()), selected, cases_by_uid
                )
            else:
                for record in response_map.values():
                    validate_intermediate_response(stage, record["text"])
            _verify_managed_download_semantics(run_dir, manifest, stage)
            completed_model_stages.append(stage)
    if completed_model_stages != list(STAGES[: len(completed_model_stages)]):
        raise StageOrderError("Completed model-response stages are not a contiguous prefix")
    if completed_request_stages != list(STAGES[: len(completed_request_stages)]):
        raise StageOrderError("Completed request stages are not a contiguous prefix")
    final_record = manifest.get("stages", {}).get("08_finalize")
    if isinstance(final_record, dict) and final_record.get("status") == "complete":
        _verify_final_artifacts(run_dir, selected_count=len(selected))
    if manifest.get("status") == "complete":
        require_stage(manifest, "08_finalize")
        if selected and completed_model_stages != list(STAGES):
            raise StageOrderError("A nonempty completed run is missing model-response stages")
    return {
        "run_spec_id": manifest["run_spec_id"],
        "status": manifest["status"],
        "selected_pairs": len(selected),
        "completed_request_stages": completed_request_stages,
        "completed_model_stages": completed_model_stages,
        "recorded_artifacts": len(manifest.get("artifacts", {})),
    }


def _batch_manager(
    run_dir: Path, stage: str, attempt: int, client: Any
) -> BatchManager:
    return BatchManager(
        client=client,
        state_dir=_attempt_dir(run_dir, stage, attempt) / "remote",
    )


def submit_model_stage(
    output_dir: str | Path,
    stage: str,
    *,
    client: Any,
    acknowledge_external_api: bool,
    attempt: int = 1,
) -> list[dict[str, Any]]:
    stage = _validate_model_stage(stage)
    attempt = _validate_attempt(attempt)
    run_dir = Path(output_dir)
    manifest = verify_manifest(run_dir)
    require_stage(manifest, _request_stage(stage))
    _require_attempt_generation(run_dir, manifest, stage, attempt)
    stage_dir = batch_stage_dir(run_dir, stage)
    request_manifest = load_json(stage_dir / "request_manifest.json")
    receipt_path = _attempt_submission_receipt(run_dir, stage, attempt)
    if receipt_path.is_file():
        receipt = load_json(receipt_path)
        _record_attempt_audit(run_dir, stage)
        return receipt["submissions"]
    manager = _batch_manager(run_dir, stage, attempt, client)
    records: list[dict[str, Any]] = []
    for shard in request_manifest["shards"]:
        records.append(
            manager.submit_shard(
                stage_dir / shard["path"],
                expected_sha256=shard["sha256"],
                stage=stage,
                metadata={
                    "attempt": str(attempt),
                    "run_spec_id": manifest["run_spec_id"][:64],
                },
                acknowledge_external_api=acknowledge_external_api,
            )
        )
    submissions = [
        {
            "batch": record.get("batch"),
            "batch_id": record.get("batch_id"),
            "completion_window": record.get("completion_window"),
            "endpoint": record.get("endpoint"),
            "initial_remote_status": record.get("remote_status"),
            "metadata": record.get("metadata"),
            "shard_bytes": record.get("shard_bytes"),
            "shard_sha256": record.get("shard_sha256"),
            "uploaded_file_id": record.get("uploaded_file_id"),
        }
        for record in records
    ]
    receipt = {
        "attempt": attempt,
        "request_manifest_sha256": sha256_file(
            stage_dir / "request_manifest.json"
        ),
        "run_spec_id": manifest["run_spec_id"],
        "schema_version": 2,
        "stage": stage,
        "submissions": submissions,
    }
    _save_immutable_json(receipt, receipt_path)
    _record_attempt_audit(run_dir, stage)
    return submissions


def retry_model_stage(
    output_dir: str | Path,
    stage: str,
    *,
    attempt: int,
    client: Any,
    acknowledge_external_api: bool,
) -> list[dict[str, Any]]:
    """Submit a new immutable attempt after the immediately prior validation failure."""

    attempt = _validate_attempt(attempt)
    if attempt < 2:
        raise ValueError("retry requires --attempt 2 or greater")
    return submit_model_stage(
        output_dir,
        stage,
        client=client,
        acknowledge_external_api=acknowledge_external_api,
        attempt=attempt,
    )


def status_model_stage(
    output_dir: str | Path,
    stage: str,
    *,
    client: Any,
    acknowledge_external_api: bool,
    attempt: int = 1,
) -> list[dict[str, Any]]:
    stage = _validate_model_stage(stage)
    attempt = _validate_attempt(attempt)
    run_dir = Path(output_dir)
    manifest = verify_manifest(run_dir)
    require_stage(manifest, _submission_stage(stage))
    submission_receipt = _attempt_submission_receipt(run_dir, stage, attempt)
    if not submission_receipt.is_file():
        raise StageOrderError(f"Managed attempt {attempt} has not been submitted")
    request_manifest = load_json(batch_stage_dir(run_dir, stage) / "request_manifest.json")
    manager = _batch_manager(run_dir, stage, attempt, client)
    statuses = [
        manager.retrieve_status(
            shard["sha256"],
            acknowledge_external_api=acknowledge_external_api,
        )
        for shard in request_manifest["shards"]
    ]
    receipt = {
        "attempt": attempt,
        "run_spec_id": manifest["run_spec_id"],
        "schema_version": 1,
        "stage": stage,
        "statuses": statuses,
        "submission_receipt_sha256": sha256_file(submission_receipt),
    }
    receipt_id = sha256_bytes(canonical_json_bytes(receipt))
    status_path = (
        _attempt_dir(run_dir, stage, attempt)
        / "statuses"
        / f"status-{receipt_id}.json"
    )
    _save_immutable_json(receipt, status_path)
    _record_attempt_audit(run_dir, stage)
    return statuses


def resolve_ambiguous_submission(
    output_dir: str | Path,
    stage: str,
    *,
    shard_sha256: str,
    batch_id: str,
    client: Any,
    acknowledge_external_api: bool,
    attempt: int = 1,
) -> dict[str, Any]:
    stage = _validate_model_stage(stage)
    attempt = _validate_attempt(attempt)
    run_dir = Path(output_dir)
    manifest = verify_manifest(run_dir)
    require_stage(manifest, _request_stage(stage))
    _require_attempt_generation(run_dir, manifest, stage, attempt)
    request_manifest = load_json(batch_stage_dir(run_dir, stage) / "request_manifest.json")
    expected = {item["sha256"] for item in request_manifest["shards"]}
    if shard_sha256 not in expected:
        raise ArtifactIntegrityError("Shard SHA-256 is not part of this stage")
    resolved = _batch_manager(run_dir, stage, attempt, client).resolve_ambiguous_create(
        shard_sha256,
        batch_id=batch_id,
        acknowledge_external_api=acknowledge_external_api,
    )
    resolution = {
        "attempt": attempt,
        "batch_id": batch_id,
        "resolved_submission": resolved,
        "run_spec_id": manifest["run_spec_id"],
        "schema_version": 1,
        "shard_sha256": shard_sha256,
        "stage": stage,
    }
    resolution_path = (
        _attempt_dir(run_dir, stage, attempt)
        / "resolutions"
        / f"{shard_sha256}.json"
    )
    _save_immutable_json(resolution, resolution_path)
    _record_attempt_audit(run_dir, stage)
    return resolved


def download_and_ingest_model_stage(
    output_dir: str | Path,
    stage: str,
    *,
    client: Any,
    acknowledge_external_api: bool,
    attempt: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage = _validate_model_stage(stage)
    attempt = _validate_attempt(attempt)
    run_dir = Path(output_dir)
    manifest = verify_manifest(run_dir)
    require_stage(manifest, _submission_stage(stage))
    stage_dir = batch_stage_dir(run_dir, stage)
    completed_response = manifest.get("stages", {}).get(_response_stage(stage))
    if isinstance(completed_response, dict) and completed_response.get("status") == "complete":
        bound_attempts = _bound_managed_attempts(run_dir, stage, completed_response)
        if not bound_attempts:
            raise StageOrderError(
                f"{stage} responses were ingested offline; refusing a later managed download"
            )
        if bound_attempts != [attempt]:
            raise StageOrderError(
                f"{stage} canonical responses came from managed attempt "
                f"{bound_attempts}, not attempt {attempt}"
            )
        _commit_accepted_attempt(run_dir, stage, attempt)
        manifest = verify_manifest(run_dir)
        _verify_managed_download_semantics(run_dir, manifest, stage)
        return (
            load_jsonl(stage_dir / "responses.jsonl"),
            load_json(stage_dir / "response_manifest.json"),
        )
    submission_receipt = _attempt_submission_receipt(run_dir, stage, attempt)
    if not submission_receipt.is_file():
        raise StageOrderError(f"Managed attempt {attempt} has not been submitted")
    failure_path = _attempt_failure_receipt(run_dir, stage, attempt)
    if failure_path.is_file():
        raise StageOrderError(
            f"Managed attempt {attempt} already failed validation; use retry "
            f"with attempt {attempt + 1}"
        )
    request_manifest = load_json(stage_dir / "request_manifest.json")
    manager = _batch_manager(run_dir, stage, attempt, client)
    response_files: list[Path] = []
    download_records: list[dict[str, Any]] = []
    for shard in request_manifest["shards"]:
        downloaded = manager.download_completed(
            shard["sha256"],
            acknowledge_external_api=acknowledge_external_api,
        )
        download_dir = manager.download_dir / downloaded["batch_id"]
        response_files.append(download_dir / "output.jsonl")
        if downloaded.get("error_file_id") is not None:
            response_files.append(download_dir / "errors.jsonl")
        download_records.append(
            {
                "batch_id": downloaded["batch_id"],
                "error_file_id": downloaded.get("error_file_id"),
                "files": downloaded["files"],
                "download_dir": f"remote/downloads/{downloaded['batch_id']}",
                "output_file_id": downloaded["output_file_id"],
                "shard_sha256": downloaded["shard_sha256"],
            }
        )
    managed_receipt = {
        "attempt": attempt,
        "run_spec_id": manifest["run_spec_id"],
        "schema_version": 2,
        "stage": stage,
        "submission_receipt_sha256": sha256_file(submission_receipt),
        "downloads": download_records,
    }
    managed_receipt_path = _attempt_download_receipt(run_dir, stage, attempt)
    _save_immutable_json(managed_receipt, managed_receipt_path)
    _record_attempt_audit(run_dir, stage)
    try:
        records, diagnostics = ingest_model_responses(
            run_dir,
            stage,
            response_files,
            provenance_paths=(submission_receipt, managed_receipt_path),
        )
    except ResponseValidationError as exc:
        failure = {
            "attempt": attempt,
            "download_receipt_sha256": sha256_file(managed_receipt_path),
            "error_message": str(exc),
            "error_type": "ResponseValidationError",
            "run_spec_id": manifest["run_spec_id"],
            "schema_version": 1,
            "stage": stage,
        }
        _save_immutable_json(failure, failure_path)
        _record_attempt_audit(
            run_dir,
            stage,
            run_status=f"waiting_for_{stage}_retry",
        )
        raise

    _commit_accepted_attempt(run_dir, stage, attempt)
    return records, diagnostics


__all__ = [
    "build_model_stage",
    "download_and_ingest_model_stage",
    "finalize_run",
    "ingest_model_responses",
    "retry_model_stage",
    "resolve_ambiguous_submission",
    "status_model_stage",
    "submit_model_stage",
    "verify_run",
]
