"""Import and validate the immutable Pipeline 02 handoff."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from .artifacts import (
    CASES,
    CITATIONS,
    HANDOFF_CASES,
    HANDOFF_MANIFEST,
    HANDOFF_PAIR_SPECIES,
    HANDOFF_PAIRS,
    HANDOFF_SPECIES_CLASSIFICATIONS,
    HANDOFF_SPECIES_STATS,
    HANDOFF_SOURCE_MANIFEST,
    HANDOFF_STAGE,
    SELECTION,
    SELECTION_HANDOFF_SCHEMA_VERSION,
    SELECTION_PIPELINE_ID,
    SELECTION_PIPELINE_VERSION,
    SOURCE_ELIGIBLE_CASES,
    SOURCE_ELIGIBLE_PAIRS,
    SOURCE_HANDOFF_MANIFEST,
    SOURCE_PAIR_SPECIES,
    SOURCE_SPECIES_CLASSIFICATIONS,
    SOURCE_SPECIES_STATS,
    SUMMARIES,
)
from .config import load_config
from .errors import ArtifactIntegrityError
from .io_utils import (
    load_json,
    load_jsonl,
    save_json,
    save_jsonl,
    sha256_file,
)
from .manifest import external_input_metadata, new_manifest, record_stage, save_manifest


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PATIENT_UID_RE = re.compile(r"^(?P<article>\d+)-(?P<case>\d+)$")
_CASE_FIELDS = {
    "schema_version",
    "patient_uid",
    "pmc_id",
    "nxml_path",
    "nxml_sha256",
    "case_description",
    "captions",
    "population_scope",
}
_POPULATION_SCOPE_FIELDS = {"ruleset", "label"}
_SPECIES_RULESET = "human_animal_v1"
_SPECIES_LABELS = {"human", "animal"}
_SPECIES_CLASSIFICATION_FIELDS = {
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
_PAIR_SPECIES_FIELDS = {
    "schema_version",
    "pair_index",
    "pair_key",
    "case_a_uid",
    "case_b_uid",
    "label",
    "endpoint_labels",
}
_CAPTION_FIELDS = {
    "caption_id",
    "caption",
    "caption_path",
    "caption_sha256",
    "image_path",
    "image_sha256",
}
_PAIR_FIELDS = {
    "schema_version",
    "pair_index",
    "pair_key",
    "case_a_uid",
    "case_b_uid",
    "citation",
}
_CITATION_FIELDS = {
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
}


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactIntegrityError(f"{label} must be a JSON object")
    return value


def _exact_fields(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ArtifactIntegrityError(
            f"{label} has incorrect fields; "
            f"missing={sorted(expected - set(value))}, extra={sorted(set(value) - expected)}"
        )


def _string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or value != value.strip() or (
        not allow_empty and not value
    ):
        qualifier = "a string" if allow_empty else "a non-empty trimmed string"
        raise ArtifactIntegrityError(f"{label} must be {qualifier}")
    return value


def _digest(value: Any, label: str) -> str:
    digest = _string(value, label)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ArtifactIntegrityError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _relative_path(value: Any, label: str) -> str:
    path = _string(value, label)
    if "\\" in path:
        raise ArtifactIntegrityError(f"{label} must use POSIX separators")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or path in {"", "."}:
        raise ArtifactIntegrityError(f"{label} must be a safe relative POSIX path")
    return path


def _uid_sort_key(uid: str) -> tuple[int, int, str]:
    match = _PATIENT_UID_RE.fullmatch(uid)
    if match is None:
        raise ArtifactIntegrityError(f"Invalid canonical patient_uid: {uid!r}")
    normalized = f"{int(match.group('article'))}-{int(match.group('case'))}"
    if uid != normalized:
        raise ArtifactIntegrityError(f"Non-canonical patient_uid: {uid!r}")
    return int(match.group("article")), int(match.group("case")), uid


def _expected_pair_key(case_a_uid: str, case_b_uid: str) -> str:
    return hashlib.sha256(f"{case_a_uid}\0{case_b_uid}".encode("utf-8")).hexdigest()


def _source_file(selection_root: Path, relative: str) -> Path:
    candidate = selection_root / relative
    if not candidate.is_file():
        raise FileNotFoundError(f"Selection handoff artifact is missing: {candidate}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(selection_root)
    except ValueError as exc:
        raise ArtifactIntegrityError(
            f"Selection handoff artifact escapes its run directory: {relative}"
        ) from exc
    return resolved


def _artifact_contract(
    handoff: dict[str, Any], role: str, expected_path: str, source: Path
) -> None:
    artifacts = _mapping(handoff.get("artifacts"), "handoff.artifacts")
    entry = _mapping(artifacts.get(role), f"handoff.artifacts.{role}")
    if set(entry) != {"path", "records", "bytes", "sha256"}:
        raise ArtifactIntegrityError(f"handoff artifact contract is malformed: {role}")
    if entry.get("path") != expected_path:
        raise ArtifactIntegrityError(f"handoff {role} path is not canonical")
    records = entry.get("records")
    if isinstance(records, bool) or not isinstance(records, int) or records < 0:
        raise ArtifactIntegrityError(f"handoff {role} record count is invalid")
    if entry.get("bytes") != source.stat().st_size:
        raise ArtifactIntegrityError(f"handoff {role} byte count does not match")
    if _digest(entry.get("sha256"), f"handoff.artifacts.{role}.sha256") != sha256_file(source):
        raise ArtifactIntegrityError(f"handoff {role} SHA-256 does not match")


def _validate_handoff_header(
    selection_manifest: dict[str, Any], handoff: dict[str, Any]
) -> None:
    if handoff.get("schema_version") != SELECTION_HANDOFF_SCHEMA_VERSION:
        raise ArtifactIntegrityError("Unsupported selection handoff schema")
    producer = _mapping(handoff.get("producer"), "handoff.producer")
    if set(producer) != {"pipeline_id", "pipeline_version", "run_spec_id"}:
        raise ArtifactIntegrityError("handoff producer contract is malformed")
    if producer.get("pipeline_id") != SELECTION_PIPELINE_ID:
        raise ArtifactIntegrityError("Handoff came from a different pipeline")
    if producer.get("pipeline_version") != SELECTION_PIPELINE_VERSION:
        raise ArtifactIntegrityError("Unsupported case-pair selection pipeline version")
    _digest(producer.get("run_spec_id"), "handoff.producer.run_spec_id")
    if (
        selection_manifest.get("schema_version") != 1
        or selection_manifest.get("pipeline_id") != SELECTION_PIPELINE_ID
        or selection_manifest.get("pipeline_version") != SELECTION_PIPELINE_VERSION
        or selection_manifest.get("status") != "complete"
        or selection_manifest.get("run_spec_id") != producer["run_spec_id"]
        or selection_manifest.get("handoff_contract") != handoff
    ):
        raise ArtifactIntegrityError(
            "Selection run manifest is incomplete or does not bind this handoff"
        )
    contracts = _mapping(handoff.get("contracts"), "handoff.contracts")
    expected_contracts = {
        "case_schema_version": 2,
        "pair_schema_version": 1,
        "population_filter": "complete_case_pair_human_animal_screen_required",
        "case_order": "patient_uid_numeric_ascending",
        "pair_order": "pair_index_ascending",
        "pair_indices": "stable_non_contiguous",
        "case_orientation": "canonical_numeric_uid_order",
    }
    if contracts != expected_contracts:
        raise ArtifactIntegrityError("Selection handoff semantic contract is unsupported")
    artifacts = _mapping(handoff.get("artifacts"), "handoff.artifacts")
    expected_roles = {
        "eligible_cases",
        "eligible_pairs",
        "species_classifications",
        "pair_species",
        "species_stats",
    }
    if set(artifacts) != expected_roles:
        raise ArtifactIntegrityError(
            "Selection handoff artifact roles are incomplete or unsupported"
        )


def _validate_producer_artifacts(
    selection_manifest: dict[str, Any],
    sources: dict[str, Path],
) -> None:
    stages = _mapping(selection_manifest.get("stages"), "selection run stages")
    handoff_stage = _mapping(stages.get("03_handoff"), "selection 03_handoff stage")
    recorded_stage_paths = handoff_stage.get("artifacts")
    if handoff_stage.get("status") != "complete" or not isinstance(
        recorded_stage_paths, list
    ):
        raise ArtifactIntegrityError("Selection run has no completed 03_handoff stage")
    artifacts = _mapping(
        selection_manifest.get("artifacts"), "selection run artifacts"
    )
    for relative, source in sources.items():
        if relative not in recorded_stage_paths:
            raise ArtifactIntegrityError(
                f"Selection 03_handoff stage does not record {relative}"
            )
        metadata = _mapping(
            artifacts.get(relative), f"selection artifact metadata for {relative}"
        )
        if (
            metadata.get("path") != relative
            or metadata.get("bytes") != source.stat().st_size
            or metadata.get("sha256") != sha256_file(source)
        ):
            raise ArtifactIntegrityError(
                f"Selection run manifest does not bind {relative}"
            )


def _configured_species_ruleset(selection_manifest: dict[str, Any]) -> str:
    config = _mapping(selection_manifest.get("config"), "selection run config")
    species_filter = _mapping(
        config.get("species_filter"),
        "selection run species_filter config",
    )
    _exact_fields(
        species_filter,
        {"ruleset"},
        "selection run species_filter config",
    )
    ruleset = _string(
        species_filter.get("ruleset"),
        "selection run species_filter.ruleset",
    )
    if ruleset != _SPECIES_RULESET:
        raise ArtifactIntegrityError(
            f"Unsupported species-filter ruleset: {ruleset!r}"
        )
    return ruleset


def _validate_species_stage_artifacts(
    selection_manifest: dict[str, Any],
    sources: dict[str, Path],
) -> str:
    expected_paths = {
        SOURCE_SPECIES_CLASSIFICATIONS,
        SOURCE_PAIR_SPECIES,
        SOURCE_SPECIES_STATS,
    }
    if set(sources) != expected_paths:
        raise ArtifactIntegrityError(
            "Internal human/animal-screen source table is incomplete"
        )
    stages = _mapping(selection_manifest.get("stages"), "selection run stages")
    stage = _mapping(
        stages.get("02_human_animal_screen"),
        "selection 02_human_animal_screen stage",
    )
    recorded_paths = stage.get("artifacts")
    if (
        stage.get("status") != "complete"
        or not isinstance(recorded_paths, list)
        or set(recorded_paths) != expected_paths
        or len(recorded_paths) != len(expected_paths)
    ):
        raise ArtifactIntegrityError(
            "Selection run has no complete canonical 02_human_animal_screen stage"
        )
    artifacts = _mapping(
        selection_manifest.get("artifacts"), "selection run artifacts"
    )
    for relative, source in sources.items():
        metadata = _mapping(
            artifacts.get(relative), f"selection artifact metadata for {relative}"
        )
        if (
            set(metadata) != {"path", "bytes", "sha256"}
            or metadata.get("path") != relative
            or metadata.get("bytes") != source.stat().st_size
            or _digest(
                metadata.get("sha256"),
                f"selection artifact metadata for {relative}.sha256",
            )
            != sha256_file(source)
        ):
            raise ArtifactIntegrityError(
                f"Selection run manifest does not bind {relative}"
            )

    ruleset = _configured_species_ruleset(selection_manifest)
    details = _mapping(stage.get("details"), "selection 02_human_animal_screen details")
    if details.get("ruleset") != ruleset:
        raise ArtifactIntegrityError(
            "Human/animal-screen stage ruleset differs from the configured ruleset"
        )
    return ruleset


def _validate_cases(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    caption_ids: set[str] = set()
    ordered_uids: list[str] = []
    for position, raw in enumerate(records):
        case = _mapping(raw, f"eligible case {position}")
        _exact_fields(case, _CASE_FIELDS, f"eligible case {position}")
        if case.get("schema_version") != 2:
            raise ArtifactIntegrityError("Eligible case schema_version must be 2")
        uid = _string(case.get("patient_uid"), f"eligible case {position}.patient_uid")
        _uid_sort_key(uid)
        if uid in cases:
            raise ArtifactIntegrityError(f"Duplicate eligible case UID: {uid}")
        pmc_id = _string(case.get("pmc_id"), f"eligible case {uid}.pmc_id")
        if not pmc_id.isdigit() or pmc_id != str(int(pmc_id)) or uid.split("-", 1)[0] != pmc_id:
            raise ArtifactIntegrityError(f"Eligible case {uid} has a non-canonical pmc_id")
        _relative_path(case.get("nxml_path"), f"eligible case {uid}.nxml_path")
        _digest(case.get("nxml_sha256"), f"eligible case {uid}.nxml_sha256")
        _string(case.get("case_description"), f"eligible case {uid}.case_description")
        population_scope = _mapping(
            case.get("population_scope"),
            f"eligible case {uid}.population_scope",
        )
        _exact_fields(
            population_scope,
            _POPULATION_SCOPE_FIELDS,
            f"eligible case {uid}.population_scope",
        )
        _string(
            population_scope.get("ruleset"),
            f"eligible case {uid}.population_scope.ruleset",
        )
        if population_scope.get("label") != "human":
            raise ArtifactIntegrityError(
                f"Eligible case {uid} population_scope.label must be 'human'"
            )
        captions = case.get("captions")
        if not isinstance(captions, list) or not captions:
            raise ArtifactIntegrityError(f"Eligible case {uid} must have captions")
        local_caption_ids: set[str] = set()
        for caption_position, raw_caption in enumerate(captions):
            caption = _mapping(raw_caption, f"eligible case {uid} caption {caption_position}")
            _exact_fields(
                caption,
                _CAPTION_FIELDS,
                f"eligible case {uid} caption {caption_position}",
            )
            caption_id = _string(
                caption.get("caption_id"),
                f"eligible case {uid} caption {caption_position}.caption_id",
            )
            if caption_id in local_caption_ids or caption_id in caption_ids:
                raise ArtifactIntegrityError(f"Duplicate handoff caption_id: {caption_id}")
            local_caption_ids.add(caption_id)
            caption_ids.add(caption_id)
            _string(
                caption.get("caption"),
                f"eligible case {uid} caption {caption_id}.caption",
            )
            for path_field in ("caption_path", "image_path"):
                _relative_path(
                    caption.get(path_field),
                    f"eligible case {uid} caption {caption_id}.{path_field}",
                )
            for hash_field in ("caption_sha256", "image_sha256"):
                _digest(
                    caption.get(hash_field),
                    f"eligible case {uid} caption {caption_id}.{hash_field}",
                )
        cases[uid] = case
        ordered_uids.append(uid)
    if ordered_uids != sorted(ordered_uids, key=_uid_sort_key):
        raise ArtifactIntegrityError("Eligible cases are not in canonical numeric UID order")
    return cases


def _validate_pairs(
    records: list[dict[str, Any]], cases: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    indices: set[int] = set()
    keys: set[str] = set()
    referenced_cases: set[str] = set()
    previous_index = -1
    for position, raw in enumerate(records):
        pair = _mapping(raw, f"eligible pair {position}")
        _exact_fields(pair, _PAIR_FIELDS, f"eligible pair {position}")
        if pair.get("schema_version") != 1:
            raise ArtifactIntegrityError("Eligible pair schema_version must be 1")
        pair_index = pair.get("pair_index")
        if (
            isinstance(pair_index, bool)
            or not isinstance(pair_index, int)
            or pair_index < 0
            or pair_index <= previous_index
            or pair_index in indices
        ):
            raise ArtifactIntegrityError(
                "Eligible pair indices must be unique, nonnegative, and ascending"
            )
        previous_index = pair_index
        indices.add(pair_index)
        case_a_uid = _string(pair.get("case_a_uid"), f"eligible pair {pair_index}.case_a_uid")
        case_b_uid = _string(pair.get("case_b_uid"), f"eligible pair {pair_index}.case_b_uid")
        if _uid_sort_key(case_a_uid) >= _uid_sort_key(case_b_uid):
            raise ArtifactIntegrityError(f"Eligible pair {pair_index} is not canonically oriented")
        if case_a_uid not in cases or case_b_uid not in cases:
            raise ArtifactIntegrityError(f"Eligible pair {pair_index} references a missing case")
        referenced_cases.update((case_a_uid, case_b_uid))
        pair_key = _digest(pair.get("pair_key"), f"eligible pair {pair_index}.pair_key")
        if pair_key != _expected_pair_key(case_a_uid, case_b_uid) or pair_key in keys:
            raise ArtifactIntegrityError(f"Eligible pair {pair_index} has an invalid pair_key")
        keys.add(pair_key)

        citation = _mapping(pair.get("citation"), f"eligible pair {pair_index}.citation")
        _exact_fields(citation, _CITATION_FIELDS, f"eligible pair {pair_index}.citation")
        if citation.get("status") != "matched":
            raise ArtifactIntegrityError(f"Eligible pair {pair_index} has no matched citation")
        for field in (
            "direction",
            "citing_uid",
            "cited_uid",
            "match_type",
            "ref_id",
            "citing_title",
            "citing_abstract",
            "cited_title",
            "cited_abstract",
        ):
            _string(
                citation.get(field),
                f"eligible pair {pair_index}.citation.{field}",
                allow_empty=field
                in {
                    "ref_id",
                    "citing_title",
                    "citing_abstract",
                    "cited_title",
                    "cited_abstract",
                },
            )
        if {citation["citing_uid"], citation["cited_uid"]} != {case_a_uid, case_b_uid}:
            raise ArtifactIntegrityError(f"Eligible pair {pair_index} citation UIDs are misaligned")
        paragraphs = citation.get("citation_paragraphs")
        if not isinstance(paragraphs, list) or any(
            not isinstance(item, str) or item != item.strip() or not item
            for item in paragraphs
        ):
            raise ArtifactIntegrityError(
                f"Eligible pair {pair_index} citation paragraphs are malformed"
            )
    if set(cases) != referenced_cases:
        raise ArtifactIntegrityError("Eligible handoff contains unreferenced or missing cases")
    return records


def _validate_species_boundary(
    selection_manifest: dict[str, Any],
    cases: dict[str, dict[str, Any]],
    pairs: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    pair_species: list[dict[str, Any]],
    stats: dict[str, Any],
    *,
    ruleset: str,
) -> dict[str, int | str]:
    if ruleset != _SPECIES_RULESET:
        raise ArtifactIntegrityError(
            f"Unsupported species-filter ruleset: {ruleset!r}"
        )

    classifications_by_uid: dict[str, dict[str, Any]] = {}
    ordered_uids: list[str] = []
    for position, raw in enumerate(classifications):
        record = _mapping(raw, f"species classification {position}")
        _exact_fields(
            record,
            _SPECIES_CLASSIFICATION_FIELDS,
            f"species classification {position}",
        )
        if record.get("schema_version") != 1:
            raise ArtifactIntegrityError(
                "Species classification schema_version must be 1"
            )
        uid = _string(
            record.get("patient_uid"),
            f"species classification {position}.patient_uid",
        )
        _uid_sort_key(uid)
        if uid in classifications_by_uid:
            raise ArtifactIntegrityError(
                f"Duplicate species classification UID: {uid}"
            )
        pmc_id = _string(
            record.get("pmc_id"), f"species classification {uid}.pmc_id"
        )
        if (
            not pmc_id.isdigit()
            or pmc_id != str(int(pmc_id))
            or uid.split("-", 1)[0] != pmc_id
        ):
            raise ArtifactIntegrityError(
                f"Species classification {uid} has a non-canonical pmc_id"
            )
        _digest(
            record.get("nxml_sha256"),
            f"species classification {uid}.nxml_sha256",
        )
        if record.get("ruleset") != ruleset:
            raise ArtifactIntegrityError(
                f"Species classification {uid} uses a different ruleset"
            )
        automatic = record.get("automatic_label")
        final = record.get("final_label")
        if automatic not in _SPECIES_LABELS or final not in _SPECIES_LABELS:
            raise ArtifactIntegrityError(
                f"Species classification {uid} must be human or animal"
            )
        matched_rules = record.get("matched_rules")
        if (
            not isinstance(matched_rules, list)
            or any(not isinstance(item, str) or not item for item in matched_rules)
            or matched_rules != sorted(set(matched_rules))
        ):
            raise ArtifactIntegrityError(
                f"Species classification {uid} has invalid matched_rules"
            )
        if automatic == "animal" and not matched_rules:
            raise ArtifactIntegrityError(
                f"Animal classification {uid} has no supporting rule"
            )
        for field in ("title", "journal_title"):
            _string(
                record.get(field),
                f"species classification {uid}.{field}",
                allow_empty=True,
            )
        adjudication = record.get("adjudication")
        if adjudication is None:
            if final != automatic:
                raise ArtifactIntegrityError(
                    f"Species classification {uid} changes label without adjudication"
                )
        else:
            adjudication = _mapping(
                adjudication, f"species classification {uid}.adjudication"
            )
            _exact_fields(
                adjudication,
                {"label", "reason"},
                f"species classification {uid}.adjudication",
            )
            if (
                adjudication.get("label") not in _SPECIES_LABELS
                or final != adjudication.get("label")
            ):
                raise ArtifactIntegrityError(
                    f"Species classification {uid} has an invalid adjudication label"
                )
            _string(
                adjudication.get("reason"),
                f"species classification {uid}.adjudication.reason",
            )
        classifications_by_uid[uid] = record
        ordered_uids.append(uid)

    if ordered_uids != sorted(ordered_uids, key=_uid_sort_key):
        raise ArtifactIntegrityError(
            "Species classifications are not in canonical UID order"
        )

    species_by_index: dict[int, dict[str, Any]] = {}
    pair_keys: set[str] = set()
    pair_identities: set[tuple[str, str]] = set()
    pair_counts: Counter[str] = Counter()
    previous_index = -1
    for position, raw in enumerate(pair_species):
        record = _mapping(raw, f"pair species {position}")
        _exact_fields(record, _PAIR_SPECIES_FIELDS, f"pair species {position}")
        if record.get("schema_version") != 1:
            raise ArtifactIntegrityError("Pair-species schema_version must be 1")
        pair_index = record.get("pair_index")
        if (
            isinstance(pair_index, bool)
            or not isinstance(pair_index, int)
            or pair_index < 0
            or pair_index <= previous_index
            or pair_index in species_by_index
        ):
            raise ArtifactIntegrityError(
                "Pair-species rows require unique ascending pair indices"
            )
        previous_index = pair_index
        case_a_uid = _string(
            record.get("case_a_uid"), f"pair species {pair_index}.case_a_uid"
        )
        case_b_uid = _string(
            record.get("case_b_uid"), f"pair species {pair_index}.case_b_uid"
        )
        if _uid_sort_key(case_a_uid) >= _uid_sort_key(case_b_uid):
            raise ArtifactIntegrityError(
                f"Pair-species row {pair_index} is not canonically oriented"
            )
        if (
            case_a_uid not in classifications_by_uid
            or case_b_uid not in classifications_by_uid
        ):
            raise ArtifactIntegrityError(
                f"Pair-species row {pair_index} references an unclassified case"
            )
        pair_key = _digest(
            record.get("pair_key"), f"pair species {pair_index}.pair_key"
        )
        if pair_key != _expected_pair_key(case_a_uid, case_b_uid):
            raise ArtifactIntegrityError(
                f"Pair-species row {pair_index} has an invalid pair_key"
            )
        identity = (case_a_uid, case_b_uid)
        if pair_key in pair_keys or identity in pair_identities:
            raise ArtifactIntegrityError("Duplicate pair-species relation")
        expected_endpoint_labels = {
            "case_a": classifications_by_uid[case_a_uid]["final_label"],
            "case_b": classifications_by_uid[case_b_uid]["final_label"],
        }
        endpoint_labels = _mapping(
            record.get("endpoint_labels"),
            f"pair species {pair_index}.endpoint_labels",
        )
        if endpoint_labels != expected_endpoint_labels:
            raise ArtifactIntegrityError(
                f"Pair-species row {pair_index} has inconsistent endpoint labels"
            )
        expected_label = (
            "human"
            if set(expected_endpoint_labels.values()) == {"human"}
            else "animal"
        )
        if record.get("label") != expected_label:
            raise ArtifactIntegrityError(
                f"Pair-species row {pair_index} has the wrong binary label"
            )
        species_by_index[pair_index] = record
        pair_keys.add(pair_key)
        pair_identities.add(identity)
        pair_counts[expected_label] += 1

    for pair in pairs:
        classification = species_by_index.get(pair["pair_index"])
        if classification is None:
            raise ArtifactIntegrityError(
                f"Eligible pair {pair['pair_index']} has no pair-species row"
            )
        if (
            classification["pair_key"] != pair["pair_key"]
            or classification["case_a_uid"] != pair["case_a_uid"]
            or classification["case_b_uid"] != pair["case_b_uid"]
            or classification["label"] != "human"
        ):
            raise ArtifactIntegrityError(
                f"Eligible pair {pair['pair_index']} does not map exactly to a "
                "human pair-species row"
            )

    for uid, case in cases.items():
        classification = classifications_by_uid.get(uid)
        if classification is None:
            raise ArtifactIntegrityError(
                f"Eligible case {uid} has no species classification"
            )
        if (
            classification["pmc_id"] != case["pmc_id"]
            or classification["nxml_sha256"] != case["nxml_sha256"]
            or classification["ruleset"] != ruleset
            or classification["final_label"] != "human"
            or case["population_scope"]
            != {"ruleset": ruleset, "label": "human"}
        ):
            raise ArtifactIntegrityError(
                f"Eligible case {uid} is inconsistent with its human classification"
            )

    stats = _mapping(stats, "species statistics")
    expected_stats_fields = {
        "schema_version",
        "ruleset",
        "workers",
        "cases_screened",
        "automatic_label_counts",
        "final_label_counts",
        "cases_adjudicated",
        "rule_counts",
        "pairs_before_screen",
        "pair_label_counts",
        "human_pairs",
        "animal_pairs",
    }
    _exact_fields(stats, expected_stats_fields, "species statistics")
    if stats.get("schema_version") != 1 or stats.get("ruleset") != ruleset:
        raise ArtifactIntegrityError(
            "Species statistics use the wrong schema or ruleset"
        )
    automatic_counts = Counter(
        record["automatic_label"] for record in classifications
    )
    final_counts = Counter(record["final_label"] for record in classifications)
    rule_counts = Counter(
        reason
        for record in classifications
        for reason in record["matched_rules"]
    )
    required_integer_stats = (
        "workers",
        "cases_screened",
        "cases_adjudicated",
        "pairs_before_screen",
        "human_pairs",
        "animal_pairs",
    )
    if any(
        isinstance(stats.get(field), bool)
        or not isinstance(stats.get(field), int)
        or stats[field] < (1 if field == "workers" else 0)
        for field in required_integer_stats
    ):
        raise ArtifactIntegrityError("Species statistics contain invalid counts")
    if (
        stats["cases_screened"] != len(classifications)
        or stats["workers"] != selection_manifest.get("workers")
        or stats["automatic_label_counts"]
        != dict(sorted(automatic_counts.items()))
        or stats["final_label_counts"] != dict(sorted(final_counts.items()))
        or stats["cases_adjudicated"]
        != sum(record["adjudication"] is not None for record in classifications)
        or stats["rule_counts"] != dict(sorted(rule_counts.items()))
        or stats["pairs_before_screen"] != len(pair_species)
        or stats["pair_label_counts"] != dict(sorted(pair_counts.items()))
        or stats["human_pairs"] != pair_counts.get("human", 0)
        or stats["animal_pairs"] != pair_counts.get("animal", 0)
        or stats["pairs_before_screen"]
        != stats["human_pairs"] + stats["animal_pairs"]
    ):
        raise ArtifactIntegrityError(
            "Species statistics do not align with the complete binary screen"
        )

    stage = _mapping(
        _mapping(selection_manifest.get("stages"), "selection run stages").get(
            "02_human_animal_screen"
        ),
        "selection 02_human_animal_screen stage",
    )
    details = _mapping(
        stage.get("details"), "selection 02_human_animal_screen details"
    )
    if details != {
        "ruleset": ruleset,
        "cases_screened": len(classifications),
        "pairs_screened": len(pair_species),
        "human_pairs": stats["human_pairs"],
        "animal_pairs": stats["animal_pairs"],
    }:
        raise ArtifactIntegrityError(
            "Human/animal-screen stage details do not match its artifacts"
        )
    return {
        "ruleset": ruleset,
        "cases_screened": len(classifications),
        "pairs_screened": len(pair_species),
        "human_pairs": pair_counts.get("human", 0),
        "animal_pairs": pair_counts.get("animal", 0),
    }


def _compatibility_views(
    cases: list[dict[str, Any]], pairs: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    case_rows = [
        {
            "schema_version": 1,
            "patient_uid": case["patient_uid"],
            "pmc_id": case["pmc_id"],
            "nxml_path": case["nxml_path"],
            "nxml_sha256": case["nxml_sha256"],
            "captions": case["captions"],
        }
        for case in cases
    ]
    summaries = [
        {
            "schema_version": 1,
            "patient_uid": case["patient_uid"],
            "status": "valid",
            "summary": case["case_description"],
            "error": None,
        }
        for case in cases
    ]
    selected_pairs = [
        {
            "schema_version": 1,
            "pair_index": pair["pair_index"],
            "pair_key": pair["pair_key"],
            "case_a_uid": pair["case_a_uid"],
            "case_b_uid": pair["case_b_uid"],
        }
        for pair in pairs
    ]
    citations = [
        {
            "pair_index": pair["pair_index"],
            "pair_key": pair["pair_key"],
            "case_a_uid": pair["case_a_uid"],
            "case_b_uid": pair["case_b_uid"],
            **pair["citation"],
        }
        for pair in pairs
    ]
    selection = {
        "schema_version": 1,
        "eligibility_rule": "imported_from_pipeline_02_handoff",
        "limit": None,
        "eligible_count": len(selected_pairs),
        "selected_count": len(selected_pairs),
        "pairs": selected_pairs,
    }
    return case_rows, summaries, citations, selection


def _require_separate_output(selection_run: Path, output_dir: Path) -> None:
    selection_root = selection_run.resolve()
    destination = output_dir.resolve()
    for parent, child, message in (
        (selection_root, destination, "inside the Pipeline 02 selection run"),
        (destination, selection_root, "a parent of the Pipeline 02 selection run"),
    ):
        try:
            child.relative_to(parent)
        except ValueError:
            continue
        raise ValueError(f"Pipeline 03 output must not be {message}: {destination}")


def _initialize_in_directory(
    selection_run: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    selection_root = Path(selection_run).resolve()
    if not selection_root.is_dir():
        raise FileNotFoundError(f"Pipeline 02 selection run is missing: {selection_root}")
    run_dir = Path(output_dir)
    _require_separate_output(selection_root, run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    source_run_manifest = _source_file(selection_root, "run_manifest.json")
    source_handoff = _source_file(selection_root, SOURCE_HANDOFF_MANIFEST)
    source_cases = _source_file(selection_root, SOURCE_ELIGIBLE_CASES)
    source_pairs = _source_file(selection_root, SOURCE_ELIGIBLE_PAIRS)
    source_species_classifications = _source_file(
        selection_root, SOURCE_SPECIES_CLASSIFICATIONS
    )
    source_pair_species = _source_file(selection_root, SOURCE_PAIR_SPECIES)
    source_species_stats = _source_file(selection_root, SOURCE_SPECIES_STATS)
    selection_manifest = _mapping(load_json(source_run_manifest), "selection run manifest")
    handoff = _mapping(load_json(source_handoff), "selection handoff manifest")
    _validate_handoff_header(selection_manifest, handoff)
    _validate_producer_artifacts(
        selection_manifest,
        {
            SOURCE_HANDOFF_MANIFEST: source_handoff,
            SOURCE_ELIGIBLE_CASES: source_cases,
            SOURCE_ELIGIBLE_PAIRS: source_pairs,
            SOURCE_SPECIES_CLASSIFICATIONS: source_species_classifications,
            SOURCE_PAIR_SPECIES: source_pair_species,
            SOURCE_SPECIES_STATS: source_species_stats,
        },
    )
    _artifact_contract(handoff, "eligible_cases", SOURCE_ELIGIBLE_CASES, source_cases)
    _artifact_contract(handoff, "eligible_pairs", SOURCE_ELIGIBLE_PAIRS, source_pairs)
    _artifact_contract(
        handoff,
        "species_classifications",
        SOURCE_SPECIES_CLASSIFICATIONS,
        source_species_classifications,
    )
    _artifact_contract(
        handoff,
        "pair_species",
        SOURCE_PAIR_SPECIES,
        source_pair_species,
    )
    _artifact_contract(
        handoff,
        "species_stats",
        SOURCE_SPECIES_STATS,
        source_species_stats,
    )
    species_sources = {
        SOURCE_SPECIES_CLASSIFICATIONS: source_species_classifications,
        SOURCE_PAIR_SPECIES: source_pair_species,
        SOURCE_SPECIES_STATS: source_species_stats,
    }
    species_ruleset = _validate_species_stage_artifacts(
        selection_manifest,
        species_sources,
    )

    cases = load_jsonl(source_cases)
    pairs = load_jsonl(source_pairs)
    species_classifications = load_jsonl(source_species_classifications)
    pair_species = load_jsonl(source_pair_species)
    species_stats = load_json(source_species_stats)
    cases_by_uid = _validate_cases(cases)
    _validate_pairs(pairs, cases_by_uid)
    species_boundary = _validate_species_boundary(
        selection_manifest,
        cases_by_uid,
        pairs,
        species_classifications,
        pair_species,
        species_stats,
        ruleset=species_ruleset,
    )
    if handoff["artifacts"]["eligible_cases"]["records"] != len(cases):
        raise ArtifactIntegrityError("Eligible case count does not match the handoff")
    if handoff["artifacts"]["eligible_pairs"]["records"] != len(pairs):
        raise ArtifactIntegrityError("Eligible pair count does not match the handoff")
    expected_species_records = {
        "species_classifications": len(species_classifications),
        "pair_species": len(pair_species),
        "species_stats": 1,
    }
    if any(
        handoff["artifacts"][role]["records"] != records
        for role, records in expected_species_records.items()
    ):
        raise ArtifactIntegrityError(
            "Species-screen artifact record counts do not match the handoff"
        )

    config_source = Path(config_path).resolve()
    config = load_config(config_source)
    inputs = [
        external_input_metadata(source_run_manifest, label="selection_run_manifest"),
        external_input_metadata(source_handoff, label="selection_handoff_manifest"),
        external_input_metadata(source_cases, label="selection_eligible_cases"),
        external_input_metadata(source_pairs, label="selection_eligible_pairs"),
        external_input_metadata(
            source_species_classifications,
            label="selection_species_classifications",
        ),
        external_input_metadata(
            source_pair_species,
            label="selection_pair_species",
        ),
        external_input_metadata(
            source_species_stats,
            label="selection_species_stats",
        ),
        external_input_metadata(config_source, label="labeling_config_source"),
    ]
    manifest = new_manifest(config=config, inputs=inputs, selection_handoff=handoff)
    manifest["status"] = "initializing"
    save_manifest(run_dir, manifest)

    snapshot_sources = (
        (source_run_manifest, run_dir / HANDOFF_SOURCE_MANIFEST),
        (source_handoff, run_dir / HANDOFF_MANIFEST),
        (source_cases, run_dir / HANDOFF_CASES),
        (source_pairs, run_dir / HANDOFF_PAIRS),
        (
            source_species_classifications,
            run_dir / HANDOFF_SPECIES_CLASSIFICATIONS,
        ),
        (
            source_pair_species,
            run_dir / HANDOFF_PAIR_SPECIES,
        ),
        (source_species_stats, run_dir / HANDOFF_SPECIES_STATS),
    )
    for position, (source, destination) in enumerate(snapshot_sources):
        shutil.copyfile(source, destination)
        expected = inputs[position]
        if (
            destination.stat().st_size != expected["bytes"]
            or sha256_file(destination) != expected["sha256"]
        ):
            raise ArtifactIntegrityError(
                f"Selection input changed while it was being snapshotted: {source.name}"
            )

    case_rows, summaries, citations, selection = _compatibility_views(cases, pairs)
    save_jsonl(case_rows, run_dir / CASES)
    save_jsonl(summaries, run_dir / SUMMARIES)
    save_jsonl(citations, run_dir / CITATIONS)
    save_json(selection, run_dir / SELECTION)

    manifest["status"] = "ready_for_citation" if pairs else "ready_to_finalize"
    record_stage(
        run_dir,
        manifest,
        HANDOFF_STAGE,
        status="complete",
        artifact_paths=(
            *(destination for _, destination in snapshot_sources),
            run_dir / CASES,
            run_dir / SUMMARIES,
            run_dir / CITATIONS,
            run_dir / SELECTION,
        ),
        details={
            "selection_run_spec_id": handoff["producer"]["run_spec_id"],
            "eligible_cases": len(cases),
            "eligible_pairs": len(pairs),
            **species_boundary,
        },
    )
    return manifest


def initialize_run(
    selection_run: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """Atomically create a Pipeline 03 run from one completed Pipeline 02 run."""

    selection_root = Path(selection_run).resolve()
    destination = Path(output_dir)
    _require_separate_output(selection_root, destination)
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(f"Output path is not a directory: {destination}")
        if any(destination.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty; choose a new run directory: {destination}"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.initializing-", dir=destination.parent)
    )
    staging.rmdir()
    try:
        manifest = _initialize_in_directory(selection_root, staging, config_path)
        if destination.exists():
            if any(destination.iterdir()):
                raise FileExistsError(
                    f"Output directory became nonempty during initialization: {destination}"
                )
            destination.rmdir()
        os.replace(staging, destination)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def verify_imported_population_boundary(
    output_dir: str | Path,
) -> dict[str, int | str]:
    """Revalidate the snapshotted Pipeline 02 population-screen boundary."""

    run_dir = Path(output_dir)
    source_run_manifest = _source_file(run_dir.resolve(), HANDOFF_SOURCE_MANIFEST)
    source_handoff = _source_file(run_dir.resolve(), HANDOFF_MANIFEST)
    source_cases = _source_file(run_dir.resolve(), HANDOFF_CASES)
    source_pairs = _source_file(run_dir.resolve(), HANDOFF_PAIRS)
    source_species_classifications = _source_file(
        run_dir.resolve(), HANDOFF_SPECIES_CLASSIFICATIONS
    )
    source_pair_species = _source_file(run_dir.resolve(), HANDOFF_PAIR_SPECIES)
    source_species_stats = _source_file(run_dir.resolve(), HANDOFF_SPECIES_STATS)

    selection_manifest = _mapping(load_json(source_run_manifest), "selection run manifest")
    handoff = _mapping(load_json(source_handoff), "selection handoff manifest")
    _validate_handoff_header(selection_manifest, handoff)
    _validate_producer_artifacts(
        selection_manifest,
        {
            SOURCE_HANDOFF_MANIFEST: source_handoff,
            SOURCE_ELIGIBLE_CASES: source_cases,
            SOURCE_ELIGIBLE_PAIRS: source_pairs,
            SOURCE_SPECIES_CLASSIFICATIONS: source_species_classifications,
            SOURCE_PAIR_SPECIES: source_pair_species,
            SOURCE_SPECIES_STATS: source_species_stats,
        },
    )
    _artifact_contract(handoff, "eligible_cases", SOURCE_ELIGIBLE_CASES, source_cases)
    _artifact_contract(handoff, "eligible_pairs", SOURCE_ELIGIBLE_PAIRS, source_pairs)
    _artifact_contract(
        handoff,
        "species_classifications",
        SOURCE_SPECIES_CLASSIFICATIONS,
        source_species_classifications,
    )
    _artifact_contract(
        handoff,
        "pair_species",
        SOURCE_PAIR_SPECIES,
        source_pair_species,
    )
    _artifact_contract(
        handoff,
        "species_stats",
        SOURCE_SPECIES_STATS,
        source_species_stats,
    )
    ruleset = _validate_species_stage_artifacts(
        selection_manifest,
        {
            SOURCE_SPECIES_CLASSIFICATIONS: source_species_classifications,
            SOURCE_PAIR_SPECIES: source_pair_species,
            SOURCE_SPECIES_STATS: source_species_stats,
        },
    )
    cases = load_jsonl(source_cases)
    pairs = load_jsonl(source_pairs)
    classifications = load_jsonl(source_species_classifications)
    pair_species = load_jsonl(source_pair_species)
    stats = load_json(source_species_stats)
    cases_by_uid = _validate_cases(cases)
    _validate_pairs(pairs, cases_by_uid)
    boundary = _validate_species_boundary(
        selection_manifest,
        cases_by_uid,
        pairs,
        classifications,
        pair_species,
        stats,
        ruleset=ruleset,
    )
    expected_records = {
        "eligible_cases": len(cases),
        "eligible_pairs": len(pairs),
        "species_classifications": len(classifications),
        "pair_species": len(pair_species),
        "species_stats": 1,
    }
    if any(
        handoff["artifacts"][role]["records"] != records
        for role, records in expected_records.items()
    ):
        raise ArtifactIntegrityError("Artifact record counts do not match the handoff")
    return boundary


__all__ = ["initialize_run", "verify_imported_population_boundary"]
