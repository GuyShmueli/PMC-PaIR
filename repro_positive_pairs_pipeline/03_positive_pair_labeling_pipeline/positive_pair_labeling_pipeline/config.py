from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .errors import ConfigurationError
from .io_utils import canonical_json_bytes, sha256_bytes


PIPELINE_ID = "positive_pair_labeling_pipeline"
PIPELINE_VERSION = "6.1.0"
CONFIG_SCHEMA_VERSION = 1
BATCH_ENDPOINT = "/v1/chat/completions"
BATCH_FILE_LIMIT_BYTES = 200_000_000
PIPELINE_MAX_SHARD_REQUESTS = 50_000
DEFAULT_MAX_SHARD_BYTES = 190_000_000
DEFAULT_MAX_SHARD_REQUESTS = 45_000
STAGES = ("citation", "question", "answer", "positive")
_SNAPSHOT_RE = re.compile(r"-(?P<date>\d{4}-\d{2}-\d{2})$")


@dataclass(frozen=True)
class ModelSpec:
    model: str
    reasoning_effort: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
        }


@dataclass(frozen=True)
class PipelineConfig:
    max_shard_bytes: int
    max_shard_requests: int
    models: dict[str, ModelSpec]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "batch": {
                "endpoint": BATCH_ENDPOINT,
                "max_shard_bytes": self.max_shard_bytes,
                "max_shard_requests": self.max_shard_requests,
            },
            "models": {
                stage: self.models[stage].as_dict()
                for stage in STAGES
            },
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.as_dict()))


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} must be a TOML table")
    return value


def _reject_unknown_keys(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ConfigurationError(f"{label} contains unknown keys: {sorted(unknown)}")


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{label} must be an integer")
    return value


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    _reject_unknown_keys(raw, {"schema_version", "batch", "models"}, "root")
    schema_version = raw.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != CONFIG_SCHEMA_VERSION
    ):
        raise ConfigurationError(
            f"schema_version must be {CONFIG_SCHEMA_VERSION}"
        )

    batch = _require_mapping(raw.get("batch"), "batch")
    models_raw = _require_mapping(raw.get("models"), "models")
    _reject_unknown_keys(
        batch,
        {"max_shard_bytes", "max_shard_requests"},
        "batch",
    )
    _reject_unknown_keys(models_raw, set(STAGES), "models")

    max_shard_bytes = _require_int(
        batch.get("max_shard_bytes", DEFAULT_MAX_SHARD_BYTES), "batch.max_shard_bytes"
    )
    max_shard_requests = _require_int(
        batch.get("max_shard_requests", DEFAULT_MAX_SHARD_REQUESTS), "batch.max_shard_requests"
    )
    if not 0 < max_shard_bytes < BATCH_FILE_LIMIT_BYTES:
        raise ConfigurationError(
            f"max_shard_bytes must be between 1 and {BATCH_FILE_LIMIT_BYTES - 1}"
        )
    if not 0 < max_shard_requests <= PIPELINE_MAX_SHARD_REQUESTS:
        raise ConfigurationError(
            "max_shard_requests must be between 1 and the pipeline safety "
            f"ceiling of {PIPELINE_MAX_SHARD_REQUESTS}"
        )

    models: dict[str, ModelSpec] = {}
    for stage in STAGES:
        stage_raw = _require_mapping(models_raw.get(stage), f"models.{stage}")
        _reject_unknown_keys(
            stage_raw, {"model", "reasoning_effort"}, f"models.{stage}"
        )
        model_raw = stage_raw.get("model")
        if not isinstance(model_raw, str) or not model_raw.strip():
            raise ConfigurationError(f"models.{stage}.model is required")
        model = model_raw.strip()
        snapshot_match = _SNAPSHOT_RE.search(model)
        if snapshot_match is None:
            raise ConfigurationError(
                f"models.{stage}.model must be an immutable dated snapshot; got {model!r}"
            )
        try:
            date.fromisoformat(snapshot_match.group("date"))
        except ValueError as exc:
            raise ConfigurationError(
                f"models.{stage}.model has an invalid snapshot date: {model!r}"
            ) from exc
        effort_raw = stage_raw.get("reasoning_effort")
        if effort_raw is not None and (
            not isinstance(effort_raw, str) or not effort_raw.strip()
        ):
            raise ConfigurationError(
                f"models.{stage}.reasoning_effort must be a non-empty string or omitted"
            )
        effort = effort_raw.strip() if isinstance(effort_raw, str) else None
        models[stage] = ModelSpec(model=model, reasoning_effort=effort)

    return PipelineConfig(
        max_shard_bytes=max_shard_bytes,
        max_shard_requests=max_shard_requests,
        models=models,
    )
