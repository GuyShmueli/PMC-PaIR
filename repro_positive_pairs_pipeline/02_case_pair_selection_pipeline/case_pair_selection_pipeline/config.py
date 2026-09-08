from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigurationError
from .io_utils import canonical_json_bytes, sha256_bytes


PIPELINE_ID = "case_pair_selection_pipeline"
PIPELINE_VERSION = "6.1.0"
CONFIG_SCHEMA_VERSION = 1

SPECIES_RULESETS = {"human_animal_v1"}


@dataclass(frozen=True)
class SelectionConfig:
    white_pixel_threshold: int
    white_ratio_threshold: float
    thumbnail_max_size: int
    ocr_text_length_threshold: int
    ocr_language: str
    ocr_config: str
    ocr_timeout_seconds: float
    title_fallback: bool
    title_min_chars: int
    title_min_tokens: int
    species_ruleset: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "diagram_filter": {
                "white_pixel_threshold": self.white_pixel_threshold,
                "white_ratio_threshold": self.white_ratio_threshold,
                "thumbnail_max_size": self.thumbnail_max_size,
                "ocr_text_length_threshold": self.ocr_text_length_threshold,
                "ocr_language": self.ocr_language,
                "ocr_config": self.ocr_config,
                "ocr_timeout_seconds": self.ocr_timeout_seconds,
            },
            "selection": {
                "title_fallback": self.title_fallback,
                "title_min_chars": self.title_min_chars,
                "title_min_tokens": self.title_min_tokens,
            },
            "species_filter": {
                "ruleset": self.species_ruleset,
            },
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.as_dict()))


def _reject_unknown_keys(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ConfigurationError(f"{label} contains unknown keys: {sorted(unknown)}")


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{label} must be an integer")
    return value


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError(f"{label} must be finite")
    return result


def load_config(path: str | Path) -> SelectionConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    if not isinstance(raw, dict):
        raise ConfigurationError("Configuration root must be a TOML table")
    _reject_unknown_keys(
        raw,
        {"schema_version", "diagram_filter", "selection", "species_filter"},
        "root",
    )
    if raw.get("schema_version") != CONFIG_SCHEMA_VERSION or isinstance(
        raw.get("schema_version"), bool
    ):
        raise ConfigurationError(
            f"schema_version must be {CONFIG_SCHEMA_VERSION}"
        )

    diagram_filter = raw.get("diagram_filter", {})
    if not isinstance(diagram_filter, dict):
        raise ConfigurationError("diagram_filter must be a TOML table")
    _reject_unknown_keys(
        diagram_filter,
        {
            "white_pixel_threshold",
            "white_ratio_threshold",
            "thumbnail_max_size",
            "ocr_text_length_threshold",
            "ocr_language",
            "ocr_config",
            "ocr_timeout_seconds",
        },
        "diagram_filter",
    )
    white_pixel_threshold = _require_int(
        diagram_filter.get("white_pixel_threshold", 250),
        "diagram_filter.white_pixel_threshold",
    )
    white_ratio_threshold = _require_number(
        diagram_filter.get("white_ratio_threshold", 0.5),
        "diagram_filter.white_ratio_threshold",
    )
    thumbnail_max_size = _require_int(
        diagram_filter.get("thumbnail_max_size", 384),
        "diagram_filter.thumbnail_max_size",
    )
    ocr_text_length_threshold = _require_int(
        diagram_filter.get("ocr_text_length_threshold", 200),
        "diagram_filter.ocr_text_length_threshold",
    )
    ocr_language = diagram_filter.get("ocr_language", "eng")
    if (
        not isinstance(ocr_language, str)
        or not ocr_language.strip()
        or ocr_language != ocr_language.strip()
    ):
        raise ConfigurationError(
            "diagram_filter.ocr_language must be a nonempty trimmed string"
        )
    ocr_config = diagram_filter.get(
        "ocr_config",
        "--oem 1 --psm 11 -c tessedit_do_invert=0",
    )
    if not isinstance(ocr_config, str):
        raise ConfigurationError("diagram_filter.ocr_config must be a string")
    ocr_timeout_seconds = _require_number(
        diagram_filter.get("ocr_timeout_seconds", 2.0),
        "diagram_filter.ocr_timeout_seconds",
    )
    if not 0 <= white_pixel_threshold <= 255:
        raise ConfigurationError(
            "diagram_filter.white_pixel_threshold must be between 0 and 255"
        )
    if not 0.0 <= white_ratio_threshold <= 1.0:
        raise ConfigurationError(
            "diagram_filter.white_ratio_threshold must be between 0 and 1"
        )
    if thumbnail_max_size < 1:
        raise ConfigurationError(
            "diagram_filter.thumbnail_max_size must be positive"
        )
    if ocr_text_length_threshold < 0:
        raise ConfigurationError(
            "diagram_filter.ocr_text_length_threshold cannot be negative"
        )
    if ocr_timeout_seconds <= 0:
        raise ConfigurationError(
            "diagram_filter.ocr_timeout_seconds must be positive"
        )

    selection = raw.get("selection")
    if not isinstance(selection, dict):
        raise ConfigurationError("selection must be a TOML table")
    _reject_unknown_keys(
        selection,
        {"title_fallback", "title_min_chars", "title_min_tokens"},
        "selection",
    )

    title_fallback = selection.get("title_fallback", False)
    if not isinstance(title_fallback, bool):
        raise ConfigurationError("selection.title_fallback must be a boolean")
    title_min_chars = _require_int(
        selection.get("title_min_chars", 24), "selection.title_min_chars"
    )
    title_min_tokens = _require_int(
        selection.get("title_min_tokens", 4), "selection.title_min_tokens"
    )
    if title_min_chars < 0 or title_min_tokens < 0:
        raise ConfigurationError("Title thresholds cannot be negative")
    if title_fallback and (title_min_chars == 0 or title_min_tokens == 0):
        raise ConfigurationError(
            "Title fallback requires nonzero character and token thresholds"
        )

    species_filter = raw.get("species_filter")
    if not isinstance(species_filter, dict):
        raise ConfigurationError("species_filter must be a TOML table")
    _reject_unknown_keys(
        species_filter,
        {"ruleset"},
        "species_filter",
    )
    species_ruleset = species_filter.get("ruleset", "human_animal_v1")
    if species_ruleset not in SPECIES_RULESETS:
        raise ConfigurationError(
            "species_filter.ruleset must be one of "
            f"{sorted(SPECIES_RULESETS)}"
        )
    return SelectionConfig(
        white_pixel_threshold=white_pixel_threshold,
        white_ratio_threshold=white_ratio_threshold,
        thumbnail_max_size=thumbnail_max_size,
        ocr_text_length_threshold=ocr_text_length_threshold,
        ocr_language=ocr_language,
        ocr_config=ocr_config,
        ocr_timeout_seconds=ocr_timeout_seconds,
        title_fallback=title_fallback,
        title_min_chars=title_min_chars,
        title_min_tokens=title_min_tokens,
        species_ruleset=species_ruleset,
    )


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "PIPELINE_ID",
    "PIPELINE_VERSION",
    "SPECIES_RULESETS",
    "SelectionConfig",
    "load_config",
]
