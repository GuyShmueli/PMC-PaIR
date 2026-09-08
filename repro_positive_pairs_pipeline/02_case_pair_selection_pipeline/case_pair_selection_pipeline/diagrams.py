"""Deterministic white-background/OCR diagram classification.

White-background classification is evaluated first; OCR is run only for
images that do not cross the white-ratio threshold. Every run classifies the
complete image manifest supplied by the unified Pipeline-01 handoff.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from PIL import Image

from .artifacts import UPSTREAM_MERGED
from .candidates import _resolve_run_path
from .config import SelectionConfig
from .errors import DiagramFilteringError
from .io_utils import sha256_file


SCHEMA_VERSION = 1
_CHUNKSIZE = 64
_TESSDATA_RE = re.compile(
    r'^List of available languages in ["\']?(?P<path>.+?)["\']? \(\d+\):$'
)
_LOGGER = logging.getLogger(__name__)
_PYTESSERACT: Any | None = None


def _pytesseract_module() -> Any:
    global _PYTESSERACT
    if _PYTESSERACT is None:
        try:
            import pytesseract
        except ImportError as exc:
            raise DiagramFilteringError(
                "pytesseract is required for computed diagram filtering"
            ) from exc
        _PYTESSERACT = pytesseract
    return _PYTESSERACT


def _manifest_images(retrieval_run: Path) -> list[tuple[Path, str]]:
    merged_path = retrieval_run / UPSTREAM_MERGED
    if not merged_path.is_file():
        raise FileNotFoundError(f"Retrieval run has no {UPSTREAM_MERGED}: {retrieval_run}")
    frame = pd.read_csv(merged_path, dtype=str, keep_default_na=False)
    if "image_path" not in frame.columns:
        raise ValueError(f"{UPSTREAM_MERGED} is missing required column 'image_path'")

    images: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for row_number, raw_path in enumerate(frame["image_path"], start=2):
        image, relative = _resolve_run_path(
            raw_path,
            retrieval_run,
            label=f"{UPSTREAM_MERGED} row {row_number} image_path",
            suffix=".jpg",
            require_file=True,
        )
        if relative in seen:
            raise ValueError(f"Duplicate image_path in {UPSTREAM_MERGED}: {relative}")
        seen.add(relative)
        images.append((image, relative))
    return images


def _flatten_alpha_on_white(image: Image.Image) -> Image.Image:
    if "A" not in image.getbands():
        return image
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def white_ratio(
    path: str | Path,
    *,
    white_pixel_threshold: int,
    thumbnail_max_size: int,
) -> float:
    """Return the fraction of thumbnail grayscale pixels at or above threshold."""

    with Image.open(path) as image:
        if image.format == "JPEG":
            image.draft("L", (thumbnail_max_size, thumbnail_max_size))
        image = _flatten_alpha_on_white(image)
        image.thumbnail(
            (thumbnail_max_size, thumbnail_max_size),
            Image.Resampling.BOX,
            reducing_gap=2.0,
        )
        grayscale = image.convert("L")
        total = grayscale.width * grayscale.height
        if total <= 0:
            raise ValueError("image has no pixels")
        histogram = grayscale.histogram()
        white = sum(histogram[white_pixel_threshold:])
        return white / total


def _classify_one(
    item: tuple[int, str, str, SelectionConfig],
) -> tuple[int, dict[str, Any]]:
    index, path, relative, config = item
    try:
        ratio = white_ratio(
            path,
            white_pixel_threshold=config.white_pixel_threshold,
            thumbnail_max_size=config.thumbnail_max_size,
        )
    except Exception as exc:
        return index, {
            "image_path": relative,
            "stage": "white_ratio",
            "error": f"{type(exc).__name__}: {exc}",
        }

    if ratio > config.white_ratio_threshold:
        return index, {
            "schema_version": SCHEMA_VERSION,
            "image_path": relative,
            "mode": "computed",
            "white_ratio": ratio,
            "ocr_status": "skipped_white_ratio",
            "ocr_text_length": None,
            "is_diagram": True,
            "reason": "white_ratio",
        }

    try:
        text = _ocr_text(path, config)
    except Exception as exc:
        return index, {
            "image_path": relative,
            "stage": "ocr",
            "error": f"{type(exc).__name__}: {exc}",
        }

    text_length = len(text)
    is_diagram = text_length > config.ocr_text_length_threshold
    return index, {
        "schema_version": SCHEMA_VERSION,
        "image_path": relative,
        "mode": "computed",
        "white_ratio": ratio,
        "ocr_status": "complete",
        "ocr_text_length": text_length,
        "is_diagram": is_diagram,
        "reason": "ocr_text_length" if is_diagram else "retained",
    }


def _ocr_text(path: str, config: SelectionConfig) -> str:
    pytesseract = _pytesseract_module()
    return str(
        pytesseract.image_to_string(
            path,
            lang=config.ocr_language,
            config=config.ocr_config,
            timeout=config.ocr_timeout_seconds,
        )
    )


def _init_worker() -> None:
    os.environ["OMP_THREAD_LIMIT"] = "1"


def tesseract_runtime_metadata(language: str) -> dict[str, Any]:
    """Record the local OCR executable and language-data provenance."""

    pytesseract = _pytesseract_module()
    configured = str(pytesseract.pytesseract.tesseract_cmd)
    executable = shutil.which(configured)
    if executable is None:
        candidate = Path(configured)
        if candidate.is_file():
            executable = str(candidate.resolve())
    if executable is None:
        raise DiagramFilteringError(
            "Tesseract is required for computed diagram filtering but was not found"
        )

    try:
        version_result = subprocess.run(
            [executable, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        languages_result = subprocess.run(
            [executable, "--list-langs"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DiagramFilteringError(
            f"Could not inspect the Tesseract runtime: {type(exc).__name__}: {exc}"
        ) from exc

    language_lines = [
        line.strip()
        for line in (languages_result.stdout + languages_result.stderr).splitlines()
        if line.strip()
    ]
    available = set(language_lines[1:]) if language_lines else set()
    if language not in available:
        raise DiagramFilteringError(
            f"Tesseract language {language!r} is unavailable; available={sorted(available)}"
        )

    tessdata_path: Path | None = None
    if language_lines:
        match = _TESSDATA_RE.fullmatch(language_lines[0])
        if match is not None:
            candidate = Path(match.group("path")) / f"{language}.traineddata"
            if candidate.is_file():
                tessdata_path = candidate.resolve()
    if tessdata_path is None:
        raise DiagramFilteringError(
            f"Could not locate the Tesseract trained-data file for {language!r}"
        )

    version_lines = [
        line.rstrip()
        for line in (version_result.stdout + version_result.stderr).splitlines()
        if line.strip()
    ]
    return {
        "executable": str(Path(executable).resolve()),
        "executable_sha256": sha256_file(executable),
        "version": version_lines[0] if version_lines else "unknown",
        "version_output": "\n".join(version_lines),
        "language": language,
        "traineddata_path": str(tessdata_path) if tessdata_path else None,
        "traineddata_sha256": sha256_file(tessdata_path) if tessdata_path else None,
    }


def tesseract_runtime_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return the path-independent OCR fields that define a computed run."""

    fields = {
        "executable_sha256",
        "version",
        "version_output",
        "language",
        "traineddata_sha256",
    }
    missing = sorted(fields - set(metadata))
    if missing:
        raise DiagramFilteringError(
            f"Tesseract runtime metadata is missing fields: {missing}"
        )
    for field in ("executable_sha256", "traineddata_sha256"):
        value = metadata[field]
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise DiagramFilteringError(
                f"Tesseract runtime metadata has an invalid {field}"
            )
    for field in ("version", "version_output", "language"):
        value = metadata[field]
        if not isinstance(value, str) or not value.strip():
            raise DiagramFilteringError(
                f"Tesseract runtime metadata has an invalid {field}"
            )
    return {field: metadata[field] for field in sorted(fields)}


def _computed_classifications(
    images: list[tuple[Path, str]],
    *,
    config: SelectionConfig,
    workers: int,
    runtime_metadata: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runtime = (
        tesseract_runtime_metadata(config.ocr_language)
        if runtime_metadata is None
        else dict(runtime_metadata)
    )
    tasks = [
        (index, str(path), relative, config)
        for index, (path, relative) in enumerate(images)
    ]
    records: list[dict[str, Any]] = []

    def accept_result(
        expected_index: int,
        result: tuple[int, dict[str, Any]],
    ) -> None:
        actual_index, record = result
        if actual_index != expected_index:
            raise DiagramFilteringError(
                "Diagram-classification results are not in manifest order"
            )
        if "error" in record:
            raise DiagramFilteringError(
                "Diagram classification failed for "
                f"{record['image_path']} during {record['stage']}: {record['error']}"
            )
        records.append(record)

    if workers == 1:
        previous_omp_limit = os.environ.get("OMP_THREAD_LIMIT")
        _init_worker()
        try:
            results: Iterable[tuple[int, dict[str, Any]]] = map(_classify_one, tasks)
            for processed, result in enumerate(results, start=1):
                accept_result(processed - 1, result)
                if processed % 5000 == 0 or processed == len(tasks):
                    _LOGGER.info(
                        "Diagram filter processed %d/%d images",
                        processed,
                        len(tasks),
                    )
        finally:
            if previous_omp_limit is None:
                os.environ.pop("OMP_THREAD_LIMIT", None)
            else:
                os.environ["OMP_THREAD_LIMIT"] = previous_omp_limit
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
        )
        try:
            results = executor.map(_classify_one, tasks, chunksize=_CHUNKSIZE)
            for processed, result in enumerate(results, start=1):
                accept_result(processed - 1, result)
                if processed % 5000 == 0 or processed == len(tasks):
                    _LOGGER.info(
                        "Diagram filter processed %d/%d images",
                        processed,
                        len(tasks),
                    )
        except BaseException:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    return records, tesseract_runtime_contract(runtime)


def build_diagram_filter(
    retrieval_run: str | Path,
    *,
    config: SelectionConfig,
    workers: int,
    runtime_metadata: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Classify every manifest image and return an ordered normalized blacklist."""

    root = Path(retrieval_run).resolve()
    images = _manifest_images(root)
    records, runtime = _computed_classifications(
        images,
        config=config,
        workers=workers,
        runtime_metadata=runtime_metadata,
    )

    blacklist = [record["image_path"] for record in records if record["is_diagram"]]
    reason_counts: dict[str, int] = {}
    for record in records:
        reason = str(record["reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    stats = {
        "schema_version": SCHEMA_VERSION,
        "mode": "computed",
        "images_input": len(records),
        "diagrams_excluded": len(blacklist),
        "images_retained": len(records) - len(blacklist),
        "reason_counts": dict(sorted(reason_counts.items())),
        "workers": workers,
        "strict_comparisons": {
            "diagram_if_white_ratio_gt": config.white_ratio_threshold,
            "otherwise_diagram_if_ocr_text_length_gt": (
                config.ocr_text_length_threshold
            ),
            "white_pixel_if_grayscale_gte": config.white_pixel_threshold,
        },
        "runtime": runtime,
    }
    return records, blacklist, stats


__all__ = [
    "SCHEMA_VERSION",
    "build_diagram_filter",
    "tesseract_runtime_metadata",
    "tesseract_runtime_contract",
    "white_ratio",
]
