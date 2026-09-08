from __future__ import annotations

import ftplib
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable
from urllib.parse import unquote, urlparse

import pandas as pd
import requests
from PIL import Image, ImageOps, __version__ as PILLOW_VERSION, features

from .artifacts import PIPELINE_VERSION, artifact_path
from .io_utils import (
    figure_path_sort_key,
    load_json,
    normalize_pmc_id,
    pmc_sort_key,
    save_bytes,
    save_dataframe,
    save_json,
    save_text,
    sha256_file,
    sha256_tree,
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".gif",
    ".bmp",
    ".webp",
}
DEFAULT_MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 50_000
MAX_EXPANDED_BYTES = 4 * 1024 * 1024 * 1024
MAX_NXML_BYTES = 128 * 1024 * 1024
MAX_IMAGE_BYTES = 1024 * 1024 * 1024
MAX_IMAGE_PIXELS = 200_000_000
EXTRACTION_LIMITS = {
    "archive_members": MAX_ARCHIVE_MEMBERS,
    "expanded_bytes": MAX_EXPANDED_BYTES,
    "nxml_bytes": MAX_NXML_BYTES,
    "image_bytes": MAX_IMAGE_BYTES,
    "image_pixels": MAX_IMAGE_PIXELS,
}
IMAGE_ENCODER_CONTRACT = {
    "python_version": sys.version.split()[0],
    "pillow_version": PILLOW_VERSION,
    "jpeg_codec_version": features.version_codec("jpg"),
    "jpeg_2000_codec_version": features.version_codec("jpg_2000"),
    "zlib_codec_version": features.version_codec("zlib"),
    "libtiff_codec_version": features.version_codec("libtiff"),
    "webp_module_version": features.version_module("webp"),
    "format": "JPEG",
    "mode": "RGB",
    "subsampling": 0,
    "optimize": False,
    "progressive": False,
    "alpha_background": [255, 255, 255, 255],
    "frame": 0,
}
ARCHIVE_RETRY_ERRORS = (
    requests.RequestException,
) + ftplib.all_errors + (EOFError, OSError)
MANIFEST_COLUMNS = [
    "image_path",
    "caption_path",
    "patient_uid",
    "pmc_id",
    "article_path",
]


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _descendants(element: ET.Element, local_name: str) -> list[ET.Element]:
    return [item for item in element.iter() if _local_name(item.tag) == local_name]


def _caption_text(figure: ET.Element) -> str:
    captions = _descendants(figure, "caption")
    if not captions:
        return ""

    def text_with_breaks(element: ET.Element) -> str:
        text = element.text or ""
        for child in element:
            text += text_with_breaks(child)
            if _local_name(child.tag) in {"p", "title", "list-item", "break"}:
                text += " "
            text += child.tail or ""
        return text

    return " ".join(text_with_breaks(captions[0]).split())


def _graphic_hrefs(figure: ET.Element) -> list[str]:
    hrefs: list[str] = []
    for graphic in _descendants(figure, "graphic"):
        for key, value in graphic.attrib.items():
            if key.split("}", 1)[-1].lower() == "href" and str(value).strip():
                hrefs.append(str(value).strip())
                break
    return hrefs


def _safe_member_path(name: str) -> PurePosixPath | None:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path


def _regular_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members: list[tarfile.TarInfo] = []
    expanded_bytes = 0
    for member_count, member in enumerate(tar, start=1):
        if member_count > MAX_ARCHIVE_MEMBERS:
            raise ValueError(
                f"archive exceeds the {MAX_ARCHIVE_MEMBERS} member limit"
            )
        if not member.isfile():
            continue
        if member.size < 0:
            raise ValueError(f"archive member has a negative size: {member.name!r}")
        expanded_bytes += member.size
        if expanded_bytes > MAX_EXPANDED_BYTES:
            raise ValueError(
                "archive exceeds the expanded regular-file byte limit"
            )
        if _safe_member_path(member.name) is None:
            continue
        members.append(member)
    return sorted(members, key=lambda member: member.name.lower())


def _pmc_ids_in_nxml(root: ET.Element) -> set[str]:
    pmc_ids: set[str] = set()
    for element in root.findall("./{*}front/{*}article-meta/{*}article-id"):
        id_type = str(element.attrib.get("pub-id-type", "")).strip().lower()
        if id_type not in {"pmc", "pmcid"}:
            continue
        text = "".join(element.itertext()).strip()
        try:
            pmc_ids.add(normalize_pmc_id(text))
        except ValueError:
            continue
    return pmc_ids


def _select_nxml_document(
    tar: tarfile.TarFile,
    members: Iterable[tarfile.TarInfo],
    pmc_id: str,
) -> tuple[tarfile.TarInfo, bytes, ET.Element]:
    candidates = sorted(
        (
            member
            for member in members
            if member.name.lower().endswith(".nxml")
        ),
        key=lambda member: (-member.size, member.name.lower()),
    )
    if not candidates:
        raise ValueError("archive contains no safe regular .nxml file")

    exact_match: tuple[tarfile.TarInfo, bytes, ET.Element] | None = None
    single_document: tuple[
        tarfile.TarInfo,
        bytes,
        ET.Element,
        set[str],
    ] | None = None
    parse_errors: list[str] = []
    for member in candidates:
        try:
            payload = _read_tar_member(
                tar,
                member,
                max_bytes=MAX_NXML_BYTES,
            )
            root = ET.fromstring(payload)
        except Exception as error:
            parse_errors.append(
                f"{member.name}: {type(error).__name__}: {error}"
            )
            continue

        declared_ids = _pmc_ids_in_nxml(root)
        if len(candidates) == 1:
            single_document = (member, payload, root, declared_ids)
        if pmc_id in declared_ids:
            if exact_match is not None:
                raise ValueError(
                    "archive contains multiple NXML files for "
                    f"PMC{pmc_id}: "
                    f"{[exact_match[0].name, member.name]}"
                )
            exact_match = (member, payload, root)

    if exact_match is not None:
        return exact_match

    if single_document is not None:
        member, payload, root, declared_ids = single_document
        if declared_ids:
            raise ValueError(
                f"NXML identity mismatch for PMC{pmc_id}; declared PMC IDs: "
                f"{sorted(declared_ids, key=pmc_sort_key)}"
            )
        return member, payload, root

    details = f" Parse errors: {parse_errors}" if parse_errors else ""
    raise ValueError(
        f"could not identify one NXML document for PMC{pmc_id} among "
        f"{len(candidates)} candidates.{details}"
    )


def _image_members(members: Iterable[tarfile.TarInfo]) -> list[tarfile.TarInfo]:
    return [
        member
        for member in members
        if PurePosixPath(member.name).suffix.lower() in IMAGE_EXTENSIONS
    ]


def _clean_href(href: str) -> str:
    value = unquote(str(href).strip()).replace("\\", "/")
    value = value.split("?", 1)[0].split("#", 1)[0]
    while value.startswith("./"):
        value = value[2:]
    path = _safe_member_path(value)
    if path is None or not value:
        raise ValueError(f"unsafe or empty graphic href: {href!r}")
    return path.as_posix()


def _resolve_image_member(
    href: str,
    members: Iterable[tarfile.TarInfo],
    *,
    nxml_member_name: str,
) -> tarfile.TarInfo | None:
    clean_href = _clean_href(href)
    href_path = PurePosixPath(clean_href)
    href_full = clean_href.lower()
    href_base = href_path.name.lower()
    href_suffix = href_path.suffix.lower()
    href_stem = (
        href_path.stem.lower() if href_suffix in IMAGE_EXTENSIONS else href_base
    )

    safe_nxml_path = _safe_member_path(nxml_member_name)
    if safe_nxml_path is None:
        raise ValueError(f"unsafe NXML member path: {nxml_member_name!r}")
    nxml_parent = safe_nxml_path.parent
    expected_full = (nxml_parent / href_path).as_posix().lower()

    ranked: list[tuple[int, str, tarfile.TarInfo]] = []
    for member in members:
        member_path = _safe_member_path(member.name)
        if member_path is None:
            continue
        member_full = member_path.as_posix().lower()
        member_base = member_path.name.lower()
        member_stem = member_path.stem.lower()
        same_nxml_directory = member_path.parent == nxml_parent

        rank: int | None = None
        if member_full == expected_full:
            rank = 0
        elif same_nxml_directory and member_stem == href_stem:
            rank = 1
        elif member_full == href_full:
            rank = 2
        elif same_nxml_directory and member_base == href_base:
            rank = 3
        elif member_base == href_base:
            rank = 4
        elif member_stem == href_stem:
            rank = 5

        if rank is not None:
            ranked.append((rank, member_full, member))

    if not ranked:
        return None
    best_rank = min(item[0] for item in ranked)
    best = sorted(
        (item for item in ranked if item[0] == best_rank),
        key=lambda item: item[1],
    )
    if len(best) > 1:
        raise ValueError(
            f"ambiguous graphic href {href!r}: "
            f"{[item[2].name for item in best]}"
        )
    return best[0][2]


def _read_tar_member(
    tar: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    max_bytes: int,
) -> bytes:
    if member.size > max_bytes:
        raise ValueError(
            f"archive member {member.name!r} exceeds the {max_bytes}-byte limit"
        )
    extracted = tar.extractfile(member)
    if extracted is None:
        raise ValueError(f"could not read archive member {member.name!r}")
    return extracted.read()


def canonical_jpeg_bytes(source: bytes, *, quality: int = 95) -> bytes:
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100")

    with Image.open(io.BytesIO(source)) as image:
        image.seek(0)
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError(
                f"image exceeds the {MAX_IMAGE_PIXELS}-pixel limit"
            )
        image = ImageOps.exif_transpose(image)

        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")

        output = io.BytesIO()
        image.save(
            output,
            format="JPEG",
            quality=quality,
            subsampling=0,
            optimize=False,
            progressive=False,
        )
        return output.getvalue()


class ArchiveClient:
    def __init__(
        self,
        *,
        archives_dir: str | Path | None = None,
        timeout_seconds: float = 120.0,
        retries: int = 3,
        backoff_seconds: float = 2.0,
        session: requests.Session | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
        user_agent: str = "repro-pmc-data-retrieval/1.0",
    ) -> None:
        if retries < 1:
            raise ValueError("retries must be at least 1")
        self.archives_dir = Path(archives_dir) if archives_dir is not None else None
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.sleeper = sleeper
        if max_archive_bytes < 1:
            raise ValueError("max_archive_bytes must be positive")
        self.max_archive_bytes = max_archive_bytes
        self._ftp: ftplib.FTP | None = None

    def _read_local_archive(self, archive_path: Path) -> bytes:
        size = archive_path.stat().st_size
        if size > self.max_archive_bytes:
            raise ValueError(
                f"Archive exceeds the {self.max_archive_bytes}-byte limit: "
                f"{archive_path}"
            )
        return archive_path.read_bytes()

    def _append_limited(self, payload: io.BytesIO, chunk: bytes) -> None:
        if payload.tell() + len(chunk) > self.max_archive_bytes:
            raise ValueError(
                f"Archive exceeds the {self.max_archive_bytes}-byte limit"
            )
        payload.write(chunk)

    def _drop_ftp(self) -> None:
        if self._ftp is None:
            return
        try:
            self._ftp.quit()
        except Exception:
            self._ftp.close()
        finally:
            self._ftp = None

    def _ftp_connection(self, hostname: str) -> ftplib.FTP:
        if self._ftp is None:
            ftp = ftplib.FTP(hostname, timeout=self.timeout_seconds)
            ftp.login()
            ftp.set_pasv(True)
            self._ftp = ftp
        return self._ftp

    def _fetch_once(self, record: dict[str, str]) -> bytes:
        if self.archives_dir is not None:
            archive_name = Path(record["archive_name"]).name
            archive_path = self.archives_dir / archive_name
            if not archive_path.is_file():
                raise FileNotFoundError(f"Local archive not found: {archive_path}")
            return self._read_local_archive(archive_path)

        download_url = record["download_url"]
        parsed = urlparse(download_url)
        if not parsed.hostname or parsed.hostname.lower() != "ftp.ncbi.nlm.nih.gov":
            raise ValueError(
                f"Unexpected OA package host: {parsed.hostname!r}"
            )
        if parsed.scheme in {"http", "https"}:
            payload = io.BytesIO()
            with self.session.get(
                download_url,
                timeout=self.timeout_seconds,
                stream=True,
            ) as response:
                response.raise_for_status()
                content_length = response.headers.get("Content-Length")
                if (
                    content_length is not None
                    and int(content_length) > self.max_archive_bytes
                ):
                    raise ValueError(
                        "Archive Content-Length exceeds the configured byte limit"
                    )
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        self._append_limited(payload, chunk)
            return payload.getvalue()
        if parsed.scheme == "ftp":
            payload = io.BytesIO()
            try:
                ftp = self._ftp_connection(parsed.hostname)
                ftp.retrbinary(
                    f"RETR {parsed.path}",
                    lambda chunk: self._append_limited(payload, chunk),
                )
            except Exception:
                self._drop_ftp()
                raise
            return payload.getvalue()
        raise ValueError(f"Unsupported archive URL scheme: {parsed.scheme!r}")

    def fetch(self, record: dict[str, str]) -> bytes:
        if self.archives_dir is not None:
            return self._fetch_once(record)

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                return self._fetch_once(record)
            except ARCHIVE_RETRY_ERRORS as error:
                last_error = error
                if isinstance(error, requests.HTTPError):
                    status = getattr(error.response, "status_code", None)
                    if status is not None and 400 <= status < 500 and status != 429:
                        raise
                if isinstance(error, ftplib.error_perm):
                    raise
                if attempt == self.retries:
                    raise
                self.sleeper(self.backoff_seconds * attempt)
        if last_error is not None:
            raise last_error
        raise RuntimeError("archive download failed without an exception")

    def close(self) -> None:
        self._drop_ftp()
        self.session.close()


def _directories_identical(left: Path, right: Path) -> bool:
    def inventory(root: Path) -> dict[str, str]:
        result: dict[str, str] = {}
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            result[path.relative_to(root).as_posix()] = sha256_file(path)
        return result

    return inventory(left) == inventory(right)


def _cleanup_stale_temp_directories(
    assets_dir: Path,
    package_ids: Iterable[str],
) -> None:
    for pmc_id in package_ids:
        article_dir = assets_dir / f"PMC{pmc_id}"
        if not article_dir.is_dir():
            continue
        for candidate in article_dir.glob(f".{pmc_id}_1.*"):
            if candidate.is_symlink() or not candidate.is_dir():
                raise RuntimeError(
                    f"Unexpected Stage 03 temporary path: {candidate}"
                )
            shutil.rmtree(candidate)


def extract_article_archive(
    record: dict[str, str],
    archive_payload: bytes,
    output_dir: str | Path,
    *,
    jpeg_quality: int = 95,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    pmc_id = normalize_pmc_id(record["pmc_id"])
    patient_folder_name = f"{pmc_id}_1"
    assets_root = artifact_path(output_dir, "assets_dir")
    article_dir = assets_root / f"PMC{pmc_id}"
    article_dir.mkdir(parents=True, exist_ok=True)

    temporary_folder = Path(
        tempfile.mkdtemp(prefix=f".{patient_folder_name}.", dir=article_dir)
    )
    final_folder = article_dir / patient_folder_name

    rows: list[dict[str, str]] = []
    figure_failures: list[dict[str, Any]] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_payload), mode="r:gz") as tar:
            members = _regular_members(tar)
            nxml_member, nxml_payload, root = _select_nxml_document(
                tar,
                members,
                pmc_id,
            )
            image_members = _image_members(members)

            nxml_path = temporary_folder / "article.nxml"
            save_bytes(nxml_payload, nxml_path)

            figures = [item for item in root.iter() if _local_name(item.tag) == "fig"]
            for figure_index, figure in enumerate(figures, start=1):
                caption = _caption_text(figure)
                hrefs = _graphic_hrefs(figure)
                if not caption or not hrefs:
                    figure_failures.append(
                        {
                            "pmc_id": pmc_id,
                            "figure_index": figure_index,
                            "reason": "missing or empty caption/graphic href",
                        }
                    )
                    continue

                chosen_member = None
                chosen_href = None
                resolution_errors: list[str] = []
                for href in hrefs:
                    try:
                        chosen_member = _resolve_image_member(
                            href,
                            image_members,
                            nxml_member_name=nxml_member.name,
                        )
                    except ValueError as error:
                        resolution_errors.append(str(error))
                        continue
                    if chosen_member is not None:
                        chosen_href = href
                        break

                if chosen_member is None:
                    figure_failures.append(
                        {
                            "pmc_id": pmc_id,
                            "figure_index": figure_index,
                            "reason": "referenced image not found in archive",
                            "graphic_hrefs": hrefs,
                            "resolution_errors": resolution_errors,
                        }
                    )
                    continue

                figure_stem = f"{pmc_id}_1_{figure_index}"
                image_path = temporary_folder / f"{figure_stem}.jpg"
                caption_path = temporary_folder / f"{figure_stem}.txt"
                try:
                    source_image = _read_tar_member(
                        tar,
                        chosen_member,
                        max_bytes=MAX_IMAGE_BYTES,
                    )
                    save_bytes(
                        canonical_jpeg_bytes(source_image, quality=jpeg_quality),
                        image_path,
                    )
                    save_text(caption, caption_path)
                except Exception as error:
                    image_path.unlink(missing_ok=True)
                    caption_path.unlink(missing_ok=True)
                    figure_failures.append(
                        {
                            "pmc_id": pmc_id,
                            "figure_index": figure_index,
                            "reason": "image conversion or caption write failed",
                            "graphic_href": chosen_href,
                            "error_type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                    continue

                final_image_path = final_folder / image_path.name
                final_caption_path = final_folder / caption_path.name
                rows.append(
                    {
                        "image_path": final_image_path.relative_to(output_dir).as_posix(),
                        "caption_path": final_caption_path.relative_to(output_dir).as_posix(),
                        "patient_uid": f"{pmc_id}_1",
                        "pmc_id": pmc_id,
                        "article_path": (
                            f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmc_id}/"
                        ),
                    }
                )

        if final_folder.exists():
            if not _directories_identical(temporary_folder, final_folder):
                raise RuntimeError(
                    f"Existing incomplete output differs for PMC{pmc_id}: {final_folder}"
                )
            shutil.rmtree(temporary_folder)
        else:
            os.replace(temporary_folder, final_folder)

        article_state = {
            "archive_sha256": hashlib.sha256(archive_payload).hexdigest(),
            "nxml_sha256": hashlib.sha256(nxml_payload).hexdigest(),
            "output_tree_sha256": sha256_tree(final_folder),
            "package_record_sha256": _canonical_json_sha256(record),
            "manifest_rows_sha256": _canonical_json_sha256(_sorted_rows(rows)),
            "figures_in_nxml": len(figures),
            "figure_rows": len(rows),
            "figure_failures": len(figure_failures),
        }
        return rows, figure_failures, article_state
    except Exception:
        shutil.rmtree(temporary_folder, ignore_errors=True)
        raise


def _sorted_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            int(normalize_pmc_id(row["pmc_id"])),
            figure_path_sort_key(row["image_path"]),
        ),
    )


def _sorted_failures(failures: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        failures,
        key=lambda item: (
            int(normalize_pmc_id(item["pmc_id"])),
            int(item.get("figure_index", 0)),
            str(item.get("reason", "")),
        ),
    )


def _checkpoint_extraction(
    rows_by_id: dict[str, list[dict[str, str]]],
    article_failures_by_id: dict[str, dict[str, Any]],
    figure_failures_by_id: dict[str, list[dict[str, Any]]],
    completed: dict[str, dict[str, Any]],
    packages_sha256: str,
    package_ids: list[str],
    limit: int | None,
    jpeg_quality: int,
    output_dir: str | Path,
) -> None:
    rows = _sorted_rows(
        row for article_rows in rows_by_id.values() for row in article_rows
    )
    dataframe = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    save_dataframe(dataframe, artifact_path(output_dir, "images_captions_csv"))
    save_json(
        _sorted_failures(article_failures_by_id.values()),
        artifact_path(output_dir, "article_failures_json"),
    )
    save_json(
        _sorted_failures(
            failure
            for article_failures in figure_failures_by_id.values()
            for failure in article_failures
        ),
        artifact_path(output_dir, "figure_failures_json"),
    )
    save_json(
        {
            "pipeline_version": PIPELINE_VERSION,
            "packages_sha256": packages_sha256,
            "package_ids": package_ids,
            "limit": limit,
            "jpeg_quality": jpeg_quality,
            "image_encoder": IMAGE_ENCODER_CONTRACT,
            "extraction_limits": EXTRACTION_LIMITS,
            "completed": {
                pmc_id: completed[pmc_id]
                for pmc_id in sorted(completed, key=pmc_sort_key)
            },
        },
        artifact_path(output_dir, "extraction_state_json"),
    )


def extract_figure_assets(
    packages_json: str | Path,
    output_dir: str | Path,
    *,
    fetch_archive: Callable[[dict[str, str]], bytes] | None = None,
    client: ArchiveClient | None = None,
    resume: bool = False,
    checkpoint_every: int = 100,
    jpeg_quality: int = 95,
    limit: int | None = None,
    source_label: str = "remote OA packages",
) -> dict[str, Any]:
    packages_path = Path(packages_json)
    package_records = load_json(packages_path)
    if not isinstance(package_records, list):
        raise ValueError(f"{packages_path} must contain a JSON list")

    packages_by_id: dict[str, dict[str, str]] = {}
    for record in package_records:
        pmc_id = normalize_pmc_id(record["pmc_id"])
        if pmc_id in packages_by_id:
            raise ValueError(f"Duplicate package record for PMC{pmc_id}")
        packages_by_id[pmc_id] = record
    package_ids = sorted(packages_by_id, key=pmc_sort_key)
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        package_ids = package_ids[:limit]

    owns_client = False
    if fetch_archive is None:
        if client is None:
            client = ArchiveClient()
            owns_client = True
        fetch_archive = client.fetch

    packages_sha256 = sha256_file(packages_path)
    state_path = artifact_path(output_dir, "extraction_state_json")
    manifest_path = artifact_path(output_dir, "images_captions_csv")
    article_failures_path = artifact_path(output_dir, "article_failures_json")
    figure_failures_path = artifact_path(output_dir, "figure_failures_json")
    assets_dir = artifact_path(output_dir, "assets_dir")

    if resume:
        _cleanup_stale_temp_directories(assets_dir, package_ids)

    if not resume:
        existing_assets = assets_dir.exists() and any(assets_dir.iterdir())
        existing_artifacts = any(
            path.exists()
            for path in (
                state_path,
                manifest_path,
                article_failures_path,
                figure_failures_path,
            )
        )
        if existing_assets or existing_artifacts:
            raise FileExistsError(
                "Stage 03 outputs already exist; use --resume or a new output directory"
            )

    rows_by_id: dict[str, list[dict[str, str]]] = {}
    article_failures_by_id: dict[str, dict[str, Any]] = {}
    figure_failures_by_id: dict[str, list[dict[str, Any]]] = {}
    completed: dict[str, dict[str, Any]] = {}

    if resume and state_path.exists():
        state = load_json(state_path)
        if state.get("pipeline_version") != PIPELINE_VERSION:
            raise ValueError("Cannot resume Stage 03: pipeline version has changed")
        prior_package_ids = [
            normalize_pmc_id(pmc_id)
            for pmc_id in state.get("package_ids", [])
        ]
        if state.get("limit") != limit:
            raise ValueError("Cannot resume Stage 03: --limit has changed")
        if not set(prior_package_ids).issubset(package_ids):
            raise ValueError(
                "Cannot resume Stage 03: a previously resolved package is "
                "missing from the current Stage 02 output"
            )
        if int(state.get("jpeg_quality", -1)) != jpeg_quality:
            raise ValueError("Cannot resume Stage 03: JPEG quality has changed")
        if state.get("image_encoder") != IMAGE_ENCODER_CONTRACT:
            raise ValueError(
                "Cannot resume Stage 03: image runtime/encoder contract has changed"
            )
        if state.get("extraction_limits") != EXTRACTION_LIMITS:
            raise ValueError(
                "Cannot resume Stage 03: extraction resource limits have changed"
            )
        allowed_ids = set(package_ids)
        completed = {
            normalize_pmc_id(pmc_id): value
            for pmc_id, value in state.get("completed", {}).items()
            if normalize_pmc_id(pmc_id) in allowed_ids
        }

        for pmc_id, article_state in completed.items():
            expected_record_hash = article_state.get("package_record_sha256")
            actual_record_hash = _canonical_json_sha256(packages_by_id[pmc_id])
            if expected_record_hash != actual_record_hash:
                raise ValueError(
                    f"Cannot resume Stage 03: package record changed for PMC{pmc_id}"
                )

            patient_folder = (
                assets_dir / f"PMC{pmc_id}" / f"{pmc_id}_1"
            )
            expected_tree_hash = article_state.get("output_tree_sha256")
            actual_tree_hash = (
                sha256_tree(patient_folder) if patient_folder.is_dir() else None
            )
            if not expected_tree_hash or actual_tree_hash != expected_tree_hash:
                raise RuntimeError(
                    "Cannot resume Stage 03: completed asset integrity check "
                    f"failed for PMC{pmc_id}"
                )

        if manifest_path.exists():
            prior_manifest = pd.read_csv(manifest_path, dtype={"pmc_id": str})
            for pmc_id, group in prior_manifest.groupby("pmc_id", sort=False):
                normalized_id = normalize_pmc_id(pmc_id)
                if normalized_id in completed and normalized_id in allowed_ids:
                    rows_by_id[normalized_id] = group[MANIFEST_COLUMNS].to_dict(
                        orient="records"
                    )
        for pmc_id, article_state in completed.items():
            prior_rows = _sorted_rows(rows_by_id.get(pmc_id, []))
            if len(prior_rows) != int(article_state.get("figure_rows", -1)):
                raise RuntimeError(
                    "Cannot resume Stage 03: manifest row count changed for "
                    f"PMC{pmc_id}"
                )
            if _canonical_json_sha256(prior_rows) != article_state.get(
                "manifest_rows_sha256"
            ):
                raise RuntimeError(
                    "Cannot resume Stage 03: manifest rows changed for "
                    f"PMC{pmc_id}"
                )
        if article_failures_path.exists():
            article_failures_by_id = {
                normalize_pmc_id(item["pmc_id"]): item
                for item in load_json(article_failures_path)
                if normalize_pmc_id(item["pmc_id"]) in allowed_ids
            }
        if figure_failures_path.exists():
            for item in load_json(figure_failures_path):
                pmc_id = normalize_pmc_id(item["pmc_id"])
                if pmc_id in completed:
                    figure_failures_by_id.setdefault(pmc_id, []).append(item)

    processed_this_run = 0
    try:
        for pmc_id in package_ids:
            if pmc_id in completed:
                continue

            record = packages_by_id[pmc_id]
            article_failures_by_id.pop(pmc_id, None)
            figure_failures_by_id.pop(pmc_id, None)
            rows_by_id.pop(pmc_id, None)
            try:
                archive_payload = fetch_archive(record)
                rows, figure_failures, article_state = extract_article_archive(
                    record,
                    archive_payload,
                    output_dir,
                    jpeg_quality=jpeg_quality,
                )
                article_state["archive_source"] = source_label
                rows_by_id[pmc_id] = rows
                figure_failures_by_id[pmc_id] = figure_failures
                completed[pmc_id] = article_state
            except Exception as error:
                article_failures_by_id[pmc_id] = {
                    "pmc_id": pmc_id,
                    "archive_source": source_label,
                    "reason": "article download/extraction failed",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }

            processed_this_run += 1
            if checkpoint_every > 0 and processed_this_run % checkpoint_every == 0:
                _checkpoint_extraction(
                    rows_by_id,
                    article_failures_by_id,
                    figure_failures_by_id,
                    completed,
                    packages_sha256,
                    package_ids,
                    limit,
                    jpeg_quality,
                    output_dir,
                )
    finally:
        if owns_client and client is not None:
            client.close()

    _checkpoint_extraction(
        rows_by_id,
        article_failures_by_id,
        figure_failures_by_id,
        completed,
        packages_sha256,
        package_ids,
        limit,
        jpeg_quality,
        output_dir,
    )

    rows = sum(len(value) for value in rows_by_id.values())
    article_failures = sum(
        1 for pmc_id in package_ids if pmc_id in article_failures_by_id
    )
    figure_failures = sum(
        len(value) for pmc_id, value in figure_failures_by_id.items()
        if pmc_id in package_ids
    )
    archive_source_counts: dict[str, int] = {}
    for article_state in completed.values():
        source = str(article_state.get("archive_source", "unknown"))
        archive_source_counts[source] = archive_source_counts.get(source, 0) + 1
    archive_failure_source_counts: dict[str, int] = {}
    for failure in article_failures_by_id.values():
        source = str(failure.get("archive_source", "unknown"))
        archive_failure_source_counts[source] = (
            archive_failure_source_counts.get(source, 0) + 1
        )
    stats_path = artifact_path(output_dir, "extraction_stats_json")
    stats = {
        "pipeline_version": PIPELINE_VERSION,
        "stage": "03_extract_image_captions",
        "configuration": {
            "checkpoint_every": checkpoint_every,
            "jpeg_quality": jpeg_quality,
            "limit": limit,
        },
        "input": {
            "name": packages_path.name,
            "sha256": packages_sha256,
            "packages_considered": len(package_ids),
        },
        "articles_completed": sum(1 for pmc_id in package_ids if pmc_id in completed),
        "article_failures": article_failures,
        "figure_rows": rows,
        "figure_failures": figure_failures,
        "archive_source_counts": archive_source_counts,
        "archive_failure_source_counts": archive_failure_source_counts,
        "outputs": {
            manifest_path.name: sha256_file(manifest_path),
            article_failures_path.name: sha256_file(article_failures_path),
            figure_failures_path.name: sha256_file(figure_failures_path),
            state_path.name: sha256_file(state_path),
        },
    }
    save_json(stats, stats_path)
    return stats
