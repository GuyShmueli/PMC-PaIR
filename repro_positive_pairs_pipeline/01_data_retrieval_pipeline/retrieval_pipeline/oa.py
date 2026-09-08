from __future__ import annotations

import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import urlparse

import requests

from .artifacts import PIPELINE_VERSION, artifact_path
from .io_utils import (
    load_json,
    normalize_pmc_id,
    pmc_sort_key,
    save_json,
    sha256_file,
)


DEFAULT_OA_ENDPOINT = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
DEFAULT_USER_AGENT = "repro-pmc-data-retrieval/1.0"
DEFAULT_MAX_OA_RESPONSE_BYTES = 5 * 1024 * 1024


def parse_oa_package_response(pmc_id: str, payload: bytes) -> dict[str, str]:
    normalized_id = normalize_pmc_id(pmc_id)
    root = ET.fromstring(payload)

    records = [
        element
        for element in root.iter()
        if element.tag.split("}", 1)[-1] == "record"
    ]
    matching_records = []
    for record in records:
        raw_record_id = str(record.attrib.get("id", "")).strip()
        try:
            if normalize_pmc_id(raw_record_id) == normalized_id:
                matching_records.append(record)
        except ValueError:
            continue
    if records and not matching_records:
        raise ValueError(
            f"NCBI OA response record does not match PMC{normalized_id}"
        )

    search_roots = matching_records or [root]
    hrefs: list[str] = []
    for search_root in search_roots:
        for element in search_root.iter():
            if element.tag.split("}", 1)[-1] != "link":
                continue
            if str(element.attrib.get("format", "")).lower() != "tgz":
                continue
            href = str(element.attrib.get("href", "")).strip()
            if href:
                hrefs.append(href)

    if not hrefs:
        raise ValueError("NCBI OA response has no format='tgz' link")

    unique_hrefs = sorted(set(hrefs))
    if len(unique_hrefs) != 1:
        raise ValueError(
            f"NCBI OA response has multiple distinct TGZ links: {unique_hrefs}"
        )
    download_url = unique_hrefs[0]
    parsed = urlparse(download_url)
    if parsed.scheme not in {"ftp", "http", "https"}:
        raise ValueError(f"Unsupported OA package URL scheme: {parsed.scheme!r}")
    if not parsed.hostname:
        raise ValueError("OA package URL has no host")
    if parsed.hostname.lower() != "ftp.ncbi.nlm.nih.gov":
        raise ValueError(f"Unexpected OA package host: {parsed.hostname!r}")

    archive_name = PurePosixPath(parsed.path).name
    if not archive_name:
        raise ValueError("OA package URL has no archive filename")

    return {
        "pmc_id": normalized_id,
        "archive_name": archive_name,
        "download_url": download_url,
        "host": parsed.hostname.lower(),
        "ftp_path": parsed.path.lstrip("/"),
    }


class OAClient:
    def __init__(
        self,
        *,
        endpoint: str = DEFAULT_OA_ENDPOINT,
        timeout_seconds: float = 30.0,
        retries: int = 3,
        backoff_seconds: float = 1.0,
        user_agent: str = DEFAULT_USER_AGENT,
        session: requests.Session | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        max_response_bytes: int = DEFAULT_MAX_OA_RESPONSE_BYTES,
    ) -> None:
        if retries < 1:
            raise ValueError("retries must be at least 1")
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.sleeper = sleeper
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        self.max_response_bytes = max_response_bytes

    def fetch_xml(self, pmc_id: str) -> bytes:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with self.session.get(
                    self.endpoint,
                    params={"id": normalize_pmc_id(pmc_id)},
                    timeout=self.timeout_seconds,
                    stream=True,
                ) as response:
                    if response.status_code == 429 or response.status_code >= 500:
                        raise requests.HTTPError(
                            f"transient HTTP status {response.status_code}",
                            response=response,
                        )
                    response.raise_for_status()
                    content_length = response.headers.get("Content-Length")
                    if (
                        content_length is not None
                        and int(content_length) > self.max_response_bytes
                    ):
                        raise ValueError(
                            "OA response Content-Length exceeds the byte limit"
                        )
                    payload = bytearray()
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        payload.extend(chunk)
                        if len(payload) > self.max_response_bytes:
                            raise ValueError("OA response exceeds the byte limit")
                    return bytes(payload)
            except requests.RequestException as error:
                last_error = error
                status = getattr(getattr(error, "response", None), "status_code", None)
                transient = status is None or status == 429 or status >= 500
                if not transient or attempt == self.retries:
                    raise
                self.sleeper(self.backoff_seconds * attempt)

        if last_error is not None:
            raise last_error
        raise RuntimeError("OA request failed without an exception")

    def close(self) -> None:
        self.session.close()


def response_directory_fetcher(
    response_dir: str | Path,
    *,
    max_response_bytes: int = DEFAULT_MAX_OA_RESPONSE_BYTES,
) -> Callable[[str], bytes]:
    directory = Path(response_dir)

    def fetch(pmc_id: str) -> bytes:
        normalized_id = normalize_pmc_id(pmc_id)
        candidates = [
            directory / f"{normalized_id}.xml",
            directory / f"PMC{normalized_id}.xml",
        ]
        for candidate in candidates:
            if candidate.is_file():
                if candidate.stat().st_size > max_response_bytes:
                    raise ValueError(
                        f"Cached OA response exceeds the byte limit: {candidate}"
                    )
                return candidate.read_bytes()
        raise FileNotFoundError(
            f"No cached OA response for PMC{normalized_id} in {directory}"
        )

    return fetch


def _checkpoint_packages(
    packages_by_id: dict[str, dict[str, str]],
    failures_by_id: dict[str, dict[str, str]],
    state: dict[str, Any],
    output_dir: str | Path,
) -> None:
    packages = [
        packages_by_id[pmc_id]
        for pmc_id in sorted(packages_by_id, key=pmc_sort_key)
    ]
    failures = [
        failures_by_id[pmc_id]
        for pmc_id in sorted(failures_by_id, key=pmc_sort_key)
    ]
    save_json(state, artifact_path(output_dir, "package_state_json"))
    save_json(packages, artifact_path(output_dir, "packages_json"))
    save_json(failures, artifact_path(output_dir, "package_failures_json"))


def resolve_oa_packages(
    article_ids_json: str | Path,
    output_dir: str | Path,
    *,
    fetch_xml: Callable[[str], bytes] | None = None,
    client: OAClient | None = None,
    resume: bool = False,
    checkpoint_every: int = 100,
    limit: int | None = None,
    source_label: str = "NCBI OA API",
) -> dict[str, Any]:
    ids_path = Path(article_ids_json)
    raw_ids = load_json(ids_path)
    if not isinstance(raw_ids, list):
        raise ValueError(f"{ids_path} must contain a JSON list")

    article_ids = sorted(
        {normalize_pmc_id(value) for value in raw_ids},
        key=pmc_sort_key,
    )
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        article_ids = article_ids[:limit]

    stage_output_paths = [
        artifact_path(output_dir, key)
        for key in (
            "packages_json",
            "package_failures_json",
            "package_state_json",
            "package_stats_json",
        )
    ]
    if not resume and any(path.exists() for path in stage_output_paths):
        raise FileExistsError(
            "Stage 02 outputs already exist; use --resume or a new output directory"
        )

    owns_client = False
    if fetch_xml is None:
        if client is None:
            client = OAClient()
            owns_client = True
        fetch_xml = client.fetch_xml

    packages_path = artifact_path(output_dir, "packages_json")
    failures_path = artifact_path(output_dir, "package_failures_json")
    state_path = artifact_path(output_dir, "package_state_json")
    prior_stats_path = artifact_path(output_dir, "package_stats_json")
    input_sha256 = sha256_file(ids_path)
    selected_ids_payload = json.dumps(
        article_ids,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    state = {
        "pipeline_version": PIPELINE_VERSION,
        "input_sha256": input_sha256,
        "selected_article_ids": len(article_ids),
        "selected_article_ids_sha256": hashlib.sha256(
            selected_ids_payload
        ).hexdigest(),
        "limit": limit,
    }

    packages_by_id: dict[str, dict[str, str]] = {}
    failures_by_id: dict[str, dict[str, str]] = {}
    if resume:
        prior_outputs_exist = packages_path.exists() or failures_path.exists()
        if state_path.exists():
            prior_state = load_json(state_path)
            if prior_state != state:
                raise ValueError(
                    "Cannot resume Stage 02: checkpoint input/configuration "
                    "contract has changed"
                )
        elif prior_outputs_exist:
            raise ValueError(
                "Cannot resume Stage 02: checkpoint outputs exist without "
                f"{state_path.name}"
            )
        if prior_stats_path.exists():
            prior_stats = load_json(prior_stats_path)
            prior_input_hash = prior_stats.get("input", {}).get("sha256")
            if prior_input_hash != input_sha256:
                raise ValueError(
                    "Cannot resume Stage 02: article-ID input hash has changed"
                )
            prior_limit = prior_stats.get("configuration", {}).get("limit")
            if prior_limit != limit:
                raise ValueError(
                    "Cannot resume Stage 02: --limit has changed"
                )
        allowed_ids = set(article_ids)
        if packages_path.exists():
            packages_by_id = {
                normalize_pmc_id(record["pmc_id"]): record
                for record in load_json(packages_path)
                if normalize_pmc_id(record["pmc_id"]) in allowed_ids
            }
        if failures_path.exists():
            failures_by_id = {
                normalize_pmc_id(record["pmc_id"]): record
                for record in load_json(failures_path)
                if normalize_pmc_id(record["pmc_id"]) in allowed_ids
            }
        # A process can be terminated between the two atomic checkpoint-file
        # replacements. A resolved record is authoritative over a stale copy
        # of the same PMCID in the failure file.
        for resolved_id in packages_by_id:
            failures_by_id.pop(resolved_id, None)

    processed_this_run = 0
    try:
        for pmc_id in article_ids:
            if pmc_id in packages_by_id:
                continue

            failures_by_id.pop(pmc_id, None)
            try:
                payload = fetch_xml(pmc_id)
                package = parse_oa_package_response(
                    pmc_id,
                    payload,
                )
                package["oa_response_sha256"] = hashlib.sha256(payload).hexdigest()
                package["metadata_source"] = source_label
                packages_by_id[pmc_id] = package
            except Exception as error:
                failures_by_id[pmc_id] = {
                    "pmc_id": pmc_id,
                    "metadata_source": source_label,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }

            processed_this_run += 1
            if checkpoint_every > 0 and processed_this_run % checkpoint_every == 0:
                _checkpoint_packages(
                    packages_by_id,
                    failures_by_id,
                    state,
                    output_dir,
                )
    finally:
        if owns_client and client is not None:
            client.close()

    _checkpoint_packages(packages_by_id, failures_by_id, state, output_dir)

    packages = load_json(packages_path)
    failures = load_json(failures_path)
    package_source_counts: dict[str, int] = {}
    for package in packages:
        source = str(package.get("metadata_source", "unknown"))
        package_source_counts[source] = package_source_counts.get(source, 0) + 1
    failure_source_counts: dict[str, int] = {}
    for failure in failures:
        source = str(failure.get("metadata_source", "unknown"))
        failure_source_counts[source] = failure_source_counts.get(source, 0) + 1
    stats = {
        "pipeline_version": PIPELINE_VERSION,
        "stage": "02_retrieve_file_info",
        "configuration": {
            "checkpoint_every": checkpoint_every,
            "limit": limit,
        },
        "input": {
            "name": ids_path.name,
            "sha256": input_sha256,
            "article_ids_considered": len(article_ids),
        },
        "packages_resolved": len(packages),
        "failures": len(failures),
        "package_source_counts": package_source_counts,
        "failure_source_counts": failure_source_counts,
        "outputs": {
            packages_path.name: sha256_file(packages_path),
            failures_path.name: sha256_file(failures_path),
            state_path.name: sha256_file(state_path),
        },
    }
    save_json(stats, prior_stats_path)
    return stats
