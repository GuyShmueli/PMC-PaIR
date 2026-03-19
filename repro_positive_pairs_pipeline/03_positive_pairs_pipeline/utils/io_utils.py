from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence


_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", re.S)


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path, *, indent: int = 2) -> None:
    p = ensure_parent(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    p = ensure_parent(path)
    with p.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = _CODE_FENCE_RE.sub("", text).strip()
    return text


def load_jsonish(path: str | Path) -> Any:
    raw = Path(path).read_text(encoding="utf-8")
    raw = strip_code_fences(raw)

    try:
        return json.loads(raw)
    except Exception:
        pass

    try:
        return ast.literal_eval(raw)
    except Exception as exc:
        raise ValueError(f"Could not parse JSON-ish file: {path}") from exc


def custom_id_sort_key(custom_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)(?!.*\d)", custom_id or "")
    if match:
        return int(match.group(1)), custom_id
    return 10**18, custom_id or ""


def read_request_custom_ids(request_jsonl_path: str | Path) -> list[str]:
    items = load_jsonl(request_jsonl_path)
    custom_ids = []
    for item in items:
        custom_id = item.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            raise ValueError(f"Request is missing a valid custom_id: {item}")
        custom_ids.append(custom_id)
    return custom_ids


def assistant_message_content_from_batch_line(item: dict[str, Any]) -> Any:
    response = item.get("response") or {}
    body = response.get("body") or {}
    choices = body.get("choices") or []

    if not choices:
        return None

    message = choices[0].get("message") or {}
    return message.get("content")


def extract_aligned_batch_texts(
    request_jsonl_path: str | Path,
    response_jsonl_path: str | Path,
) -> tuple[list[Any], dict[str, Any]]:
    expected_custom_ids = read_request_custom_ids(request_jsonl_path)
    raw_responses = load_jsonl(response_jsonl_path)

    by_custom_id: dict[str, Any] = {}
    duplicate_custom_ids: list[str] = []
    malformed_rows = 0

    for row in raw_responses:
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            malformed_rows += 1
            continue

        content = assistant_message_content_from_batch_line(row)
        if custom_id in by_custom_id:
            duplicate_custom_ids.append(custom_id)
            continue

        by_custom_id[custom_id] = content

    aligned = [by_custom_id.get(custom_id) for custom_id in expected_custom_ids]
    missing = [custom_id for custom_id in expected_custom_ids if custom_id not in by_custom_id]
    unexpected = sorted(
        [custom_id for custom_id in by_custom_id if custom_id not in set(expected_custom_ids)],
        key=custom_id_sort_key,
    )

    diagnostics = {
        "request_count": len(expected_custom_ids),
        "response_rows_seen": len(raw_responses),
        "missing_custom_ids": missing,
        "unexpected_custom_ids": unexpected,
        "duplicate_custom_ids": sorted(set(duplicate_custom_ids), key=custom_id_sort_key),
        "malformed_response_rows": malformed_rows,
        "non_null_texts": sum(x is not None for x in aligned),
    }
    return aligned, diagnostics


def normalize_text_pair_item(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(item).strip() for item in x if str(item).strip()]
    value = str(x).strip()
    return [value] if value else []


def require_no_missing(values: Sequence[Any], *, label: str) -> None:
    missing_positions = [idx for idx, value in enumerate(values) if value is None]
    if missing_positions:
        preview = missing_positions[:20]
        raise ValueError(
            f"{label} contains missing responses at positions {preview}"
            + (" ..." if len(missing_positions) > len(preview) else "")
        )
