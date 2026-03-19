from __future__ import annotations

from pathlib import Path
from typing import Any


def _to_serializable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def submit_batch(
    *,
    request_jsonl_path: str | Path,
    description: str,
    endpoint: str = "/v1/chat/completions",
    completion_window: str = "24h",
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()

    with Path(request_jsonl_path).open("rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata=metadata or {"description": description},
    )

    return {
        "request_jsonl_path": str(request_jsonl_path),
        "uploaded_file": _to_serializable(file_obj),
        "batch": _to_serializable(batch),
        "description": description,
        "endpoint": endpoint,
        "completion_window": completion_window,
    }


def retrieve_batch(batch_id: str) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    return _to_serializable(batch)


def download_batch_files(batch_id: str) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    batch_dict = _to_serializable(batch)

    output_text = None
    error_text = None

    output_file_id = getattr(batch, "output_file_id", None)
    error_file_id = getattr(batch, "error_file_id", None)

    if output_file_id:
        output_resp = client.files.content(output_file_id)
        output_text = getattr(output_resp, "text", None)

    if error_file_id:
        error_resp = client.files.content(error_file_id)
        error_text = getattr(error_resp, "text", None)

    return {
        "batch": batch_dict,
        "output_text": output_text,
        "error_text": error_text,
    }
