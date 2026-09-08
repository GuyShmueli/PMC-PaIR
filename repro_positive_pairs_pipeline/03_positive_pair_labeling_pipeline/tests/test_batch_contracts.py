from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from positive_pair_labeling_pipeline.errors import ResponseValidationError
from positive_pair_labeling_pipeline.io_utils import load_json, load_jsonl, save_jsonl
from positive_pair_labeling_pipeline.responses import reconcile_responses
from positive_pair_labeling_pipeline.sharding import write_request_shards


def _body() -> dict:
    return {
        "messages": [{"content": "same deterministic prompt", "role": "user"}],
        "model": "gpt-test-snapshot",
    }


def _make_stage(root: Path) -> tuple[Path, list[dict]]:
    stage_dir = root / "stage"
    write_request_shards(
        [
            {"pair_index": 11, "body": _body()},
            {"pair_index": 29, "body": {**_body(), "seed": 7}},
        ],
        stage="citation",
        output_dir=stage_dir,
        max_shard_bytes=10_000,
        max_shard_requests=10,
    )
    return stage_dir, load_jsonl(stage_dir / "request_index.jsonl")


def _response(custom_id: str, text: str, *, choice_index: int = 0) -> dict:
    return {
        "custom_id": custom_id,
        "error": None,
        "response": {
            "body": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": choice_index,
                        "message": {"content": text, "role": "assistant"},
                    }
                ]
            },
            "status_code": 200,
        },
    }


class BatchContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def test_identical_api_bodies_for_distinct_pairs_are_allowed(self) -> None:
        stage_dir = self.root / "stage"
        manifest = write_request_shards(
            [
                {"pair_index": 4, "body": _body()},
                {"pair_index": 9, "body": _body()},
            ],
            stage="question",
            output_dir=stage_dir,
            max_shard_bytes=10_000,
            max_shard_requests=10,
        )
        index = load_jsonl(stage_dir / "request_index.jsonl")

        self.assertEqual(manifest["request_count"], 2)
        self.assertEqual(index[0]["request_sha256"], index[1]["request_sha256"])
        self.assertNotEqual(index[0]["custom_id"], index[1]["custom_id"])
        self.assertEqual([row["pair_index"] for row in index], [4, 9])

    def test_duplicate_id_is_rejected_even_when_first_row_is_an_error(self) -> None:
        stage_dir, index = _make_stage(self.root)
        first_id = index[0]["custom_id"]
        raw_path = self.root / "raw.jsonl"
        save_jsonl(
            [
                {"custom_id": first_id, "error": {"message": "failed"}},
                _response(first_id, "later success"),
                _response(index[1]["custom_id"], "ok"),
            ],
            raw_path,
        )

        with self.assertRaisesRegex(ResponseValidationError, "duplicate custom_id"):
            reconcile_responses(stage_dir, raw_path)
        self.assertFalse((stage_dir / "responses.jsonl").exists())

    def test_response_requires_one_choice_at_index_zero(self) -> None:
        invalid_choice_sets = [
            [
                {
                    "index": 1,
                    "message": {"content": "wrong index", "role": "assistant"},
                }
            ],
            [
                {
                    "index": 0,
                    "message": {"content": "first", "role": "assistant"},
                },
                {
                    "index": 1,
                    "message": {"content": "second", "role": "assistant"},
                },
            ],
        ]
        for choices in invalid_choice_sets:
            with self.subTest(choices=choices):
                case_root = self.root / f"case-{len(choices)}"
                case_root.mkdir()
                stage_dir, index = _make_stage(case_root)
                raw_path = case_root / "raw.jsonl"
                rows = [
                    _response(index[0]["custom_id"], "unused"),
                    _response(index[1]["custom_id"], "ok"),
                ]
                rows[0]["response"]["body"]["choices"] = choices
                save_jsonl(rows, raw_path)

                with self.assertRaisesRegex(
                    ResponseValidationError, "exactly one choice"
                ):
                    reconcile_responses(stage_dir, raw_path)
                self.assertFalse((stage_dir / "responses.jsonl").exists())

    def test_overlapping_response_sources_are_rejected(self) -> None:
        stage_dir, index = _make_stage(self.root)
        responses = self.root / "responses"
        raw_path = responses / "output.jsonl"
        save_jsonl([_response(row["custom_id"], "ok") for row in index], raw_path)

        with self.assertRaisesRegex(ResponseValidationError, "Duplicate response file"):
            reconcile_responses(stage_dir, [responses, raw_path])
        self.assertFalse((stage_dir / "responses.jsonl").exists())

    def test_response_manifest_contains_no_absolute_paths(self) -> None:
        stage_dir, index = _make_stage(self.root)
        external_dir = self.root / "external"
        external_dir.mkdir()
        raw_path = external_dir / "batch-output.jsonl"
        save_jsonl(
            [
                _response(index[1]["custom_id"], "second"),
                _response(index[0]["custom_id"], "first"),
            ],
            raw_path,
        )

        records, diagnostics = reconcile_responses(stage_dir, raw_path)

        self.assertEqual([record["pair_index"] for record in records], [11, 29])
        self.assertEqual(diagnostics["response_files"][0]["path"], raw_path.name)
        self.assertEqual(diagnostics["output"]["path"], "responses.jsonl")
        self.assertEqual(
            diagnostics["request_manifest"]["path"], "request_manifest.json"
        )
        persisted = load_json(stage_dir / "response_manifest.json")
        for item in [
            persisted["output"],
            persisted["request_manifest"],
            *persisted["response_files"],
        ]:
            self.assertFalse(Path(item["path"]).is_absolute())

    def test_incomplete_or_invalid_response_sets_write_nothing(self) -> None:
        def missing(index: list[dict]) -> list[dict]:
            return [_response(index[0]["custom_id"], "first")]

        def unexpected(index: list[dict]) -> list[dict]:
            return [
                _response(index[0]["custom_id"], "first"),
                _response(index[1]["custom_id"], "second"),
                _response("not-in-the-request-index", "unexpected"),
            ]

        def non_2xx(index: list[dict]) -> list[dict]:
            bad = _response(index[0]["custom_id"], "not accepted")
            bad["response"]["status_code"] = 500
            return [bad, _response(index[1]["custom_id"], "second")]

        def error_row(index: list[dict]) -> list[dict]:
            return [
                {
                    "custom_id": index[0]["custom_id"],
                    "error": {"message": "remote request failed"},
                },
                _response(index[1]["custom_id"], "second"),
            ]

        def empty_content(index: list[dict]) -> list[dict]:
            return [
                _response(index[0]["custom_id"], " \n\t"),
                _response(index[1]["custom_id"], "second"),
            ]

        def truncated(index: list[dict]) -> list[dict]:
            bad = _response(index[0]["custom_id"], "partial output")
            bad["response"]["body"]["choices"][0]["finish_reason"] = "length"
            return [bad, _response(index[1]["custom_id"], "second")]

        def refusal(index: list[dict]) -> list[dict]:
            bad = _response(index[0]["custom_id"], "refused")
            bad["response"]["body"]["choices"][0]["message"]["refusal"] = "cannot comply"
            return [bad, _response(index[1]["custom_id"], "second")]

        cases = {
            "missing": (missing, "missing custom_ids"),
            "unexpected": (unexpected, "unexpected custom_id"),
            "non-2xx": (non_2xx, "non-success status code"),
            "error": (error_row, "contains an error"),
            "empty": (empty_content, "assistant content is empty"),
            "truncated": (truncated, "did not finish normally"),
            "refusal": (refusal, "explicit refusal"),
        }
        for case_number, (label, (build_rows, error_pattern)) in enumerate(
            cases.items()
        ):
            with self.subTest(case=label):
                case_root = self.root / f"invalid-{case_number}"
                case_root.mkdir()
                stage_dir, index = _make_stage(case_root)
                raw_path = case_root / "raw.jsonl"
                save_jsonl(build_rows(index), raw_path)

                with self.assertRaisesRegex(
                    ResponseValidationError, error_pattern
                ):
                    reconcile_responses(stage_dir, raw_path)
                self.assertFalse((stage_dir / "responses.jsonl").exists())
                self.assertFalse((stage_dir / "response_manifest.json").exists())

    def test_request_shards_respect_count_and_canonical_utf8_bytes(self) -> None:
        requests = [
            {
                "pair_index": pair_index,
                "body": {
                    "messages": [
                        {
                            "content": f"בדיקה רפואית {pair_index}",
                            "role": "user",
                        }
                    ],
                    "model": "gpt-test-snapshot",
                },
            }
            for pair_index in range(3)
        ]
        count_dir = self.root / "count-shards"
        count_manifest = write_request_shards(
            requests,
            stage="answer",
            output_dir=count_dir,
            max_shard_bytes=50_000,
            max_shard_requests=2,
        )

        self.assertEqual(count_manifest["shard_count"], 2)
        self.assertEqual(
            [item["request_count"] for item in count_manifest["shards"]], [2, 1]
        )
        first_payload = (
            count_dir / count_manifest["shards"][0]["path"]
        ).read_bytes()
        self.assertIn("בדיקה רפואית".encode("utf-8"), first_payload)
        self.assertNotIn(b"\\u05d1", first_payload)

        index = load_jsonl(count_dir / "request_index.jsonl")
        byte_limit = max(item["line_bytes"] for item in index)
        byte_dir = self.root / "byte-shards"
        byte_manifest = write_request_shards(
            requests[:2],
            stage="answer",
            output_dir=byte_dir,
            max_shard_bytes=byte_limit,
            max_shard_requests=10,
        )

        self.assertEqual(byte_manifest["shard_count"], 2)
        for shard in byte_manifest["shards"]:
            shard_path = byte_dir / shard["path"]
            self.assertLessEqual(shard_path.stat().st_size, byte_limit)
            self.assertEqual(shard["bytes"], shard_path.stat().st_size)

    def test_single_oversize_request_is_rejected_without_artifacts(self) -> None:
        request = {
            "pair_index": 5,
            "body": {
                "messages": [
                    {"content": "א" * 100, "role": "user"}
                ],
                "model": "gpt-test-snapshot",
            },
        }
        probe_dir = self.root / "probe"
        write_request_shards(
            [request],
            stage="positive",
            output_dir=probe_dir,
            max_shard_bytes=10_000,
            max_shard_requests=10,
        )
        line_bytes = load_jsonl(probe_dir / "request_index.jsonl")[0]["line_bytes"]
        rejected_dir = self.root / "rejected"

        with self.assertRaisesRegex(ValueError, "cannot fit"):
            write_request_shards(
                [request],
                stage="positive",
                output_dir=rejected_dir,
                max_shard_bytes=line_bytes - 1,
                max_shard_requests=10,
            )
        self.assertFalse(rejected_dir.exists())


if __name__ == "__main__":
    unittest.main()
