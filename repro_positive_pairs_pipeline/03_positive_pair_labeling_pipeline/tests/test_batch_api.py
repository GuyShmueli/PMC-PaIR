from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from positive_pair_labeling_pipeline.batch_api import BatchManager
from positive_pair_labeling_pipeline.errors import BatchStateError, ConfigurationError
from positive_pair_labeling_pipeline.io_utils import load_json, save_bytes, sha256_file


class FakeFiles:
    def __init__(self) -> None:
        self.create_calls = 0
        self.content_calls: list[str] = []
        self.output_payload = b'{"custom_id":"request-1"}\n'
        self.error_payload = b""

    def create(self, *, file, purpose: str):
        self.create_calls += 1
        if purpose != "batch" or not file.read(1):
            raise AssertionError("Unexpected upload call")
        return SimpleNamespace(id="file_input")

    def content(self, file_id: str):
        self.content_calls.append(file_id)
        if file_id == "file_output":
            return SimpleNamespace(content=self.output_payload)
        if file_id == "file_errors":
            return SimpleNamespace(content=self.error_payload)
        raise AssertionError(f"Unexpected content file ID: {file_id}")


class FakeBatches:
    def __init__(self) -> None:
        self.create_calls = 0
        self.retrieve_calls = 0
        self.create_kwargs: dict | None = None
        self.create_error: Exception | None = None
        self.status = "validating"
        self.error_file_id: str | None = None

    def create(self, **kwargs):
        self.create_calls += 1
        self.create_kwargs = kwargs
        if self.create_error is not None:
            raise self.create_error
        return SimpleNamespace(
            endpoint=kwargs["endpoint"],
            id="batch_fake",
            input_file_id=kwargs["input_file_id"],
            status=self.status,
        )

    def retrieve(self, batch_id: str):
        self.retrieve_calls += 1
        if batch_id != "batch_fake":
            raise AssertionError(f"Unexpected batch ID: {batch_id}")
        return SimpleNamespace(
            endpoint="/v1/chat/completions",
            error_file_id=self.error_file_id,
            id=batch_id,
            input_file_id="file_input",
            output_file_id="file_output",
            status=self.status,
        )


class FakeClient:
    def __init__(self) -> None:
        self.files = FakeFiles()
        self.batches = FakeBatches()


class BatchManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.shard_path = self.root / "requests-00000.jsonl"
        save_bytes(b'{"body":{},"custom_id":"x","method":"POST"}\n', self.shard_path)
        self.shard_sha256 = sha256_file(self.shard_path)

    def manager(self, client: FakeClient) -> BatchManager:
        return BatchManager(client=client, state_dir=self.root / "remote")

    def test_explicit_acknowledgement_is_required_before_external_calls(self) -> None:
        client = FakeClient()
        manager = self.manager(client)

        with self.assertRaisesRegex(ConfigurationError, "not acknowledged"):
            manager.submit_shard(self.shard_path)

        self.assertEqual(client.files.create_calls, 0)
        self.assertEqual(client.batches.create_calls, 0)
        self.assertFalse((self.root / "remote").exists())

    def test_successful_submission_is_idempotent_by_shard_hash(self) -> None:
        client = FakeClient()
        manager = self.manager(client)

        first = manager.submit_shard(
            self.shard_path,
            expected_sha256=self.shard_sha256,
            stage="citation",
            acknowledge_external_api=True,
        )
        second = manager.submit_shard(
            self.shard_path,
            expected_sha256=self.shard_sha256,
            stage="citation",
            acknowledge_external_api=True,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["batch_id"], "batch_fake")
        self.assertEqual(client.files.create_calls, 1)
        self.assertEqual(client.batches.create_calls, 1)
        self.assertEqual(
            client.batches.create_kwargs["endpoint"], "/v1/chat/completions"
        )
        self.assertEqual(client.batches.create_kwargs["completion_window"], "24h")
        record_path = (
            self.root / "remote" / "submissions" / f"{self.shard_sha256}.json"
        )
        self.assertEqual(load_json(record_path)["state"], "submitted")

    def test_ambiguous_create_state_blocks_automatic_retry(self) -> None:
        client = FakeClient()
        client.batches.create_error = TimeoutError("create result unknown")
        manager = self.manager(client)

        with self.assertRaisesRegex(BatchStateError, "ambiguous"):
            manager.submit_shard(
                self.shard_path,
                acknowledge_external_api=True,
            )
        record_path = (
            self.root / "remote" / "submissions" / f"{self.shard_sha256}.json"
        )
        self.assertEqual(load_json(record_path)["state"], "ambiguous_create")

        client.batches.create_error = None
        with self.assertRaisesRegex(BatchStateError, "ambiguous-create"):
            manager.submit_shard(
                self.shard_path,
                acknowledge_external_api=True,
            )
        self.assertEqual(client.files.create_calls, 1)
        self.assertEqual(client.batches.create_calls, 1)

    def test_completed_only_atomic_download_and_idempotent_reread(self) -> None:
        client = FakeClient()
        manager = self.manager(client)
        manager.submit_shard(
            self.shard_path,
            acknowledge_external_api=True,
        )

        with self.assertRaisesRegex(ConfigurationError, "not acknowledged"):
            manager.download_completed(self.shard_sha256)
        retrieve_calls_before = client.batches.retrieve_calls
        with self.assertRaisesRegex(BatchStateError, "not completed"):
            manager.download_completed(
                self.shard_sha256,
                acknowledge_external_api=True,
            )
        self.assertEqual(client.files.content_calls, [])
        self.assertEqual(
            client.batches.retrieve_calls, retrieve_calls_before + 1
        )
        self.assertFalse((self.root / "remote" / "downloads").exists())

        client.batches.status = "completed"
        client.batches.error_file_id = "file_errors"
        first_manifest = manager.download_completed(
            self.shard_sha256,
            acknowledge_external_api=True,
        )
        destination = self.root / "remote" / "downloads" / "batch_fake"
        self.assertTrue((destination / "output.jsonl").is_file())
        self.assertTrue((destination / "errors.jsonl").is_file())
        self.assertTrue((destination / "batch.json").is_file())
        self.assertTrue((destination / "download_manifest.json").is_file())
        self.assertFalse(
            any(
                path.name.startswith(".batch_fake.staging-")
                for path in destination.parent.iterdir()
            )
        )
        original_output = (destination / "output.jsonl").read_bytes()
        self.assertEqual(client.files.content_calls, ["file_output", "file_errors"])

        client.files.output_payload = b"changed remote content"
        second_manifest = manager.download_completed(
            self.shard_sha256,
            acknowledge_external_api=True,
        )
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual((destination / "output.jsonl").read_bytes(), original_output)
        self.assertEqual(client.files.content_calls, ["file_output", "file_errors"])
