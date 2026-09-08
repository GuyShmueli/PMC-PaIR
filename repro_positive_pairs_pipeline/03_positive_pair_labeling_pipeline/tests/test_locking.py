from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from positive_pair_labeling_pipeline.errors import PipelineError
from positive_pair_labeling_pipeline.locking import run_lock


class RunLockTests(unittest.TestCase):
    def test_nested_lock_fails_then_lock_can_be_reacquired(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)

            with run_lock(run_dir):
                with self.assertRaisesRegex(
                    PipelineError, "Another pipeline command is already using this run"
                ):
                    with run_lock(run_dir):
                        self.fail("A nested lock unexpectedly succeeded")

            with run_lock(run_dir):
                self.assertTrue((run_dir / ".pipeline.lock").is_file())


if __name__ == "__main__":
    unittest.main()
