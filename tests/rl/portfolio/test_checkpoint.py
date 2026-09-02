# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tempfile
import unittest
from pathlib import Path

from qlib.rl.checkpoint import RollingCheckpointRun


def publish_windows(checkpoint_dir: Path, count: int) -> None:
    with RollingCheckpointRun(checkpoint_dir, range(count)) as checkpoint_run:
        for window_id in range(count):
            checkpoint_run.checkpoint_path(window_id).write_bytes(f"window-{window_id}".encode())
        checkpoint_run.publish()


class RollingCheckpointRunTest(unittest.TestCase):
    def test_replaces_ten_windows_with_six_without_stale_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "strategy" / "rolling" / "checkpoints"
            publish_windows(checkpoint_dir, 10)
            publish_windows(checkpoint_dir, 6)

            self.assertEqual(
                sorted(path.name for path in checkpoint_dir.iterdir()),
                [f"window_{window_id:03d}" for window_id in range(6)],
            )

    def test_replaces_six_windows_with_ten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "strategy" / "rolling" / "checkpoints"
            publish_windows(checkpoint_dir, 6)
            publish_windows(checkpoint_dir, 10)

            self.assertEqual(len(list(checkpoint_dir.iterdir())), 10)
            self.assertEqual(
                (checkpoint_dir / "window_009" / "best_policy.pt").read_bytes(),
                b"window-9",
            )

    def test_incomplete_run_preserves_previous_completed_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "strategy" / "rolling" / "checkpoints"
            publish_windows(checkpoint_dir, 4)

            with self.assertRaisesRegex(RuntimeError, "missing windows"):
                with RollingCheckpointRun(checkpoint_dir, range(3)) as checkpoint_run:
                    checkpoint_run.checkpoint_path(0).write_bytes(b"new-window-0")
                    checkpoint_run.publish()

            self.assertEqual(len(list(checkpoint_dir.iterdir())), 4)
            self.assertEqual(
                (checkpoint_dir / "window_000" / "best_policy.pt").read_bytes(),
                b"window-0",
            )
            self.assertFalse((checkpoint_dir.parent / ".checkpoints.next").exists())

    def test_unexpected_window_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "checkpoints"
            with RollingCheckpointRun(checkpoint_dir, [2, 4]) as checkpoint_run:
                with self.assertRaisesRegex(ValueError, "Unexpected rolling window"):
                    checkpoint_run.checkpoint_path(3)


if __name__ == "__main__":
    unittest.main()
