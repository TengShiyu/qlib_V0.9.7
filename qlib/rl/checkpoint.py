# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Failure-safe checkpoint publication shared by rolling RL strategies."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable


class RollingCheckpointRun:
    """Stage an arbitrary rolling checkpoint set and replace the old set on success.

    The official directory is left untouched until :meth:`publish` verifies
    that every expected rolling window produced its checkpoint. A failed or
    incomplete run removes only its staging directory.
    """

    def __init__(self, checkpoint_dir: Path, window_ids: Iterable[int]) -> None:
        self.checkpoint_dir = Path(checkpoint_dir).expanduser()
        if self.checkpoint_dir.name in {"", ".", ".."}:
            raise ValueError("checkpoint_dir must name a specific directory.")
        resolved_ids = tuple(int(window_id) for window_id in window_ids)
        if not resolved_ids or any(window_id < 0 for window_id in resolved_ids):
            raise ValueError("window_ids must contain non-negative integers.")
        if len(set(resolved_ids)) != len(resolved_ids):
            raise ValueError("window_ids must be unique.")
        self.window_ids = resolved_ids
        self.staging_dir = self.checkpoint_dir.with_name(f".{self.checkpoint_dir.name}.next")
        self.backup_dir = self.checkpoint_dir.with_name(f".{self.checkpoint_dir.name}.previous")
        self._published = False

    def __enter__(self) -> "RollingCheckpointRun":
        self.checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
        self._recover_interrupted_publication()
        self._remove_generated_directory(self.staging_dir)
        self.staging_dir.mkdir()
        return self

    def checkpoint_path(self, window_id: int, filename: str = "best_policy.pt") -> Path:
        """Return the staging path for one expected window checkpoint."""

        resolved_id = int(window_id)
        if resolved_id not in self.window_ids:
            raise ValueError(f"Unexpected rolling window ID: {resolved_id}")
        if Path(filename).name != filename or filename in {"", ".", ".."}:
            raise ValueError("filename must be one plain file name.")
        window_dir = self.staging_dir / f"window_{resolved_id:03d}"
        window_dir.mkdir(parents=True, exist_ok=True)
        return window_dir / filename

    def official_checkpoint_path(self, window_id: int, filename: str = "best_policy.pt") -> Path:
        """Return the path a staged checkpoint receives after publication."""

        resolved_id = int(window_id)
        if resolved_id not in self.window_ids:
            raise ValueError(f"Unexpected rolling window ID: {resolved_id}")
        return self.checkpoint_dir / f"window_{resolved_id:03d}" / filename

    def publish(self, filename: str = "best_policy.pt") -> Path:
        """Replace the completed checkpoint set and permanently clear the old set."""

        missing = [
            window_id
            for window_id in self.window_ids
            if not (self.staging_dir / f"window_{window_id:03d}" / filename).is_file()
        ]
        if missing:
            raise RuntimeError(f"Rolling checkpoint set is incomplete; missing windows: {missing}")

        self._remove_generated_directory(self.backup_dir)
        if self.checkpoint_dir.exists():
            self.checkpoint_dir.replace(self.backup_dir)
        try:
            self.staging_dir.replace(self.checkpoint_dir)
        except Exception:
            if self.backup_dir.exists() and not self.checkpoint_dir.exists():
                self.backup_dir.replace(self.checkpoint_dir)
            raise
        self._remove_generated_directory(self.backup_dir)
        self._published = True
        return self.checkpoint_dir

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._published:
            self._remove_generated_directory(self.staging_dir)

    def _recover_interrupted_publication(self) -> None:
        if self.backup_dir.exists() and not self.checkpoint_dir.exists():
            self.backup_dir.replace(self.checkpoint_dir)
        elif self.backup_dir.exists():
            self._remove_generated_directory(self.backup_dir)

    def _remove_generated_directory(self, path: Path) -> None:
        expected = {self.staging_dir, self.backup_dir}
        if path not in expected:
            raise ValueError(f"Refusing to clear unmanaged directory: {path}")
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
