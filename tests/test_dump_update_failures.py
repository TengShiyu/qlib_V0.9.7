"""Local-only update failures, metadata publication, and safe retry coverage."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dump_bin


@pytest.fixture
def dataset(tmp_path):
    root, source = tmp_path / "data", tmp_path / "source"
    source.mkdir()
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "calendars/day.txt").write_text("2025-01-02\n2025-01-03\n")
    (root / "instruments/all.txt").write_text("".join(
        f"{symbol}\t2025-01-02\t2025-01-03\n" for symbol in ["AAA", "BBB", "MISSING"]))
    for symbol, values in {"aaa": [100, 101], "bbb": [200, 201], "missing": [50, 51]}.items():
        directory = root / "features" / symbol
        directory.mkdir(parents=True)
        np.array([0, *values], dtype="<f4").tofile(directory / "close.day.bin")
    for symbol, rows in {
        "AAA": [("2025-01-03", 111), ("2025-01-06", 102), ("2025-01-08", 104)],
        "BBB": [("2025-01-06", 202), ("2025-01-07", 203)],
        "CCC": [("2025-01-08", 300)],
    }.items():
        (source / f"{symbol}.csv").write_text("symbol,date,close\n" + "".join(
            f"{symbol},{date},{price}\n" for date, price in rows))
    return root, source


def updater(dataset):
    root, source = dataset
    return dump_bin.DumpDataUpdate(data_path=source, qlib_dir=root, max_workers=2, include_fields="close")


def metadata(root):
    return [(root / name).read_bytes() for name in ["calendars/day.txt", "instruments/all.txt"]]


def binaries(root):
    return {path.relative_to(root): path.read_bytes() for path in (root / "features").rglob("*.bin")}


def assert_updated(root):
    assert (root / "calendars/day.txt").read_text().splitlines() == [
        "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]
    for symbol, expected in {
        "aaa": [0, 100, 111, 102, np.nan, 104],
        "bbb": [0, 200, 201, 202, 203],
        "ccc": [4, 300],  # New fields use the global calendar offset.
        "missing": [0, 50, 51],  # No Yahoo rows is not a local write failure.
    }.items():
        np.testing.assert_allclose(np.fromfile(root / f"features/{symbol}/close.day.bin", dtype="<f4"),
                                   expected, equal_nan=True)
    instruments = (root / "instruments/all.txt").read_text()
    assert "AAA\t2025-01-02\t2025-01-08" in instruments
    assert "MISSING\t2025-01-02\t2025-01-03" in instruments
    assert not list(root.rglob(".close.day.bin.*"))
    assert not list(root.glob(".dump-update-*"))


def test_success_and_identical_retry_with_real_workers(dataset):
    root, _ = dataset
    updater(dataset).dump()
    assert_updated(root)
    before = binaries(root), metadata(root)
    updater(dataset).dump()
    assert (binaries(root), metadata(root)) == before


def test_missing_source_price_preserves_existing_observation(dataset):
    root, source = dataset
    path = source / "AAA.csv"
    path.write_text(path.read_text().replace("2025-01-03,111", "2025-01-03,"))
    updater(dataset).dump()
    np.testing.assert_allclose(np.fromfile(root / "features/aaa/close.day.bin", dtype="<f4"),
                               [0, 100, 101, 102, np.nan, 104], equal_nan=True)


def test_new_feature_of_existing_instrument_uses_global_offset(dataset):
    root, _ = dataset
    (root / "features/aaa/close.day.bin").unlink()
    updater(dataset).dump()
    np.testing.assert_allclose(np.fromfile(root / "features/aaa/close.day.bin", dtype="<f4"),
                               [1, 111, 102, np.nan, 104], equal_nan=True)


@pytest.mark.parametrize("failed_symbol", ["bbb", "ccc"])
def test_failed_feature_preserves_metadata_and_retry_does_not_duplicate(dataset, monkeypatch, failed_symbol):
    root, _ = dataset
    before = metadata(root)
    old_bbb = (root / "features/bbb/close.day.bin").read_bytes()
    replace = Path.replace

    def fail_replace(path, target):
        if Path(target) == root / f"features/{failed_symbol}/close.day.bin":
            raise OSError("simulated disk write failure")
        return replace(path, target)

    # Threads let the injected filesystem failure be observed by worker futures;
    # separate success and CLI tests exercise the real process pool.
    with monkeypatch.context() as patch:
        patch.setattr(dump_bin, "ProcessPoolExecutor", ThreadPoolExecutor)
        patch.setattr(Path, "replace", fail_replace)
        with pytest.raises(RuntimeError, match=failed_symbol.upper()):
            updater(dataset).dump()
    assert metadata(root) == before
    assert len(np.fromfile(root / "features/aaa/close.day.bin", dtype="<f4")) > 3
    if failed_symbol == "bbb":
        assert (root / "features/bbb/close.day.bin").read_bytes() == old_bbb
    assert not list(root.rglob(".close.day.bin.*"))
    updater(dataset).dump()
    assert_updated(root)


def test_failed_temp_file_flush_preserves_previous_binaries(dataset, monkeypatch):
    root, _ = dataset
    before = binaries(root), metadata(root)

    def fail_flush(*args):
        raise OSError("simulated full disk")

    with monkeypatch.context() as patch:
        patch.setattr(dump_bin, "ProcessPoolExecutor", ThreadPoolExecutor)
        patch.setattr(dump_bin.os, "fsync", fail_flush)
        with pytest.raises(RuntimeError, match="Binary feature update failed"):
            updater(dataset).dump()
    assert (binaries(root), metadata(root)) == before
    assert not list(root.rglob(".close.day.bin.*"))


def test_failed_metadata_staging_preserves_both_metadata_files(dataset, monkeypatch):
    root, _ = dataset
    before = metadata(root)

    def fail(*args):
        raise OSError("simulated metadata disk error")

    with monkeypatch.context() as patch:
        patch.setattr(dump_bin.DumpDataBase, "save_instruments", fail)
        with pytest.raises(OSError, match="metadata disk error"):
            updater(dataset).dump()
    assert metadata(root) == before
    updater(dataset).dump()
    assert_updated(root)


def test_failed_calendar_publication_keeps_watermark_and_retry_is_safe(dataset, monkeypatch):
    root, _ = dataset
    calendar = root / "calendars/day.txt"
    before = calendar.read_bytes()
    replace = Path.replace

    def fail(path, target):
        if Path(target) == calendar:
            raise OSError("calendar replace failed")
        return replace(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail)
        with pytest.raises(OSError, match="calendar replace failed"):
            updater(dataset).dump()
    assert calendar.read_bytes() == before
    updater(dataset).dump()
    assert_updated(root)


def test_retry_rebuilds_uncommitted_tail_using_new_calendar(dataset, monkeypatch):
    root, source = dataset
    replace = Path.replace

    def fail(path, target):
        if Path(target) == root / "features/bbb/close.day.bin":
            raise OSError("disk failure")
        return replace(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(dump_bin, "ProcessPoolExecutor", ThreadPoolExecutor)
        patch.setattr(Path, "replace", fail)
        with pytest.raises(RuntimeError):
            updater(dataset).dump()
    # This new date changes the meaning of all later uncommitted offsets.
    (source / "DDD.csv").write_text("symbol,date,close\nDDD,2025-01-04,400\n")
    updater(dataset).dump()
    np.testing.assert_allclose(np.fromfile(root / "features/aaa/close.day.bin", dtype="<f4"),
                               [0, 100, 111, np.nan, 102, np.nan, 104], equal_nan=True)
    np.testing.assert_allclose(np.fromfile(root / "features/ccc/close.day.bin", dtype="<f4"), [5, 300])


def test_cli_failure_is_nonzero_and_stops_a_shell_pipeline(dataset, tmp_path):
    root, source = dataset
    before = metadata(root)
    blocked = root / "features/bbb/close.day.bin"
    blocked.unlink()
    blocked.mkdir()  # A real local filesystem error, not an unavailable Yahoo symbol.
    marker = tmp_path / "ml_was_started"
    result = subprocess.run([
        "bash", "-c", 'set -e; "$@"; touch "$ML_STAGE_MARKER"', "test-update",
        sys.executable, "-B", str(Path(dump_bin.__file__)), "dump_update",
        "--data_path", str(source), "--qlib_dir", str(root), "--max_workers", "2", "--include_fields", "close",
    ], env={**os.environ, "ML_STAGE_MARKER": str(marker), "PYTHONDONTWRITEBYTECODE": "1"},
        text=True, capture_output=True, timeout=30)
    assert result.returncode != 0
    assert "Binary feature update failed" in result.stderr and "BBB" in result.stderr
    assert not marker.exists()
    assert metadata(root) == before
