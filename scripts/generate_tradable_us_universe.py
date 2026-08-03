#!/usr/bin/env python3
"""Generate the merged Qlib US execution universe."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


INSTRUMENTS_DIR = Path("/mnt/hdd/qlib_data/us_data/instruments")
DEFAULT_SOURCES = ("all", "sp500", "nasdaq100", "djia", "sp400")
OUTPUT_PATH = INSTRUMENTS_DIR / "tradable_us.txt"


def read_instrument_file(path: Path) -> dict[str, tuple[str, str]]:
    """Read a Qlib instrument file and merge duplicate symbol date ranges."""
    instruments: dict[str, tuple[str, str]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.rstrip("\r\n")
            if not stripped:
                continue
            fields = stripped.split("\t")
            if len(fields) != 3:
                raise ValueError(
                    f"{path}:{line_number}: expected 3 tab-separated fields, "
                    f"found {len(fields)}"
                )
            symbol, start_date, end_date = fields
            if not symbol:
                continue
            if symbol in instruments:
                old_start, old_end = instruments[symbol]
                start_date = min(old_start, start_date)
                end_date = max(old_end, end_date)
            instruments[symbol] = (start_date, end_date)
    return instruments


def merge_instruments(
    instruments_dir: Path, source_names: tuple[str, ...]
) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Merge sources, keeping each symbol's earliest start and latest end."""
    merged: dict[str, tuple[str, str]] = {}
    used_sources: list[str] = []

    for source_name in source_names:
        source_path = instruments_dir / f"{source_name}.txt"
        if not source_path.is_file():
            print(f"Warning: source does not exist; skipping {source_path}")
            continue

        source_instruments = read_instrument_file(source_path)
        if not source_instruments:
            print(f"Warning: source is empty; skipping {source_path}")
            continue

        used_sources.append(source_name)
        for symbol, (start_date, end_date) in source_instruments.items():
            if symbol in merged:
                old_start, old_end = merged[symbol]
                start_date = min(old_start, start_date)
                end_date = max(old_end, end_date)
            merged[symbol] = (start_date, end_date)

    if not used_sources:
        raise ValueError("no non-empty source instrument files were found")
    return merged, used_sources


def write_instruments(
    instruments: dict[str, tuple[str, str]], output_path: Path
) -> None:
    """Atomically write sorted, tab-separated Qlib instrument records."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            for symbol in sorted(instruments):
                start_date, end_date = instruments[symbol]
                stream.write(f"{symbol}\t{start_date}\t{end_date}\n")
        os.replace(temporary_name, output_path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge Qlib US instrument universes, keeping the earliest start "
            "date and latest end date for each symbol."
        )
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(DEFAULT_SOURCES),
        metavar="NAME",
        help="source filenames without .txt (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not INSTRUMENTS_DIR.is_dir():
        raise NotADirectoryError(
            f"instruments directory does not exist: {INSTRUMENTS_DIR}"
        )

    source_names = tuple(dict.fromkeys(args.sources))
    instruments, used_sources = merge_instruments(INSTRUMENTS_DIR, source_names)
    write_instruments(instruments, OUTPUT_PATH)
    print(
        f"Wrote {len(instruments)} symbols to {OUTPUT_PATH} "
        f"from {', '.join(used_sources)}"
    )


if __name__ == "__main__":
    main()
