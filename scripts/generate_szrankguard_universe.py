#!/usr/bin/env python3
"""Generate a Qlib trading universe from SZRankGuard support CSV filenames."""

from __future__ import annotations

import argparse
import os
import tempfile
from datetime import date
from pathlib import Path


DEFAULT_SOURCE = Path("/mnt/hdd/qlib_data/SZRankGuard_symbol_support")
DEFAULT_OUTPUT = Path("/mnt/hdd/qlib_data/us_data/instruments/szrankguard.txt")
DEFAULT_REFERENCE = Path("/mnt/hdd/qlib_data/us_data/instruments/all.txt")
FILE_SUFFIX = "_support.csv"


def iso_date(value: str) -> str:
    """Validate and normalize an ISO date supplied on the command line."""
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a valid date; expected YYYY-MM-DD"
        ) from exc


def read_symbols(source_dir: Path) -> list[str]:
    """Read symbols from files named ``<symbol>_support.csv``."""
    if not source_dir.is_dir():
        raise NotADirectoryError(f"source directory does not exist: {source_dir}")

    symbols: set[str] = set()
    for path in source_dir.iterdir():
        if not path.is_file() or not path.name.lower().endswith(FILE_SUFFIX):
            continue
        symbol = path.name[: -len(FILE_SUFFIX)].strip().upper()
        if not symbol:
            raise ValueError(f"cannot extract a symbol from {path.name!r}")
        if any(char in symbol for char in "\t\r\n"):
            raise ValueError(f"invalid symbol extracted from {path.name!r}")
        symbols.add(symbol)

    if not symbols:
        raise ValueError(f"no *{FILE_SUFFIX} files found in {source_dir}")
    return sorted(symbols)


def write_universe(
    symbols: list[str], output_path: Path, start_date: str, end_date: str
) -> None:
    """Atomically write tab-separated Qlib instrument records."""
    if start_date > end_date:
        raise ValueError(f"start date {start_date} is after end date {end_date}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            for symbol in symbols:
                stream.write(f"{symbol}\t{start_date}\t{end_date}\n")
        os.replace(temporary_name, output_path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def read_reference_symbols(reference_path: Path) -> set[str]:
    """Read symbols from the first column of a Qlib instrument file."""
    if not reference_path.is_file():
        raise FileNotFoundError(f"reference universe does not exist: {reference_path}")

    symbols: set[str] = set()
    with reference_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            symbol = stripped.split("\t")[0].strip().upper()
            if not symbol:
                raise ValueError(
                    f"empty symbol in {reference_path} at line {line_number}"
                )
            symbols.add(symbol)
    return symbols


def print_unmatched(symbols: list[str], reference_path: Path) -> None:
    """Print every source symbol absent from the reference universe."""
    reference_symbols = read_reference_symbols(reference_path)
    unmatched = [symbol for symbol in symbols if symbol not in reference_symbols]

    if unmatched:
        print(f"Unmatched/unfound symbols ({len(unmatched)}):")
        for symbol in unmatched:
            print(symbol)
    else:
        print("Unmatched/unfound symbols (0): none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Qlib universe from <symbol>_support.csv filenames. "
            "The output has: symbol<TAB>start_date<TAB>end_date."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reference-universe",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="Qlib universe used to identify unmatched symbols (default: %(default)s)",
    )
    parser.add_argument("--start-date", type=iso_date, default="1999-01-01")
    parser.add_argument("--end-date", type=iso_date, default="2099-12-31")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = read_symbols(args.source_dir.expanduser())
    output = args.output.expanduser()
    write_universe(symbols, output, args.start_date, args.end_date)
    print(f"Wrote {len(symbols)} symbols to {output}")
    print_unmatched(symbols, args.reference_universe.expanduser())


if __name__ == "__main__":
    main()
