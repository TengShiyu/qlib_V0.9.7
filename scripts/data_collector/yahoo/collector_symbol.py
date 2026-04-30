# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from typing import Iterable
from concurrent.futures import ProcessPoolExecutor

import fire
from loguru import logger
from tqdm import tqdm
from qlib.constant import REG_CN as REGION_CN
from qlib.utils import code_to_fname

import collector as yahoo_collector


DEFAULT_QLIB_DIR = Path("/mnt/hdd/qlib_data/us_data_symbol")
DEFAULT_SOURCE_DIR = DEFAULT_QLIB_DIR.joinpath("metadata", "source")
DEFAULT_NORMALIZE_DIR = DEFAULT_QLIB_DIR.joinpath("metadata", "normalize")


def _init_default_folder_structure(qlib_dir: Path = DEFAULT_QLIB_DIR):
    qlib_dir = Path(qlib_dir).expanduser()
    for sub_dir in ["calendars", "features", "instruments", "metadata/source", "metadata/normalize"]:
        qlib_dir.joinpath(sub_dir).mkdir(parents=True, exist_ok=True)


def _resolve_default_child_dir(path, child_dir: Path):
    if path is None:
        return child_dir

    path = Path(path).expanduser()
    if path.resolve() == DEFAULT_QLIB_DIR.expanduser().resolve():
        return child_dir
    return path


def _normalize_input_symbol(symbol) -> str:
    symbol = str(symbol).strip()
    if "." not in symbol:
        return symbol.upper()

    code, suffix = symbol.rsplit(".", maxsplit=1)
    if suffix.lower() in {"ss", "sz"}:
        return f"{code.upper()}.{suffix.lower()}"
    return f"{code.upper()}.{suffix.upper()}"


def _parse_symbols(symbols) -> list:
    if symbols is None:
        raise ValueError("symbols must contain at least one stock symbol, for example: --symbols AAPL")

    if isinstance(symbols, str):
        symbols = symbols.replace(",", " ").split()
    elif isinstance(symbols, Iterable):
        symbols = list(symbols)
    else:
        symbols = [symbols]

    symbols = [_normalize_input_symbol(symbol) for symbol in symbols if str(symbol).strip()]
    if not symbols:
        raise ValueError("symbols must contain at least one stock symbol, for example: --symbols AAPL")
    return sorted(set(symbols))


class YahooSymbolCollectorMixin:
    def __init__(self, *args, symbols=None, **kwargs):
        self.symbols = _parse_symbols(symbols)
        super().__init__(*args, **kwargs)

    def get_instrument_list(self):
        logger.info(f"get {len(self.symbols)} assigned symbols: {self.symbols}")
        return self.symbols


class YahooSymbolCollectorCN1d(YahooSymbolCollectorMixin, yahoo_collector.YahooCollectorCN1d):
    pass


class YahooSymbolCollectorCN1min(YahooSymbolCollectorMixin, yahoo_collector.YahooCollectorCN1min):
    pass


class YahooSymbolCollectorUS1d(YahooSymbolCollectorMixin, yahoo_collector.YahooCollectorUS1d):
    pass


class YahooSymbolCollectorUS1min(YahooSymbolCollectorMixin, yahoo_collector.YahooCollectorUS1min):
    pass


class YahooSymbolCollectorIN1d(YahooSymbolCollectorMixin, yahoo_collector.YahooCollectorIN1d):
    pass


class YahooSymbolCollectorIN1min(YahooSymbolCollectorMixin, yahoo_collector.YahooCollectorIN1min):
    pass


class YahooSymbolCollectorBR1d(YahooSymbolCollectorMixin, yahoo_collector.YahooCollectorBR1d):
    pass


class YahooSymbolCollectorBR1min(YahooSymbolCollectorMixin, yahoo_collector.YahooCollectorBR1min):
    pass


class SymbolNormalize(yahoo_collector.Normalize):
    def __init__(self, *args, file_names=None, **kwargs):
        self._file_names = set(file_names or [])
        super().__init__(*args, **kwargs)

    def normalize(self):
        logger.info("normalize assigned symbol data......")

        if self._file_names:
            file_list = [self._source_dir.joinpath(file_name) for file_name in sorted(self._file_names)]
            missing_files = [file_path.name for file_path in file_list if not file_path.exists()]
            if missing_files:
                logger.warning(f"missing source csv files will be skipped: {missing_files}")
            file_list = [file_path for file_path in file_list if file_path.exists()]
        else:
            file_list = list(self._source_dir.glob("*.csv"))

        if not file_list:
            raise FileNotFoundError(f"no source csv files found to normalize in {self._source_dir}")

        with ProcessPoolExecutor(max_workers=self._max_workers) as worker:
            with tqdm(total=len(file_list)) as p_bar:
                for _ in worker.map(self._executor, file_list):
                    p_bar.update()


for _symbol_collector_class in [
    YahooSymbolCollectorCN1d,
    YahooSymbolCollectorCN1min,
    YahooSymbolCollectorUS1d,
    YahooSymbolCollectorUS1min,
    YahooSymbolCollectorIN1d,
    YahooSymbolCollectorIN1min,
    YahooSymbolCollectorBR1d,
    YahooSymbolCollectorBR1min,
]:
    setattr(yahoo_collector, _symbol_collector_class.__name__, _symbol_collector_class)


class Run(yahoo_collector.Run):
    def __init__(
        self,
        symbols=None,
        source_dir=None,
        normalize_dir=None,
        max_workers=1,
        interval="1d",
        region=REGION_CN,
    ):
        """
        Parameters
        ----------
        symbols : str or list
            One or more Yahoo symbols to collect, for example AAPL or "AAPL,MSFT".
            It can also be provided to download_data, download_today_data, or update_data_to_bin.
        source_dir : str
            The directory where raw data is saved.
            Defaults to /mnt/hdd/qlib_data/us_data_symbol/metadata/source.
        normalize_dir : str
            Directory for normalized data.
            Defaults to /mnt/hdd/qlib_data/us_data_symbol/metadata/normalize.
        max_workers : int
            Concurrent number, default is 1.
        interval : str
            Frequency, value from [1min, 1d], default 1d.
        region : str
            Region, value from ["CN", "US", "IN", "BR"], default "CN".
        """
        self.symbols = _parse_symbols(symbols) if symbols is not None else None
        _init_default_folder_structure()
        super().__init__(
            source_dir=_resolve_default_child_dir(source_dir, DEFAULT_SOURCE_DIR),
            normalize_dir=_resolve_default_child_dir(normalize_dir, DEFAULT_NORMALIZE_DIR),
            max_workers=max_workers,
            interval=interval,
            region=region,
        )
        self._source_dir_provided = source_dir is not None
        self._normalize_dir_provided = normalize_dir is not None

    @property
    def collector_class_name(self):
        return f"YahooSymbolCollector{self.region.upper()}{self.interval}"

    def _set_symbols(self, symbols=None):
        if symbols is not None:
            self.symbols = _parse_symbols(symbols)
        if self.symbols is None:
            raise ValueError("symbols must contain at least one stock symbol, for example: --symbols AAPL")
        return self.symbols

    def download_data(
        self,
        symbols=None,
        max_collector_count=2,
        delay=0.5,
        start=None,
        end=None,
        check_data_length=None,
        limit_nums=None,
    ):
        """Download Yahoo data for assigned symbols only."""
        symbols = self._set_symbols(symbols)

        if self.interval == "1d" and yahoo_collector.pd.Timestamp(end) > yahoo_collector.pd.Timestamp(
            yahoo_collector.datetime.datetime.now().strftime("%Y-%m-%d")
        ):
            raise ValueError(f"end_date: {end} is greater than the current date.")

        super(yahoo_collector.Run, self).download_data(
            max_collector_count=max_collector_count,
            delay=delay,
            start=start,
            end=end,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
            symbols=symbols,
        )

    def download_today_data(
        self,
        symbols=None,
        max_collector_count=2,
        delay=0.5,
        check_data_length=None,
        limit_nums=None,
    ):
        """Download today's Yahoo data for assigned symbols only."""
        self._set_symbols(symbols)
        start = yahoo_collector.datetime.datetime.now().date()
        end = yahoo_collector.pd.Timestamp(start + yahoo_collector.pd.Timedelta(days=1)).date()
        self.download_data(
            max_collector_count=max_collector_count,
            delay=delay,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )

    def normalize_data(
        self,
        symbols=None,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        end_date: str = None,
        qlib_data_1d_dir: str = None,
    ):
        """Normalize data for assigned symbols only."""
        self._set_symbols(symbols)
        if self.interval.lower() == "1min":
            if qlib_data_1d_dir is None or not Path(qlib_data_1d_dir).expanduser().exists():
                raise ValueError(
                    "If normalize 1min, the qlib_data_1d_dir parameter must be set: "
                    "--qlib_data_1d_dir <user qlib 1d data >"
                )

        normalize_class = getattr(yahoo_collector, self.normalize_class_name)
        SymbolNormalize(
            source_dir=self.source_dir,
            target_dir=self.normalize_dir,
            normalize_class=normalize_class,
            max_workers=self.max_workers,
            date_field_name=date_field_name,
            symbol_field_name=symbol_field_name,
            end_date=end_date,
            qlib_data_1d_dir=qlib_data_1d_dir,
            file_names=self._get_expected_symbol_csv_files(),
        ).normalize()

    def normalize_data_1d_extend(
        self,
        old_qlib_data_dir,
        symbols=None,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
    ):
        """Normalize extend data for assigned symbols only."""
        self._set_symbols(symbols)
        normalize_class = getattr(yahoo_collector, f"{self.normalize_class_name}Extend")
        SymbolNormalize(
            source_dir=self.source_dir,
            target_dir=self.normalize_dir,
            normalize_class=normalize_class,
            max_workers=self.max_workers,
            date_field_name=date_field_name,
            symbol_field_name=symbol_field_name,
            old_qlib_data_dir=old_qlib_data_dir,
            file_names=self._get_expected_symbol_csv_files(),
        ).normalize()

    def update_data_to_bin(
        self,
        qlib_data_1d_dir: str = DEFAULT_QLIB_DIR,
        symbols=None,
        trading_date: str = None,
        end_date: str = None,
        check_data_length: int = None,
        delay: float = 0.1,
        stale_ratio: float = 0.001,
        exists_skip: bool = False,
    ):
        """Update qlib binary data using Yahoo data for assigned symbols only."""
        self._set_symbols(symbols)

        if self.interval.lower() != "1d":
            logger.warning("currently supports 1d data updates: --interval 1d")
        if stale_ratio < 0 or stale_ratio > 1:
            raise ValueError(f"stale_ratio should be within [0, 1], got {stale_ratio}")

        qlib_data_1d_dir = str(Path(qlib_data_1d_dir).expanduser().resolve())
        metadata_dir = Path(qlib_data_1d_dir).joinpath("metadata")
        metadata_dir.mkdir(parents=True, exist_ok=True)
        if not self._source_dir_provided:
            self.source_dir = metadata_dir.joinpath("source")
            self.source_dir.mkdir(parents=True, exist_ok=True)
        if not self._normalize_dir_provided:
            self.normalize_dir = metadata_dir.joinpath("normalize")
            self.normalize_dir.mkdir(parents=True, exist_ok=True)

        if not yahoo_collector.exists_qlib_data(qlib_data_1d_dir):
            yahoo_collector.GetData().qlib_data(
                target_dir=qlib_data_1d_dir, interval=self.interval, region=self.region, exists_skip=exists_skip
            )

        if trading_date is None:
            calendar_df = yahoo_collector.pd.read_csv(Path(qlib_data_1d_dir).joinpath("calendars/day.txt"))
            trading_date = (
                yahoo_collector.pd.Timestamp(calendar_df.iloc[-1, 0]) - yahoo_collector.pd.Timedelta(days=1)
            ).strftime("%Y-%m-%d")
        else:
            trading_date = yahoo_collector.pd.Timestamp(trading_date).strftime("%Y-%m-%d")

        if end_date is None:
            end_date = (yahoo_collector.pd.Timestamp(trading_date) + yahoo_collector.pd.Timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )

        source_csv_paths = list(self.source_dir.glob("*.csv"))
        raw_precheck_skipped_no_source = False
        expected_files = None
        if not source_csv_paths:
            logger.info(f"skip precheck: no csv files found in {self.source_dir}, start downloading directly")
            raw_precheck_skipped_no_source = True
            self.download_data(delay=delay, start=trading_date, end=end_date, check_data_length=check_data_length)
        else:
            expected_files = self._get_expected_symbol_csv_files()

        if expected_files and self._is_raw_source_up_to_date(
            end_date, stale_ratio=stale_ratio, expected_files=expected_files
        ):
            logger.info(f"skip download: source csv files are already up-to-date for end_date={end_date}")
        elif not raw_precheck_skipped_no_source:
            self.download_data(delay=delay, start=trading_date, end=end_date, check_data_length=check_data_length)

        self.max_workers = (
            max(yahoo_collector.multiprocessing.cpu_count() - 2, 1)
            if self.max_workers is None or self.max_workers <= 1
            else self.max_workers
        )

        self.normalize_data_1d_extend(qlib_data_1d_dir)

        if raw_precheck_skipped_no_source:
            logger.info("skip normalized precheck: raw precheck was skipped because source csv dir was empty")
        else:
            if expected_files is None:
                expected_files = self._get_expected_symbol_csv_files()
            self._precheck_normalized_source(end_date=end_date, stale_ratio=stale_ratio, expected_files=expected_files)

        dump_data = yahoo_collector.DumpDataUpdate(
            data_path=self.normalize_dir,
            qlib_dir=qlib_data_1d_dir,
            exclude_fields="symbol,date",
            max_workers=self.max_workers,
        )
        dump_data.dump()
        logger.info("skip full-market index instrument parsing for assigned-symbol dataset")

    def _get_expected_symbol_csv_files(self) -> set:
        self._set_symbols()
        collector_cls = getattr(yahoo_collector, f"YahooCollector{self.region.upper()}{self.interval}")
        collector = collector_cls.__new__(collector_cls)
        return {f"{code_to_fname(collector.normalize_symbol(symbol))}.csv" for symbol in self.symbols}


if __name__ == "__main__":
    fire.Fire(Run)
