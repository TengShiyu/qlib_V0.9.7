# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Leakage-safe data alignment for the portfolio RL environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DateRange:
    """Inclusive chronological boundaries for one independent RL episode."""

    start: pd.Timestamp
    end: pd.Timestamp

    def __init__(self, start: Union[str, pd.Timestamp], end: Union[str, pd.Timestamp]) -> None:
        normalized_start = pd.Timestamp(start).normalize()
        normalized_end = pd.Timestamp(end).normalize()
        if normalized_start > normalized_end:
            raise ValueError("DateRange start must not be after end.")
        object.__setattr__(self, "start", normalized_start)
        object.__setattr__(self, "end", normalized_end)


ENGINEERING_SPLITS: Dict[str, DateRange] = {
    "train": DateRange("2025-01-02", "2025-09-30"),
    "valid": DateRange("2025-10-01", "2025-12-31"),
    "test": DateRange("2026-01-02", "2026-03-30"),
}


@dataclass(frozen=True)
class PortfolioDataSplit:
    """Aligned arrays for one leakage-isolated two-session RL episode.

    Array shapes are ``(decision_count, instrument_count)``. Scores and
    volatility belong to decision date ``t``. Execution fields belong to
    ``t+1``. Asset returns cover close ``t+1`` through close ``t+3``.
    """

    instruments: Tuple[str, ...]
    decision_dates: pd.DatetimeIndex
    execution_dates: pd.DatetimeIndex
    reward_end_dates: pd.DatetimeIndex
    scores: np.ndarray
    volatility: np.ndarray
    observation_tradable: np.ndarray
    execution_tradable: np.ndarray
    execution_close: np.ndarray
    reward_end_close: np.ndarray
    asset_returns: np.ndarray

    def __post_init__(self) -> None:
        decision_count = len(self.decision_dates)
        instrument_count = len(self.instruments)
        expected_shape = (decision_count, instrument_count)

        if decision_count == 0 or instrument_count == 0:
            raise ValueError("PortfolioDataSplit must contain decisions and instruments.")
        if len(self.execution_dates) != decision_count or len(self.reward_end_dates) != decision_count:
            raise ValueError("All transition date indexes must have equal length.")

        arrays = {
            "scores": self.scores,
            "volatility": self.volatility,
            "observation_tradable": self.observation_tradable,
            "execution_tradable": self.execution_tradable,
            "execution_close": self.execution_close,
            "reward_end_close": self.reward_end_close,
            "asset_returns": self.asset_returns,
        }
        for name, values in arrays.items():
            if values.shape != expected_shape:
                raise ValueError(f"{name} has shape {values.shape}; expected {expected_shape}.")

        if self.observation_tradable.dtype != np.bool_ or self.execution_tradable.dtype != np.bool_:
            raise ValueError("Tradability arrays must have boolean dtype.")
        if not self.decision_dates.is_monotonic_increasing:
            raise ValueError("Decision dates must be ordered chronologically.")
        if not np.all(self.decision_dates < self.execution_dates):
            raise ValueError("Every execution date must be after its decision date.")
        if not np.all(self.execution_dates < self.reward_end_dates):
            raise ValueError("Every reward-end date must be after its execution date.")


def load_prediction_frame(
    path: Union[str, Path],
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
) -> pd.DataFrame:
    """Load and validate a trusted Qlib ``pred.pkl`` artifact.

    Pickle files can execute code while loading. Callers must use only trusted
    artifacts created by their own Qlib experiment.
    """

    prediction_path = Path(path).expanduser()
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Prediction artifact does not exist: {prediction_path}")

    predictions = pd.read_pickle(prediction_path)
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("Prediction artifact must contain a pandas DataFrame.")
    if list(predictions.columns) != ["score"]:
        raise ValueError("Prediction DataFrame must contain exactly one column named 'score'.")
    if not isinstance(predictions.index, pd.MultiIndex) or predictions.index.names != ["datetime", "instrument"]:
        raise ValueError("Prediction index must be a MultiIndex named ['datetime', 'instrument'].")
    if predictions.index.duplicated().any():
        raise ValueError("Prediction DataFrame contains duplicate date/instrument keys.")

    values = predictions["score"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Stored prediction scores must be finite.")

    date_range = DateRange(start, end)
    datetimes = pd.DatetimeIndex(predictions.index.get_level_values("datetime")).normalize()
    mask = (datetimes >= date_range.start) & (datetimes <= date_range.end)
    selected = predictions.loc[mask].copy()
    if selected.empty:
        raise ValueError(f"Prediction artifact contains no rows from {date_range.start} through {date_range.end}.")

    selected.index = pd.MultiIndex.from_arrays(
        [
            pd.DatetimeIndex(selected.index.get_level_values("datetime")).normalize(),
            selected.index.get_level_values("instrument").astype(str),
        ],
        names=["datetime", "instrument"],
    )
    return selected.sort_index()


def build_portfolio_data_split(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    calendar: Sequence[Union[str, pd.Timestamp]],
    date_range: DateRange,
    instruments: Sequence[str],
    volatility_window: int = 20,
) -> PortfolioDataSplit:
    """Align scores and market outcomes without exposing future data.

    ``market`` must have a ``(datetime, instrument)`` MultiIndex and columns
    ``close`` and ``volume``. It should include enough history before the split
    to calculate backward-looking volatility.
    """

    if volatility_window < 2:
        raise ValueError("volatility_window must be at least two trading sessions.")

    normalized_predictions = _validate_prediction_frame(predictions)
    normalized_market = _validate_market_frame(market)
    stable_instruments = tuple(sorted(str(instrument) for instrument in instruments))
    if not stable_instruments or len(set(stable_instruments)) != len(stable_instruments):
        raise ValueError("instruments must be a non-empty sequence of unique identifiers.")

    trading_calendar = pd.DatetimeIndex(calendar).normalize().drop_duplicates().sort_values()
    if trading_calendar.empty:
        raise ValueError("calendar must contain at least one trading date.")

    split_calendar = trading_calendar[
        (trading_calendar >= date_range.start) & (trading_calendar <= date_range.end)
    ]
    if len(split_calendar) < 4:
        raise ValueError("A split needs at least four trading dates for one complete transition.")

    decision_offsets = np.arange(0, len(split_calendar) - 3, 2, dtype=np.int64)
    decision_dates = split_calendar[decision_offsets]
    execution_dates = split_calendar[decision_offsets + 1]
    reward_end_dates = split_calendar[decision_offsets + 3]

    scores = _pivot(normalized_predictions, "score", decision_dates, stable_instruments)
    close = _pivot(normalized_market, "close", trading_calendar, stable_instruments)
    volume = _pivot(normalized_market, "volume", trading_calendar, stable_instruments)

    close_frame = pd.DataFrame(close, index=trading_calendar, columns=stable_instruments)
    returns = close_frame.pct_change(fill_method=None)
    volatility_frame = returns.rolling(window=volatility_window, min_periods=volatility_window).std(ddof=0)

    decision_positions = trading_calendar.get_indexer(decision_dates)
    execution_positions = trading_calendar.get_indexer(execution_dates)
    reward_end_positions = trading_calendar.get_indexer(reward_end_dates)

    decision_close = close[decision_positions]
    decision_volume = volume[decision_positions]
    execution_close = close[execution_positions]
    execution_volume = volume[execution_positions]
    reward_end_close = close[reward_end_positions]
    volatility = volatility_frame.to_numpy(dtype=np.float64)[decision_positions]

    observation_tradable = _tradable_mask(decision_close, decision_volume)
    execution_tradable = _tradable_mask(execution_close, execution_volume)
    valid_return = execution_tradable & np.isfinite(reward_end_close) & (reward_end_close > 0.0)
    asset_returns = np.divide(
        reward_end_close,
        execution_close,
        out=np.full_like(execution_close, np.nan),
        where=valid_return,
    )
    asset_returns = np.where(valid_return, asset_returns - 1.0, np.nan)

    return PortfolioDataSplit(
        instruments=stable_instruments,
        decision_dates=decision_dates,
        execution_dates=execution_dates,
        reward_end_dates=reward_end_dates,
        scores=scores,
        volatility=volatility,
        observation_tradable=observation_tradable,
        execution_tradable=execution_tradable,
        execution_close=execution_close,
        reward_end_close=reward_end_close,
        asset_returns=asset_returns,
    )


def load_qlib_calendar(
    provider_uri: Union[str, Path],
    region: str,
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
) -> pd.DatetimeIndex:
    """Load a daily Qlib calendar through a delayed optional dependency."""

    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(Path(provider_uri).expanduser()), region=region)
    return pd.DatetimeIndex(D.calendar(start_time=start, end_time=end, freq="day")).normalize()


def load_qlib_market_frame(
    provider_uri: Union[str, Path],
    region: str,
    instruments: Sequence[str],
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
) -> pd.DataFrame:
    """Load close and volume fields through a delayed Qlib adapter."""

    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(Path(provider_uri).expanduser()), region=region)
    fields = ["$close", "$volume"]
    market = D.features(list(instruments), fields, start_time=start, end_time=end, freq="day")
    market = market.rename(columns={"$close": "close", "$volume": "volume"})
    return market.reorder_levels(["datetime", "instrument"]).sort_index()


def _validate_prediction_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(predictions, pd.DataFrame) or "score" not in predictions.columns:
        raise ValueError("predictions must be a DataFrame containing a 'score' column.")
    return _normalize_multiindex(predictions[["score"]], "predictions")


def _validate_market_frame(market: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(market, pd.DataFrame) or not {"close", "volume"}.issubset(market.columns):
        raise ValueError("market must be a DataFrame containing 'close' and 'volume' columns.")
    return _normalize_multiindex(market[["close", "volume"]], "market")


def _normalize_multiindex(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(frame.index, pd.MultiIndex) or set(frame.index.names) != {"datetime", "instrument"}:
        raise ValueError(f"{name} index must have datetime and instrument levels.")
    if frame.index.duplicated().any():
        raise ValueError(f"{name} contains duplicate date/instrument keys.")

    normalized = frame.reorder_levels(["datetime", "instrument"]).copy()
    normalized.index = pd.MultiIndex.from_arrays(
        [
            pd.DatetimeIndex(normalized.index.get_level_values("datetime")).normalize(),
            normalized.index.get_level_values("instrument").astype(str),
        ],
        names=["datetime", "instrument"],
    )
    if normalized.index.duplicated().any():
        raise ValueError(f"{name} contains duplicate keys after date normalization.")
    return normalized.sort_index()


def _pivot(
    frame: pd.DataFrame,
    column: str,
    dates: pd.DatetimeIndex,
    instruments: Tuple[str, ...],
) -> np.ndarray:
    matrix = frame[column].unstack("instrument").reindex(index=dates, columns=instruments)
    return matrix.to_numpy(dtype=np.float64)


def _tradable_mask(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    return np.isfinite(close) & (close > 0.0) & np.isfinite(volume) & (volume > 0.0)
