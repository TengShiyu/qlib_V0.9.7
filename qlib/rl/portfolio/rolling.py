# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate portfolio-DQN windows with Qlib's ML rolling semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import pandas as pd

from qlib.data import D
from qlib.workflow.task.gen import RollingGen

from .data import DateRange


@dataclass(frozen=True)
class PortfolioRollingWindow:
    """One chronological DQN train, validation, and execution window."""

    window_id: int
    score_block: DateRange
    train: DateRange
    valid: DateRange
    test: DateRange

    def as_record(self) -> dict[str, object]:
        return {
            "window_id": self.window_id,
            "score_block_start": self.score_block.start,
            "score_block_end": self.score_block.end,
            "train_start": self.train.start,
            "train_end": self.train.end,
            "valid_start": self.valid.start,
            "valid_end": self.valid.end,
            "test_start": self.test.start,
            "test_end": self.test.end,
        }


def generate_portfolio_rolling_windows(
    segments: Mapping[str, DateRange],
    *,
    trading_interval: int,
    step: int,
    rolling_type: str,
    rl_split_ratio: tuple[int, int, int] | list[int] = (3, 1, 1),
    calendar_provider: Callable[..., object] | None = None,
) -> list[PortfolioRollingWindow]:
    """Apply the same ``RollingGen`` rules used by the upstream ML runner.

    Qlib first generates the exact upstream ML windows. Each ML test-score
    block is then divided chronologically into DQN train, validation, and test
    segments using ``rl_split_ratio``. The last score block is capped at the
    configured test end when Qlib's future calendar is unavailable.
    """

    if set(segments) != {"train", "valid", "test"}:
        raise ValueError("RL segments must define train, valid, and test.")
    if trading_interval <= 0:
        raise ValueError("trading_interval must be positive.")
    if step <= 0:
        raise ValueError("step must be positive.")
    normalized_type = str(rolling_type).upper()
    type_map = {"ROLL_SD": RollingGen.ROLL_SD, "ROLL_EX": RollingGen.ROLL_EX}
    if normalized_type not in type_map:
        raise ValueError("rolling_type must be ROLL_SD or ROLL_EX.")
    split_ratio = tuple(int(value) for value in rl_split_ratio)
    if len(split_ratio) != 3 or any(value <= 0 for value in split_ratio):
        raise ValueError("rl_split_ratio must contain three positive integers.")
    get_calendar = D.calendar if calendar_provider is None else calendar_provider

    task = {
        "dataset": {
            "kwargs": {
                "segments": {
                    name: (date_range.start, date_range.end)
                    for name, date_range in segments.items()
                }
            }
        }
    }
    generator = RollingGen(
        step=step,
        rtype=type_map[normalized_type],
        trunc_days=trading_interval + 1,
        ds_extra_mod_func=None,
    )
    generated = generator.generate(task)
    configured_test_end = segments["test"].end
    windows = []
    for window_id, generated_task in enumerate(generated):
        values = generated_task["dataset"]["kwargs"]["segments"]
        test_start, generated_test_end = values["test"]
        if test_start is None or pd.Timestamp(test_start) > configured_test_end:
            continue
        test_end = configured_test_end if generated_test_end is None else min(
            pd.Timestamp(generated_test_end), configured_test_end
        )
        score_calendar = pd.DatetimeIndex(
            get_calendar(start_time=test_start, end_time=test_end, freq="day")
        ).normalize()
        split_sizes = _allocate_split_sizes(
            len(score_calendar), split_ratio, minimum_size=trading_interval + 2
        )
        train_end = split_sizes[0]
        valid_end = train_end + split_sizes[1]
        windows.append(
            PortfolioRollingWindow(
                window_id=window_id,
                score_block=DateRange(test_start, test_end),
                train=DateRange(score_calendar[0], score_calendar[train_end - 1]),
                valid=DateRange(score_calendar[train_end], score_calendar[valid_end - 1]),
                test=DateRange(score_calendar[valid_end], score_calendar[-1]),
            )
        )
    if not windows:
        raise ValueError("The configured RL segments produced no rolling windows.")
    return windows


def _allocate_split_sizes(
    session_count: int,
    split_ratio: tuple[int, int, int],
    *,
    minimum_size: int,
) -> tuple[int, int, int]:
    """Allocate proportional chronological splits while preserving feasibility."""

    if session_count < minimum_size * 3:
        raise ValueError(
            f"A score block needs at least {minimum_size * 3} sessions for three complete "
            f"RL splits; found {session_count}."
        )
    ratio_total = sum(split_ratio)
    exact = [session_count * value / ratio_total for value in split_ratio]
    sizes = [int(value) for value in exact]
    for index in sorted(range(3), key=lambda idx: exact[idx] - sizes[idx], reverse=True):
        if sum(sizes) == session_count:
            break
        sizes[index] += 1

    for receiver in range(3):
        while sizes[receiver] < minimum_size:
            donors = [index for index in range(3) if sizes[index] > minimum_size]
            if not donors:  # pragma: no cover - guarded by total-size validation
                raise ValueError("Unable to construct feasible RL splits.")
            donor = max(donors, key=lambda index: sizes[index] - minimum_size)
            sizes[donor] -= 1
            sizes[receiver] += 1
    return tuple(sizes)


def rolling_windows_frame(windows: list[PortfolioRollingWindow]) -> pd.DataFrame:
    """Return an auditable chronological table for exported artifacts."""

    if not windows:
        raise ValueError("windows must not be empty.")
    return pd.DataFrame(window.as_record() for window in windows)
