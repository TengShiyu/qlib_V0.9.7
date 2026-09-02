# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
from unittest.mock import patch

import pandas as pd

from qlib.rl.portfolio.data import DateRange
from qlib.rl.portfolio.rolling import generate_portfolio_rolling_windows, rolling_windows_frame
from qlib.workflow.task.gen import RollingGen


class FakeRollingGen:
    ROLL_SD = RollingGen.ROLL_SD
    ROLL_EX = RollingGen.ROLL_EX
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    def generate(self, task):
        return [
            {
                "dataset": {
                    "kwargs": {
                        "segments": {
                            "train": (pd.Timestamp("2025-01-02"), pd.Timestamp("2025-10-31")),
                            "valid": (pd.Timestamp("2025-11-03"), pd.Timestamp("2025-12-29")),
                            "test": (pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-29")),
                        }
                    }
                }
            },
            {
                "dataset": {
                    "kwargs": {
                        "segments": {
                            "train": (pd.Timestamp("2025-01-17"), pd.Timestamp("2025-11-14")),
                            "valid": (pd.Timestamp("2025-11-17"), pd.Timestamp("2026-01-13")),
                            "test": (pd.Timestamp("2026-01-30"), None),
                        }
                    }
                }
            },
        ]


class PortfolioRollingWindowTest(unittest.TestCase):
    def setUp(self):
        self.segments = {
            "train": DateRange("2025-01-02", "2025-10-31"),
            "valid": DateRange("2025-11-01", "2025-12-31"),
            "test": DateRange("2026-01-02", "2026-02-19"),
        }

    @patch("qlib.rl.portfolio.rolling.RollingGen", FakeRollingGen)
    def test_delegates_to_ml_generator_splits_ratio_and_caps_last_window(self):
        windows = generate_portfolio_rolling_windows(
            self.segments,
            trading_interval=2,
            step=20,
            rolling_type="ROLL_SD",
            rl_split_ratio=[3, 1, 1],
            calendar_provider=lambda start_time, end_time, freq: pd.bdate_range(start_time, end_time),
        )

        self.assertEqual(FakeRollingGen.last_kwargs["step"], 20)
        self.assertEqual(FakeRollingGen.last_kwargs["rtype"], RollingGen.ROLL_SD)
        self.assertEqual(FakeRollingGen.last_kwargs["trunc_days"], 3)
        self.assertIsNone(FakeRollingGen.last_kwargs["ds_extra_mod_func"])
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].score_block, DateRange("2026-01-02", "2026-01-29"))
        self.assertEqual(windows[0].train, DateRange("2026-01-02", "2026-01-19"))
        self.assertEqual(windows[0].valid, DateRange("2026-01-20", "2026-01-23"))
        self.assertEqual(windows[0].test, DateRange("2026-01-26", "2026-01-29"))
        self.assertEqual(windows[1].score_block, DateRange("2026-01-30", "2026-02-19"))
        self.assertEqual(windows[1].train, DateRange("2026-01-30", "2026-02-09"))
        self.assertEqual(windows[1].valid, DateRange("2026-02-10", "2026-02-13"))
        self.assertEqual(windows[1].test, DateRange("2026-02-16", "2026-02-19"))

        frame = rolling_windows_frame(windows)
        self.assertEqual(frame["window_id"].tolist(), [0, 1])
        self.assertEqual(frame.loc[0, "score_block_end"], pd.Timestamp("2026-01-29"))
        self.assertEqual(frame.loc[0, "valid_end"], pd.Timestamp("2026-01-23"))

    def test_rejects_invalid_shared_rolling_settings(self):
        with self.assertRaisesRegex(ValueError, "trading_interval"):
            generate_portfolio_rolling_windows(
                self.segments, trading_interval=0, step=10, rolling_type="ROLL_SD"
            )
        with self.assertRaisesRegex(ValueError, "rolling_type"):
            generate_portfolio_rolling_windows(
                self.segments, trading_interval=2, step=10, rolling_type="OTHER"
            )
        with self.assertRaisesRegex(ValueError, "rl_split_ratio"):
            generate_portfolio_rolling_windows(
                self.segments,
                trading_interval=2,
                step=20,
                rolling_type="ROLL_SD",
                rl_split_ratio=[3, 0, 1],
            )


if __name__ == "__main__":
    unittest.main()
