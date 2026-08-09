# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from qlib.rl.portfolio.data import DateRange, build_portfolio_data_split, load_prediction_frame


class PortfolioDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.calendar = pd.bdate_range("2025-01-01", periods=12)
        self.instruments = ["BBB", "AAA"]
        index = pd.MultiIndex.from_product(
            [self.calendar, ["AAA", "BBB"]], names=["datetime", "instrument"]
        )
        self.predictions = pd.DataFrame({"score": np.arange(len(index), dtype=float)}, index=index)

        market_index = pd.MultiIndex.from_product(
            [self.calendar, ["AAA", "BBB"]], names=["datetime", "instrument"]
        )
        close = np.column_stack(
            [
                np.arange(100.0, 112.0),
                np.arange(200.0, 224.0, 2.0),
            ]
        ).reshape(-1)
        self.market = pd.DataFrame(
            {"close": close, "volume": np.full(len(market_index), 1000.0)},
            index=market_index,
        )

    def build(self, start: str = "2025-01-01", end: str = "2025-01-14", volatility_window: int = 2):
        return build_portfolio_data_split(
            predictions=self.predictions,
            market=self.market,
            calendar=self.calendar,
            date_range=DateRange(start, end),
            instruments=self.instruments,
            volatility_window=volatility_window,
        )

    def test_instruments_are_sorted_and_transition_dates_are_aligned(self) -> None:
        data = self.build()

        self.assertEqual(data.instruments, ("AAA", "BBB"))
        self.assertEqual(data.decision_dates.tolist(), self.calendar[[0, 2, 4, 6]].tolist())
        self.assertEqual(data.execution_dates.tolist(), self.calendar[[1, 3, 5, 7]].tolist())
        self.assertEqual(data.reward_end_dates.tolist(), self.calendar[[3, 5, 7, 9]].tolist())

    def test_returns_use_execution_close_through_reward_end_close(self) -> None:
        data = self.build()

        expected_aaa = self.market.loc[(self.calendar[3], "AAA"), "close"]
        expected_aaa /= self.market.loc[(self.calendar[1], "AAA"), "close"]
        self.assertAlmostEqual(data.asset_returns[0, 0], expected_aaa - 1.0)

    def test_missing_execution_data_makes_asset_non_tradable(self) -> None:
        execution_key = (self.calendar[1], "AAA")
        self.market.loc[execution_key, "volume"] = 0.0
        data = self.build()

        self.assertFalse(data.execution_tradable[0, 0])
        self.assertTrue(np.isnan(data.asset_returns[0, 0]))
        self.assertTrue(data.execution_tradable[0, 1])

    def test_missing_score_is_preserved_as_nan_on_fixed_axis(self) -> None:
        self.predictions = self.predictions.drop(index=(self.calendar[0], "BBB"))
        data = self.build()

        self.assertTrue(np.isnan(data.scores[0, 1]))
        self.assertTrue(np.isfinite(data.scores[0, 0]))

    def test_volatility_uses_no_prices_after_decision_date(self) -> None:
        original = self.build(volatility_window=2)
        original_first_volatility = original.volatility[1].copy()

        future_dates = self.calendar[self.calendar > original.decision_dates[1]]
        self.market.loc[(future_dates, slice(None)), "close"] *= 100.0
        changed = self.build(volatility_window=2)

        np.testing.assert_allclose(changed.volatility[1], original_first_volatility)

    def test_transition_does_not_cross_split_end(self) -> None:
        data = self.build(start="2025-01-01", end="2025-01-08")

        self.assertTrue(np.all(data.reward_end_dates <= pd.Timestamp("2025-01-08")))
        self.assertEqual(len(data.decision_dates), 2)

    def test_short_split_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least four trading dates"):
            self.build(start="2025-01-01", end="2025-01-03")

    def test_duplicate_market_key_is_rejected(self) -> None:
        self.market = pd.concat([self.market, self.market.iloc[[0]]])

        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.build()

    def test_load_prediction_frame_filters_and_sorts_trusted_artifact(self) -> None:
        shuffled = self.predictions.sample(frac=1.0, random_state=7)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "pred.pkl"
            shuffled.to_pickle(path)
            loaded = load_prediction_frame(path, "2025-01-03", "2025-01-06")

        dates = loaded.index.get_level_values("datetime")
        self.assertEqual(dates.min(), pd.Timestamp("2025-01-03"))
        self.assertEqual(dates.max(), pd.Timestamp("2025-01-06"))
        self.assertTrue(loaded.index.is_monotonic_increasing)

    def test_load_prediction_frame_rejects_non_finite_stored_score(self) -> None:
        invalid = self.predictions.copy()
        invalid.iloc[0, 0] = np.nan
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "pred.pkl"
            invalid.to_pickle(path)
            with self.assertRaisesRegex(ValueError, "finite"):
                load_prediction_frame(path, "2025-01-01", "2025-01-31")


if __name__ == "__main__":
    unittest.main()
