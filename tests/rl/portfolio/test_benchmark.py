# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import numpy as np
import pandas as pd

from qlib.rl.portfolio import PortfolioSimulatorConfig
from qlib.rl.portfolio.benchmark import calculate_metrics, results_frame, run_required_benchmarks
from tests.rl.portfolio.test_integration import make_data


class PortfolioBenchmarkTest(unittest.TestCase):
    def test_required_benchmarks_are_reproducible(self) -> None:
        data = make_data(np.array([[0.01, 0.02], [-0.01, 0.03], [0.02, 0.01]]))
        first = run_required_benchmarks(data, random_seed=11)
        second = run_required_benchmarks(data, random_seed=11)

        self.assertEqual(
            set(first),
            {"cash", "hold", "equal_weight", "standard_score", "volatility_adjusted", "random"},
        )
        pd.testing.assert_frame_equal(results_frame(first), results_frame(second))
        self.assertEqual(first["random"].action_counts, second["random"].action_counts)
        self.assertEqual(first["cash"].metrics["total_return"], 0.0)
        self.assertEqual(first["cash"].metrics["average_cash_exposure"], 1.0)
        self.assertEqual(first["hold"].metrics["total_return"], 0.0)

    def test_transaction_cost_sensitivity_reduces_value(self) -> None:
        data = make_data(np.zeros((2, 2)))
        zero_cost = run_required_benchmarks(data)["equal_weight"]
        nonzero_cost = run_required_benchmarks(
            data,
            simulator_config=PortfolioSimulatorConfig(buy_cost=0.0005, sell_cost=0.0015),
        )["equal_weight"]

        self.assertEqual(zero_cost.metrics["total_return"], 0.0)
        self.assertLess(nonzero_cost.metrics["total_return"], zero_cost.metrics["total_return"])
        self.assertGreater(nonzero_cost.metrics["transaction_cost"], 0.0)

    def test_market_benchmark_uses_matching_transition_boundaries(self) -> None:
        data = make_data(np.zeros((2, 2)))
        dates = data.execution_dates.union(data.reward_end_dates)
        market_close = pd.Series(np.arange(100.0, 100.0 + len(dates)), index=dates)
        result = run_required_benchmarks(data, market_close=market_close)["market"]
        expected = (
            market_close.reindex(data.reward_end_dates).to_numpy()
            / market_close.reindex(data.execution_dates).to_numpy()
            - 1.0
        )

        np.testing.assert_allclose(result.transitions["net_return"], expected)
        np.testing.assert_allclose(result.transitions["portfolio_net_return"], expected)
        np.testing.assert_allclose(result.transitions["learning_reward"], expected)
        np.testing.assert_allclose(result.transitions["turnover_penalty"], 0.0)
        self.assertEqual(result.metrics["average_cash_exposure"], 0.0)

    def test_metrics_include_initial_value_in_drawdown(self) -> None:
        metrics = calculate_metrics(
            returns=np.array([-0.1, 0.1]),
            portfolio_values=np.array([90.0, 99.0]),
            initial_value=100.0,
            turnover=np.zeros(2),
            transaction_cost=np.zeros(2),
            cash_exposure=np.zeros(2),
            missing_return_weight=np.zeros(2),
        )
        self.assertAlmostEqual(metrics["total_return"], -0.01)
        self.assertAlmostEqual(metrics["max_drawdown"], -0.1)


if __name__ == "__main__":
    unittest.main()
