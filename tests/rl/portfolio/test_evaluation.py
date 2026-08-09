# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import numpy as np
import pandas as pd

from qlib.rl.portfolio import PortfolioSimulatorConfig
from qlib.rl.portfolio.benchmark import run_required_benchmarks
from qlib.rl.portfolio.evaluation import cost_sensitivity, regime_diagnostics, result_diagnostics
from tests.rl.portfolio.test_integration import make_data


class PortfolioEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data = make_data(np.array([[0.01, 0.02], [-0.02, 0.01], [0.03, -0.01]]))
        dates = self.data.execution_dates.union(self.data.reward_end_dates)
        self.market_close = pd.Series([100.0, 102.0, 101.0, 103.0], index=dates)

    def test_result_diagnostics_reports_action_and_exposure(self) -> None:
        result = run_required_benchmarks(self.data)["equal_weight"]
        diagnostics = result_diagnostics(result)

        self.assertEqual(diagnostics["unique_action_count"], 1.0)
        self.assertEqual(diagnostics["dominant_action_frequency"], 1.0)
        self.assertEqual(diagnostics["action_persistence"], 1.0)
        self.assertGreaterEqual(diagnostics["average_max_asset_weight"], 0.0)

    def test_regime_diagnostics_assigns_every_transition(self) -> None:
        results = run_required_benchmarks(self.data, market_close=self.market_close)
        regimes = regime_diagnostics({"equal_weight": results["equal_weight"]}, results["market"])

        self.assertEqual(regimes["transition_count"].sum(), len(self.data.decision_dates))
        self.assertEqual(set(regimes["regime"]), {"market_up", "market_down_or_flat"})

    def test_cost_sensitivity_reports_return_difference(self) -> None:
        reproduction = run_required_benchmarks(self.data)
        sensitivity = run_required_benchmarks(
            self.data,
            simulator_config=PortfolioSimulatorConfig(buy_cost=0.001),
        )
        comparison = cost_sensitivity(reproduction, sensitivity).set_index("strategy")

        self.assertLess(comparison.loc["equal_weight", "return_difference"], 0.0)
        self.assertEqual(comparison.loc["cash", "return_difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
