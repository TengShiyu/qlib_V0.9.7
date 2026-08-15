# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import pandas as pd

from qlib.backtest.position import Position
from qlib.rl.portfolio.workflow import _market_vs_dqn, _positions_to_frame, _strategy_diagnostics


class PortfolioWorkflowExportTest(unittest.TestCase):
    def test_exports_daily_position_holding_sessions(self) -> None:
        position = Position(cash=500.0)
        position.position["AAPL"] = {
            "amount": 5.0,
            "price": 100.0,
            "weight": 0.5,
            "count": 139,
        }

        result = _positions_to_frame({pd.Timestamp("2026-07-30"): position})

        self.assertEqual(result.loc[0, "instrument"], "AAPL")
        self.assertEqual(result.loc[0, "market_value"], 500.0)
        self.assertEqual(result.loc[0, "holding_sessions"], 139)

    def test_exports_market_and_dqn_from_qlib_report(self) -> None:
        report = pd.DataFrame(
            {
                "account": [100_000.0, 110_000.0],
                "bench": [0.05, 0.02],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
        )

        comparison = _market_vs_dqn(report, 100_000.0).set_index("strategy")

        self.assertAlmostEqual(comparison.loc["dqn", "total_return"], 0.10)
        self.assertAlmostEqual(comparison.loc["market", "total_return"], 0.071)

    def test_diagnostics_exposes_longest_holding(self) -> None:
        actions = pd.DataFrame(
            {
                "action": ["HOLD", "HOLD", "CASH"],
                "executed_order_count": [0, 0, 2],
            }
        )
        holdings = pd.DataFrame({"holding_sessions": [3, 139]})

        diagnostics = _strategy_diagnostics(actions, holdings).set_index("metric")["value"]

        self.assertEqual(diagnostics["dominant_action"], "HOLD")
        self.assertEqual(diagnostics["maximum_holding_sessions"], 139)


if __name__ == "__main__":
    unittest.main()
