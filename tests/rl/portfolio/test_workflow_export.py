# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import tempfile
from pathlib import Path

import pandas as pd

from qlib.backtest.position import Position
from qlib.rl.portfolio.workflow import (
    _market_vs_dqn,
    _parse_rolling_config,
    _promote_checkpoint,
    _positions_to_frame,
    _rolling_phase_by_date,
    _strategy_diagnostics,
)


class PortfolioWorkflowExportTest(unittest.TestCase):
    def test_promotes_only_a_completed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "rolling" / "window_003" / "best_policy.pt"
            destination = root / "training" / "best_policy.pt"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"window-3")

            _promote_checkpoint(source, destination)

            self.assertEqual(destination.read_bytes(), b"window-3")
            self.assertFalse(destination.with_suffix(".pt.new").exists())
            with self.assertRaises(FileNotFoundError):
                _promote_checkpoint(root / "missing.pt", destination)
            self.assertEqual(destination.read_bytes(), b"window-3")

    def test_shared_rolling_configuration_is_required_and_validated(self) -> None:
        self.assertEqual(
            _parse_rolling_config(
                {
                    "trading_interval": 2,
                    "step": 10,
                    "rolling_type": "ROLL_SD",
                    "rl_split_ratio": [3, 1, 1],
                }
            ),
            {
                "trading_interval": 2,
                "step": 10,
                "rolling_type": "ROLL_SD",
                "rl_split_ratio": [3, 1, 1],
            },
        )
        with self.assertRaisesRegex(ValueError, "requires the shared"):
            _parse_rolling_config(None)
        with self.assertRaisesRegex(ValueError, "trading_interval must be positive"):
            _parse_rolling_config(
                {
                    "trading_interval": 0,
                    "step": 10,
                    "rolling_type": "ROLL_SD",
                    "rl_split_ratio": [3, 1, 1],
                }
            )

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
                "return": [0.0, 0.10],
                "bench": [0.05, 0.02],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
        )

        comparison = _market_vs_dqn(report, 100_000.0).set_index("strategy")

        self.assertAlmostEqual(comparison.loc["dqn", "total_return"], 0.10)
        self.assertAlmostEqual(comparison.loc["market", "total_return"], 0.071)
        self.assertAlmostEqual(comparison.loc["dqn", "dqn_test_session_return"], 0.10)

    def test_labels_train_validation_and_active_test_sessions(self) -> None:
        dates = pd.date_range("2025-01-02", periods=6, freq="D")
        windows = pd.DataFrame(
            [
                {
                    "train_start": "2025-01-02",
                    "train_end": "2025-01-03",
                    "valid_start": "2025-01-04",
                    "valid_end": "2025-01-05",
                    "test_start": "2025-01-06",
                    "test_end": "2025-01-07",
                }
            ]
        )

        phases = _rolling_phase_by_date(dates, windows)

        self.assertEqual(
            phases.tolist(),
            ["rl_train", "rl_train", "rl_valid", "rl_valid", "dqn_test", "dqn_test"],
        )

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
