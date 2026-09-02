# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tempfile
import unittest
from pathlib import Path

from qlib.rl.portfolio.config import (
    load_portfolio_workflow_config,
    portfolio_artifact_root,
)


class PortfolioConfigTest(unittest.TestCase):
    def test_artifact_root_is_derived_from_experiment_config_directory(self) -> None:
        config = Path("/experiments/new_signal_experiment/configs/workflow.yaml")
        self.assertEqual(
            portfolio_artifact_root(config),
            Path("/experiments/new_signal_experiment/portfolio_dqn_rl"),
        )

    def test_artifact_root_rejects_config_outside_configs_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "configs directory"):
            portfolio_artifact_root("/experiments/new_signal_experiment/workflow.yaml")

    def test_loads_backtest_costs_from_workflow_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workflow.yaml"
            path.write_text(
                """
qlib_init:
    provider_uri: /data/qlib
    region: us
task:
    rolling:
        trading_interval: 2
        step: 10
        rolling_type: ROLL_SD
        rl_split_ratio: [3, 1, 1]
    dataset:
        kwargs:
            segments:
                train: [2025-01-02, 2025-09-30]
                valid: [2025-10-01, 2025-12-31]
                test: [2026-01-02, 2026-03-30]
port_analysis_config:
    strategy:
        class: PortfolioDQNStrategy
        kwargs:
            max_holdings: 12
            increase_exposure_ratio: 0.30
            concentrated_holdings: 4
            concentrated_equity: 0.90
            rotation_count: 3
            defensive_equity: 0.40
            turnover_penalty: 0.001
    backtest:
        account: 250000
        benchmark: ^NDX
        exchange_kwargs:
            open_cost: 0.0005
            close_cost: 0.0015
            min_cost: 5
""".strip(),
                encoding="utf-8",
            )
            config = load_portfolio_workflow_config(path)

        self.assertEqual(config.provider_uri, "/data/qlib")
        self.assertEqual(config.region, "us")
        self.assertEqual(config.benchmark, "^NDX")
        self.assertEqual(config.simulator.initial_value, 250_000.0)
        self.assertEqual(config.simulator.buy_cost, 0.0005)
        self.assertEqual(config.simulator.sell_cost, 0.0015)
        self.assertEqual(config.simulator.min_cost, 5.0)
        self.assertEqual(config.simulator.turnover_penalty, 0.001)
        self.assertEqual(config.action.max_holdings, 12)
        self.assertEqual(config.action.increase_exposure_ratio, 0.30)
        self.assertEqual(config.action.concentrated_holdings, 4)
        self.assertEqual(config.action.concentrated_equity, 0.90)
        self.assertEqual(config.action.rotation_count, 3)
        self.assertEqual(config.action.defensive_equity, 0.40)
        self.assertEqual(config.trading_interval, 2)
        self.assertEqual(config.rolling_step, 10)
        self.assertEqual(config.rolling_type, "ROLL_SD")
        self.assertEqual(config.rl_split_ratio, (3, 1, 1))
        self.assertEqual(str(config.rolling_segments["train"].start.date()), "2025-01-02")
        self.assertEqual(str(config.rolling_segments["test"].end.date()), "2026-03-30")


if __name__ == "__main__":
    unittest.main()
