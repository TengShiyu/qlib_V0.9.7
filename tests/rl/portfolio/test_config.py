# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tempfile
import unittest
from pathlib import Path

from qlib.rl.portfolio.config import load_portfolio_workflow_config


class PortfolioConfigTest(unittest.TestCase):
    def test_loads_backtest_costs_from_workflow_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workflow.yaml"
            path.write_text(
                """
qlib_init:
    provider_uri: /data/qlib
    region: us
port_analysis_config:
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


if __name__ == "__main__":
    unittest.main()
