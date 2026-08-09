# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import numpy as np

from qlib.rl.portfolio.reward import calculate_portfolio_reward, calculate_turnover


class PortfolioRewardTest(unittest.TestCase):
    def test_turnover_separates_buys_and_sells(self) -> None:
        turnover = calculate_turnover(np.array([0.4, 0.1, 0.0]), np.array([0.2, 0.3, 0.1]))

        self.assertAlmostEqual(turnover.buy, 0.3)
        self.assertAlmostEqual(turnover.sell, 0.2)
        self.assertAlmostEqual(turnover.total, 0.5)
        self.assertAlmostEqual(turnover.one_way, 0.3)

    def test_reward_is_weighted_return_minus_cost_once(self) -> None:
        turnover = calculate_turnover(np.zeros(2), np.array([0.6, 0.2]))
        reward = calculate_portfolio_reward(
            target_asset_weights=np.array([0.6, 0.2]),
            target_cash_weight=0.2,
            asset_returns=np.array([0.10, -0.05]),
            turnover=turnover,
            buy_cost=0.001,
        )

        self.assertAlmostEqual(reward.gross_return, 0.05)
        self.assertAlmostEqual(reward.transaction_cost, 0.0008)
        self.assertAlmostEqual(reward.net_return, 0.0492)

    def test_missing_return_uses_configured_value_and_reports_weight(self) -> None:
        turnover = calculate_turnover(np.zeros(2), np.array([0.4, 0.4]))
        reward = calculate_portfolio_reward(
            target_asset_weights=np.array([0.4, 0.4]),
            target_cash_weight=0.2,
            asset_returns=np.array([0.10, np.nan]),
            turnover=turnover,
        )

        self.assertAlmostEqual(reward.gross_return, 0.04)
        self.assertAlmostEqual(reward.missing_return_weight, 0.4)
        np.testing.assert_allclose(reward.effective_asset_returns, [0.1, 0.0])

    def test_return_below_total_loss_is_rejected(self) -> None:
        turnover = calculate_turnover(np.zeros(1), np.ones(1))

        with self.assertRaisesRegex(ValueError, "less than -1"):
            calculate_portfolio_reward(np.ones(1), 0.0, np.array([-1.1]), turnover)


if __name__ == "__main__":
    unittest.main()
