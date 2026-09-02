# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import numpy as np
import pandas as pd

from qlib.rl.portfolio import PortfolioAction
from qlib.rl.portfolio.data import PortfolioDataSplit
from qlib.rl.portfolio.simulator import PortfolioSimulator, PortfolioSimulatorConfig


def make_data(
    asset_returns: np.ndarray,
    execution_tradable: np.ndarray = None,
    scores: np.ndarray = None,
) -> PortfolioDataSplit:
    returns = np.asarray(asset_returns, dtype=np.float64)
    decision_count, instrument_count = returns.shape
    base_dates = pd.bdate_range("2025-01-02", periods=decision_count * 2 + 2)
    decisions = base_dates[np.arange(0, decision_count * 2, 2)]
    executions = base_dates[np.arange(1, decision_count * 2 + 1, 2)]
    reward_ends = base_dates[np.arange(3, decision_count * 2 + 3, 2)]
    tradable = np.ones_like(returns, dtype=bool) if execution_tradable is None else execution_tradable
    score_values = np.ones_like(returns) if scores is None else scores
    execution_close = np.ones_like(returns)
    reward_end_close = execution_close * (1.0 + np.where(np.isfinite(returns), returns, 0.0))

    return PortfolioDataSplit(
        instruments=tuple(f"S{index}" for index in range(instrument_count)),
        decision_dates=decisions,
        execution_dates=executions,
        reward_end_dates=reward_ends,
        scores=score_values,
        volatility=np.ones_like(returns),
        observation_tradable=np.ones_like(returns, dtype=bool),
        execution_tradable=np.asarray(tradable, dtype=bool),
        execution_close=execution_close,
        reward_end_close=reward_end_close,
        asset_returns=returns,
    )


class PortfolioSimulatorTest(unittest.TestCase):
    def test_rejects_invalid_turnover_penalty(self) -> None:
        with self.assertRaisesRegex(ValueError, "turnover_penalty"):
            PortfolioSimulatorConfig(turnover_penalty=-0.001)

    def test_reset_starts_all_cash(self) -> None:
        simulator = PortfolioSimulator(make_data(np.array([[0.1, 0.2]])))
        state = simulator.reset()

        np.testing.assert_array_equal(state.asset_weights, [0.0, 0.0])
        self.assertAlmostEqual(state.cash_weight, 1.0)
        self.assertAlmostEqual(state.portfolio_value, 100_000.0)
        self.assertFalse(state.done)

    def test_equal_weight_return_and_weight_drift(self) -> None:
        simulator = PortfolioSimulator(make_data(np.array([[0.10, 0.0]])))
        transition = simulator.step(PortfolioAction.EQUAL_WEIGHT)

        np.testing.assert_allclose(transition.target.asset_weights, [0.5, 0.5])
        self.assertAlmostEqual(transition.reward.gross_return, 0.05)
        self.assertAlmostEqual(transition.ending_value, 105_000.0)
        np.testing.assert_allclose(transition.ending_asset_weights, [1.1 / 2.1, 1.0 / 2.1])
        self.assertTrue(simulator.get_state().done)

    def test_cash_action_has_zero_market_return(self) -> None:
        simulator = PortfolioSimulator(make_data(np.array([[0.5, -0.5]])))
        transition = simulator.step(PortfolioAction.CASH)

        self.assertAlmostEqual(transition.reward.net_return, 0.0)
        self.assertAlmostEqual(transition.ending_value, 100_000.0)
        self.assertAlmostEqual(transition.ending_cash_weight, 1.0)

    def test_transaction_cost_is_charged_once(self) -> None:
        simulator = PortfolioSimulator(
            make_data(np.array([[0.0, 0.0]])),
            config=PortfolioSimulatorConfig(buy_cost=0.001),
        )
        transition = simulator.step(PortfolioAction.EQUAL_WEIGHT)

        self.assertAlmostEqual(transition.turnover.buy, 1.0)
        self.assertAlmostEqual(transition.reward.transaction_cost, 0.001)
        self.assertAlmostEqual(transition.ending_value, 99_900.0)
        self.assertAlmostEqual(transition.ending_cash_weight, 0.0)
        self.assertAlmostEqual(float(transition.ending_asset_weights.sum()), 1.0)

    def test_nonzero_cost_weights_remain_valid_for_the_next_decision(self) -> None:
        simulator = PortfolioSimulator(
            make_data(np.zeros((2, 2))),
            config=PortfolioSimulatorConfig(buy_cost=0.001, sell_cost=0.002),
        )

        first = simulator.step(PortfolioAction.EQUAL_WEIGHT)
        second = simulator.step(PortfolioAction.CASH)

        self.assertAlmostEqual(first.ending_value, 99_900.0)
        self.assertAlmostEqual(float(first.ending_asset_weights.sum()), 1.0)
        self.assertAlmostEqual(first.ending_cash_weight, 0.0)
        self.assertAlmostEqual(second.ending_value, 99_700.2)
        self.assertAlmostEqual(float(second.ending_asset_weights.sum()), 0.0)
        self.assertAlmostEqual(second.ending_cash_weight, 1.0)

    def test_turnover_penalty_changes_learning_reward_not_account_value(self) -> None:
        simulator = PortfolioSimulator(
            make_data(np.array([[0.0, 0.0]])),
            config=PortfolioSimulatorConfig(turnover_penalty=0.002),
        )
        transition = simulator.step(PortfolioAction.EQUAL_WEIGHT)

        self.assertAlmostEqual(transition.turnover.one_way, 1.0)
        self.assertAlmostEqual(transition.reward.turnover_penalty, 0.002)
        self.assertAlmostEqual(transition.reward.net_return, 0.0)
        self.assertAlmostEqual(transition.reward.learning_reward, -0.002)
        self.assertAlmostEqual(transition.ending_value, 100_000.0)

    def test_account_uses_real_net_return_not_learning_penalty(self) -> None:
        simulator = PortfolioSimulator(
            make_data(np.array([[0.10]])),
            config=PortfolioSimulatorConfig(
                buy_cost=0.001,
                turnover_penalty=0.002,
            ),
        )
        transition = simulator.step(PortfolioAction.EQUAL_WEIGHT)

        self.assertAlmostEqual(transition.reward.gross_return, 0.10)
        self.assertAlmostEqual(transition.reward.transaction_cost, 0.001)
        self.assertAlmostEqual(transition.reward.net_return, 0.099)
        self.assertAlmostEqual(transition.reward.turnover_penalty, 0.002)
        self.assertAlmostEqual(transition.reward.learning_reward, 0.097)
        self.assertAlmostEqual(transition.ending_value, 109_900.0)

    def test_sell_cost_is_charged_once_when_liquidating(self) -> None:
        simulator = PortfolioSimulator(
            make_data(np.zeros((2, 2))),
            config=PortfolioSimulatorConfig(sell_cost=0.002),
        )
        simulator.step(PortfolioAction.EQUAL_WEIGHT)
        transition = simulator.step(PortfolioAction.CASH)

        self.assertAlmostEqual(transition.turnover.sell, 1.0)
        self.assertAlmostEqual(transition.reward.transaction_cost, 0.002)
        self.assertAlmostEqual(transition.ending_value, 99_800.0)

    def test_non_tradable_current_position_remains_locked(self) -> None:
        data = make_data(
            np.array([[0.0, 0.0], [0.0, 0.0]]),
            execution_tradable=np.array([[True, True], [False, True]]),
        )
        simulator = PortfolioSimulator(data)
        simulator.step(PortfolioAction.EQUAL_WEIGHT)
        transition = simulator.step(PortfolioAction.CASH)

        self.assertAlmostEqual(transition.target.asset_weights[0], 0.5)
        self.assertAlmostEqual(transition.target.asset_weights[1], 0.0)
        self.assertAlmostEqual(transition.target.cash_weight, 0.5)

    def test_missing_return_is_carried_at_zero_and_reported(self) -> None:
        simulator = PortfolioSimulator(make_data(np.array([[np.nan, 0.10]])))
        transition = simulator.step(PortfolioAction.EQUAL_WEIGHT)

        self.assertAlmostEqual(transition.reward.missing_return_weight, 0.5)
        self.assertAlmostEqual(transition.reward.gross_return, 0.05)

    def test_steps_form_a_contiguous_execution_chain(self) -> None:
        simulator = PortfolioSimulator(make_data(np.zeros((2, 2))))
        first = simulator.step(PortfolioAction.HOLD)
        second = simulator.step(PortfolioAction.HOLD)

        self.assertEqual(first.reward_end_date, second.execution_date)
        self.assertTrue(simulator.get_state().done)

    def test_completed_episode_rejects_another_step(self) -> None:
        simulator = PortfolioSimulator(make_data(np.zeros((1, 1))))
        simulator.step(PortfolioAction.HOLD)

        with self.assertRaisesRegex(RuntimeError, "completed"):
            simulator.step(PortfolioAction.HOLD)


if __name__ == "__main__":
    unittest.main()
