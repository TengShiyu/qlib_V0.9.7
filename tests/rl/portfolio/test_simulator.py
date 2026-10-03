# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from qlib.rl.portfolio import PortfolioAction
from qlib.rl.portfolio.data import PortfolioDataSplit
from qlib.rl.portfolio.simulator import PortfolioSimulator, PortfolioSimulatorConfig
from qlib.rl.portfolio.interpreter import PortfolioStateInterpreter


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
    reward_ends = base_dates[np.arange(2, decision_count * 2 + 2, 2)]
    tradable = np.ones_like(returns, dtype=bool) if execution_tradable is None else execution_tradable
    score_values = np.ones_like(returns) if scores is None else scores
    decision_close = np.vstack([np.ones(instrument_count), np.cumprod(1.0 + np.nan_to_num(returns), axis=0)[:-1]])
    execution_close = decision_close.copy()
    reward_end_close = execution_close * (1.0 + returns)

    return PortfolioDataSplit(
        instruments=tuple(f"S{index}" for index in range(instrument_count)),
        decision_dates=decisions,
        execution_dates=executions,
        reward_end_dates=reward_ends,
        scores=score_values,
        volatility=np.ones_like(returns),
        observation_tradable=np.ones_like(returns, dtype=bool),
        execution_tradable=np.asarray(tradable, dtype=bool),
        decision_close=decision_close,
        execution_close=execution_close,
        reward_end_close=reward_end_close,
        asset_returns=returns,
    )


class PortfolioSimulatorTest(unittest.TestCase):
    def test_rejects_invalid_turnover_penalty(self) -> None:
        with self.assertRaisesRegex(ValueError, "turnover_penalty"):
            PortfolioSimulatorConfig(turnover_penalty=-0.001)

    def test_missing_prices_use_carry_forward_not_a_synthetic_future_return(self):
        with self.assertRaisesRegex(ValueError, "missing_return_value must be 0"):
            PortfolioSimulator(make_data(np.zeros((1, 1))), config=PortfolioSimulatorConfig(missing_return_value=0.01))

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

        invested = 100_000.0 / 1.001
        self.assertAlmostEqual(transition.turnover.buy, invested / 100_000.0)
        self.assertAlmostEqual(transition.reward.transaction_cost, invested * 0.001 / 100_000.0)
        self.assertAlmostEqual(transition.ending_value, invested)
        self.assertAlmostEqual(transition.ending_cash_weight, 0.0)
        self.assertAlmostEqual(float(transition.ending_asset_weights.sum()), 1.0)

    def test_nonzero_cost_weights_remain_valid_for_the_next_decision(self) -> None:
        simulator = PortfolioSimulator(
            make_data(np.zeros((2, 2))),
            config=PortfolioSimulatorConfig(buy_cost=0.001, sell_cost=0.002),
        )

        first = simulator.step(PortfolioAction.EQUAL_WEIGHT)
        second = simulator.step(PortfolioAction.CASH)

        invested = 100_000.0 / 1.002
        remaining_cash = 100_000.0 - invested * 1.001
        self.assertAlmostEqual(first.ending_value, invested + remaining_cash)
        self.assertAlmostEqual(float(first.ending_asset_weights.sum()), invested / first.ending_value)
        self.assertAlmostEqual(first.ending_cash_weight, remaining_cash / first.ending_value)
        self.assertAlmostEqual(second.ending_value, 100_000.0 - invested * 0.003)
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

        invested_fraction = 1.0 / 1.001
        self.assertAlmostEqual(transition.reward.gross_return, 0.10 * invested_fraction)
        self.assertAlmostEqual(transition.reward.transaction_cost, 0.001 * invested_fraction)
        self.assertAlmostEqual(transition.reward.net_return, 0.099 * invested_fraction)
        self.assertAlmostEqual(transition.reward.turnover_penalty, 0.002 * invested_fraction)
        self.assertAlmostEqual(transition.reward.learning_reward, 0.097 * invested_fraction)
        self.assertAlmostEqual(transition.ending_value, 100_000.0 * 1.1 / 1.001)

    def test_sell_cost_is_charged_once_when_liquidating(self) -> None:
        simulator = PortfolioSimulator(
            make_data(np.zeros((2, 2))),
            config=PortfolioSimulatorConfig(sell_cost=0.002),
        )
        simulator.step(PortfolioAction.EQUAL_WEIGHT)
        transition = simulator.step(PortfolioAction.CASH)

        self.assertAlmostEqual(transition.turnover.sell, 1.0 / 1.002)
        self.assertAlmostEqual(transition.reward.transaction_cost, 0.002 / 1.002)
        self.assertAlmostEqual(transition.ending_value, 100_000.0 * (1.0 - 0.002 / 1.002))

    def test_non_tradable_current_position_remains_locked(self) -> None:
        data = make_data(
            np.array([[0.0, 0.0], [0.0, 0.0]]),
            execution_tradable=np.array([[True, True], [False, True]]),
        )
        simulator = PortfolioSimulator(data)
        simulator.step(PortfolioAction.EQUAL_WEIGHT)
        transition = simulator.step(PortfolioAction.CASH)

        # CASH was selected without foreknowledge of the execution suspension.
        self.assertAlmostEqual(transition.target.asset_weights[0], 0.0)
        self.assertAlmostEqual(transition.target.asset_weights[1], 0.0)
        self.assertAlmostEqual(transition.target.cash_weight, 1.0)
        np.testing.assert_allclose(transition.ending_asset_weights, [0.5, 0.0])
        self.assertAlmostEqual(transition.ending_cash_weight, 0.5)

    def test_missing_return_is_carried_at_zero_and_reported(self) -> None:
        simulator = PortfolioSimulator(make_data(np.array([[np.nan, 0.10]])))
        transition = simulator.step(PortfolioAction.EQUAL_WEIGHT)

        self.assertAlmostEqual(transition.reward.missing_return_weight, 0.5)
        self.assertAlmostEqual(transition.reward.gross_return, 0.05)

    def test_steps_form_a_contiguous_decision_chain(self) -> None:
        simulator = PortfolioSimulator(make_data(np.zeros((2, 2))))
        first = simulator.step(PortfolioAction.HOLD)
        second = simulator.step(PortfolioAction.HOLD)

        self.assertEqual(first.reward_end_date, second.decision_date)
        self.assertTrue(simulator.get_state().done)

    def test_future_execution_data_cannot_change_observation_target_or_order(self):
        data = make_data(np.zeros((2, 2)))
        changed = replace(
            data,
            execution_close=data.execution_close * 50.0,
            execution_tradable=np.zeros_like(data.execution_tradable),
            reward_end_close=data.reward_end_close * 100.0,
        )
        left, right = PortfolioSimulator(data), PortfolioSimulator(changed)
        interpreter = PortfolioStateInterpreter()
        np.testing.assert_array_equal(interpreter.interpret(left.get_state()), interpreter.interpret(right.get_state()))
        a, b = left.step(PortfolioAction.EQUAL_WEIGHT), right.step(PortfolioAction.EQUAL_WEIGHT)
        np.testing.assert_array_equal(a.target.asset_weights, b.target.asset_weights)
        np.testing.assert_array_equal(a.requested_amounts, b.requested_amounts)
        self.assertGreater(a.filled_amounts.sum(), 0.0)
        self.assertEqual(b.filled_amounts.sum(), 0.0)

    def test_next_observation_does_not_see_next_execution_or_forward_return(self):
        data = make_data(np.array([[0.1, 0.0], [0.0, 0.0]]))
        execution = data.execution_close.copy()
        execution[1] *= 10.0
        end = data.reward_end_close.copy()
        end[1] *= 20.0
        tradable = data.execution_tradable.copy()
        tradable[1] = False
        changed = replace(data, execution_close=execution, reward_end_close=end, execution_tradable=tradable)
        left, right = PortfolioSimulator(data), PortfolioSimulator(changed)
        for simulator in (left, right):
            simulator.step(PortfolioAction.EQUAL_WEIGHT)
            self.assertEqual(simulator.get_state().date, data.decision_dates[1])
        interpreter = PortfolioStateInterpreter()
        np.testing.assert_array_equal(interpreter.interpret(left.get_state()), interpreter.interpret(right.get_state()))
        np.testing.assert_allclose(left.get_state().net_return_history, [0.05])
        a, b = left.step(PortfolioAction.CASH), right.step(PortfolioAction.CASH)
        np.testing.assert_array_equal(a.requested_amounts, b.requested_amounts)

    def test_execution_gap_reduces_fills_without_resizing_the_requested_order(self):
        data = make_data(np.zeros((1, 1)))
        data = replace(data, execution_close=np.array([[2.0]]), reward_end_close=np.array([[2.0]]))
        simulator = PortfolioSimulator(data)
        transition = simulator.step(PortfolioAction.EQUAL_WEIGHT)
        np.testing.assert_allclose(transition.requested_amounts, [100_000.0])
        np.testing.assert_allclose(transition.filled_amounts, [50_000.0])
        self.assertAlmostEqual(transition.ending_value, 100_000.0)

    def test_hold_keeps_quantities_and_accounts_for_pre_execution_price_move(self):
        data = make_data(np.zeros((2, 1)))
        execution = data.execution_close.copy()
        execution[1] = 1.2
        end = data.reward_end_close.copy()
        end[1] = 1.3
        simulator = PortfolioSimulator(replace(data, execution_close=execution, reward_end_close=end))
        simulator.step(PortfolioAction.EQUAL_WEIGHT)
        transition = simulator.step(PortfolioAction.HOLD)
        np.testing.assert_array_equal(transition.requested_amounts, [0.0])
        self.assertAlmostEqual(transition.ending_value, 130_000.0)
        self.assertAlmostEqual(transition.reward.net_return, 0.3)

    def test_minimum_fees_never_overdraw_cash(self):
        simulator = PortfolioSimulator(
            make_data(np.zeros((1, 2))), config=PortfolioSimulatorConfig(initial_value=100.0, min_cost=10.0)
        )
        transition = simulator.step(PortfolioAction.EQUAL_WEIGHT)
        np.testing.assert_allclose(transition.filled_amounts, [50.0, 30.0])
        self.assertAlmostEqual(transition.ending_value, 80.0)
        self.assertAlmostEqual(transition.ending_cash_weight, 0.0)

    def test_completed_episode_rejects_another_step(self) -> None:
        simulator = PortfolioSimulator(make_data(np.zeros((1, 1))))
        simulator.step(PortfolioAction.HOLD)

        with self.assertRaisesRegex(RuntimeError, "completed"):
            simulator.step(PortfolioAction.HOLD)


if __name__ == "__main__":
    unittest.main()
