# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from qlib.backtest.decision import Order
from qlib.contrib.strategy.rl_dqn import (
    PortfolioDQNStrategy,
    resolve_portfolio_dqn_checkpoint,
    resolve_portfolio_experiment_root,
)
from qlib.rl.portfolio import PortfolioDQNConfig, make_dqn_policy, save_dqn_checkpoint
from qlib.rl.portfolio.interpreter import PortfolioStateInterpreter, build_portfolio_observation


class FixedActionPolicy:
    def __init__(self, action: int) -> None:
        self.action = action

    def __call__(self, batch):
        return SimpleNamespace(act=np.asarray([self.action], dtype=np.int64))


class FakeCalendar:
    dates = pd.DatetimeIndex(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"])

    def __init__(self) -> None:
        self.step = 0

    def get_trade_step(self) -> int:
        return self.step

    def get_step_time(self, trade_step=None, shift=0):
        selected = self.step if trade_step is None else trade_step
        date = self.dates[selected - shift]
        return date, date


class FakePosition:
    def __init__(self) -> None:
        self.cash = 100_000.0

    def get_stock_amount_dict(self) -> dict:
        return {}

    def get_cash(self, include_settle=False) -> float:
        return self.cash

    def calculate_value(self) -> float:
        return self.cash


class FakeAccount:
    def __init__(self, position: FakePosition) -> None:
        self.current_position = position


class FakeExchange:
    open_cost = 0.0
    close_cost = 0.0

    def __init__(self) -> None:
        self.last_target_position = None

    def is_stock_tradable(self, **kwargs) -> bool:
        return True

    def get_close(self, *args, **kwargs) -> float:
        return 100.0

    def get_deal_price(self, *args, **kwargs) -> float:
        return 100.0

    def generate_order_for_target_amount_position(self, target_position, current_position, start_time, end_time):
        self.last_target_position = target_position
        return [
            Order(
                stock_id="AAPL",
                amount=1.0,
                direction=Order.BUY,
                start_time=start_time,
                end_time=end_time,
            )
        ]


class PortfolioDQNStrategyConfigTest(unittest.TestCase):
    def test_combines_experiment_base_directory_and_name(self) -> None:
        self.assertEqual(
            resolve_portfolio_experiment_root(
                experiment_name="alpha158_szrankguard_rolling_horizon2_step10",
                experiment_base_dir="/home/shiyu/qlib_experiment",
            ),
            Path("/home/shiyu/qlib_experiment/alpha158_szrankguard_rolling_horizon2_step10"),
        )

    def test_rejects_incomplete_or_conflicting_experiment_location(self) -> None:
        with self.assertRaisesRegex(ValueError, "provided together"):
            resolve_portfolio_experiment_root(experiment_name="experiment")
        with self.assertRaisesRegex(ValueError, "not both"):
            resolve_portfolio_experiment_root(
                experiment_root="/experiments/experiment",
                experiment_name="experiment",
                experiment_base_dir="/experiments",
            )

    def test_resolves_standard_experiment_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "portfolio_dqn_rl" / "training" / "best_policy.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()

            self.assertEqual(resolve_portfolio_dqn_checkpoint(root), checkpoint)

    def test_missing_checkpoint_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory) / "portfolio_dqn_rl" / "training" / "best_policy.pt"
            with self.assertRaisesRegex(FileNotFoundError, str(expected)):
                resolve_portfolio_dqn_checkpoint(directory)

    def test_loads_checkpoint_in_deterministic_inference_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "portfolio_dqn_rl" / "training" / "best_policy.pt"
            config = PortfolioDQNConfig(hidden_dims=(16,))
            save_dqn_checkpoint(make_dqn_policy(config), checkpoint)
            signal = pd.Series(
                [0.1],
                index=pd.MultiIndex.from_tuples(
                    [(pd.Timestamp("2025-01-02"), "AAPL")],
                    names=["datetime", "instrument"],
                ),
            )

            strategy = PortfolioDQNStrategy(
                signal=signal,
                experiment_root=root,
                hidden_dims=[16],
            )

            self.assertEqual(strategy.checkpoint_path, checkpoint)
            self.assertEqual(strategy.policy.eps, 0.0)
            self.assertFalse(strategy.policy.training)

    def test_shared_observation_matches_training_interpreter(self) -> None:
        state = SimpleNamespace(
            asset_weights=np.asarray([0.4, 0.2]),
            cash_weight=0.4,
            scores=np.asarray([0.2, -0.1]),
            volatility=np.asarray([0.1, 0.2]),
            tradable=np.asarray([True, True]),
            last_transition=None,
            net_return_history=np.asarray([0.01, -0.02]),
        )
        expected = PortfolioStateInterpreter().interpret(state)
        actual = build_portfolio_observation(
            asset_weights=state.asset_weights,
            cash_weight=state.cash_weight,
            scores=state.scores,
            volatility=state.volatility,
            tradable=state.tradable,
            net_return_history=state.net_return_history,
        )

        np.testing.assert_array_equal(actual, expected)

    def test_two_session_schedule_selects_action_and_generates_qlib_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "portfolio_dqn_rl" / "training" / "best_policy.pt"
            save_dqn_checkpoint(make_dqn_policy(), checkpoint)
            signal = pd.Series(
                [0.2, 0.3],
                index=pd.MultiIndex.from_tuples(
                    [
                        (pd.Timestamp("2025-01-02"), "AAPL"),
                        (pd.Timestamp("2025-01-06"), "AAPL"),
                    ],
                    names=["datetime", "instrument"],
                ),
            )
            strategy = PortfolioDQNStrategy(signal=signal, experiment_root=root, holding_interval=2)
            strategy.policy = FixedActionPolicy(2)
            strategy._load_volatility = lambda instruments, decision_date: np.full(len(instruments), 0.1)
            calendar = FakeCalendar()
            position = FakePosition()
            exchange = FakeExchange()
            strategy.reset(
                level_infra={"trade_calendar": calendar},
                common_infra={"trade_account": FakeAccount(position), "trade_exchange": exchange},
            )

            self.assertEqual(strategy.generate_trade_decision().get_decision(), [])
            calendar.step = 1
            first_decision = strategy.generate_trade_decision()
            self.assertEqual(len(first_decision.get_decision()), 1)
            self.assertEqual(strategy.last_action.name, "EQUAL_WEIGHT")
            self.assertEqual(exchange.last_target_position, {"AAPL": 1000.0})
            order = first_decision.get_decision()[0]
            order.deal_amount = 1.0
            strategy.post_exe_step([(order, 100.0, 0.0, 100.0)])

            calendar.step = 2
            self.assertEqual(strategy.generate_trade_decision().get_decision(), [])
            calendar.step = 3
            second_decision = strategy.generate_trade_decision()
            self.assertEqual(len(second_decision.get_decision()), 1)
            self.assertEqual(len(strategy.action_history), 2)
            self.assertEqual(strategy._net_return_history, [0.0])
            second_order = second_decision.get_decision()[0]
            second_order.deal_amount = 1.0
            strategy.post_exe_step([(second_order, 100.0, 0.0, 100.0)])
            strategy.post_upper_level_exe_step()

            evaluation = root / "portfolio_dqn_rl" / "evaluation"
            actions = pd.read_csv(evaluation / "dqn_actions.csv")
            executions = pd.read_csv(evaluation / "executed_orders.csv")
            self.assertEqual(actions["action"].tolist(), ["EQUAL_WEIGHT", "EQUAL_WEIGHT"])
            self.assertEqual(actions["executed_order_count"].tolist(), [1, 1])
            self.assertEqual(executions["direction"].tolist(), ["BUY", "BUY"])

            strategy.reset(
                level_infra={"trade_calendar": calendar},
                common_infra={"trade_account": FakeAccount(position), "trade_exchange": exchange},
            )
            self.assertEqual(strategy.action_history, [])
            self.assertEqual(strategy._net_return_history, [])


if __name__ == "__main__":
    unittest.main()
