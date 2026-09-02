# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from qlib.backtest.decision import Order
from qlib.backtest.utils import TradeCalendarManager
from qlib.contrib.strategy.rl_dqn import (
    PortfolioDQNStrategy,
    RollingPolicyRoute,
    load_rolling_policy_routes,
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

    def set_eps(self, epsilon):
        self.eps = epsilon

    def eval(self):
        return self


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


class FinalBoundaryCalendar:
    def __init__(self) -> None:
        self._calendar = pd.DatetimeIndex(["2026-07-30", "2026-07-31"])
        self.start_index = 0

    def get_trade_step(self) -> int:
        return 1

    def get_step_time(self, trade_step=None, shift=0):
        calendar_index = self.start_index + trade_step - shift
        return self._calendar[calendar_index], self._calendar[calendar_index + 1]


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
    def test_calendar_fallback_moves_when_future_session_is_added(self) -> None:
        calendar = TradeCalendarManager.__new__(TradeCalendarManager)
        calendar.start_index = 0
        calendar.trade_step = 1
        calendar._calendar = pd.DatetimeIndex(["2026-07-30", "2026-07-31"]).to_numpy()

        self.assertEqual(
            calendar.get_step_time(),
            (pd.Timestamp("2026-07-31"), pd.Timestamp("2026-07-31")),
        )

        calendar._calendar = pd.DatetimeIndex(
            ["2026-07-30", "2026-07-31", "2026-08-03"]
        ).to_numpy()
        start_time, end_time = calendar.get_step_time()

        self.assertEqual(start_time, pd.Timestamp("2026-07-31"))
        self.assertGreater(end_time, start_time)
        self.assertLess(end_time, pd.Timestamp("2026-08-03"))

    def test_final_calendar_session_is_a_valid_execution_interval(self) -> None:
        strategy = PortfolioDQNStrategy.__new__(PortfolioDQNStrategy)
        strategy.level_infra = {"trade_calendar": FinalBoundaryCalendar()}
        strategy._decision_routes = {}

        start_time, end_time = strategy._trade_step_time(1)
        decision = strategy.generate_trade_decision()

        self.assertEqual(start_time, pd.Timestamp("2026-07-31"))
        self.assertEqual(end_time, pd.Timestamp("2026-07-31"))
        self.assertEqual(decision.get_decision(), [])
        self.assertEqual(decision.start_time, pd.Timestamp("2026-07-31"))
        self.assertEqual(decision.end_time, pd.Timestamp("2026-07-31"))

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

            route = RollingPolicyRoute(
                window_id=0,
                test_start=pd.Timestamp("2025-01-02"),
                test_end=pd.Timestamp("2025-01-02"),
                checkpoint_path=checkpoint,
                decision_dates=frozenset({pd.Timestamp("2025-01-02")}),
            )
            with mock.patch(
                "qlib.contrib.strategy.rl_dqn.load_rolling_policy_routes", return_value=[route]
            ):
                strategy = PortfolioDQNStrategy(
                    signal=signal,
                    experiment_root=root,
                    hidden_dims=[16],
                )

            self.assertEqual(strategy.checkpoint_path, checkpoint)
            self.assertIsNone(strategy.policy)
            strategy._activate_window(route)
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
            route = RollingPolicyRoute(
                window_id=0,
                test_start=pd.Timestamp("2025-01-02"),
                test_end=pd.Timestamp("2025-01-07"),
                checkpoint_path=checkpoint,
                decision_dates=frozenset(
                    {pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-06")}
                ),
            )
            with mock.patch(
                "qlib.contrib.strategy.rl_dqn.load_rolling_policy_routes", return_value=[route]
            ):
                strategy = PortfolioDQNStrategy(
                    signal=signal, experiment_root=root, trading_interval=2
                )
            strategy._load_volatility = lambda instruments, decision_date: np.full(len(instruments), 0.1)
            calendar = FakeCalendar()
            position = FakePosition()
            exchange = FakeExchange()
            strategy.reset(
                level_infra={"trade_calendar": calendar},
                common_infra={"trade_account": FakeAccount(position), "trade_exchange": exchange},
            )
            strategy.policy = FixedActionPolicy(2)
            strategy._active_window_id = 0

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

    def test_loads_chronological_checkpoint_routes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rolling = root / "portfolio_dqn_rl" / "rolling"
            checkpoint_0 = rolling / "checkpoints" / "window_000" / "best_policy.pt"
            checkpoint_1 = rolling / "checkpoints" / "window_001" / "best_policy.pt"
            checkpoint_0.parent.mkdir(parents=True)
            checkpoint_1.parent.mkdir(parents=True)
            checkpoint_0.touch()
            checkpoint_1.touch()
            pd.DataFrame(
                [
                    {"window_id": 0, "test_start": "2025-01-02", "test_end": "2025-01-07"},
                    {"window_id": 1, "test_start": "2025-01-20", "test_end": "2025-01-23"},
                ]
            ).to_csv(rolling / "rolling_windows.csv", index=False)
            pd.DataFrame(
                [
                    {"window_id": 0, "checkpoint_path": str(checkpoint_0)},
                    {"window_id": 1, "checkpoint_path": str(checkpoint_1)},
                ]
            ).to_csv(rolling / "checkpoint_history.csv", index=False)

            routes = load_rolling_policy_routes(
                root,
                trading_interval=2,
                calendar_provider=lambda start_time, end_time, freq: pd.bdate_range(
                    start_time, end_time
                ),
            )

            self.assertEqual([route.window_id for route in routes], [0, 1])
            self.assertEqual(
                routes[0].decision_dates,
                frozenset({pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-06")}),
            )
            self.assertEqual(routes[1].checkpoint_path, checkpoint_1)

    def test_switches_checkpoint_only_when_matching_test_window_arrives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main = root / "portfolio_dqn_rl" / "training" / "best_policy.pt"
            checkpoint_0 = root / "window_000.pt"
            checkpoint_1 = root / "window_001.pt"
            main.parent.mkdir(parents=True)
            for path in (main, checkpoint_0, checkpoint_1):
                path.touch()
            routes = [
                RollingPolicyRoute(
                    0,
                    pd.Timestamp("2025-01-02"),
                    pd.Timestamp("2025-01-03"),
                    checkpoint_0,
                    frozenset({pd.Timestamp("2025-01-02")}),
                ),
                RollingPolicyRoute(
                    1,
                    pd.Timestamp("2025-01-06"),
                    pd.Timestamp("2025-01-07"),
                    checkpoint_1,
                    frozenset({pd.Timestamp("2025-01-06")}),
                ),
            ]
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
            with mock.patch(
                "qlib.contrib.strategy.rl_dqn.load_rolling_policy_routes", return_value=routes
            ), mock.patch(
                "qlib.contrib.strategy.rl_dqn.load_dqn_checkpoint",
                side_effect=[FixedActionPolicy(2), FixedActionPolicy(4)],
            ) as loader:
                strategy = PortfolioDQNStrategy(signal=signal, experiment_root=root)
                strategy._load_volatility = lambda instruments, decision_date: np.full(
                    len(instruments), 0.1
                )
                calendar = FakeCalendar()
                strategy.reset(
                    level_infra={"trade_calendar": calendar},
                    common_infra={
                        "trade_account": FakeAccount(FakePosition()),
                        "trade_exchange": FakeExchange(),
                    },
                )
                calendar.step = 1
                strategy.generate_trade_decision()
                calendar.step = 2
                self.assertEqual(strategy.generate_trade_decision().get_decision(), [])
                self.assertEqual(loader.call_count, 1)
                calendar.step = 3
                strategy.generate_trade_decision()

            self.assertEqual([call.args[0] for call in loader.call_args_list], [checkpoint_0, checkpoint_1])
            self.assertEqual([row["window_id"] for row in strategy._decision_audit], [0, 1])


if __name__ == "__main__":
    unittest.main()
