# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Portfolio DQN strategy for Qlib backtests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from tianshou.data import Batch

from qlib.backtest.decision import OrderDir, TradeDecisionWO
from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
from qlib.data import D
from qlib.rl.portfolio import (
    PortfolioAction,
    PortfolioActionConfig,
    PortfolioDQNConfig,
    build_target_weights,
    calculate_turnover,
    load_dqn_checkpoint,
)
from qlib.rl.portfolio.interpreter import build_portfolio_observation


DEFAULT_CHECKPOINT = Path("portfolio_dqn_rl/training/best_policy.pt")


@dataclass(frozen=True)
class RollingPolicyRoute:
    window_id: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    checkpoint_path: Path
    decision_dates: frozenset[pd.Timestamp]


def resolve_portfolio_experiment_root(
    experiment_root: Optional[Union[str, Path]] = None,
    experiment_name: Optional[str] = None,
    experiment_base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve an explicit root or combine a reusable base directory and name."""

    if experiment_root is not None:
        if experiment_name is not None or experiment_base_dir is not None:
            raise ValueError("Use experiment_root or experiment_name with experiment_base_dir, not both.")
        return Path(experiment_root).expanduser().resolve()
    if experiment_name is None or experiment_base_dir is None:
        raise ValueError("experiment_name and experiment_base_dir must be provided together.")
    if not experiment_name.strip() or Path(experiment_name).name != experiment_name:
        raise ValueError("experiment_name must be a non-empty directory name, not a path.")
    return Path(experiment_base_dir).expanduser().resolve() / experiment_name


def resolve_portfolio_dqn_checkpoint(
    experiment_root: Union[str, Path], checkpoint_path: Optional[Union[str, Path]] = None
) -> Path:
    """Resolve the standard experiment-relative checkpoint or an explicit override."""

    root = Path(experiment_root).expanduser().resolve()
    resolved = (
        Path(checkpoint_path).expanduser().resolve()
        if checkpoint_path is not None
        else root / DEFAULT_CHECKPOINT
    )
    if not resolved.is_file():
        raise FileNotFoundError(f"Portfolio DQN checkpoint does not exist: {resolved}")
    return resolved


class PortfolioDQNStrategy(BaseSignalStrategy):
    """Use a trained portfolio DQN to generate Qlib target-position orders."""

    def __init__(
        self,
        *,
        experiment_root: Optional[Union[str, Path]] = None,
        experiment_name: Optional[str] = None,
        experiment_base_dir: Optional[Union[str, Path]] = None,
        trading_interval: int = 2,
        hidden_dims: Sequence[int] = (64, 64),
        learning_rate: float = 1e-3,
        discount_factor: float = 0.99,
        epochs: int = 30,
        epsilon_decay_epochs: int = 20,
        checkpoint_path: Optional[Union[str, Path]] = None,
        volatility_window: int = 20,
        rolling_retrain: bool = True,
        max_holdings: Optional[int] = None,
        increase_exposure_ratio: float = 0.25,
        concentrated_holdings: int = 5,
        concentrated_equity: float = 0.95,
        rotation_count: int = 2,
        defensive_equity: float = 0.50,
        turnover_penalty: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if trading_interval <= 0:
            raise ValueError("trading_interval must be positive.")
        if volatility_window < 2:
            raise ValueError("volatility_window must be at least two sessions.")
        if not np.isfinite(turnover_penalty) or turnover_penalty < 0.0:
            raise ValueError("turnover_penalty must be a finite non-negative value.")

        self.experiment_root = resolve_portfolio_experiment_root(
            experiment_root=experiment_root,
            experiment_name=experiment_name,
            experiment_base_dir=experiment_base_dir,
        )
        self.trading_interval = int(trading_interval)
        self.volatility_window = int(volatility_window)
        self.rolling_retrain = bool(rolling_retrain)
        # This is a learning-only setting read from the shared strategy config.
        # Consume it here so it is not forwarded to BaseStrategy.
        self.turnover_penalty = float(turnover_penalty)
        self.action_config = PortfolioActionConfig(
            max_holdings=max_holdings,
            increase_exposure_ratio=increase_exposure_ratio,
            concentrated_holdings=concentrated_holdings,
            concentrated_equity=concentrated_equity,
            rotation_count=rotation_count,
            defensive_equity=defensive_equity,
        )
        self.dqn_config = PortfolioDQNConfig(
            hidden_dims=tuple(int(size) for size in hidden_dims),
            learning_rate=float(learning_rate),
            discount_factor=float(discount_factor),
            epochs=int(epochs),
            epsilon_decay_epochs=int(epsilon_decay_epochs),
        )
        self.checkpoint_path = resolve_portfolio_dqn_checkpoint(
            self.experiment_root,
            checkpoint_path=checkpoint_path,
        )
        self.policy_routes = load_rolling_policy_routes(
            self.experiment_root,
            trading_interval=self.trading_interval,
        )
        self._decision_routes = {
            decision_date: route
            for route in self.policy_routes
            for decision_date in route.decision_dates
        }
        self.policy = None
        self._reset_episode_state()

    def generate_trade_decision(self, execute_result=None) -> TradeDecisionWO:
        """Select an action every holding interval and return executable orders."""

        trade_step = self.trade_calendar.get_trade_step()
        if trade_step == 0:
            return TradeDecisionWO([], self)

        trade_start_time, trade_end_time = self._trade_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        route = self._decision_routes.get(pd.Timestamp(pred_start_time).normalize())
        if route is None:
            return self._make_trade_decision([], trade_start_time, trade_end_time)
        self._activate_window(route)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if pred_score is None:
            return self._make_trade_decision([], trade_start_time, trade_end_time)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        scores_by_instrument = pd.Series(pred_score, dtype=np.float64)
        scores_by_instrument.index = scores_by_instrument.index.astype(str)

        current_amounts = self.trade_position.get_stock_amount_dict()
        instruments = tuple(sorted(set(scores_by_instrument.index) | set(current_amounts)))
        if not instruments:
            return self._make_trade_decision([], trade_start_time, trade_end_time)

        current_weights, cash_weight, portfolio_value = self._current_weights(
            instruments,
            trade_start_time,
            trade_end_time,
        )
        self._record_completed_holding_return(portfolio_value)
        scores = scores_by_instrument.reindex(instruments).to_numpy(dtype=np.float64)
        volatility = self._load_volatility(instruments, pred_start_time)
        tradable = np.asarray(
            [
                self.trade_exchange.is_stock_tradable(
                    stock_id=instrument,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                )
                for instrument in instruments
            ],
            dtype=np.bool_,
        )
        observation = build_portfolio_observation(
            asset_weights=current_weights,
            cash_weight=cash_weight,
            scores=scores,
            volatility=volatility,
            tradable=tradable,
            last_turnover=self._last_turnover,
            net_return_history=np.asarray(self._net_return_history, dtype=np.float64),
            action_config=self.action_config,
        )
        with torch.no_grad():
            output = self.policy(Batch(obs=observation[None, :], info=Batch()))
        action = PortfolioAction(int(output.act[0]))
        target = build_target_weights(
            action=action,
            current_asset_weights=current_weights,
            current_cash_weight=cash_weight,
            scores=scores,
            tradable=tradable,
            volatility=volatility,
            config=self.action_config,
        )
        turnover = calculate_turnover(current_weights, target.asset_weights)
        self._last_turnover = turnover.total
        self.last_observation = observation.copy()
        self.last_action = action
        self.action_history.append((pd.Timestamp(pred_start_time), pd.Timestamp(trade_start_time), action))
        self._decision_audit.append(
            {
                "window_id": route.window_id,
                "checkpoint_path": str(route.checkpoint_path),
                "decision_date": pd.Timestamp(pred_start_time),
                "execution_date": pd.Timestamp(trade_start_time),
                "action_id": int(action),
                "action": action.name,
                "requested_turnover": float(turnover.total),
                "target_equity_weight": float(target.asset_weights.sum()),
                "target_cash_weight": float(target.cash_weight),
                "requested_order_count": 0,
                "executed_order_count": 0,
                "executed_buy_count": 0,
                "executed_sell_count": 0,
                "executed_trade_value": 0.0,
                "transaction_cost": 0.0,
            }
        )
        self._decision_pending = True

        if action == PortfolioAction.HOLD:
            return self._make_trade_decision([], trade_start_time, trade_end_time)

        target_amounts = self._target_amounts(
            instruments,
            target.asset_weights,
            portfolio_value,
            trade_start_time,
            trade_end_time,
        )
        order_list = self.trade_exchange.generate_order_for_target_amount_position(
            target_position=target_amounts,
            current_position=current_amounts,
            start_time=trade_start_time,
            end_time=trade_end_time,
        )
        self._decision_audit[-1]["requested_order_count"] = len(order_list)
        return self._make_trade_decision(order_list, trade_start_time, trade_end_time)

    def _make_trade_decision(self, orders, start_time, end_time) -> TradeDecisionWO:
        """Keep the strategy's resolved interval instead of asking the calendar twice."""

        return TradeDecisionWO(
            orders,
            self,
            trade_time=(pd.Timestamp(start_time), pd.Timestamp(end_time)),
        )

    def _trade_step_time(self, trade_step: int) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return the execution interval, including Qlib's final calendar session.

        Qlib normally represents an interval with the current and next calendar
        entries. When a backtest ends on the last available data session, that
        extra right-boundary entry does not exist. A daily strategy can still
        execute safely by using the final session as both closed endpoints.
        """

        try:
            return self.trade_calendar.get_step_time(trade_step)
        except IndexError:
            calendar = self.trade_calendar._calendar
            calendar_index = self.trade_calendar.start_index + trade_step
            if calendar_index != len(calendar) - 1:
                raise
            final_session = pd.Timestamp(calendar[calendar_index])
            return final_session, final_session

    def reset(self, *args, **kwargs) -> None:
        """Reset both Qlib infrastructure and state carried between DQN decisions."""

        super().reset(*args, **kwargs)
        self._reset_episode_state()

    def post_exe_step(self, execute_result=None) -> None:
        """Record portfolio value immediately after a scheduled rebalance."""

        if self._decision_pending:
            executions = execute_result or []
            buys = 0
            sells = 0
            trade_value = 0.0
            transaction_cost = 0.0
            for order, value, cost, price in executions:
                direction = "BUY" if order.direction == OrderDir.BUY else "SELL"
                buys += direction == "BUY"
                sells += direction == "SELL"
                trade_value += float(value)
                transaction_cost += float(cost)
                self._execution_audit.append(
                    {
                        "window_id": self._decision_audit[-1]["window_id"],
                        "checkpoint_path": self._decision_audit[-1]["checkpoint_path"],
                        "execution_date": pd.Timestamp(order.start_time),
                        "instrument": str(order.stock_id),
                        "direction": direction,
                        "requested_amount": float(order.amount),
                        "executed_amount": float(order.deal_amount),
                        "executed_price": float(price),
                        "executed_value": float(value),
                        "transaction_cost": float(cost),
                    }
                )
            audit = self._decision_audit[-1]
            audit["executed_order_count"] = len(executions)
            audit["executed_buy_count"] = buys
            audit["executed_sell_count"] = sells
            audit["executed_trade_value"] = trade_value
            audit["transaction_cost"] = transaction_cost
            self._last_decision_value = float(self.trade_position.calculate_value())
            self._decision_pending = False

    def post_upper_level_exe_step(self) -> None:
        """Publish the completed live-Qlib action and execution audit."""

        super().post_upper_level_exe_step()
        evaluation_dir = self.experiment_root / "portfolio_dqn_rl" / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        _write_csv_atomically(pd.DataFrame(self._decision_audit), evaluation_dir / "dqn_actions.csv")
        _write_csv_atomically(pd.DataFrame(self._execution_audit), evaluation_dir / "executed_orders.csv")

    def _reset_episode_state(self) -> None:
        self._last_decision_value: Optional[float] = None
        self._last_turnover = 0.0
        self._net_return_history: list[float] = []
        self._decision_pending = False
        self._active_window_id: Optional[int] = None
        self._active_checkpoint_path: Optional[Path] = None
        self.last_observation: Optional[np.ndarray] = None
        self.last_action: Optional[PortfolioAction] = None
        self.action_history: list[tuple[pd.Timestamp, pd.Timestamp, PortfolioAction]] = []
        self._decision_audit: list[dict] = []
        self._execution_audit: list[dict] = []

    def _activate_window(self, route: RollingPolicyRoute) -> None:
        if self._active_window_id == route.window_id:
            return
        policy = load_dqn_checkpoint(route.checkpoint_path, self.dqn_config)
        policy.set_eps(0.0)
        policy.eval()
        self.policy = policy
        self._active_window_id = route.window_id
        self._active_checkpoint_path = route.checkpoint_path

    def _record_completed_holding_return(self, current_value: float) -> None:
        if self._last_decision_value is not None:
            self._net_return_history.append(current_value / self._last_decision_value - 1.0)

    def _current_weights(
        self,
        instruments: Sequence[str],
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
    ) -> tuple[np.ndarray, float, float]:
        amounts = self.trade_position.get_stock_amount_dict()
        values = np.zeros(len(instruments), dtype=np.float64)
        for index, instrument in enumerate(instruments):
            amount = float(amounts.get(instrument, 0.0))
            if amount == 0.0:
                continue
            price = self.trade_exchange.get_close(instrument, trade_start_time, trade_end_time)
            if price is None or not np.isfinite(price) or price <= 0.0:
                price = self.trade_position.get_stock_price(instrument)
            values[index] = amount * float(price)
        cash = float(self.trade_position.get_cash(include_settle=True))
        total_value = float(values.sum() + cash)
        if not np.isfinite(total_value) or total_value <= 0.0:
            raise ValueError("Qlib portfolio value must be positive and finite.")
        return values / total_value, cash / total_value, total_value

    def _load_volatility(self, instruments: Sequence[str], decision_date: pd.Timestamp) -> np.ndarray:
        calendar = pd.DatetimeIndex(
            D.calendar(
                start_time=pd.Timestamp(decision_date) - pd.Timedelta(days=max(60, self.volatility_window * 4)),
                end_time=decision_date,
                freq="day",
            )
        )
        required_dates = calendar[-(self.volatility_window + 1) :]
        if len(required_dates) < self.volatility_window + 1:
            return np.full(len(instruments), np.nan, dtype=np.float64)
        close = D.features(
            list(instruments),
            ["$close"],
            start_time=required_dates[0],
            end_time=required_dates[-1],
            freq="day",
        )["$close"]
        close_frame = close.unstack("instrument").reindex(index=required_dates, columns=instruments)
        volatility = close_frame.pct_change(fill_method=None).std(ddof=0)
        return volatility.to_numpy(dtype=np.float64)

    def _target_amounts(
        self,
        instruments: Sequence[str],
        target_weights: np.ndarray,
        portfolio_value: float,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
    ) -> dict[str, float]:
        cost_buffer = 1.0 + max(float(self.trade_exchange.open_cost), float(self.trade_exchange.close_cost))
        investable_value = portfolio_value / cost_buffer
        target_amounts: dict[str, float] = {}
        for instrument, weight in zip(instruments, target_weights):
            if weight <= 0.0:
                continue
            price = self.trade_exchange.get_deal_price(
                stock_id=instrument,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.BUY,
            )
            if price is None or not np.isfinite(price) or price <= 0.0:
                continue
            target_amounts[instrument] = investable_value * float(weight) / float(price)
        return target_amounts


def _write_csv_atomically(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".new")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_rolling_policy_routes(
    experiment_root: Union[str, Path], *, trading_interval: int, calendar_provider=None
) -> list[RollingPolicyRoute]:
    """Load and validate the one-checkpoint-per-test-block execution schedule."""

    root = Path(experiment_root).expanduser().resolve()
    rolling_dir = root / "portfolio_dqn_rl" / "rolling"
    windows_path = rolling_dir / "rolling_windows.csv"
    history_path = rolling_dir / "checkpoint_history.csv"
    if not windows_path.is_file() or not history_path.is_file():
        raise FileNotFoundError(
            "Rolling DQN execution requires rolling_windows.csv and checkpoint_history.csv."
        )
    windows = pd.read_csv(windows_path)
    history = pd.read_csv(history_path)
    required_windows = {"window_id", "test_start", "test_end"}
    required_history = {"window_id", "checkpoint_path"}
    if not required_windows.issubset(windows) or not required_history.issubset(history):
        raise ValueError("Rolling DQN audit files are missing required schedule columns.")
    if windows["window_id"].duplicated().any() or history["window_id"].duplicated().any():
        raise ValueError("Rolling DQN audit files contain duplicate window IDs.")
    schedule = windows[list(required_windows)].merge(
        history[list(required_history)], on="window_id", how="left", validate="one_to_one"
    ).sort_values("window_id")
    if schedule["checkpoint_path"].isna().any() or len(schedule) != len(windows):
        raise ValueError("Every rolling test window must have one completed checkpoint.")

    routes = []
    get_calendar = D.calendar if calendar_provider is None else calendar_provider
    previous_end = None
    for row in schedule.itertuples(index=False):
        test_start = pd.Timestamp(row.test_start).normalize()
        test_end = pd.Timestamp(row.test_end).normalize()
        if previous_end is not None and test_start <= previous_end:
            raise ValueError("Rolling DQN test windows must be chronological and non-overlapping.")
        checkpoint_path = Path(row.checkpoint_path).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = (root / checkpoint_path).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Rolling DQN checkpoint does not exist: {checkpoint_path}")
        test_calendar = pd.DatetimeIndex(
            get_calendar(start_time=test_start, end_time=test_end, freq="day")
        ).normalize()
        decision_dates = frozenset(test_calendar[::trading_interval])
        routes.append(
            RollingPolicyRoute(
                window_id=int(row.window_id),
                test_start=test_start,
                test_end=test_end,
                checkpoint_path=checkpoint_path,
                decision_dates=decision_dates,
            )
        )
        previous_end = test_end
    return routes
