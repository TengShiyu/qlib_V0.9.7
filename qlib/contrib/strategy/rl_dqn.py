# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Portfolio DQN strategy for Qlib backtests."""

from __future__ import annotations

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
        holding_interval: int = 2,
        hidden_dims: Sequence[int] = (64, 64),
        learning_rate: float = 1e-3,
        discount_factor: float = 0.99,
        epochs: int = 30,
        epsilon_decay_epochs: int = 20,
        checkpoint_path: Optional[Union[str, Path]] = None,
        volatility_window: int = 20,
        retrain: bool = True,
        max_holdings: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if holding_interval <= 0:
            raise ValueError("holding_interval must be positive.")
        if volatility_window < 2:
            raise ValueError("volatility_window must be at least two sessions.")

        self.experiment_root = resolve_portfolio_experiment_root(
            experiment_root=experiment_root,
            experiment_name=experiment_name,
            experiment_base_dir=experiment_base_dir,
        )
        self.holding_interval = int(holding_interval)
        self.volatility_window = int(volatility_window)
        self.retrain = bool(retrain)
        self.action_config = PortfolioActionConfig(max_holdings=max_holdings)
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
        self.policy = load_dqn_checkpoint(self.checkpoint_path, self.dqn_config)
        self.policy.set_eps(0.0)
        self.policy.eval()
        self._reset_episode_state()

    def generate_trade_decision(self, execute_result=None) -> TradeDecisionWO:
        """Select an action every holding interval and return executable orders."""

        trade_step = self.trade_calendar.get_trade_step()
        if trade_step == 0 or (trade_step - 1) % self.holding_interval != 0:
            return TradeDecisionWO([], self)

        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if pred_score is None:
            return TradeDecisionWO([], self)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        scores_by_instrument = pd.Series(pred_score, dtype=np.float64)
        scores_by_instrument.index = scores_by_instrument.index.astype(str)

        current_amounts = self.trade_position.get_stock_amount_dict()
        instruments = tuple(sorted(set(scores_by_instrument.index) | set(current_amounts)))
        if not instruments:
            return TradeDecisionWO([], self)

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
            return TradeDecisionWO([], self)

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
        return TradeDecisionWO(order_list, self)

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
        self.last_observation: Optional[np.ndarray] = None
        self.last_action: Optional[PortfolioAction] = None
        self.action_history: list[tuple[pd.Timestamp, pd.Timestamp, PortfolioAction]] = []
        self._decision_audit: list[dict] = []
        self._execution_audit: list[dict] = []

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
