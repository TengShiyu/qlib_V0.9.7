# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Deterministic two-session portfolio simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

from .action import PortfolioAction, PortfolioActionConfig, PortfolioTarget, build_target_weights
from .data import PortfolioDataSplit
from .reward import PortfolioReward, PortfolioTurnover, calculate_portfolio_reward, calculate_turnover


@dataclass(frozen=True)
class PortfolioSimulatorConfig:
    """Accounting settings for the initial engineering simulator."""

    initial_value: float = 100_000.0
    buy_cost: float = 0.0
    sell_cost: float = 0.0
    missing_return_value: float = 0.0
    tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if not np.isfinite(self.initial_value) or self.initial_value <= 0.0:
            raise ValueError("initial_value must be a positive finite value.")
        if not np.isfinite(self.buy_cost) or self.buy_cost < 0.0:
            raise ValueError("buy_cost must be a finite non-negative value.")
        if not np.isfinite(self.sell_cost) or self.sell_cost < 0.0:
            raise ValueError("sell_cost must be a finite non-negative value.")
        if not np.isfinite(self.missing_return_value) or self.missing_return_value < -1.0:
            raise ValueError("missing_return_value must be finite and no less than -1.")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be a positive finite value.")


@dataclass(frozen=True)
class PortfolioState:
    """Portfolio state at an execution boundary."""

    step: int
    date: pd.Timestamp
    portfolio_value: float
    asset_weights: np.ndarray
    cash_weight: float
    done: bool


@dataclass(frozen=True)
class PortfolioTransition:
    """Complete accounting record for one simulator step."""

    action: PortfolioAction
    decision_date: pd.Timestamp
    execution_date: pd.Timestamp
    reward_end_date: pd.Timestamp
    starting_value: float
    ending_value: float
    current_asset_weights: np.ndarray
    current_cash_weight: float
    target: PortfolioTarget
    turnover: PortfolioTurnover
    reward: PortfolioReward
    ending_asset_weights: np.ndarray
    ending_cash_weight: float


class PortfolioSimulator:
    """Apply discrete portfolio commands to one prepared data split."""

    def __init__(
        self,
        data: PortfolioDataSplit,
        config: PortfolioSimulatorConfig = PortfolioSimulatorConfig(),
        action_config: PortfolioActionConfig = PortfolioActionConfig(),
    ) -> None:
        self.data = data
        self.config = config
        self.action_config = action_config
        self._validate_transition_chain()
        self._state = self._initial_state()

    def reset(self) -> PortfolioState:
        """Reset the episode to an all-cash portfolio."""

        self._state = self._initial_state()
        return self.get_state()

    def get_state(self) -> PortfolioState:
        """Return a defensive copy of the current state."""

        return PortfolioState(
            step=self._state.step,
            date=self._state.date,
            portfolio_value=self._state.portfolio_value,
            asset_weights=self._state.asset_weights.copy(),
            cash_weight=self._state.cash_weight,
            done=self._state.done,
        )

    def step(self, action: Union[PortfolioAction, int]) -> PortfolioTransition:
        """Execute one action and advance through its two-session return."""

        if self._state.done:
            raise RuntimeError("Cannot step a completed portfolio episode; call reset().")

        step = self._state.step
        resolved_action = PortfolioAction(action)
        current_assets = self._state.asset_weights.copy()
        current_cash = self._state.cash_weight
        starting_value = self._state.portfolio_value

        target = build_target_weights(
            action=resolved_action,
            current_asset_weights=current_assets,
            current_cash_weight=current_cash,
            scores=self.data.scores[step],
            tradable=self.data.execution_tradable[step],
            volatility=self.data.volatility[step],
            config=self.action_config,
        )
        turnover = calculate_turnover(current_assets, target.asset_weights)
        reward = calculate_portfolio_reward(
            target_asset_weights=target.asset_weights,
            target_cash_weight=target.cash_weight,
            asset_returns=self.data.asset_returns[step],
            turnover=turnover,
            buy_cost=self.config.buy_cost,
            sell_cost=self.config.sell_cost,
            missing_return_value=self.config.missing_return_value,
            tolerance=self.config.tolerance,
        )

        ending_value = starting_value * (1.0 + reward.net_return)
        if not np.isfinite(ending_value) or ending_value <= 0.0:
            raise ValueError("Portfolio value became non-positive or non-finite.")

        ending_assets, ending_cash = self._drift_weights(target, reward)
        next_step = step + 1
        done = next_step == len(self.data.decision_dates)
        next_date = self.data.reward_end_dates[step]
        self._state = PortfolioState(
            step=next_step,
            date=next_date,
            portfolio_value=ending_value,
            asset_weights=ending_assets,
            cash_weight=ending_cash,
            done=done,
        )

        return PortfolioTransition(
            action=resolved_action,
            decision_date=self.data.decision_dates[step],
            execution_date=self.data.execution_dates[step],
            reward_end_date=self.data.reward_end_dates[step],
            starting_value=starting_value,
            ending_value=ending_value,
            current_asset_weights=current_assets,
            current_cash_weight=current_cash,
            target=target,
            turnover=turnover,
            reward=reward,
            ending_asset_weights=ending_assets.copy(),
            ending_cash_weight=ending_cash,
        )

    def _initial_state(self) -> PortfolioState:
        return PortfolioState(
            step=0,
            date=self.data.execution_dates[0],
            portfolio_value=self.config.initial_value,
            asset_weights=np.zeros(len(self.data.instruments), dtype=np.float64),
            cash_weight=1.0,
            done=False,
        )

    def _drift_weights(self, target: PortfolioTarget, reward: PortfolioReward) -> tuple[np.ndarray, float]:
        asset_values = target.asset_weights * (1.0 + reward.effective_asset_returns)
        cash_value = target.cash_weight
        gross_value = float(asset_values.sum() + cash_value)
        if not np.isfinite(gross_value) or gross_value <= 0.0:
            raise ValueError("Gross portfolio value became non-positive or non-finite.")

        ending_assets = asset_values / gross_value
        ending_cash = cash_value / gross_value
        if not np.isclose(ending_assets.sum() + ending_cash, 1.0, atol=self.config.tolerance, rtol=0.0):
            raise ValueError("Post-return asset and cash weights do not sum to one.")
        return ending_assets, float(ending_cash)

    def _validate_transition_chain(self) -> None:
        if len(self.data.decision_dates) > 1 and not np.all(
            self.data.execution_dates[1:] == self.data.reward_end_dates[:-1]
        ):
            raise ValueError("Each execution date must equal the previous transition's reward-end date.")
