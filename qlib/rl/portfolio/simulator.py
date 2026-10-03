# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Deterministic two-session portfolio simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from qlib.rl.simulator import Simulator

from .action import PortfolioAction, PortfolioActionConfig, PortfolioTarget, build_target_weights
from .data import PortfolioDataSplit
from .reward import PortfolioReward, PortfolioTurnover
from .sizing import size_target_amounts


# Policies trained with execution-time observations must not be silently reused.
DECISION_TIMING_VERSION = "decision_close_v2"


@dataclass(frozen=True)
class PortfolioSimulatorConfig:
    """Accounting settings for the initial engineering simulator."""

    initial_value: float = 100_000.0
    buy_cost: float = 0.0
    sell_cost: float = 0.0
    min_cost: float = 0.0
    turnover_penalty: float = 0.0
    missing_return_value: float = 0.0
    tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if not np.isfinite(self.initial_value) or self.initial_value <= 0.0:
            raise ValueError("initial_value must be a positive finite value.")
        if not np.isfinite(self.buy_cost) or self.buy_cost < 0.0:
            raise ValueError("buy_cost must be a finite non-negative value.")
        if not np.isfinite(self.sell_cost) or self.sell_cost < 0.0:
            raise ValueError("sell_cost must be a finite non-negative value.")
        if not np.isfinite(self.min_cost) or self.min_cost < 0.0:
            raise ValueError("min_cost must be a finite non-negative dollar amount.")
        if not np.isfinite(self.turnover_penalty) or self.turnover_penalty < 0.0:
            raise ValueError("turnover_penalty must be a finite non-negative value.")
        if not np.isfinite(self.missing_return_value) or self.missing_return_value < -1.0:
            raise ValueError("missing_return_value must be finite and no less than -1.")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be a positive finite value.")


@dataclass(frozen=True)
class PortfolioState:
    """Portfolio state at a decision close, before next-session execution."""

    step: int
    date: pd.Timestamp
    portfolio_value: float
    asset_weights: np.ndarray
    cash_weight: float
    done: bool
    scores: np.ndarray
    volatility: np.ndarray
    tradable: np.ndarray
    last_transition: Optional["PortfolioTransition"]
    net_return_history: np.ndarray


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
    requested_amounts: np.ndarray
    filled_amounts: np.ndarray


class PortfolioSimulator(Simulator[PortfolioDataSplit, PortfolioState, PortfolioAction]):
    """Apply discrete portfolio commands to one prepared data split."""

    def __init__(
        self,
        data: PortfolioDataSplit,
        config: PortfolioSimulatorConfig = PortfolioSimulatorConfig(),
        action_config: PortfolioActionConfig = PortfolioActionConfig(),
    ) -> None:
        self.data = data
        self.config = config
        if config.missing_return_value != 0.0:
            raise ValueError(
                "The chronological simulator carries missing prices forward; missing_return_value must be 0."
            )
        self.action_config = action_config
        self._validate_transition_chain()
        self._last_transition: Optional[PortfolioTransition] = None
        self._net_return_history: list[float] = []
        self._amounts = np.zeros(len(data.instruments), dtype=np.float64)
        self._cash = config.initial_value
        self._state = self._initial_state()

    def reset(self) -> PortfolioState:
        """Reset the episode to an all-cash portfolio."""

        self._last_transition = None
        self._net_return_history = []
        self._amounts[:] = 0.0
        self._cash = self.config.initial_value
        self._state = self._initial_state()
        return self.get_state()

    def get_state(self) -> PortfolioState:
        """Return a defensive copy of the current state."""

        observation_step = min(self._state.step, len(self.data.decision_dates) - 1)
        return PortfolioState(
            step=self._state.step,
            date=self._state.date,
            portfolio_value=self._state.portfolio_value,
            asset_weights=self._state.asset_weights.copy(),
            cash_weight=self._state.cash_weight,
            done=self._state.done,
            scores=self.data.scores[observation_step].copy(),
            volatility=self.data.volatility[observation_step].copy(),
            tradable=self.data.observation_tradable[observation_step].copy(),
            last_transition=self._last_transition,
            net_return_history=np.asarray(self._net_return_history, dtype=np.float64),
        )

    def done(self) -> bool:
        """Return whether all transitions in this episode have been consumed."""

        return self._state.done

    def step(self, action: Union[PortfolioAction, int]) -> PortfolioTransition:
        """Freeze orders at T, fill at T+1, and mark the account at T+2.

        The next observation sees only T+2 account information. In particular,
        the T+1 to T+3 supervised forward label never enters this state.
        """

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
            tradable=self.data.observation_tradable[step],
            volatility=self.data.volatility[step],
            config=self.action_config,
        )
        decision_prices = self.data.decision_close[step]
        desired = (
            self._amounts.copy()
            if resolved_action == PortfolioAction.HOLD
            else size_target_amounts(
                self._amounts,
                target.asset_weights,
                starting_value,
                decision_prices,
                self.data.observation_tradable[step],
                max(self.config.buy_cost, self.config.sell_cost),
            )
        )
        requested = desired - self._amounts
        filled, fees = self._execute(requested, step)
        execution_prices = self._mark_prices(self.data.execution_close[step], decision_prices)
        ending_prices = self._mark_prices(self.data.reward_end_close[step], execution_prices)
        asset_values = self._amounts * np.nan_to_num(ending_prices, nan=0.0)
        ending_value = float(asset_values.sum() + self._cash)
        if not np.isfinite(ending_value) or ending_value <= 0.0:
            raise ValueError("Portfolio value became non-positive or non-finite.")
        ending_assets = asset_values / ending_value
        ending_cash = self._cash / ending_value
        # Turnover and costs describe actual fills, not unfilled intentions.
        values = filled * np.nan_to_num(execution_prices, nan=0.0) / starting_value
        buy_orders, sell_orders = np.maximum(values, 0.0), np.maximum(-values, 0.0)
        turnover = PortfolioTurnover(float(buy_orders.sum()), float(sell_orders.sum()), buy_orders, sell_orders)
        net_return = ending_value / starting_value - 1.0
        cost = fees / starting_value
        penalty = self.config.turnover_penalty * turnover.one_way
        missing = ~np.isfinite(self.data.reward_end_close[step]) | (self.data.reward_end_close[step] <= 0.0)
        reward = PortfolioReward(
            gross_return=net_return + cost,
            transaction_cost=cost,
            turnover_penalty=penalty,
            net_return=net_return,
            learning_reward=net_return - penalty,
            missing_return_weight=float(asset_values[missing].sum() / starting_value),
            effective_asset_returns=np.divide(
                ending_prices,
                decision_prices,
                out=np.ones_like(ending_prices),
                where=np.isfinite(decision_prices) & (decision_prices > 0.0),
            )
            - 1.0,
        )
        next_step = step + 1
        done = next_step == len(self.data.decision_dates)
        next_date = self.data.reward_end_dates[step]
        transition = PortfolioTransition(
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
            requested_amounts=requested.copy(),
            filled_amounts=filled.copy(),
        )
        self._last_transition = transition
        self._net_return_history.append(reward.net_return)
        self._state = PortfolioState(
            step=next_step,
            date=next_date,
            portfolio_value=ending_value,
            asset_weights=ending_assets,
            cash_weight=ending_cash,
            done=done,
            scores=np.empty(0, dtype=np.float64),
            volatility=np.empty(0, dtype=np.float64),
            tradable=np.empty(0, dtype=np.bool_),
            last_transition=transition,
            net_return_history=np.asarray(self._net_return_history, dtype=np.float64),
        )

        return transition

    def _initial_state(self) -> PortfolioState:
        return PortfolioState(
            step=0,
            date=self.data.decision_dates[0],
            portfolio_value=self.config.initial_value,
            asset_weights=np.zeros(len(self.data.instruments), dtype=np.float64),
            cash_weight=1.0,
            done=False,
            scores=np.empty(0, dtype=np.float64),
            volatility=np.empty(0, dtype=np.float64),
            tradable=np.empty(0, dtype=np.bool_),
            last_transition=None,
            net_return_history=np.empty(0, dtype=np.float64),
        )

    def _mark_prices(self, prices: np.ndarray, previous: np.ndarray) -> np.ndarray:
        valid = np.isfinite(prices) & (prices > 0.0)
        marks = np.where(valid, prices, previous)
        if np.any((self._amounts > 0.0) & (~np.isfinite(marks) | (marks <= 0.0))):
            raise ValueError("A held asset has no past price for valuation.")
        return marks

    def _execute(self, requested: np.ndarray, step: int) -> tuple[np.ndarray, float]:
        """Apply frozen quantities at execution prices, sells before buys.

        An unavailable stock does not fill. Cash shortages reduce buys in the
        stable instrument order. No future return is used to filter orders.
        """
        prices = self.data.execution_close[step]
        tradable = self.data.execution_tradable[step] & np.isfinite(prices) & (prices > 0.0)
        filled = np.zeros_like(requested)
        fees = 0.0
        for index in np.flatnonzero((requested < 0.0) & tradable):
            amount = min(-requested[index], self._amounts[index])
            value = amount * prices[index]
            fee = max(value * self.config.sell_cost, self.config.min_cost)
            if self._cash + value < fee:
                continue
            self._amounts[index] -= amount
            self._cash += value - fee
            filled[index] = -amount
            fees += fee
        for index in np.flatnonzero((requested > 0.0) & tradable):
            budget = max(
                0.0,
                min(
                    self._cash / (1.0 + self.config.buy_cost),
                    self._cash - self.config.min_cost,
                ),
            )
            amount = min(requested[index], budget / prices[index])
            if amount <= 0.0:
                continue
            value = amount * prices[index]
            fee = max(value * self.config.buy_cost, self.config.min_cost)
            self._amounts[index] += amount
            self._cash = max(0.0, self._cash - value - fee)
            filled[index] = amount
            fees += fee
        return filled, fees

    def _validate_transition_chain(self) -> None:
        if len(self.data.decision_dates) > 1 and not np.all(
            self.data.decision_dates[1:] == self.data.reward_end_dates[:-1]
        ):
            raise ValueError("Each transition must end at the next decision date.")
