# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Portfolio turnover and two-session reward accounting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PortfolioTurnover:
    """Executed risky-asset purchases and sales as fractions of NAV."""

    buy: float
    sell: float
    buy_orders: np.ndarray
    sell_orders: np.ndarray

    @property
    def total(self) -> float:
        """Two-way turnover: purchases plus sales."""

        return self.buy + self.sell

    @property
    def one_way(self) -> float:
        """One-way turnover commonly used for portfolio reporting."""

        return max(self.buy, self.sell)


@dataclass(frozen=True)
class PortfolioReward:
    """Real portfolio return and learning-only penalty for one holding period."""

    gross_return: float
    transaction_cost: float
    turnover_penalty: float
    net_return: float
    learning_reward: float
    missing_return_weight: float
    effective_asset_returns: np.ndarray


def calculate_turnover(current_asset_weights: np.ndarray, target_asset_weights: np.ndarray) -> PortfolioTurnover:
    """Calculate purchases and sales between two risky-asset allocations."""

    current = np.asarray(current_asset_weights, dtype=np.float64)
    target = np.asarray(target_asset_weights, dtype=np.float64)
    if current.ndim != 1 or target.shape != current.shape:
        raise ValueError("Current and target asset weights must be one-dimensional arrays with equal shape.")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(target)):
        raise ValueError("Current and target asset weights must be finite.")
    if np.any(current < 0.0) or np.any(target < 0.0):
        raise ValueError("Current and target asset weights must be non-negative.")

    difference = target - current
    buy_orders = np.clip(difference, 0.0, None)
    sell_orders = np.clip(-difference, 0.0, None)
    return PortfolioTurnover(
        buy=float(buy_orders.sum()),
        sell=float(sell_orders.sum()),
        buy_orders=buy_orders,
        sell_orders=sell_orders,
    )


def calculate_portfolio_reward(
    target_asset_weights: np.ndarray,
    target_cash_weight: float,
    asset_returns: np.ndarray,
    turnover: PortfolioTurnover,
    buy_cost: float = 0.0,
    sell_cost: float = 0.0,
    min_cost: float = 0.0,
    turnover_penalty_rate: float = 0.0,
    portfolio_value: float = 1.0,
    missing_return_value: float = 0.0,
    tolerance: float = 1e-8,
) -> PortfolioReward:
    """Calculate net portfolio return and charge transaction cost once.

    Missing asset returns are replaced by ``missing_return_value`` and their
    total target weight is reported. The initial engineering configuration
    uses zero, which carries the execution valuation forward.
    """

    weights = np.asarray(target_asset_weights, dtype=np.float64)
    returns = np.asarray(asset_returns, dtype=np.float64)
    if weights.ndim != 1 or returns.shape != weights.shape:
        raise ValueError("Asset weights and returns must be one-dimensional arrays with equal shape.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Target asset weights must be finite and non-negative.")
    if not np.isfinite(target_cash_weight) or target_cash_weight < 0.0:
        raise ValueError("Target cash weight must be finite and non-negative.")
    if not np.isclose(weights.sum() + target_cash_weight, 1.0, atol=tolerance, rtol=0.0):
        raise ValueError("Target asset and cash weights must sum to one.")
    if not np.isfinite(buy_cost) or buy_cost < 0.0 or not np.isfinite(sell_cost) or sell_cost < 0.0:
        raise ValueError("Transaction-cost rates must be finite and non-negative.")
    if not np.isfinite(min_cost) or min_cost < 0.0:
        raise ValueError("min_cost must be a finite non-negative dollar amount.")
    if not np.isfinite(turnover_penalty_rate) or turnover_penalty_rate < 0.0:
        raise ValueError("turnover_penalty_rate must be finite and non-negative.")
    if not np.isfinite(portfolio_value) or portfolio_value <= 0.0:
        raise ValueError("portfolio_value must be positive and finite.")
    if not np.isfinite(missing_return_value) or missing_return_value < -1.0:
        raise ValueError("missing_return_value must be finite and no less than -1.")

    finite_returns = np.isfinite(returns)
    if np.any(returns[finite_returns] < -1.0):
        raise ValueError("Asset returns cannot be less than -1.")

    effective_returns = np.where(finite_returns, returns, missing_return_value)
    missing_return_weight = float(weights[~finite_returns].sum())
    gross_return = float(np.dot(weights, effective_returns))
    buy_fees = _order_fees(turnover.buy_orders, portfolio_value, buy_cost, min_cost)
    sell_fees = _order_fees(turnover.sell_orders, portfolio_value, sell_cost, min_cost)
    transaction_cost = float((buy_fees.sum() + sell_fees.sum()) / portfolio_value)
    net_return = gross_return - transaction_cost
    turnover_penalty = turnover_penalty_rate * turnover.one_way
    learning_reward = net_return - turnover_penalty

    return PortfolioReward(
        gross_return=gross_return,
        transaction_cost=transaction_cost,
        turnover_penalty=turnover_penalty,
        net_return=net_return,
        learning_reward=learning_reward,
        missing_return_weight=missing_return_weight,
        effective_asset_returns=effective_returns,
    )


def _order_fees(order_weights: np.ndarray, portfolio_value: float, rate: float, min_cost: float) -> np.ndarray:
    order_values = np.asarray(order_weights, dtype=np.float64) * portfolio_value
    return np.where(order_values > 0.0, np.maximum(order_values * rate, min_cost), 0.0)
