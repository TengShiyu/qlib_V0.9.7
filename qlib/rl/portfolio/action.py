# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Translate discrete portfolio actions into feasible target weights.

This module is intentionally independent of the DQN implementation. A policy
selects one stable integer action, and :func:`build_target_weights` applies the
corresponding deterministic portfolio rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple, Union

import numpy as np


class PortfolioAction(IntEnum):
    """Stable action identifiers used by portfolio DQN checkpoints."""

    HOLD = 0
    CASH = 1
    EQUAL_WEIGHT = 2
    CONSERVATIVE_SCORE = 3
    AGGRESSIVE_SCORE = 4
    VOLATILITY_ADJUSTED = 5
    REDUCE_EXPOSURE = 6
    PARTIAL_REBALANCE = 7


@dataclass(frozen=True)
class PortfolioActionConfig:
    """Equity budgets used by portfolio-construction actions.

    ``standard_equity`` is used by volatility-adjusted and partial-rebalance
    actions. Cash receives any portfolio budget not allocated to risky assets.
    """

    conservative_equity: float = 0.50
    standard_equity: float = 0.80
    aggressive_equity: float = 0.95
    tolerance: float = 1e-8

    def __post_init__(self) -> None:
        budgets = (self.conservative_equity, self.standard_equity, self.aggressive_equity)
        if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in budgets):
            raise ValueError("Equity budgets must be finite values between zero and one.")
        if not self.conservative_equity <= self.standard_equity <= self.aggressive_equity:
            raise ValueError("Equity budgets must be ordered conservative <= standard <= aggressive.")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be a positive finite value.")


@dataclass(frozen=True)
class PortfolioTarget:
    """A feasible long-only target containing risky assets and cash."""

    asset_weights: np.ndarray
    cash_weight: float


def build_target_weights(
    action: Union[PortfolioAction, int],
    current_asset_weights: np.ndarray,
    current_cash_weight: float,
    scores: np.ndarray,
    tradable: np.ndarray,
    volatility: Optional[np.ndarray] = None,
    config: Optional[PortfolioActionConfig] = None,
) -> PortfolioTarget:
    """Construct a feasible target portfolio for one discrete action.

    Parameters
    ----------
    action
        One stable :class:`PortfolioAction` identifier.
    current_asset_weights
        Current risky-asset weights in the environment's stable instrument
        order.
    current_cash_weight
        Current cash weight. Current risky weights and cash must sum to one.
    scores
        Cross-sectional prediction scores. Non-finite scores are treated as
        unavailable, and only finite positive scores are eligible.
    tradable
        Boolean mask indicating which assets can be traded for this decision.
    volatility
        Positive backward-looking volatility estimates. Required only by the
        volatility-adjusted action; invalid values make an asset ineligible.
    config
        Optional equity-budget configuration.

    Notes
    -----
    Existing non-tradable positions are locked at their current weights. They
    cannot be sold and do not receive additional allocation. Any feasible
    budget not assigned to eligible assets remains in cash.
    """

    cfg = config or PortfolioActionConfig()
    resolved_action = _resolve_action(action)
    current, cash, score_values, tradable_mask, volatility_values = _validate_inputs(
        current_asset_weights=current_asset_weights,
        current_cash_weight=current_cash_weight,
        scores=scores,
        tradable=tradable,
        volatility=volatility,
        tolerance=cfg.tolerance,
    )

    if resolved_action == PortfolioAction.HOLD:
        return _finalize(current, cfg.tolerance)

    locked = np.where(tradable_mask, 0.0, current)

    if resolved_action == PortfolioAction.CASH:
        return _finalize(locked, cfg.tolerance)

    if resolved_action == PortfolioAction.REDUCE_EXPOSURE:
        target = locked + np.where(tradable_mask, current * 0.5, 0.0)
        return _finalize(target, cfg.tolerance)

    positive_and_tradable = tradable_mask & (score_values > 0.0)

    if resolved_action == PortfolioAction.EQUAL_WEIGHT:
        target = _weighted_target(locked, positive_and_tradable, np.ones_like(score_values), 1.0)
    elif resolved_action == PortfolioAction.CONSERVATIVE_SCORE:
        target = _weighted_target(locked, positive_and_tradable, score_values, cfg.conservative_equity)
    elif resolved_action == PortfolioAction.AGGRESSIVE_SCORE:
        target = _weighted_target(locked, positive_and_tradable, score_values, cfg.aggressive_equity)
    elif resolved_action == PortfolioAction.VOLATILITY_ADJUSTED:
        if volatility_values is None:
            raise ValueError("volatility is required for VOLATILITY_ADJUSTED.")
        valid_volatility = np.isfinite(volatility_values) & (volatility_values > 0.0)
        eligible = positive_and_tradable & valid_volatility
        adjusted_scores = np.divide(
            score_values,
            volatility_values,
            out=np.zeros_like(score_values),
            where=eligible,
        )
        target = _weighted_target(locked, eligible, adjusted_scores, cfg.standard_equity)
    elif resolved_action == PortfolioAction.PARTIAL_REBALANCE:
        standard_target = _weighted_target(locked, positive_and_tradable, score_values, cfg.standard_equity)
        target = 0.5 * current + 0.5 * standard_target
    else:  # pragma: no cover - protects callers if the enum grows without an implementation
        raise ValueError(f"Unsupported portfolio action: {resolved_action!r}")

    return _finalize(target, cfg.tolerance)


def _resolve_action(action: Union[PortfolioAction, int]) -> PortfolioAction:
    try:
        return PortfolioAction(action)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unknown portfolio action: {action!r}") from exc


def _validate_inputs(
    current_asset_weights: np.ndarray,
    current_cash_weight: float,
    scores: np.ndarray,
    tradable: np.ndarray,
    volatility: Optional[np.ndarray],
    tolerance: float,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    current = np.asarray(current_asset_weights, dtype=np.float64)
    score_values = np.asarray(scores, dtype=np.float64)
    tradable_values = np.asarray(tradable)

    if current.ndim != 1 or current.size == 0:
        raise ValueError("current_asset_weights must be a non-empty one-dimensional array.")
    if score_values.shape != current.shape or tradable_values.shape != current.shape:
        raise ValueError("scores and tradable must have the same shape as current_asset_weights.")
    if tradable_values.dtype != np.bool_:
        raise ValueError("tradable must be a boolean array.")
    if not np.all(np.isfinite(current)) or np.any(current < 0.0):
        raise ValueError("Current asset weights must be finite and non-negative.")
    if not np.isfinite(current_cash_weight) or current_cash_weight < 0.0:
        raise ValueError("Current cash weight must be finite and non-negative.")
    if not np.isclose(current.sum() + current_cash_weight, 1.0, atol=tolerance, rtol=0.0):
        raise ValueError("Current asset and cash weights must sum to one.")

    volatility_values = None if volatility is None else np.asarray(volatility, dtype=np.float64)
    if volatility_values is not None and volatility_values.shape != current.shape:
        raise ValueError("volatility must have the same shape as current_asset_weights.")

    clean_scores = np.where(np.isfinite(score_values), score_values, 0.0)
    return current.copy(), float(current_cash_weight), clean_scores, tradable_values.copy(), volatility_values


def _weighted_target(
    locked: np.ndarray,
    eligible: np.ndarray,
    allocation_values: np.ndarray,
    total_equity_budget: float,
) -> np.ndarray:
    target = locked.copy()
    locked_weight = float(locked.sum())
    available_weight = max(0.0, 1.0 - locked_weight)
    tradable_equity_budget = min(available_weight, max(0.0, total_equity_budget - locked_weight))

    eligible_values = np.where(eligible, allocation_values, 0.0)
    value_sum = float(eligible_values.sum())
    if tradable_equity_budget > 0.0 and value_sum > 0.0:
        target += eligible_values / value_sum * tradable_equity_budget
    return target


def _finalize(asset_weights: np.ndarray, tolerance: float) -> PortfolioTarget:
    weights = np.asarray(asset_weights, dtype=np.float64).copy()
    weights[np.abs(weights) < tolerance] = 0.0

    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Constructed asset weights must be finite and non-negative.")

    asset_total = float(weights.sum())
    if asset_total > 1.0 + tolerance:
        raise ValueError("Constructed asset weights exceed the portfolio budget.")

    cash = max(0.0, 1.0 - asset_total)
    if not np.isclose(asset_total + cash, 1.0, atol=tolerance, rtol=0.0):
        raise ValueError("Constructed portfolio does not sum to one.")
    return PortfolioTarget(asset_weights=weights, cash_weight=cash)
