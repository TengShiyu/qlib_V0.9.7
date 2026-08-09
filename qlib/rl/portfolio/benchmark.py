# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Deterministic benchmark evaluation for the portfolio RL simulator."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from .action import PortfolioAction, PortfolioActionConfig
from .data import PortfolioDataSplit
from .simulator import PortfolioSimulator, PortfolioSimulatorConfig


PERIODS_PER_YEAR = 126


@dataclass(frozen=True)
class BenchmarkResult:
    """Metrics and transition history for one benchmark episode."""

    name: str
    metrics: Mapping[str, float]
    transitions: pd.DataFrame
    action_counts: Mapping[str, int]


def run_action_benchmark(
    name: str,
    data: PortfolioDataSplit,
    action: Optional[PortfolioAction] = None,
    simulator_config: PortfolioSimulatorConfig = PortfolioSimulatorConfig(),
    action_config: PortfolioActionConfig = PortfolioActionConfig(),
    random_seed: int = 7,
) -> BenchmarkResult:
    """Run one constant-action or seeded-random benchmark episode."""

    if (name == "random") == (action is not None):
        raise ValueError("Use action=None only for the random benchmark.")

    simulator = PortfolioSimulator(data, config=simulator_config, action_config=action_config)
    rng = np.random.default_rng(random_seed)
    records = []
    action_counts: Counter[str] = Counter()
    while not simulator.done():
        selected = PortfolioAction(int(rng.integers(len(PortfolioAction)))) if action is None else action
        transition = simulator.step(selected)
        action_counts[selected.name] += 1
        records.append(
            {
                "decision_date": transition.decision_date,
                "execution_date": transition.execution_date,
                "reward_end_date": transition.reward_end_date,
                "action": selected.name,
                "gross_return": transition.reward.gross_return,
                "net_return": transition.reward.net_return,
                "portfolio_value": transition.ending_value,
                "turnover": transition.turnover.one_way,
                "transaction_cost": transition.reward.transaction_cost,
                "cash_exposure": transition.target.cash_weight,
                "max_asset_weight": float(transition.target.asset_weights.max(initial=0.0)),
                "asset_hhi": float(np.square(transition.target.asset_weights).sum()),
                "missing_return_weight": transition.reward.missing_return_weight,
            }
        )

    transitions = pd.DataFrame.from_records(records)
    metrics = calculate_metrics(
        returns=transitions["net_return"].to_numpy(),
        portfolio_values=transitions["portfolio_value"].to_numpy(),
        initial_value=simulator_config.initial_value,
        turnover=transitions["turnover"].to_numpy(),
        transaction_cost=transitions["transaction_cost"].to_numpy(),
        cash_exposure=transitions["cash_exposure"].to_numpy(),
        missing_return_weight=transitions["missing_return_weight"].to_numpy(),
    )
    return BenchmarkResult(name=name, metrics=metrics, transitions=transitions, action_counts=dict(action_counts))


def run_market_benchmark(name: str, data: PortfolioDataSplit, close: pd.Series) -> BenchmarkResult:
    """Evaluate a fully invested market index over the portfolio transition dates."""

    prices = close.copy()
    prices.index = pd.DatetimeIndex(prices.index).normalize()
    prices = prices[~prices.index.duplicated(keep="last")].sort_index().astype(float)
    start_prices = prices.reindex(data.execution_dates).to_numpy(dtype=np.float64)
    end_prices = prices.reindex(data.reward_end_dates).to_numpy(dtype=np.float64)
    valid = np.isfinite(start_prices) & (start_prices > 0.0) & np.isfinite(end_prices) & (end_prices > 0.0)
    if not np.all(valid):
        missing_dates = data.execution_dates[~valid] if np.any(~valid) else []
        raise ValueError(f"Market benchmark is missing valid prices for transition dates: {list(missing_dates)}")

    returns = end_prices / start_prices - 1.0
    values = np.cumprod(1.0 + returns) * 100_000.0
    transitions = pd.DataFrame(
        {
            "decision_date": data.decision_dates,
            "execution_date": data.execution_dates,
            "reward_end_date": data.reward_end_dates,
            "action": name,
            "gross_return": returns,
            "net_return": returns,
            "portfolio_value": values,
            "turnover": np.zeros(len(returns)),
            "transaction_cost": np.zeros(len(returns)),
            "cash_exposure": np.zeros(len(returns)),
            "max_asset_weight": np.ones(len(returns)),
            "asset_hhi": np.ones(len(returns)),
            "missing_return_weight": np.zeros(len(returns)),
        }
    )
    metrics = calculate_metrics(
        returns=returns,
        portfolio_values=values,
        initial_value=100_000.0,
        turnover=np.zeros(len(returns)),
        transaction_cost=np.zeros(len(returns)),
        cash_exposure=np.zeros(len(returns)),
        missing_return_weight=np.zeros(len(returns)),
    )
    return BenchmarkResult(name=name, metrics=metrics, transitions=transitions, action_counts={name: len(returns)})


def run_required_benchmarks(
    data: PortfolioDataSplit,
    simulator_config: PortfolioSimulatorConfig = PortfolioSimulatorConfig(),
    market_close: Optional[pd.Series] = None,
    random_seed: int = 7,
) -> Dict[str, BenchmarkResult]:
    """Run all contract-required portfolio rules and an optional market index."""

    default_action_config = PortfolioActionConfig()
    standard_action_config = PortfolioActionConfig(
        conservative_equity=default_action_config.standard_equity,
        standard_equity=default_action_config.standard_equity,
        aggressive_equity=default_action_config.aggressive_equity,
    )
    specifications = {
        "cash": (PortfolioAction.CASH, default_action_config),
        "hold": (PortfolioAction.HOLD, default_action_config),
        "equal_weight": (PortfolioAction.EQUAL_WEIGHT, default_action_config),
        "standard_score": (PortfolioAction.CONSERVATIVE_SCORE, standard_action_config),
        "volatility_adjusted": (PortfolioAction.VOLATILITY_ADJUSTED, default_action_config),
        "random": (None, default_action_config),
    }
    results = {
        name: run_action_benchmark(
            name=name,
            data=data,
            action=action,
            simulator_config=simulator_config,
            action_config=action_config,
            random_seed=random_seed,
        )
        for name, (action, action_config) in specifications.items()
    }
    if market_close is not None:
        results["market"] = run_market_benchmark("market", data, market_close)
    return results


def results_frame(results: Mapping[str, BenchmarkResult]) -> pd.DataFrame:
    """Convert named benchmark results to a stable metrics table."""

    return pd.DataFrame({name: result.metrics for name, result in results.items()}).T.rename_axis("benchmark")


def calculate_metrics(
    returns: np.ndarray,
    portfolio_values: np.ndarray,
    initial_value: float,
    turnover: np.ndarray,
    transaction_cost: np.ndarray,
    cash_exposure: np.ndarray,
    missing_return_weight: np.ndarray,
) -> Dict[str, float]:
    """Calculate metrics for non-overlapping two-session returns."""

    returns = _finite_vector("returns", returns)
    values = _finite_vector("portfolio_values", portfolio_values)
    if len(returns) == 0 or len(values) != len(returns):
        raise ValueError("returns and portfolio_values must have equal nonzero length.")
    if not np.isfinite(initial_value) or initial_value <= 0.0:
        raise ValueError("initial_value must be positive and finite.")

    total_return = float(values[-1] / initial_value - 1.0)
    annualized_return = float((values[-1] / initial_value) ** (PERIODS_PER_YEAR / len(returns)) - 1.0)
    annualized_volatility = float(returns.std(ddof=0) * np.sqrt(PERIODS_PER_YEAR))
    annualized_mean = float(returns.mean() * PERIODS_PER_YEAR)
    sharpe_ratio = annualized_mean / annualized_volatility if annualized_volatility > 0.0 else 0.0
    wealth = np.concatenate([[initial_value], values])
    running_peak = np.maximum.accumulate(wealth)
    max_drawdown = float(np.min(wealth / running_peak - 1.0))

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": max_drawdown,
        "cumulative_turnover": float(_finite_vector("turnover", turnover).sum()),
        "transaction_cost": float(_finite_vector("transaction_cost", transaction_cost).sum()),
        "average_cash_exposure": float(_finite_vector("cash_exposure", cash_exposure).mean()),
        "average_missing_return_weight": float(_finite_vector("missing_return_weight", missing_return_weight).mean()),
    }


def _finite_vector(name: str, values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a one-dimensional finite array.")
    return vector
