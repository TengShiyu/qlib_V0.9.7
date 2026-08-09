# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Diagnostics for trained portfolio policies and deterministic baselines."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .benchmark import BenchmarkResult


def result_diagnostics(result: BenchmarkResult) -> Mapping[str, float]:
    """Summarize action behavior, costs, exposure, and concentration."""

    transitions = result.transitions
    actions = transitions["action"].astype(str).to_numpy()
    total = len(actions)
    counts = transitions["action"].value_counts()
    persistence = float(np.mean(actions[1:] == actions[:-1])) if total > 1 else 0.0
    gross_total_return = float(np.prod(1.0 + transitions["gross_return"].to_numpy()) - 1.0)
    net_total_return = float(np.prod(1.0 + transitions["net_return"].to_numpy()) - 1.0)
    return {
        "transition_count": float(total),
        "unique_action_count": float(len(counts)),
        "dominant_action_frequency": float(counts.iloc[0] / total),
        "action_persistence": persistence,
        "gross_total_return": gross_total_return,
        "net_total_return": net_total_return,
        "cost_return_drag": gross_total_return - net_total_return,
        "cumulative_turnover": float(transitions["turnover"].sum()),
        "cumulative_transaction_cost": float(transitions["transaction_cost"].sum()),
        "average_cash_exposure": float(transitions["cash_exposure"].mean()),
        "average_max_asset_weight": float(transitions["max_asset_weight"].mean()),
        "average_asset_hhi": float(transitions["asset_hhi"].mean()),
        "max_drawdown": float(result.metrics["max_drawdown"]),
    }


def regime_diagnostics(
    results: Mapping[str, BenchmarkResult],
    market: BenchmarkResult,
) -> pd.DataFrame:
    """Compare strategy returns during positive and non-positive market transitions."""

    market_returns = market.transitions.set_index("reward_end_date")["net_return"]
    records = []
    for strategy, result in results.items():
        transitions = result.transitions.set_index("reward_end_date")
        aligned_market = market_returns.reindex(transitions.index)
        if aligned_market.isna().any():
            raise ValueError(f"Market returns do not cover every {strategy} transition.")
        for regime, mask in {
            "market_up": aligned_market > 0.0,
            "market_down_or_flat": aligned_market <= 0.0,
        }.items():
            returns = transitions.loc[mask.to_numpy(), "net_return"].to_numpy(dtype=np.float64)
            records.append(
                {
                    "strategy": strategy,
                    "regime": regime,
                    "transition_count": len(returns),
                    "mean_return": float(returns.mean()) if len(returns) else np.nan,
                    "compounded_return": float(np.prod(1.0 + returns) - 1.0) if len(returns) else np.nan,
                }
            )
    return pd.DataFrame.from_records(records)


def cost_sensitivity(
    reproduction: Mapping[str, BenchmarkResult],
    sensitivity: Mapping[str, BenchmarkResult],
) -> pd.DataFrame:
    """Report how configured costs change each strategy's total return."""

    if set(reproduction) != set(sensitivity):
        raise ValueError("Cost scenarios must contain the same strategies.")
    records = []
    for strategy in reproduction:
        zero_return = reproduction[strategy].metrics["total_return"]
        cost_return = sensitivity[strategy].metrics["total_return"]
        records.append(
            {
                "strategy": strategy,
                "reproduction_total_return": zero_return,
                "cost_sensitivity_total_return": cost_return,
                "return_difference": cost_return - zero_return,
            }
        )
    return pd.DataFrame.from_records(records)
