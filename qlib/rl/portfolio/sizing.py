# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Freeze share targets using decision-time prices, before simulating fills."""

import numpy as np


def size_target_amounts(current_amounts, target_weights, portfolio_value, prices, tradable, cost_rate=0.0):
    """Size fractional-share targets with a proportional fee reserve.

    Unavailable assets retain their existing quantities. Execution constraints
    and available cash may subsequently reduce fills, never revise the action.
    """
    amounts = np.asarray(current_amounts, dtype=np.float64).copy()
    prices = np.asarray(prices, dtype=np.float64)
    weights = np.asarray(target_weights, dtype=np.float64)
    eligible = np.asarray(tradable, dtype=bool) & np.isfinite(prices) & (prices > 0.0)
    amounts[eligible] = portfolio_value / (1.0 + cost_rate) * weights[eligible] / prices[eligible]
    return amounts
