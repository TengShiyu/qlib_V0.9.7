# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Freeze cash-aware requested orders using decision-session information only."""

from dataclasses import dataclass

import numpy as np

from qlib.backtest.decision import Order


@dataclass
class RequestedOrderPlan:
    orders: list
    estimates: list[dict]
    estimated_cash_after: float
    adjustments: list[dict]


def plan_requested_orders(exchange, target_amounts, current_amounts, cash,
                          decision_start, decision_end, execution_start, execution_end):
    """Use Qlib eligibility/rounding, then reserve fees and cap buys at T prices.

    Quantities are Qlib adjusted amounts. Sells fund buys in stable instrument
    order. Cash estimates assume all preceding requests fill at the reference
    close; execution must still enforce actual cash, prices and availability.
    No position is mutated and no exchange deal_order method is called here.
    """
    remaining = float(cash)
    costs = (float(exchange.open_cost), float(exchange.close_cost), float(exchange.min_cost))
    if not np.isfinite(remaining) or remaining < 0 or any(not np.isfinite(c) or c < 0 for c in costs):
        raise ValueError("Cash and commission settings must be finite and non-negative.")
    if float(exchange.impact_cost) != 0:
        raise ValueError("Decision-time order planning currently requires impact_cost=0.")
    buy_cost, sell_cost, min_cost = costs
    proposed = exchange.generate_order_for_target_amount_position(
        target_position=target_amounts, current_position=current_amounts,
        start_time=decision_start, end_time=decision_end,
    )
    proposed.sort(key=lambda order: (order.direction, order.stock_id))
    orders, estimates, adjustments = [], [], []
    for order in proposed:
        amount = original = float(order.amount)
        price = exchange.get_close(order.stock_id, decision_start, decision_end)
        if price is None or not np.isfinite(price) or price <= 0 or not np.isfinite(amount) or amount <= 0:
            raise ValueError("A requested order has an invalid decision price or quantity.")
        price = float(price)
        if order.direction == Order.SELL:
            amount = min(amount, float(current_amounts.get(order.stock_id, 0.0)))
            fee = max(amount * price * sell_cost, min_cost)
            if remaining + amount * price < fee:
                amount = 0.0
            reason = "sell exceeds holdings or available cash cannot cover its commission"
        else:
            budget = max(0.0, min(remaining / (1.0 + buy_cost), remaining - min_cost))
            if amount * price > budget:
                amount = budget / price
                # Order.factor is populated by Qlib at execution, so resolve
                # the sizing factor explicitly at the decision session here.
                unit = exchange.get_amount_of_trade_unit(
                    stock_id=order.stock_id, start_time=decision_start, end_time=decision_end,
                )
                # Strict downward rounding prevents Qlib's rounding tolerance
                # from making a capped order exceed the available cash.
                if unit is not None:
                    amount = np.floor(amount / unit) * unit
            reason = "buy reduced to estimated available cash after commissions"
        if amount != original:
            adjustments.append({"instrument": order.stock_id, "requested_amount": original,
                                "planned_amount": float(amount), "reason": reason})
        if amount <= 0:
            continue
        value = float(amount * price)
        fee = max(value * (sell_cost if order.direction == Order.SELL else buy_cost), min_cost)
        remaining += value - fee if order.direction == Order.SELL else -value - fee
        if remaining < -1e-8:
            raise ValueError("Requested order plan exceeds estimated available cash.")
        remaining = max(0.0, remaining)
        order.amount = float(amount)
        order.start_time, order.end_time = execution_start, execution_end
        orders.append(order)
        estimates.append({"reference_price": price, "estimated_notional": value, "estimated_fee": fee})
    return RequestedOrderPlan(orders, estimates, remaining, adjustments)
