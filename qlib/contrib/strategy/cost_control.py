# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd

from .order_generator import OrderGenWInteract
from .signal_strategy import WeightStrategyBase


class SoftTopkStrategy(WeightStrategyBase):
    def __init__(
        self,
        model=None,
        dataset=None,
        topk=None,
        order_generator_cls_or_obj=OrderGenWInteract,
        max_sold_weight=1.0,
        trade_impact_limit=None,
        risk_degree=0.95,
        buy_method="first_fill",
        **kwargs,
    ):
        """
        Refactored SoftTopkStrategy with a budget-constrained rebalancing engine.

        Parameters
        ----------
        topk : int
            The number of top-N stocks to be held in the portfolio.
        trade_impact_limit : float
            Maximum weight change for each stock in one trade. If None, fallback to max_sold_weight.
        max_sold_weight : float
            Backward-compatible alias for trade_impact_limit. Use 1.0 to effectively disable the limit.
        risk_degree : float
            The target percentage of total value to be invested.
        """
        super(SoftTopkStrategy, self).__init__(
            model=model, dataset=dataset, order_generator_cls_or_obj=order_generator_cls_or_obj, **kwargs
        )

        self.topk = topk
        self.trade_impact_limit = trade_impact_limit if trade_impact_limit is not None else max_sold_weight
        self.risk_degree = risk_degree
        self.buy_method = buy_method

    def get_risk_degree(self, trade_step=None):
        return self.risk_degree

    def _normalize_score(self, score):
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]
        return score

    def _generate_target_weight_position(self, score, current, forced_exit=None):
        """
        Generates target position using Proportional Budget Allocation.
        Ensures deterministic sells and synchronized buys under impact limits.
        """

        if self.topk is None or self.topk <= 0:
            return {}

        score = self._normalize_score(score)

        def apply_impact_limit(weight):
            return weight if self.trade_impact_limit is None else min(weight, self.trade_impact_limit)

        ideal_per_stock = self.risk_degree / self.topk
        forced_exit = set() if forced_exit is None else set(forced_exit)
        ideal_list = [
            code for code in score.sort_values(ascending=False).index.tolist() if code not in forced_exit
        ][: self.topk]

        cur_weights = current.get_stock_weight_dict(only_stock=True)
        initial_total_weight = sum(cur_weights.values())

        # --- Case A: Cold Start ---
        if not cur_weights:
            fill = apply_impact_limit(ideal_per_stock)
            return {code: fill for code in ideal_list}

        # --- Case B: Rebalancing ---
        all_tickers = set(cur_weights.keys()) | set(ideal_list)
        next_weights = {t: cur_weights.get(t, 0.0) for t in all_tickers}

        # Phase 1: Deterministic Sell Phase
        released_cash = 0.0
        for t in list(next_weights.keys()):
            cur = next_weights[t]
            if cur <= 1e-8:
                continue

            if t in forced_exit:
                sell = apply_impact_limit(cur)
                next_weights[t] -= sell
                released_cash += sell
            elif t not in ideal_list:
                sell = apply_impact_limit(cur)
                next_weights[t] -= sell
                released_cash += sell
            elif cur > ideal_per_stock + 1e-8:
                excess = cur - ideal_per_stock
                sell = apply_impact_limit(excess)
                next_weights[t] -= sell
                released_cash += sell

        # Phase 2: Budget Calculation
        # Budget = Cash from sells + Available space from target risk degree
        total_budget = released_cash + (self.risk_degree - initial_total_weight)

        # Phase 3: Proportional Buy Allocation
        if total_budget > 1e-8:
            shortfalls = {
                t: (ideal_per_stock - next_weights.get(t, 0.0))
                for t in ideal_list
                if next_weights.get(t, 0.0) < ideal_per_stock - 1e-8
            }

            if shortfalls:
                total_shortfall = sum(shortfalls.values())
                # Normalize total_budget to not exceed total_shortfall
                available_to_spend = min(total_budget, total_shortfall)

                for t, shortfall in shortfalls.items():
                    # Every stock gets its fair share based on its distance to target
                    share_of_budget = (shortfall / total_shortfall) * available_to_spend

                    # Capped by impact limit
                    max_buy_cap = apply_impact_limit(shortfall)

                    next_weights[t] += min(share_of_budget, max_buy_cap)

        return {k: v for k, v in next_weights.items() if v > 1e-8}

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time, **kwargs):
        return self._generate_target_weight_position(score, current)


class SoftTopkStrategy_SMS(SoftTopkStrategy):
    # SMS: sell missing signal. This keeps SoftTopkStrategy's budget-constrained
    # rebalancing while forcing current holdings that have left the signal universe
    # into the sell path.
    def _get_forced_exit(self, score, current):
        score = self._normalize_score(score)
        cur_codes = set(current.get_stock_weight_dict(only_stock=True).keys())
        current_signal = set(score.index)
        forced_exit = {code for code in cur_codes if code not in current_signal}

        try:
            trade_step = self.trade_calendar.get_trade_step()
            prev_start_time, prev_end_time = self.trade_calendar.get_step_time(trade_step, shift=2)
            prev_score = self.signal.get_signal(start_time=prev_start_time, end_time=prev_end_time)
        except (AttributeError, IndexError, KeyError):
            prev_score = None

        if isinstance(prev_score, pd.DataFrame):
            prev_score = prev_score.iloc[:, 0]

        if prev_score is not None:
            left_signal_universe = set(prev_score.index) - current_signal
            forced_exit.update(code for code in cur_codes if code in left_signal_universe)

        return forced_exit

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time, **kwargs):
        forced_exit = self._get_forced_exit(score, current)
        return self._generate_target_weight_position(score, current, forced_exit=forced_exit)
