# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import numpy as np

from qlib.rl.portfolio import PortfolioAction, PortfolioActionConfig, build_target_weights


class PortfolioActionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = {
            "current_asset_weights": np.array([0.20, 0.10, 0.10, 0.00]),
            "current_cash_weight": 0.60,
            "scores": np.array([4.0, 2.0, -1.0, 1.0]),
            "tradable": np.array([True, False, True, True]),
            "volatility": np.array([0.20, 0.10, 0.30, 0.40]),
        }

    def test_all_actions_produce_feasible_deterministic_targets(self) -> None:
        for action in PortfolioAction:
            with self.subTest(action=action):
                first = build_target_weights(action, **self.inputs)
                second = build_target_weights(action, **self.inputs)

                np.testing.assert_array_equal(first.asset_weights, second.asset_weights)
                self.assertEqual(first.cash_weight, second.cash_weight)
                self.assertTrue(np.all(np.isfinite(first.asset_weights)))
                self.assertTrue(np.all(first.asset_weights >= 0.0))
                self.assertAlmostEqual(float(first.asset_weights.sum()) + first.cash_weight, 1.0)
                self.assertAlmostEqual(first.asset_weights[1], self.inputs["current_asset_weights"][1])

    def test_hold_preserves_current_portfolio(self) -> None:
        target = build_target_weights(PortfolioAction.HOLD, **self.inputs)

        np.testing.assert_array_equal(target.asset_weights, self.inputs["current_asset_weights"])
        self.assertAlmostEqual(target.cash_weight, self.inputs["current_cash_weight"])

    def test_existing_action_ids_remain_stable_and_new_ids_are_appended(self) -> None:
        self.assertEqual(PortfolioAction.PARTIAL_REBALANCE.value, 7)
        self.assertEqual(PortfolioAction.INCREASE_EXPOSURE.value, 8)
        self.assertEqual(PortfolioAction.TOP_SCORE_CONCENTRATED.value, 9)
        self.assertEqual(PortfolioAction.ROTATE_WORST_TO_BEST.value, 10)
        self.assertEqual(PortfolioAction.DEFENSIVE_VOLATILITY.value, 11)
        self.assertEqual(len(PortfolioAction), 12)

    def test_cash_keeps_only_locked_non_tradable_positions(self) -> None:
        target = build_target_weights(PortfolioAction.CASH, **self.inputs)

        np.testing.assert_allclose(target.asset_weights, [0.0, 0.1, 0.0, 0.0])
        self.assertAlmostEqual(target.cash_weight, 0.9)

    def test_equal_weight_uses_all_available_budget_for_eligible_assets(self) -> None:
        target = build_target_weights(PortfolioAction.EQUAL_WEIGHT, **self.inputs)

        np.testing.assert_allclose(target.asset_weights, [0.45, 0.10, 0.0, 0.45])
        self.assertAlmostEqual(target.cash_weight, 0.0)

    def test_score_actions_respect_configured_equity_budgets(self) -> None:
        config = PortfolioActionConfig(conservative_equity=0.4, standard_equity=0.7, aggressive_equity=0.9)

        conservative = build_target_weights(PortfolioAction.CONSERVATIVE_SCORE, config=config, **self.inputs)
        aggressive = build_target_weights(PortfolioAction.AGGRESSIVE_SCORE, config=config, **self.inputs)

        self.assertAlmostEqual(float(conservative.asset_weights.sum()), 0.4)
        self.assertAlmostEqual(conservative.cash_weight, 0.6)
        self.assertAlmostEqual(float(aggressive.asset_weights.sum()), 0.9)
        self.assertAlmostEqual(aggressive.cash_weight, 0.1)

    def test_max_holdings_keeps_highest_positive_scores(self) -> None:
        target = build_target_weights(
            PortfolioAction.EQUAL_WEIGHT,
            current_asset_weights=np.zeros(5),
            current_cash_weight=1.0,
            scores=np.array([1.0, 5.0, 3.0, 4.0, 2.0]),
            tradable=np.ones(5, dtype=np.bool_),
            volatility=np.ones(5),
            config=PortfolioActionConfig(max_holdings=2),
        )

        np.testing.assert_allclose(target.asset_weights, [0.0, 0.5, 0.0, 0.5, 0.0])
        self.assertEqual(np.count_nonzero(target.asset_weights), 2)

    def test_non_tradable_asset_is_excluded_before_score_normalization(self) -> None:
        target = build_target_weights(PortfolioAction.AGGRESSIVE_SCORE, **self.inputs)

        # The positive score of the locked second asset must not dilute the
        # allocation divided between the first and fourth assets.
        np.testing.assert_allclose(target.asset_weights, [0.68, 0.10, 0.0, 0.17])
        self.assertAlmostEqual(target.cash_weight, 0.05)

    def test_volatility_adjustment_uses_score_to_volatility_ratio(self) -> None:
        target = build_target_weights(PortfolioAction.VOLATILITY_ADJUSTED, **self.inputs)

        # The locked position is 0.1. The remaining 0.7 equity budget is split
        # using ratios 4/0.2=20 and 1/0.4=2.5.
        np.testing.assert_allclose(target.asset_weights, [0.6222222222, 0.1, 0.0, 0.0777777778])
        self.assertAlmostEqual(target.cash_weight, 0.2)

    def test_reduce_exposure_halves_only_tradable_positions(self) -> None:
        target = build_target_weights(PortfolioAction.REDUCE_EXPOSURE, **self.inputs)

        np.testing.assert_allclose(target.asset_weights, [0.1, 0.1, 0.05, 0.0])
        self.assertAlmostEqual(target.cash_weight, 0.75)

    def test_partial_rebalance_moves_halfway_to_standard_target(self) -> None:
        target = build_target_weights(PortfolioAction.PARTIAL_REBALANCE, **self.inputs)

        np.testing.assert_allclose(target.asset_weights, [0.38, 0.1, 0.05, 0.07])
        self.assertAlmostEqual(target.cash_weight, 0.4)

    def test_increase_exposure_deploys_configured_share_of_cash(self) -> None:
        target = build_target_weights(PortfolioAction.INCREASE_EXPOSURE, **self.inputs)

        np.testing.assert_allclose(target.asset_weights, [0.35, 0.10, 0.10, 0.0])
        self.assertAlmostEqual(target.cash_weight, 0.45)

    def test_increase_exposure_never_adds_to_negative_score_holding(self) -> None:
        target = build_target_weights(PortfolioAction.INCREASE_EXPOSURE, **self.inputs)

        self.assertAlmostEqual(target.asset_weights[2], self.inputs["current_asset_weights"][2])
        self.assertGreater(target.asset_weights[0], self.inputs["current_asset_weights"][0])

    def test_concentrated_action_uses_only_highest_scored_names(self) -> None:
        target = build_target_weights(
            PortfolioAction.TOP_SCORE_CONCENTRATED,
            current_asset_weights=np.zeros(6),
            current_cash_weight=1.0,
            scores=np.array([1.0, 6.0, 4.0, 5.0, 2.0, 3.0]),
            tradable=np.ones(6, dtype=np.bool_),
            volatility=np.ones(6),
            config=PortfolioActionConfig(concentrated_holdings=3),
        )

        np.testing.assert_allclose(target.asset_weights, [0.0, 0.38, 0.2533333333, 0.3166666667, 0.0, 0.0])
        self.assertAlmostEqual(target.cash_weight, 0.05)

    def test_rotate_replaces_weakest_tradable_holdings(self) -> None:
        target = build_target_weights(PortfolioAction.ROTATE_WORST_TO_BEST, **self.inputs)

        np.testing.assert_allclose(target.asset_weights, [0.20, 0.10, 0.0, 0.10])
        self.assertAlmostEqual(target.cash_weight, 0.60)

    def test_rotate_sells_only_as_many_holdings_as_it_can_replace(self) -> None:
        target = build_target_weights(PortfolioAction.ROTATE_WORST_TO_BEST, **self.inputs)

        self.assertAlmostEqual(float(target.asset_weights.sum()), 0.40)
        self.assertEqual(np.count_nonzero(target.asset_weights), 3)

    def test_defensive_volatility_uses_lower_equity_budget(self) -> None:
        target = build_target_weights(PortfolioAction.DEFENSIVE_VOLATILITY, **self.inputs)

        np.testing.assert_allclose(target.asset_weights, [0.3555555556, 0.10, 0.0, 0.0444444444])
        self.assertAlmostEqual(target.cash_weight, 0.50)

    def test_no_eligible_assets_places_available_budget_in_cash(self) -> None:
        self.inputs["scores"] = np.array([-1.0, 2.0, np.nan, 0.0])
        target = build_target_weights(PortfolioAction.AGGRESSIVE_SCORE, **self.inputs)

        np.testing.assert_allclose(target.asset_weights, [0.0, 0.1, 0.0, 0.0])
        self.assertAlmostEqual(target.cash_weight, 0.9)

    def test_locked_weight_above_equity_budget_is_preserved(self) -> None:
        self.inputs["current_asset_weights"] = np.array([0.05, 0.60, 0.05, 0.00])
        self.inputs["current_cash_weight"] = 0.30
        target = build_target_weights(PortfolioAction.CONSERVATIVE_SCORE, **self.inputs)

        np.testing.assert_allclose(target.asset_weights, [0.0, 0.6, 0.0, 0.0])
        self.assertAlmostEqual(target.cash_weight, 0.4)

    def test_volatility_is_required_for_adjusted_action(self) -> None:
        self.inputs["volatility"] = None

        with self.assertRaisesRegex(ValueError, "volatility is required"):
            build_target_weights(PortfolioAction.VOLATILITY_ADJUSTED, **self.inputs)

    def test_invalid_volatility_makes_assets_ineligible(self) -> None:
        self.inputs["volatility"] = np.array([np.nan, 0.1, 0.3, 0.0])
        target = build_target_weights(PortfolioAction.VOLATILITY_ADJUSTED, **self.inputs)

        np.testing.assert_allclose(target.asset_weights, [0.0, 0.1, 0.0, 0.0])
        self.assertAlmostEqual(target.cash_weight, 0.9)

    def test_invalid_current_portfolio_is_rejected(self) -> None:
        self.inputs["current_cash_weight"] = 0.5

        with self.assertRaisesRegex(ValueError, "sum to one"):
            build_target_weights(PortfolioAction.HOLD, **self.inputs)

    def test_non_boolean_tradable_mask_is_rejected(self) -> None:
        self.inputs["tradable"] = np.array([1, 0, 1, 1])

        with self.assertRaisesRegex(ValueError, "boolean array"):
            build_target_weights(PortfolioAction.HOLD, **self.inputs)

    def test_unknown_action_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown portfolio action"):
            build_target_weights(99, **self.inputs)

    def test_inputs_are_not_mutated(self) -> None:
        originals = {key: value.copy() for key, value in self.inputs.items() if isinstance(value, np.ndarray)}

        build_target_weights(PortfolioAction.VOLATILITY_ADJUSTED, **self.inputs)

        for key, original in originals.items():
            np.testing.assert_array_equal(self.inputs[key], original)


if __name__ == "__main__":
    unittest.main()
