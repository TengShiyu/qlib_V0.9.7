# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import numpy as np
import pandas as pd

from qlib.rl.interpreter import GymSpaceValidationError
from qlib.rl.portfolio import OBSERVATION_DIM, PortfolioAction, make_portfolio_env
from qlib.rl.portfolio.data import PortfolioDataSplit
from qlib.rl.portfolio.simulator import PortfolioSimulatorConfig


def make_data(asset_returns: np.ndarray) -> PortfolioDataSplit:
    returns = np.asarray(asset_returns, dtype=np.float64)
    decision_count, instrument_count = returns.shape
    calendar = pd.bdate_range("2025-01-02", periods=decision_count * 2 + 2)
    decisions = calendar[np.arange(0, decision_count * 2, 2)]
    executions = calendar[np.arange(1, decision_count * 2 + 1, 2)]
    reward_ends = calendar[np.arange(3, decision_count * 2 + 3, 2)]
    scores = np.tile(np.linspace(0.1, 0.2, instrument_count), (decision_count, 1))
    volatility = np.full_like(scores, 0.02)
    tradable = np.ones_like(scores, dtype=np.bool_)
    execution_close = np.full_like(scores, 100.0)
    reward_end_close = execution_close * (1.0 + returns)
    return PortfolioDataSplit(
        instruments=tuple(f"STOCK{i}" for i in range(instrument_count)),
        decision_dates=decisions,
        execution_dates=executions,
        reward_end_dates=reward_ends,
        scores=scores,
        volatility=volatility,
        observation_tradable=tradable.copy(),
        execution_tradable=tradable,
        execution_close=execution_close,
        reward_end_close=reward_end_close,
        asset_returns=returns,
    )


class PortfolioIntegrationTest(unittest.TestCase):
    def test_environment_returns_penalized_reward_without_reducing_account(self) -> None:
        env = make_portfolio_env(
            make_data(np.zeros((1, 2))),
            simulator_config=PortfolioSimulatorConfig(turnover_penalty=0.002),
        )
        env.reset()

        _, reward, done, _ = env.step(PortfolioAction.EQUAL_WEIGHT.value)
        transition = env.simulator.get_state().last_transition

        self.assertTrue(done)
        self.assertAlmostEqual(reward, -0.002)
        self.assertAlmostEqual(transition.reward.net_return, 0.0)
        self.assertAlmostEqual(transition.reward.turnover_penalty, 0.002)
        self.assertAlmostEqual(transition.ending_value, 100_000.0)

    def test_environment_spaces_and_random_episode(self) -> None:
        data = make_data(np.array([[0.01, 0.02], [-0.01, 0.03], [0.02, 0.01]]))
        env = make_portfolio_env(data)

        observation = env.reset()
        self.assertEqual(observation.shape, (OBSERVATION_DIM,))
        self.assertEqual(OBSERVATION_DIM, 65)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(env.action_space.n, len(PortfolioAction))

        rng = np.random.default_rng(7)
        rewards = []
        done = False
        while not done:
            observation, reward, done, info = env.step(int(rng.integers(env.action_space.n)))
            self.assertTrue(env.observation_space.contains(observation))
            self.assertTrue(np.isfinite(reward))
            self.assertIn("reward", info["log"])
            rewards.append(reward)

        self.assertEqual(len(rewards), len(data.decision_dates))
        self.assertTrue(env.simulator.done())

    def test_initial_observation_does_not_use_forward_returns(self) -> None:
        positive = make_portfolio_env(make_data(np.full((2, 2), 0.50))).reset()
        negative = make_portfolio_env(make_data(np.full((2, 2), -0.50))).reset()
        np.testing.assert_array_equal(positive, negative)

    def test_action_interpreter_rejects_out_of_range_action(self) -> None:
        env = make_portfolio_env(make_data(np.zeros((1, 2))))
        env.reset()
        with self.assertRaises(GymSpaceValidationError):
            env.step(len(PortfolioAction))


if __name__ == "__main__":
    unittest.main()
