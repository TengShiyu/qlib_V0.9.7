# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch

from qlib.rl.portfolio import OBSERVATION_DIM, PortfolioAction
from qlib.rl.portfolio.integration import make_portfolio_env
from qlib.rl.portfolio.policy import PortfolioDQNConfig, PortfolioQNetwork
from qlib.rl.portfolio.simulator import PortfolioSimulatorConfig
from qlib.rl.portfolio.training import (
    action_config_record,
    cumulative_learning_return,
    evaluate_dqn,
    load_dqn_checkpoint,
    save_dqn_checkpoint,
    simulator_config_record,
    train_dqn,
)
from qlib.rl.portfolio.action import PortfolioActionConfig
from tests.rl.portfolio.test_integration import make_data


class PortfolioDQNTest(unittest.TestCase):
    def test_shared_configuration_records_include_all_fields(self) -> None:
        action = PortfolioActionConfig(max_holdings=12, rotation_count=3)
        simulator = PortfolioSimulatorConfig(turnover_penalty=0.002)

        self.assertEqual(action_config_record(action)["rotation_count"], 3)
        self.assertEqual(action_config_record(action)["max_holdings"], 12)
        self.assertEqual(simulator_config_record(simulator)["turnover_penalty"], 0.002)
        self.assertIn("missing_return_value", simulator_config_record(simulator))

    def test_cumulative_learning_return_uses_penalized_reward(self) -> None:
        transitions = evaluate_dqn(
            train_dqn(
                make_data(np.zeros((2, 2))),
                make_data(np.zeros((1, 2))),
                dqn_config=PortfolioDQNConfig(
                    hidden_dims=(8,), replay_buffer_size=16, batch_size=1,
                    epochs=1, updates_per_epoch=1, target_update_freq=1,
                    epsilon_decay_epochs=1,
                ),
                simulator_config=PortfolioSimulatorConfig(turnover_penalty=0.002),
            ).policy,
            make_data(np.zeros((1, 2))),
            simulator_config=PortfolioSimulatorConfig(turnover_penalty=0.002),
        ).transitions

        self.assertAlmostEqual(
            cumulative_learning_return(transitions),
            float(np.prod(1.0 + transitions["learning_reward"]) - 1.0),
        )

    def test_training_and_validation_receive_same_turnover_penalty(self) -> None:
        train_data = make_data(np.array([[0.01, 0.00], [0.00, 0.01]]))
        valid_data = make_data(np.array([[0.00, 0.01]]))
        dqn_config = PortfolioDQNConfig(
            hidden_dims=(8,),
            replay_buffer_size=32,
            batch_size=2,
            epochs=1,
            updates_per_epoch=1,
            target_update_freq=1,
            epsilon_decay_epochs=1,
        )
        simulator_config = PortfolioSimulatorConfig(turnover_penalty=0.002)

        with mock.patch(
            "qlib.rl.portfolio.training.make_portfolio_env",
            wraps=make_portfolio_env,
        ) as environment_factory:
            result = train_dqn(
                train_data,
                valid_data,
                dqn_config=dqn_config,
                simulator_config=simulator_config,
            )

        configured_penalties = [
            call.kwargs["simulator_config"].turnover_penalty
            for call in environment_factory.call_args_list
        ]
        self.assertGreaterEqual(len(configured_penalties), 3)
        self.assertEqual(configured_penalties, [0.002] * len(configured_penalties))
        np.testing.assert_allclose(
            result.validation.transitions["turnover_penalty"],
            result.validation.transitions["one_way_turnover"] * 0.002,
        )

    def test_network_emits_one_raw_q_value_per_action(self) -> None:
        network = PortfolioQNetwork(hidden_dims=(16,))
        output, state = network(np.zeros((3, OBSERVATION_DIM), dtype=np.float32))

        self.assertEqual(tuple(output.shape), (3, len(PortfolioAction)))
        self.assertIsNone(state)
        self.assertFalse(any(isinstance(module, torch.nn.Softmax) for module in network.modules()))

    def test_training_checkpoint_reload_and_greedy_evaluation(self) -> None:
        train_data = make_data(np.array([[0.01, -0.01], [0.02, 0.00], [-0.01, 0.01]]))
        valid_data = make_data(np.array([[0.01, 0.00], [0.00, 0.01]]))
        config = PortfolioDQNConfig(
            hidden_dims=(16,),
            replay_buffer_size=64,
            batch_size=4,
            epochs=2,
            updates_per_epoch=2,
            target_update_freq=2,
            epsilon_decay_epochs=1,
        )
        result = train_dqn(train_data, valid_data, dqn_config=config)

        self.assertEqual(len(result.history), 2)
        self.assertIn(result.best_epoch, (1, 2))
        self.assertTrue(np.all(np.isfinite(result.history["mean_loss"])))
        self.assertIn("validation_learning_return", result.history)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_dqn_checkpoint(result.policy, Path(directory) / "policy.pt")
            restored = load_dqn_checkpoint(checkpoint, config)
            original_eval = evaluate_dqn(result.policy, valid_data)
            restored_eval = evaluate_dqn(restored, valid_data)

        expected_reward_columns = {
            "gross_return",
            "transaction_cost",
            "one_way_turnover",
            "turnover_penalty",
            "learning_reward",
            "portfolio_net_return",
        }
        self.assertTrue(expected_reward_columns.issubset(original_eval.transitions.columns))
        self.assertEqual(original_eval.action_counts, restored_eval.action_counts)
        self.assertEqual(original_eval.metrics, restored_eval.metrics)


if __name__ == "__main__":
    unittest.main()
