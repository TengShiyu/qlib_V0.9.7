# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch
from tianshou.data import Batch, ReplayBuffer

from qlib.rl.portfolio import OBSERVATION_DIM, PortfolioAction
from qlib.rl.portfolio.integration import make_portfolio_env
from qlib.rl.portfolio.policy import ACTION_MASK_VERSION, PortfolioDQNConfig, PortfolioQNetwork, portfolio_action_mask
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
from qlib.rl.portfolio.policy import make_dqn_policy
from qlib.rl.portfolio.action import PortfolioActionConfig
from tests.rl.portfolio.test_integration import make_data


class PortfolioDQNTest(unittest.TestCase):
    def rotation_policy(self):
        policy = make_dqn_policy(PortfolioDQNConfig(hidden_dims=(8,)))
        with torch.no_grad():
            for parameter in policy.model.parameters():
                parameter.zero_()
            policy.model.network[-1].bias[PortfolioAction.EQUAL_WEIGHT] = 5.0
            policy.model.network[-1].bias[PortfolioAction.ROTATE_WORST_TO_BEST] = 10.0
        policy.sync_weight()
        return policy

    def test_mask_only_excludes_rotation_without_stock_exposure(self):
        observations = np.zeros((4, OBSERVATION_DIM), dtype=np.float32)
        observations[:, 0] = [0., 1e-10, 1e-8, .5]
        mask = portfolio_action_mask(torch.from_numpy(observations))
        np.testing.assert_array_equal(mask[:, PortfolioAction.ROTATE_WORST_TO_BEST], [False, False, False, True])
        self.assertTrue(np.delete(mask, PortfolioAction.ROTATE_WORST_TO_BEST, axis=1).all())

    def test_greedy_selection_masks_rotation_without_mutating_raw_q_values_or_batch(self):
        policy = self.rotation_policy()
        observations = np.zeros((2, OBSERVATION_DIM), dtype=np.float32)
        observations[1, 0] = .5
        batch = Batch(obs=observations.copy(), info=Batch())
        output = policy(batch)
        np.testing.assert_array_equal(output.act, [PortfolioAction.EQUAL_WEIGHT, PortfolioAction.ROTATE_WORST_TO_BEST])
        np.testing.assert_array_equal(batch.obs, observations)
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertTrue((output.logits[:, PortfolioAction.ROTATE_WORST_TO_BEST] == 10.).all())

    def test_random_exploration_obeys_the_same_mask(self):
        policy = self.rotation_policy()
        policy.set_eps(1.0)
        observations = np.zeros((2, OBSERVATION_DIM), dtype=np.float32)
        observations[1, 0] = .5
        batch = Batch(obs=observations, info=Batch())
        actions = policy(batch).act
        random_scores = np.zeros((2, len(PortfolioAction)))
        random_scores[:, PortfolioAction.EQUAL_WEIGHT] = .8
        random_scores[:, PortfolioAction.ROTATE_WORST_TO_BEST] = .9
        with mock.patch("tianshou.policy.modelfree.dqn.np.random.rand",
                        side_effect=[np.zeros(2), random_scores]):
            explored = policy.exploration_noise(actions, batch)
        np.testing.assert_array_equal(explored, [PortfolioAction.EQUAL_WEIGHT, PortfolioAction.ROTATE_WORST_TO_BEST])
        self.assertIsInstance(batch.obs, np.ndarray)

    def test_double_dqn_target_masks_next_state_not_current_state(self):
        policy = self.rotation_policy()
        empty = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        held = empty.copy()
        held[0] = .5
        buffer = ReplayBuffer(4)
        for current, following in [(held, empty), (empty, held)]:
            buffer.add(Batch(obs=current, obs_next=following, act=0, rew=0.,
                             terminated=False, truncated=False, info={}))
        targets = policy._target_q(buffer, np.array([0, 1]))
        np.testing.assert_array_equal(targets.detach().numpy(), [5., 10.])

    def test_existing_masks_are_respected_and_hold_remains_available(self):
        policy = self.rotation_policy()
        mask = np.zeros((1, len(PortfolioAction)), dtype=bool)
        mask[:, PortfolioAction.HOLD] = True
        batch = Batch(obs=Batch(obs=np.zeros((1, OBSERVATION_DIM)), mask=mask), info=Batch())
        self.assertEqual(policy(batch).act[0], PortfolioAction.HOLD)
        np.testing.assert_array_equal(batch.obs.mask, mask)
        batch.obs.mask[:] = False
        with self.assertRaisesRegex(ValueError, "no available actions"):
            policy(batch)

    def test_evaluation_can_buy_from_cash_then_rotate(self):
        result = evaluate_dqn(self.rotation_policy(), make_data(np.zeros((2, 2))))
        self.assertEqual(result.transitions["action"].tolist(), ["EQUAL_WEIGHT", "ROTATE_WORST_TO_BEST"])

    def test_checkpoint_mask_version_and_legacy_weight_compatibility(self):
        config = PortfolioDQNConfig(hidden_dims=(8,))
        batch = Batch(obs=np.zeros((1, OBSERVATION_DIM)), info=Batch())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_dqn_checkpoint(self.rotation_policy(), Path(directory) / "policy.pt")
            state = torch.load(checkpoint, weights_only=True)
            self.assertEqual(state["action_mask_version"], ACTION_MASK_VERSION)
            self.assertEqual(load_dqn_checkpoint(checkpoint, config)(batch).act[0], PortfolioAction.EQUAL_WEIGHT)
            del state["action_mask_version"]
            torch.save(state, checkpoint)
            self.assertEqual(load_dqn_checkpoint(checkpoint, config)(batch).act[0], PortfolioAction.EQUAL_WEIGHT)
            state["action_mask_version"] = "unknown"
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, "unsupported action mask"):
                load_dqn_checkpoint(checkpoint, config)

    def test_old_unversioned_checkpoint_requires_retraining(self):
        config = PortfolioDQNConfig(hidden_dims=(8,))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "old.pt"
            torch.save(make_dqn_policy(config).state_dict(), checkpoint)
            with self.assertRaisesRegex(ValueError, "obsolete decision timing"):
                load_dqn_checkpoint(checkpoint, config)

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
                    hidden_dims=(8,),
                    replay_buffer_size=16,
                    batch_size=1,
                    epochs=1,
                    updates_per_epoch=1,
                    target_update_freq=1,
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
            call.kwargs["simulator_config"].turnover_penalty for call in environment_factory.call_args_list
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
