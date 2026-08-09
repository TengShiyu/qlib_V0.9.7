# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from qlib.rl.portfolio.policy import PortfolioDQNConfig, PortfolioQNetwork
from qlib.rl.portfolio.training import evaluate_dqn, load_dqn_checkpoint, save_dqn_checkpoint, train_dqn
from tests.rl.portfolio.test_integration import make_data


class PortfolioDQNTest(unittest.TestCase):
    def test_network_emits_eight_raw_q_values(self) -> None:
        network = PortfolioQNetwork(hidden_dims=(16,))
        output, state = network(np.zeros((3, 49), dtype=np.float32))

        self.assertEqual(tuple(output.shape), (3, 8))
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
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_dqn_checkpoint(result.policy, Path(directory) / "policy.pt")
            restored = load_dqn_checkpoint(checkpoint, config)
            original_eval = evaluate_dqn(result.policy, valid_data)
            restored_eval = evaluate_dqn(restored, valid_data)

        self.assertEqual(original_eval.action_counts, restored_eval.action_counts)
        self.assertEqual(original_eval.metrics, restored_eval.metrics)


if __name__ == "__main__":
    unittest.main()
