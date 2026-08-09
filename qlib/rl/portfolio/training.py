# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reproducible training and evaluation helpers for portfolio DQN."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Union

import numpy as np
import pandas as pd
import torch
from tianshou.data import Batch, ReplayBuffer
from tianshou.data.collector import Collector

from .benchmark import BenchmarkResult, calculate_metrics
from .data import PortfolioDataSplit
from .integration import make_portfolio_env
from .policy import PortfolioDQNConfig, PortfolioDQNPolicy, make_dqn_policy
from .simulator import PortfolioSimulatorConfig


@dataclass(frozen=True)
class PortfolioDQNTrainingResult:
    """Selected policy, training history, and validation result."""

    policy: PortfolioDQNPolicy
    history: pd.DataFrame
    validation: BenchmarkResult
    best_epoch: int


def train_dqn(
    train_data: PortfolioDataSplit,
    valid_data: PortfolioDataSplit,
    dqn_config: PortfolioDQNConfig = PortfolioDQNConfig(),
    simulator_config: PortfolioSimulatorConfig = PortfolioSimulatorConfig(),
) -> PortfolioDQNTrainingResult:
    """Train on one chronological split and select by validation total return."""

    set_global_seed(dqn_config.random_seed)
    policy = make_dqn_policy(dqn_config)
    replay_buffer = ReplayBuffer(dqn_config.replay_buffer_size)
    collector = Collector(
        policy,
        make_portfolio_env(train_data, simulator_config=simulator_config),
        replay_buffer,
        exploration_noise=True,
    )
    best_return = -np.inf
    best_epoch = 0
    best_state = copy.deepcopy(policy.state_dict())
    records = []
    for epoch in range(1, dqn_config.epochs + 1):
        epsilon = dqn_config.epsilon(epoch - 1)
        policy.set_eps(epsilon)
        policy.train()
        collection = collector.collect(n_episode=dqn_config.episodes_per_epoch)
        losses = []
        gradient_norms = []
        for _ in range(dqn_config.updates_per_epoch):
            update = policy.update(dqn_config.batch_size, replay_buffer)
            losses.append(float(update["loss"]))
            gradient_norms.append(float(update["gradient_norm"]))

        validation = evaluate_dqn(policy, valid_data, simulator_config=simulator_config, name="validation")
        validation_return = validation.metrics["total_return"]
        if validation_return > best_return:
            best_return = validation_return
            best_epoch = epoch
            best_state = copy.deepcopy(policy.state_dict())
        records.append(
            {
                "epoch": epoch,
                "epsilon": epsilon,
                "collected_steps": int(collection["n/st"]),
                "training_episode_reward": float(collection["rew"]),
                "mean_loss": float(np.mean(losses)),
                "mean_gradient_norm": float(np.mean(gradient_norms)),
                "validation_total_return": validation_return,
                "validation_sharpe_ratio": validation.metrics["sharpe_ratio"],
            }
        )

    policy.load_state_dict(best_state)
    selected_validation = evaluate_dqn(policy, valid_data, simulator_config=simulator_config, name="validation")
    return PortfolioDQNTrainingResult(
        policy=policy,
        history=pd.DataFrame.from_records(records),
        validation=selected_validation,
        best_epoch=best_epoch,
    )


def evaluate_dqn(
    policy: PortfolioDQNPolicy,
    data: PortfolioDataSplit,
    simulator_config: PortfolioSimulatorConfig = PortfolioSimulatorConfig(),
    name: str = "dqn",
) -> BenchmarkResult:
    """Run one greedy DQN episode without exploration."""

    previous_epsilon = policy.eps
    policy.set_eps(0.0)
    policy.eval()
    env = make_portfolio_env(data, simulator_config=simulator_config)
    observation = env.reset()
    records = []
    action_counts: Dict[str, int] = {}
    done = False
    while not done:
        with torch.no_grad():
            output = policy(Batch(obs=observation[None, :], info=Batch()))
        action = int(output.act[0])
        observation, reward, done, _ = env.step(action)
        transition = env.simulator.get_state().last_transition
        if transition is None:  # pragma: no cover - EnvWrapper has just completed a step
            raise RuntimeError("DQN evaluation did not produce a transition.")
        action_name = transition.action.name
        action_counts[action_name] = action_counts.get(action_name, 0) + 1
        records.append(
            {
                "decision_date": transition.decision_date,
                "execution_date": transition.execution_date,
                "reward_end_date": transition.reward_end_date,
                "action": action_name,
                "net_return": reward,
                "portfolio_value": transition.ending_value,
                "turnover": transition.turnover.one_way,
                "transaction_cost": transition.reward.transaction_cost,
                "cash_exposure": transition.target.cash_weight,
                "missing_return_weight": transition.reward.missing_return_weight,
            }
        )

    policy.set_eps(previous_epsilon)
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
    return BenchmarkResult(name=name, metrics=metrics, transitions=transitions, action_counts=action_counts)


def save_dqn_checkpoint(policy: PortfolioDQNPolicy, path: Union[str, Path]) -> Path:
    """Save model and target-network weights outside Git."""

    checkpoint_path = Path(path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), checkpoint_path)
    return checkpoint_path


def load_dqn_checkpoint(path: Union[str, Path], config: PortfolioDQNConfig) -> PortfolioDQNPolicy:
    """Build a fresh policy and load a trusted weights-only checkpoint."""

    checkpoint_path = Path(path).expanduser()
    policy = make_dqn_policy(config)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    policy.load_state_dict(state)
    return policy


def config_record(config: PortfolioDQNConfig) -> Mapping[str, object]:
    """Return a JSON-serializable DQN configuration."""

    record = asdict(config)
    record["hidden_dims"] = list(config.hidden_dims)
    return record


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for the engineering run."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
