# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Raw-Q network and DQN policy for portfolio command selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
from tianshou.data import Batch, to_torch_as
from tianshou.policy import DQNPolicy

from .action import PortfolioAction
from .interpreter import OBSERVATION_DIM


@dataclass(frozen=True)
class PortfolioDQNConfig:
    """Reproducible settings for the engineering DQN training run."""

    hidden_dims: tuple[int, ...] = (64, 64)
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    discount_factor: float = 0.99
    target_update_freq: int = 100
    replay_buffer_size: int = 4096
    batch_size: int = 32
    epochs: int = 30
    episodes_per_epoch: int = 1
    updates_per_epoch: int = 32
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_epochs: int = 20
    max_grad_norm: float = 10.0
    random_seed: int = 7

    def __post_init__(self) -> None:
        if not self.hidden_dims or any(size <= 0 for size in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive layer sizes.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        if not 0.0 <= self.discount_factor <= 1.0:
            raise ValueError("discount_factor must be between zero and one.")
        integer_settings = (
            self.target_update_freq,
            self.replay_buffer_size,
            self.batch_size,
            self.epochs,
            self.episodes_per_epoch,
            self.updates_per_epoch,
            self.epsilon_decay_epochs,
        )
        if any(value <= 0 for value in integer_settings):
            raise ValueError("DQN count and capacity settings must be positive.")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1.")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive.")

    def epsilon(self, epoch: int) -> float:
        """Linearly decay exploration over the configured number of epochs."""

        fraction = min(max(epoch, 0), self.epsilon_decay_epochs) / self.epsilon_decay_epochs
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)


class PortfolioQNetwork(torch.nn.Module):
    """Small MLP that emits one unrestricted Q-value per stable action."""

    def __init__(
        self,
        observation_dim: int = OBSERVATION_DIM,
        action_count: int = len(PortfolioAction),
        hidden_dims: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()
        dimensions = [observation_dim, *hidden_dims, action_count]
        layers = []
        for input_dim, output_dim in zip(dimensions[:-2], dimensions[1:-1]):
            layers.extend([torch.nn.Linear(input_dim, output_dim), torch.nn.ReLU()])
        layers.append(torch.nn.Linear(dimensions[-2], dimensions[-1]))
        self.network = torch.nn.Sequential(*layers)

    def forward(
        self,
        observation: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Optional[dict] = None,
    ) -> tuple[torch.Tensor, Optional[Any]]:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=next(self.parameters()).device)
        return self.network(tensor), state


class PortfolioDQNPolicy(DQNPolicy):
    """Tianshou DQN with Huber loss and explicit gradient-norm clipping."""

    def __init__(self, *args: Any, max_grad_norm: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_grad_norm = max_grad_norm

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        q_values = self(batch).logits
        selected_q = q_values[np.arange(len(q_values)), batch.act]
        returns = to_torch_as(batch.returns.flatten(), selected_q)
        td_error = returns - selected_q
        if self._clip_loss_grad:
            loss = torch.nn.functional.huber_loss(selected_q.reshape(-1, 1), returns.reshape(-1, 1))
        else:
            loss = (td_error.pow(2) * weight).mean()
        batch.weight = td_error
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item(), "gradient_norm": float(gradient_norm)}


def make_dqn_policy(config: PortfolioDQNConfig = PortfolioDQNConfig()) -> PortfolioDQNPolicy:
    """Construct the engineering Double-DQN policy on CPU."""

    model = PortfolioQNetwork(hidden_dims=config.hidden_dims)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    return PortfolioDQNPolicy(
        model=model,
        optim=optimizer,
        discount_factor=config.discount_factor,
        estimation_step=1,
        target_update_freq=config.target_update_freq,
        reward_normalization=False,
        is_double=True,
        clip_loss_grad=True,
        max_grad_norm=config.max_grad_norm,
    )
