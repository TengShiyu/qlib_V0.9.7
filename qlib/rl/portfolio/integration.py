# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Assembly helpers for the QlibRL portfolio environment."""

from __future__ import annotations

from qlib.rl.reward import Reward
from qlib.rl.utils.env_wrapper import EnvWrapper

from .action import PortfolioActionConfig
from .data import PortfolioDataSplit
from .interpreter import PortfolioActionInterpreter, PortfolioStateInterpreter
from .simulator import PortfolioSimulator, PortfolioSimulatorConfig, PortfolioState


class PortfolioNetReturnReward(Reward[PortfolioState]):
    """Expose the simulator's latest net two-session return to QlibRL."""

    def reward(self, simulator_state: PortfolioState) -> float:
        transition = simulator_state.last_transition
        if transition is None:
            raise RuntimeError("Portfolio reward requested before the first transition.")
        self.log("reward/gross_return", transition.reward.gross_return)
        self.log("reward/transaction_cost", transition.reward.transaction_cost)
        self.log("reward/missing_return_weight", transition.reward.missing_return_weight)
        return transition.reward.net_return


def make_portfolio_env(
    data: PortfolioDataSplit,
    simulator_config: PortfolioSimulatorConfig = PortfolioSimulatorConfig(),
    action_config: PortfolioActionConfig = PortfolioActionConfig(),
) -> EnvWrapper:
    """Build one Gym-compatible QlibRL environment for an isolated data split."""

    return EnvWrapper(
        simulator_fn=lambda: PortfolioSimulator(data, config=simulator_config, action_config=action_config),
        state_interpreter=PortfolioStateInterpreter(action_config=action_config),
        action_interpreter=PortfolioActionInterpreter(),
        seed_iterator=None,
        reward_fn=PortfolioNetReturnReward(),
    )
