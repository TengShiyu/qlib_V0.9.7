# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""QlibRL interpreters for the fixed-size portfolio environment."""

from __future__ import annotations

import numpy as np
from gym import spaces

from qlib.rl.interpreter import ActionInterpreter, StateInterpreter

from .action import PortfolioAction, PortfolioActionConfig, build_target_weights
from .reward import calculate_turnover
from .simulator import PortfolioState


ACTION_FEATURE_COUNT = 4
GLOBAL_FEATURE_COUNT = 17
OBSERVATION_DIM = GLOBAL_FEATURE_COUNT + ACTION_FEATURE_COUNT * len(PortfolioAction)


class PortfolioStateInterpreter(StateInterpreter[PortfolioState, np.ndarray]):
    """Convert a variable-universe portfolio state into 49 fixed features.

    The global block describes current exposure, recent realized returns, and
    cross-sectional score/volatility availability. Each action then contributes
    four ex-ante properties: equity exposure, two-way turnover, score exposure,
    and diagonal volatility. No realized forward return enters the observation.
    """

    def __init__(self, action_config: PortfolioActionConfig = PortfolioActionConfig()) -> None:
        self.action_config = action_config
        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )

    @property
    def observation_space(self) -> spaces.Box:
        return self._observation_space

    def interpret(self, state: PortfolioState) -> np.ndarray:
        scores = np.asarray(state.scores, dtype=np.float64)
        volatility = np.asarray(state.volatility, dtype=np.float64)
        tradable = np.asarray(state.tradable, dtype=np.bool_)
        finite_scores = scores[np.isfinite(scores)]
        positive_scores = finite_scores[finite_scores > 0.0]
        valid_volatility = volatility[np.isfinite(volatility) & (volatility > 0.0)]
        history = np.asarray(state.net_return_history, dtype=np.float64)

        last_turnover = 0.0 if state.last_transition is None else state.last_transition.turnover.total
        global_features = np.array(
            [
                state.asset_weights.sum(),
                state.cash_weight,
                last_turnover,
                history[-1] if history.size else 0.0,
                history.mean() if history.size else 0.0,
                history.std(ddof=0) if history.size else 0.0,
                tradable.mean() if tradable.size else 0.0,
                finite_scores.mean() if finite_scores.size else 0.0,
                finite_scores.std(ddof=0) if finite_scores.size else 0.0,
                finite_scores.min() if finite_scores.size else 0.0,
                finite_scores.max() if finite_scores.size else 0.0,
                positive_scores.size / scores.size if scores.size else 0.0,
                finite_scores.size / scores.size if scores.size else 0.0,
                valid_volatility.mean() if valid_volatility.size else 0.0,
                valid_volatility.std(ddof=0) if valid_volatility.size else 0.0,
                valid_volatility.min() if valid_volatility.size else 0.0,
                valid_volatility.max() if valid_volatility.size else 0.0,
            ],
            dtype=np.float64,
        )

        clean_scores = np.where(np.isfinite(scores), scores, 0.0)
        clean_volatility = np.where(np.isfinite(volatility) & (volatility > 0.0), volatility, 0.0)
        action_features = []
        for action in PortfolioAction:
            target = build_target_weights(
                action=action,
                current_asset_weights=state.asset_weights,
                current_cash_weight=state.cash_weight,
                scores=scores,
                tradable=tradable,
                volatility=volatility,
                config=self.action_config,
            )
            turnover = calculate_turnover(state.asset_weights, target.asset_weights)
            action_features.extend(
                [
                    target.asset_weights.sum(),
                    turnover.total,
                    np.dot(target.asset_weights, clean_scores),
                    np.sqrt(np.sum(np.square(target.asset_weights * clean_volatility))),
                ]
            )

        observation = np.concatenate([global_features, np.asarray(action_features, dtype=np.float64)])
        if not np.all(np.isfinite(observation)):
            raise ValueError("Portfolio observation contains a non-finite feature.")
        return observation.astype(np.float32)


class PortfolioActionInterpreter(ActionInterpreter[PortfolioState, int, PortfolioAction]):
    """Validate a DQN integer and convert it to its stable portfolio command."""

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(PortfolioAction))

    def interpret(self, state: PortfolioState, action: int) -> PortfolioAction:
        return PortfolioAction(int(action))
