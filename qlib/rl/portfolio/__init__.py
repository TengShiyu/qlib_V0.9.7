# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Multi-asset portfolio reinforcement-learning components."""

from .action import PortfolioAction, PortfolioActionConfig, PortfolioTarget, build_target_weights
from .benchmark import BenchmarkResult, calculate_metrics, results_frame, run_required_benchmarks
from .config import PortfolioWorkflowConfig, load_portfolio_workflow_config
from .evaluation import cost_sensitivity, regime_diagnostics, result_diagnostics
from .data import (
    ENGINEERING_SPLITS,
    DateRange,
    PortfolioDataSplit,
    build_portfolio_data_split,
    load_prediction_frame,
)
from .integration import PortfolioNetReturnReward, make_portfolio_env
from .interpreter import OBSERVATION_DIM, PortfolioActionInterpreter, PortfolioStateInterpreter
from .policy import PortfolioDQNConfig, PortfolioDQNPolicy, PortfolioQNetwork, make_dqn_policy
from .reward import PortfolioReward, PortfolioTurnover, calculate_portfolio_reward, calculate_turnover
from .simulator import PortfolioSimulator, PortfolioSimulatorConfig, PortfolioState, PortfolioTransition
from .training import (
    PortfolioDQNTrainingResult,
    evaluate_dqn,
    load_dqn_checkpoint,
    save_dqn_checkpoint,
    train_dqn,
)

__all__ = [
    "PortfolioAction",
    "PortfolioActionConfig",
    "PortfolioTarget",
    "build_target_weights",
    "BenchmarkResult",
    "calculate_metrics",
    "results_frame",
    "run_required_benchmarks",
    "PortfolioWorkflowConfig",
    "load_portfolio_workflow_config",
    "cost_sensitivity",
    "regime_diagnostics",
    "result_diagnostics",
    "ENGINEERING_SPLITS",
    "DateRange",
    "PortfolioDataSplit",
    "build_portfolio_data_split",
    "load_prediction_frame",
    "PortfolioNetReturnReward",
    "make_portfolio_env",
    "OBSERVATION_DIM",
    "PortfolioActionInterpreter",
    "PortfolioStateInterpreter",
    "PortfolioDQNConfig",
    "PortfolioDQNPolicy",
    "PortfolioQNetwork",
    "make_dqn_policy",
    "PortfolioReward",
    "PortfolioTurnover",
    "calculate_portfolio_reward",
    "calculate_turnover",
    "PortfolioSimulator",
    "PortfolioSimulatorConfig",
    "PortfolioState",
    "PortfolioTransition",
    "PortfolioDQNTrainingResult",
    "evaluate_dqn",
    "load_dqn_checkpoint",
    "save_dqn_checkpoint",
    "train_dqn",
]
