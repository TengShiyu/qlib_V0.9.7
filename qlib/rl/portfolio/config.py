# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Load portfolio settings from an upstream Qlib workflow configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Union

import yaml

from .data import DateRange
from .action import PortfolioActionConfig
from .simulator import PortfolioSimulatorConfig


@dataclass(frozen=True)
class PortfolioWorkflowConfig:
    """Portfolio-relevant values inherited from a Qlib workflow YAML."""

    provider_uri: str
    region: str
    benchmark: str
    simulator: PortfolioSimulatorConfig
    rolling_segments: Mapping[str, DateRange]
    action: PortfolioActionConfig
    trading_interval: int
    rolling_step: int
    rolling_type: str
    rl_split_ratio: tuple[int, int, int]


def load_portfolio_workflow_config(path: Union[str, Path]) -> PortfolioWorkflowConfig:
    """Map Qlib backtest costs and account settings to the RL simulator."""

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Workflow configuration does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    try:
        qlib_init = raw["qlib_init"]
        rolling = raw["task"]["rolling"]
        backtest = raw["port_analysis_config"]["backtest"]
        exchange = backtest["exchange_kwargs"]
        strategy_kwargs = raw["port_analysis_config"]["strategy"]["kwargs"]
        rolling_segments = {
            name: DateRange(values[0], values[1])
            for name, values in raw["task"]["dataset"]["kwargs"]["segments"].items()
        }
        if set(rolling_segments) != {"train", "valid", "test"}:
            raise ValueError("task.dataset segments must define train, valid, and test.")
        if not (
            rolling_segments["train"].end < rolling_segments["valid"].start
            and rolling_segments["valid"].end < rolling_segments["test"].start
        ):
            raise ValueError("RL segments must be non-overlapping and chronological.")
        simulator = PortfolioSimulatorConfig(
            initial_value=float(backtest["account"]),
            buy_cost=float(exchange["open_cost"]),
            sell_cost=float(exchange["close_cost"]),
            min_cost=float(exchange["min_cost"]),
            turnover_penalty=float(strategy_kwargs.get("turnover_penalty", 0.0)),
        )
        trading_interval = int(rolling["trading_interval"])
        rolling_step = int(rolling["step"])
        rolling_type = str(rolling["rolling_type"]).upper()
        rl_split_ratio = tuple(int(value) for value in rolling["rl_split_ratio"])
        if trading_interval <= 0 or rolling_step <= 0:
            raise ValueError("Rolling intervals must be positive.")
        if rolling_type not in {"ROLL_EX", "ROLL_SD"}:
            raise ValueError("rolling_type must be ROLL_EX or ROLL_SD.")
        if len(rl_split_ratio) != 3 or any(value <= 0 for value in rl_split_ratio):
            raise ValueError("rl_split_ratio must contain three positive integers.")
        return PortfolioWorkflowConfig(
            provider_uri=str(qlib_init["provider_uri"]),
            region=str(qlib_init["region"]),
            benchmark=str(backtest["benchmark"]),
            simulator=simulator,
            rolling_segments=rolling_segments,
            action=PortfolioActionConfig(
                max_holdings=strategy_kwargs.get("max_holdings"),
                increase_exposure_ratio=float(strategy_kwargs.get("increase_exposure_ratio", 0.25)),
                concentrated_holdings=int(strategy_kwargs.get("concentrated_holdings", 5)),
                concentrated_equity=float(strategy_kwargs.get("concentrated_equity", 0.95)),
                rotation_count=int(strategy_kwargs.get("rotation_count", 2)),
                defensive_equity=float(strategy_kwargs.get("defensive_equity", 0.50)),
            ),
            trading_interval=trading_interval,
            rolling_step=rolling_step,
            rolling_type=rolling_type,
            rl_split_ratio=rl_split_ratio,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Workflow configuration is missing valid portfolio backtest settings: {config_path}"
        ) from exc


def portfolio_artifact_root(path: Union[str, Path]) -> Path:
    """Derive ``<experiment>/portfolio_dqn_rl`` from a workflow in ``configs``."""

    config_path = Path(path).expanduser().resolve()
    if config_path.parent.name != "configs":
        raise ValueError(
            "Workflow configuration must be stored in an experiment's configs directory."
        )
    return config_path.parent.parent / "portfolio_dqn_rl"


def portfolio_artifact_root_from_strategy(strategy_kwargs: Mapping[str, object]) -> Path:
    """Derive the artifact root from PortfolioDQNStrategy location parameters."""

    from qlib.contrib.strategy.rl_dqn import resolve_portfolio_experiment_root

    experiment_root = resolve_portfolio_experiment_root(
        experiment_root=strategy_kwargs.get("experiment_root"),
        experiment_name=strategy_kwargs.get("experiment_name"),
        experiment_base_dir=strategy_kwargs.get("experiment_base_dir"),
    )
    return experiment_root / "portfolio_dqn_rl"
