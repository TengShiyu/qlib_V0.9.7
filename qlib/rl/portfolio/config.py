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
    rl_segments: Mapping[str, DateRange]
    action: PortfolioActionConfig


def load_portfolio_workflow_config(path: Union[str, Path]) -> PortfolioWorkflowConfig:
    """Map Qlib backtest costs and account settings to the RL simulator."""

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Workflow configuration does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    try:
        qlib_init = raw["qlib_init"]
        backtest = raw["port_analysis_config"]["backtest"]
        exchange = backtest["exchange_kwargs"]
        strategy_kwargs = raw["port_analysis_config"]["strategy"]["kwargs"]
        rl_segments = {
            name: DateRange(values[0], values[1])
            for name, values in backtest["rl_segments"].items()
        }
        if set(rl_segments) != {"train", "valid", "test"}:
            raise ValueError("backtest.rl_segments must define train, valid, and test.")
        if not (
            rl_segments["train"].end < rl_segments["valid"].start
            and rl_segments["valid"].end < rl_segments["test"].start
        ):
            raise ValueError("RL segments must be non-overlapping and chronological.")
        simulator = PortfolioSimulatorConfig(
            initial_value=float(backtest["account"]),
            buy_cost=float(exchange["open_cost"]),
            sell_cost=float(exchange["close_cost"]),
            min_cost=float(exchange["min_cost"]),
        )
        return PortfolioWorkflowConfig(
            provider_uri=str(qlib_init["provider_uri"]),
            region=str(qlib_init["region"]),
            benchmark=str(backtest["benchmark"]),
            simulator=simulator,
            rl_segments=rl_segments,
            action=PortfolioActionConfig(max_holdings=strategy_kwargs.get("max_holdings")),
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
