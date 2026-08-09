# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Load portfolio settings from an upstream Qlib workflow configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import yaml

from .simulator import PortfolioSimulatorConfig


@dataclass(frozen=True)
class PortfolioWorkflowConfig:
    """Portfolio-relevant values inherited from a Qlib workflow YAML."""

    provider_uri: str
    region: str
    benchmark: str
    simulator: PortfolioSimulatorConfig


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
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Workflow configuration is missing valid portfolio backtest settings: {config_path}") from exc
