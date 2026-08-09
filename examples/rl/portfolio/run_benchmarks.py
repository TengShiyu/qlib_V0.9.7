#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate engineering baseline reports for the portfolio DQN project."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from qlib.rl.portfolio import (
    ENGINEERING_SPLITS,
    build_portfolio_data_split,
    load_prediction_frame,
    load_portfolio_workflow_config,
    results_frame,
    run_required_benchmarks,
)
from qlib.rl.portfolio.data import load_qlib_calendar, load_qlib_market_frame


DEFAULT_PREDICTIONS = Path(
    "/home/shiyu/qlib_experiment/mlruns/542860662874316362/" "58da01314bfa4295a96b952d5dbb7db4/artifacts/pred.pkl"
)
DEFAULT_OUTPUT = Path("/home/shiyu/qlib_experiment/portfolio_dqn_baselines")
DEFAULT_WORKFLOW_CONFIG = Path(
    "/home/shiyu/qlib_experiment/alpha158_szrankguard_rolling_horizon2_step10/configs/"
    "workflow_config_szrankguard_topk20drop2_rolling_h2_step10.yaml"
)
DEFAULT_SENSITIVITY_CONFIG = Path(__file__).with_name("workflow_config_portfolio_dqn_cost_sensitivity.yaml")
MARKET_START = "2024-11-01"
MARKET_END = "2026-03-30"
RANDOM_SEED = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--workflow-config", type=Path, default=DEFAULT_WORKFLOW_CONFIG)
    parser.add_argument("--sensitivity-config", type=Path, default=DEFAULT_SENSITIVITY_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workflow = load_portfolio_workflow_config(args.workflow_config)
    sensitivity = load_portfolio_workflow_config(args.sensitivity_config)
    if (workflow.provider_uri, workflow.region, workflow.benchmark) != (
        sensitivity.provider_uri,
        sensitivity.region,
        sensitivity.benchmark,
    ):
        raise ValueError("Reproduction and sensitivity configurations must use the same data and benchmark.")
    predictions = load_prediction_frame(
        args.predictions, ENGINEERING_SPLITS["valid"].start, ENGINEERING_SPLITS["test"].end
    )
    instruments = tuple(sorted(predictions.index.get_level_values("instrument").unique()))
    calendar = load_qlib_calendar(workflow.provider_uri, workflow.region, MARKET_START, MARKET_END)
    market = load_qlib_market_frame(
        workflow.provider_uri,
        workflow.region,
        [*instruments, workflow.benchmark],
        MARKET_START,
        MARKET_END,
    )
    benchmark_close = market.xs(workflow.benchmark, level="instrument")["close"]
    asset_market = market.loc[market.index.get_level_values("instrument").isin(instruments)]

    cost_settings = {
        "reproduction": workflow.simulator,
        "cost_sensitivity": sensitivity.simulator,
    }
    metric_frames = []
    action_records = []
    transition_frames = []
    for split_name in ("valid", "test"):
        split_range = ENGINEERING_SPLITS[split_name]
        data = build_portfolio_data_split(
            predictions=predictions,
            market=asset_market,
            calendar=calendar,
            date_range=split_range,
            instruments=instruments,
        )
        for cost_name, simulator_config in cost_settings.items():
            results = run_required_benchmarks(
                data=data,
                simulator_config=simulator_config,
                market_close=benchmark_close,
                random_seed=args.random_seed,
            )
            metrics = results_frame(results).reset_index()
            metrics.insert(0, "cost_setting", cost_name)
            metrics.insert(0, "split", split_name)
            metric_frames.append(metrics)
            for benchmark_name, result in results.items():
                for action_name, count in result.action_counts.items():
                    action_records.append(
                        {
                            "split": split_name,
                            "cost_setting": cost_name,
                            "benchmark": benchmark_name,
                            "action": action_name,
                            "count": count,
                            "frequency": count / len(data.decision_dates),
                        }
                    )
                history = result.transitions.copy()
                history.insert(0, "benchmark", benchmark_name)
                history.insert(0, "cost_setting", cost_name)
                history.insert(0, "split", split_name)
                transition_frames.append(history)

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "engineering_baseline_metrics.csv"
    actions_path = output_dir / "engineering_baseline_action_frequency.csv"
    transitions_path = output_dir / "engineering_baseline_transitions.csv"
    pd.concat(metric_frames, ignore_index=True).to_csv(metrics_path, index=False)
    pd.DataFrame.from_records(action_records).to_csv(actions_path, index=False)
    pd.concat(transition_frames, ignore_index=True).to_csv(transitions_path, index=False)
    print(f"Metrics: {metrics_path}")
    print(f"Action frequencies: {actions_path}")
    print(f"Transitions: {transitions_path}")


if __name__ == "__main__":
    main()
