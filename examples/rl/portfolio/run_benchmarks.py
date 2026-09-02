#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate engineering baseline reports for the portfolio DQN project."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from qlib.rl.portfolio import (
    build_portfolio_data_split,
    load_prediction_frame,
    load_portfolio_workflow_config,
    portfolio_artifact_root,
    results_frame,
)
from qlib.rl.portfolio.benchmark import run_market_benchmark
from qlib.rl.portfolio.data import load_qlib_calendar, load_qlib_market_frame


DEFAULT_PREDICTIONS = Path(
    "/home/shiyu/qlib_experiment/mlruns/542860662874316362/"
    "58da01314bfa4295a96b952d5dbb7db4/artifacts/pred.pkl"
)
DEFAULT_WORKFLOW_CONFIG = Path(
    "/home/shiyu/qlib_experiment/alpha158_szrankguard_rolling_DQN/configs/"
    "workflow_config_szrankguard_topk20drop2_rolling_DQN.yaml"
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--workflow-config", type=Path, default=DEFAULT_WORKFLOW_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workflow = load_portfolio_workflow_config(args.workflow_config)
    predictions = load_prediction_frame(
        args.predictions,
        workflow.rolling_segments["valid"].start,
        workflow.rolling_segments["test"].end,
    )
    instruments = tuple(
        sorted(predictions.index.get_level_values("instrument").unique())
    )
    market_start = workflow.rolling_segments["train"].start - pd.Timedelta(days=90)
    market_end = workflow.rolling_segments["test"].end
    calendar = load_qlib_calendar(
        workflow.provider_uri, workflow.region, market_start, market_end
    )
    market = load_qlib_market_frame(
        workflow.provider_uri,
        workflow.region,
        [*instruments, workflow.benchmark],
        market_start,
        market_end,
    )
    benchmark_close = market.xs(workflow.benchmark, level="instrument")["close"]
    asset_market = market.loc[
        market.index.get_level_values("instrument").isin(instruments)
    ]

    metric_frames = []
    action_records = []
    transition_frames = []
    for split_name in ("valid", "test"):
        split_range = workflow.rolling_segments[split_name]
        data = build_portfolio_data_split(
            predictions=predictions,
            market=asset_market,
            calendar=calendar,
            date_range=split_range,
            instruments=instruments,
        )
        results = {"market": run_market_benchmark("market", data, benchmark_close)}
        metrics = results_frame(results).reset_index()
        metrics.insert(0, "split", split_name)
        metric_frames.append(metrics)
        result = results["market"]
        action_records.append(
            {
                "split": split_name,
                "benchmark": "market",
                "action": "market",
                "count": len(data.decision_dates),
                "frequency": 1.0,
            }
        )
        history = result.transitions.copy()
        history.insert(0, "benchmark", "market")
        history.insert(0, "split", split_name)
        transition_frames.append(history)

    output_dir = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else portfolio_artifact_root(args.workflow_config) / "baselines"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "engineering_baseline_metrics.csv"
    actions_path = output_dir / "engineering_baseline_action_frequency.csv"
    transitions_path = output_dir / "engineering_baseline_transitions.csv"
    pd.concat(metric_frames, ignore_index=True).to_csv(metrics_path, index=False)
    pd.DataFrame.from_records(action_records).to_csv(actions_path, index=False)
    pd.concat(transition_frames, ignore_index=True).to_csv(
        transitions_path, index=False
    )
    print(f"Metrics: {metrics_path}")
    print(f"Action frequencies: {actions_path}")
    print(f"Transitions: {transitions_path}")


if __name__ == "__main__":
    main()
