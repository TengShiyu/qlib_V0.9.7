#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Train and evaluate the engineering portfolio DQN checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qlib.rl.portfolio import (
    PortfolioDQNConfig,
    build_portfolio_data_split,
    evaluate_dqn,
    load_dqn_checkpoint,
    load_portfolio_workflow_config,
    load_prediction_frame,
    portfolio_artifact_root,
    save_dqn_checkpoint,
    train_dqn,
)
from qlib.rl.portfolio.data import load_qlib_calendar, load_qlib_market_frame
from qlib.rl.portfolio.training import config_record


DEFAULT_PREDICTIONS = Path(
    "/home/shiyu/qlib_experiment/mlruns/542860662874316362/"
    "58da01314bfa4295a96b952d5dbb7db4/artifacts/pred.pkl"
)
DEFAULT_WORKFLOW_CONFIG = Path(
    "/home/shiyu/qlib_experiment/alpha158_szrankguard_rolling_horizon2_step10/configs/"
    "workflow_config_szrankguard_topk20drop2_rolling_h2_step10.yaml"
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--workflow-config", type=Path, default=DEFAULT_WORKFLOW_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workflow = load_portfolio_workflow_config(args.workflow_config)
    dqn_config = PortfolioDQNConfig(epochs=args.epochs, random_seed=args.random_seed)
    predictions = load_prediction_frame(
        args.predictions,
        workflow.rl_segments["train"].start,
        workflow.rl_segments["test"].end,
    )
    instruments = tuple(
        sorted(predictions.index.get_level_values("instrument").unique())
    )
    market_start = workflow.rl_segments["train"].start - pd.Timedelta(days=90)
    market_end = workflow.rl_segments["test"].end
    calendar = load_qlib_calendar(
        workflow.provider_uri, workflow.region, market_start, market_end
    )
    market = load_qlib_market_frame(
        workflow.provider_uri,
        workflow.region,
        instruments,
        market_start,
        market_end,
    )
    datasets = {
        name: build_portfolio_data_split(
            predictions=predictions,
            market=market,
            calendar=calendar,
            date_range=workflow.rl_segments[name],
            instruments=instruments,
        )
        for name in ("train", "valid", "test")
    }

    result = train_dqn(
        train_data=datasets["train"],
        valid_data=datasets["valid"],
        dqn_config=dqn_config,
        simulator_config=workflow.simulator,
        action_config=workflow.action,
    )
    output_dir = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else portfolio_artifact_root(args.workflow_config) / "training"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dqn_checkpoint(result.policy, output_dir / "best_policy.pt")
    restored_policy = load_dqn_checkpoint(checkpoint_path, dqn_config)
    restored_validation = evaluate_dqn(
        restored_policy,
        datasets["valid"],
        simulator_config=workflow.simulator,
        action_config=workflow.action,
        name="dqn",
    )
    if restored_validation.action_counts != result.validation.action_counts:
        raise RuntimeError("Reloaded checkpoint does not reproduce validation actions.")
    test_result = evaluate_dqn(
        restored_policy,
        datasets["test"],
        simulator_config=workflow.simulator,
        action_config=workflow.action,
        name="dqn",
    )

    configuration = {
        "dqn": config_record(dqn_config),
        "workflow_config": str(args.workflow_config.expanduser()),
        "predictions": str(args.predictions.expanduser()),
        "provider_uri": workflow.provider_uri,
        "benchmark": workflow.benchmark,
        "simulator": {
            "initial_value": workflow.simulator.initial_value,
            "buy_cost": workflow.simulator.buy_cost,
            "sell_cost": workflow.simulator.sell_cost,
            "min_cost": workflow.simulator.min_cost,
        },
        "best_epoch": result.best_epoch,
        "split_transition_counts": {
            name: len(data.decision_dates) for name, data in datasets.items()
        },
        "rl_segments": {
            name: {"start": str(segment.start.date()), "end": str(segment.end.date())}
            for name, segment in workflow.rl_segments.items()
        },
        "action": {"max_holdings": workflow.action.max_holdings},
    }
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=2)
    result.history.to_csv(output_dir / "training_history.csv", index=False)

    evaluations = {"valid": restored_validation, "test": test_result}
    metrics = pd.DataFrame(
        {name: evaluation.metrics for name, evaluation in evaluations.items()}
    ).T
    metrics.rename_axis("split").reset_index().to_csv(
        output_dir / "dqn_metrics.csv", index=False
    )
    action_records = []
    transition_frames = []
    for split_name, evaluation in evaluations.items():
        transition_count = len(evaluation.transitions)
        for action, count in evaluation.action_counts.items():
            action_records.append(
                {
                    "split": split_name,
                    "action": action,
                    "count": count,
                    "frequency": count / transition_count,
                }
            )
        transitions = evaluation.transitions.copy()
        transitions.insert(0, "split", split_name)
        transition_frames.append(transitions)
    pd.DataFrame.from_records(action_records).to_csv(
        output_dir / "dqn_action_frequency.csv", index=False
    )
    pd.concat(transition_frames, ignore_index=True).to_csv(
        output_dir / "dqn_transitions.csv", index=False
    )

    print(f"Best epoch: {result.best_epoch}")
    print(f"Checkpoint: {checkpoint_path}")
    print(metrics.to_string())


if __name__ == "__main__":
    main()
