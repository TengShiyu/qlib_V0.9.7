#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Compare the trained engineering DQN checkpoint with all required baselines."""

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
    regime_diagnostics,
    result_diagnostics,
)
from qlib.rl.portfolio.benchmark import run_market_benchmark
from qlib.rl.portfolio.data import load_qlib_calendar, load_qlib_market_frame
from qlib.rl.portfolio.training import action_config_record, simulator_config_record


DEFAULT_WORKFLOW_CONFIG = Path(
    "/home/shiyu/qlib_experiment/alpha158_szrankguard_rolling_DQN/configs/"
    "workflow_config_szrankguard_topk20drop2_rolling_DQN.yaml"
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-config", type=Path, default=DEFAULT_WORKFLOW_CONFIG)
    parser.add_argument("--training-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_root = portfolio_artifact_root(args.workflow_config)
    training_dir = (
        args.training_dir.expanduser()
        if args.training_dir is not None
        else artifact_root / "training"
    )
    with (training_dir / "training_config.json").open(encoding="utf-8") as file:
        training_record = json.load(file)
    dqn_values = dict(training_record["dqn"])
    dqn_values["hidden_dims"] = tuple(dqn_values["hidden_dims"])
    dqn_config = PortfolioDQNConfig(**dqn_values)
    if (
        Path(training_record["workflow_config"]).expanduser().resolve()
        != args.workflow_config.expanduser().resolve()
    ):
        raise ValueError("Training record workflow does not match --workflow-config.")
    workflow = load_portfolio_workflow_config(training_record["workflow_config"])
    active_segments = {
        name: {"start": str(segment.start.date()), "end": str(segment.end.date())}
        for name, segment in workflow.rolling_segments.items()
    }
    if training_record.get("rolling_segments") != active_segments:
        raise ValueError("Configured RL segments do not match the saved training record.")
    if training_record.get("action") != action_config_record(workflow.action):
        raise ValueError("Configured portfolio action settings do not match the saved training record.")
    if training_record.get("simulator") != simulator_config_record(workflow.simulator):
        raise ValueError("Configured simulator settings do not match the saved training record.")

    policy = load_dqn_checkpoint(training_dir / "best_policy.pt", dqn_config)
    predictions = load_prediction_frame(
        training_record["predictions"],
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

    metric_records = []
    diagnostic_records = []
    action_records = []
    regime_frames = []
    transition_frames = []
    for split_name in ("valid", "test"):
        split_data = build_portfolio_data_split(
            predictions=predictions,
            market=asset_market,
            calendar=calendar,
            date_range=workflow.rolling_segments[split_name],
            instruments=instruments,
        )
        results = {
            "market": run_market_benchmark("market", split_data, benchmark_close)
        }
        results["dqn"] = evaluate_dqn(
            policy,
            split_data,
            simulator_config=workflow.simulator,
            action_config=workflow.action,
            name="dqn",
        )
        regimes = regime_diagnostics(results, results["market"])
        regimes.insert(0, "split", split_name)
        regime_frames.append(regimes)
        for strategy, result in results.items():
            metric_records.append(
                {"split": split_name, "strategy": strategy, **result.metrics}
            )
            diagnostic_records.append(
                {
                    "split": split_name,
                    "strategy": strategy,
                    **result_diagnostics(result),
                }
            )
            for action, count in result.action_counts.items():
                action_records.append(
                    {
                        "split": split_name,
                        "strategy": strategy,
                        "action": action,
                        "count": count,
                        "frequency": count / len(result.transitions),
                    }
                )
            transitions = result.transitions.copy()
            transitions.insert(0, "strategy", strategy)
            transitions.insert(0, "split", split_name)
            transition_frames.append(transitions)

    metrics = pd.DataFrame.from_records(metric_records)
    diagnostics = pd.DataFrame.from_records(diagnostic_records)
    actions = pd.DataFrame.from_records(action_records)
    regimes = pd.concat(regime_frames, ignore_index=True)
    transitions = pd.concat(transition_frames, ignore_index=True)

    output_dir = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else artifact_root / "evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "comparison_metrics.csv", index=False)
    diagnostics.to_csv(output_dir / "strategy_diagnostics.csv", index=False)
    actions.to_csv(output_dir / "action_frequency.csv", index=False)
    regimes.to_csv(output_dir / "market_regime_metrics.csv", index=False)
    transitions.to_csv(output_dir / "evaluation_transitions.csv", index=False)
    report = build_report(metrics, diagnostics, actions, regimes, training_record)
    (output_dir / "engineering_evaluation_report.md").write_text(
        report, encoding="utf-8"
    )
    print(f"Report: {output_dir / 'engineering_evaluation_report.md'}")
    print(
        metrics.loc[
            metrics["split"] == "test",
            ["strategy", "total_return", "sharpe_ratio", "max_drawdown"],
        ].to_string(index=False)
    )


def build_report(
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    actions: pd.DataFrame,
    regimes: pd.DataFrame,
    training_record: dict,
) -> str:
    """Create the concise human-readable engineering evaluation report."""

    test_metrics = metrics.loc[
        metrics["split"] == "test",
        [
            "strategy",
            "total_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "cumulative_turnover",
        ],
    ]
    dqn_diagnostics = diagnostics.loc[diagnostics["strategy"] == "dqn"]
    dqn_actions = actions.loc[actions["strategy"] == "dqn"]
    dqn_regimes = regimes.loc[
        (regimes["strategy"] == "dqn") & (regimes["split"] == "test")
    ]
    test_dqn = dqn_diagnostics.loc[dqn_diagnostics["split"] == "test"].iloc[0]
    collapsed = test_dqn["dominant_action_frequency"] >= 0.9
    return "\n".join(
        [
            "# Portfolio DQN Engineering Evaluation",
            "",
            "> This is a single-seed engineering evaluation over a short, previously inspected test period. "
            "It is not a pristine formal out-of-sample investment claim.",
            "",
            f"Selected training epoch: {training_record['best_epoch']}. Random seed: {training_record['dqn']['random_seed']}.",
            "",
            "## Test metrics: configured workflow costs",
            "",
            test_metrics.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## DQN action and portfolio diagnostics",
            "",
            dqn_diagnostics.to_markdown(index=False, floatfmt=".6f"),
            "",
            f"Action-collapse check: **{'collapsed' if collapsed else 'not collapsed'}** "
            f"(dominant test action frequency {test_dqn['dominant_action_frequency']:.2%}; threshold 90%).",
            "",
            dqn_actions.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## DQN test performance by market regime",
            "",
            dqn_regimes.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Interpretation boundary",
            "",
            "The existing 92/31/29-transition split is sufficient to verify software integration, but not to establish "
            "robustness or profitability. Formal claims require longer point-in-time data, repeated seeds, and a new "
            "untouched test period.",
            "",
        ]
    )


if __name__ == "__main__":
    main()
