# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qlib workflow record that retrains portfolio DQN before backtesting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from qlib.data import D
from qlib.workflow.record_temp import PortAnaRecord

from .config import portfolio_artifact_root_from_strategy
from .action import PortfolioActionConfig
from .data import DateRange, build_portfolio_data_split
from .policy import PortfolioDQNConfig
from .simulator import PortfolioSimulatorConfig
from .training import evaluate_dqn, save_dqn_checkpoint, train_dqn


class PortfolioDQNRecord(PortAnaRecord):
    """Retrain the configured portfolio DQN once, then run normal analysis."""

    def _generate(self, **kwargs):
        if self.strategy_config.get("class") != "PortfolioDQNStrategy":
            raise ValueError("PortfolioDQNRecord requires PortfolioDQNStrategy.")
        strategy_kwargs = self.strategy_config.get("kwargs", {})
        if strategy_kwargs.get("retrain", True):
            self._retrain_portfolio_dqn(self.load("pred.pkl"), strategy_kwargs)
        artifacts = super()._generate(**kwargs)
        self._export_qlib_evaluation(artifacts, strategy_kwargs)
        return artifacts

    def _export_qlib_evaluation(self, artifacts: dict[str, Any], strategy_kwargs: dict) -> None:
        """Export live Qlib decisions, executions, holdings, and returns beside the checkpoint."""

        output_dir = portfolio_artifact_root_from_strategy(strategy_kwargs) / "evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_key = _single_artifact_key(artifacts, "report_normal_")
        positions_key = _single_artifact_key(artifacts, "positions_normal_")
        report = artifacts[report_key].copy()
        positions = artifacts[positions_key]

        portfolio_report = report.reset_index().rename(columns={report.index.name or "index": "datetime"})
        _write_csv_atomically(portfolio_report, output_dir / "portfolio_report.csv")
        holdings = _positions_to_frame(positions)
        _write_csv_atomically(holdings, output_dir / "daily_holdings.csv")

        actions_path = output_dir / "dqn_actions.csv"
        if not actions_path.is_file():
            raise RuntimeError("PortfolioDQNStrategy did not publish dqn_actions.csv.")
        actions = pd.read_csv(actions_path)
        _write_csv_atomically(actions, output_dir / "evaluation_transitions.csv")
        action_frequency = (
            actions.groupby("action", dropna=False)
            .size()
            .rename("count")
            .reset_index()
            .sort_values(["count", "action"], ascending=[False, True])
        )
        action_frequency["frequency"] = action_frequency["count"] / max(len(actions), 1)
        _write_csv_atomically(action_frequency, output_dir / "action_frequency.csv")

        comparison = _market_vs_dqn(report, float(self.backtest_config["account"]))
        _write_csv_atomically(comparison, output_dir / "comparison_metrics.csv")
        diagnostics = _strategy_diagnostics(actions, holdings)
        _write_csv_atomically(diagnostics, output_dir / "strategy_diagnostics.csv")
        manifest = {
            "source": "live Qlib backtest",
            "report_artifact": report_key,
            "positions_artifact": positions_key,
            "backtest_start": str(pd.Timestamp(report.index.min()).date()),
            "backtest_end": str(pd.Timestamp(report.index.max()).date()),
            "daily_rows": len(report),
            "decision_count": len(actions),
            "executed_order_count": int(actions["executed_order_count"].sum()),
            "holding_rows": len(holdings),
        }
        _write_json_atomically(manifest, output_dir / "run_manifest.json")

    def _retrain_portfolio_dqn(self, predictions: pd.DataFrame, strategy_kwargs: dict) -> None:
        segments = _parse_segments(self.rl_segments_config)
        selected = _select_predictions(predictions, segments["train"].start, segments["test"].end)
        instruments = tuple(sorted(selected.index.get_level_values("instrument").astype(str).unique()))
        market_start = segments["train"].start - pd.Timedelta(days=90)
        market_end = segments["test"].end
        calendar = pd.DatetimeIndex(D.calendar(start_time=market_start, end_time=market_end, freq="day"))
        market = D.features(
            list(instruments),
            ["$close", "$volume"],
            start_time=market_start,
            end_time=market_end,
            freq="day",
        ).rename(columns={"$close": "close", "$volume": "volume"})
        market = market.reorder_levels(["datetime", "instrument"]).sort_index()
        datasets = {
            name: build_portfolio_data_split(
                predictions=selected,
                market=market,
                calendar=calendar,
                date_range=segments[name],
                instruments=instruments,
            )
            for name in ("train", "valid", "test")
        }
        dqn_config = PortfolioDQNConfig(
            hidden_dims=tuple(strategy_kwargs.get("hidden_dims", (64, 64))),
            learning_rate=float(strategy_kwargs.get("learning_rate", 1e-3)),
            discount_factor=float(strategy_kwargs.get("discount_factor", 0.99)),
            epochs=int(strategy_kwargs.get("epochs", 30)),
            epsilon_decay_epochs=int(strategy_kwargs.get("epsilon_decay_epochs", 20)),
        )
        exchange = self.backtest_config["exchange_kwargs"]
        simulator_config = PortfolioSimulatorConfig(
            initial_value=float(self.backtest_config["account"]),
            buy_cost=float(exchange["open_cost"]),
            sell_cost=float(exchange["close_cost"]),
            min_cost=float(exchange["min_cost"]),
        )
        action_config = PortfolioActionConfig(max_holdings=strategy_kwargs.get("max_holdings"))
        result = train_dqn(
            train_data=datasets["train"],
            valid_data=datasets["valid"],
            dqn_config=dqn_config,
            simulator_config=simulator_config,
            action_config=action_config,
        )
        test_result = evaluate_dqn(
            result.policy,
            datasets["test"],
            simulator_config=simulator_config,
            action_config=action_config,
            name="dqn",
        )
        output_dir = portfolio_artifact_root_from_strategy(strategy_kwargs) / "training"
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary_checkpoint = output_dir / "best_policy.pt.new"
        save_dqn_checkpoint(result.policy, temporary_checkpoint)
        temporary_checkpoint.replace(output_dir / "best_policy.pt")
        result.history.to_csv(output_dir / "training_history.csv", index=False)
        pd.DataFrame(
            [
                {"split": "valid", **result.validation.metrics},
                {"split": "test", **test_result.metrics},
            ]
        ).to_csv(output_dir / "dqn_metrics.csv", index=False)
        training_record = {
            "dqn": {
                **dqn_config.__dict__,
                "hidden_dims": list(dqn_config.hidden_dims),
            },
            "best_epoch": result.best_epoch,
            "rl_segments": {
                name: {"start": str(segment.start.date()), "end": str(segment.end.date())}
                for name, segment in segments.items()
            },
            "split_transition_counts": {
                name: len(dataset.decision_dates) for name, dataset in datasets.items()
            },
            "source": "PortfolioDQNRecord",
            "action": {"max_holdings": action_config.max_holdings},
        }
        (output_dir / "training_config.json").write_text(
            json.dumps(training_record, indent=2),
            encoding="utf-8",
        )


def _parse_segments(values: dict | None) -> dict[str, DateRange]:
    if values is None or set(values) != {"train", "valid", "test"}:
        raise ValueError("backtest.rl_segments must define train, valid, and test.")
    segments = {name: DateRange(bounds[0], bounds[1]) for name, bounds in values.items()}
    if not (
        segments["train"].end < segments["valid"].start
        and segments["valid"].end < segments["test"].start
    ):
        raise ValueError("RL segments must be non-overlapping and chronological.")
    return segments


def _select_predictions(predictions: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if isinstance(predictions, pd.Series):
        predictions = predictions.to_frame("score")
    if not isinstance(predictions, pd.DataFrame) or predictions.shape[1] != 1:
        raise ValueError("Portfolio DQN training requires one prediction-score column.")
    selected = predictions.copy()
    selected.columns = ["score"]
    selected = selected.reorder_levels(["datetime", "instrument"]).sort_index()
    dates = pd.DatetimeIndex(selected.index.get_level_values("datetime"))
    selected = selected.loc[(dates >= start) & (dates <= end)]
    if selected.empty:
        raise ValueError("Prediction record does not cover the configured RL segments.")
    return selected


def _single_artifact_key(artifacts: dict[str, Any], prefix: str) -> str:
    matches = [key for key in artifacts if key.startswith(prefix) and key.endswith(".pkl")]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {prefix} artifact, found {matches}.")
    return matches[0]


def _positions_to_frame(positions: dict) -> pd.DataFrame:
    rows = []
    reserved = {"cash", "now_account_value", "cash_delay"}
    for date, position in sorted(positions.items()):
        account_value = float(position.calculate_value())
        for instrument, values in position.position.items():
            if instrument in reserved or not isinstance(values, dict):
                continue
            amount = float(values.get("amount", 0.0))
            price = float(values.get("price", 0.0))
            market_value = amount * price
            rows.append(
                {
                    "datetime": pd.Timestamp(date),
                    "instrument": str(instrument),
                    "amount": amount,
                    "price": price,
                    "market_value": market_value,
                    "weight": float(values.get("weight", market_value / account_value if account_value else 0.0)),
                    "holding_sessions": int(values.get("count", 0)),
                }
            )
    return pd.DataFrame(rows)


def _market_vs_dqn(report: pd.DataFrame, initial_value: float) -> pd.DataFrame:
    dqn_return = float(report["account"].iloc[-1] / initial_value - 1.0)
    market_return = float((1.0 + report["bench"].fillna(0.0)).prod() - 1.0)
    return pd.DataFrame(
        [
            {"strategy": "dqn", "total_return": dqn_return, "final_value": float(report["account"].iloc[-1])},
            {"strategy": "market", "total_return": market_return, "final_value": initial_value * (1.0 + market_return)},
        ]
    )


def _strategy_diagnostics(actions: pd.DataFrame, holdings: pd.DataFrame) -> pd.DataFrame:
    counts = actions["action"].value_counts()
    dominant_action = counts.index[0] if not counts.empty else None
    dominant_frequency = float(counts.iloc[0] / len(actions)) if len(actions) else 0.0
    return pd.DataFrame(
        [
            {"metric": "decision_count", "value": len(actions)},
            {"metric": "executed_order_count", "value": int(actions["executed_order_count"].sum())},
            {"metric": "dominant_action", "value": dominant_action},
            {"metric": "dominant_action_frequency", "value": dominant_frequency},
            {
                "metric": "maximum_holding_sessions",
                "value": int(holdings["holding_sessions"].max()) if not holdings.empty else 0,
            },
        ]
    )


def _write_csv_atomically(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".new")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_json_atomically(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)
