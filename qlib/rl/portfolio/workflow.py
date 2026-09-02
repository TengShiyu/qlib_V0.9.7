# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qlib workflow record that retrains portfolio DQN before backtesting."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from qlib.data import D
from qlib.rl.checkpoint import RollingCheckpointRun
from qlib.workflow.record_temp import PortAnaRecord

from .config import portfolio_artifact_root_from_strategy
from .action import PortfolioActionConfig
from .data import DateRange, build_portfolio_data_split
from .policy import PortfolioDQNConfig
from .simulator import PortfolioSimulatorConfig
from .training import (
    action_config_record,
    config_record,
    cumulative_learning_return,
    save_dqn_checkpoint,
    simulator_config_record,
    train_dqn,
)
from .rolling import generate_portfolio_rolling_windows, rolling_windows_frame


class PortfolioDQNRecord(PortAnaRecord):
    """Retrain one portfolio DQN per rolling score block, then run analysis."""

    def __init__(
        self,
        *args,
        rolling_config: dict | None = None,
        rolling_segments: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rolling_config = _parse_rolling_config(rolling_config)
        self.rolling_segments = _parse_segments(rolling_segments)
        strategy_kwargs = self.strategy_config.get("kwargs", {})
        configured_interval = strategy_kwargs.get("trading_interval")
        if configured_interval is not None and int(configured_interval) != self.rolling_config["trading_interval"]:
            raise ValueError("Portfolio strategy trading_interval must match task.rolling.trading_interval.")
        strategy_kwargs["trading_interval"] = self.rolling_config["trading_interval"]

    def _generate(self, **kwargs):
        if self.strategy_config.get("class") != "PortfolioDQNStrategy":
            raise ValueError("PortfolioDQNRecord requires PortfolioDQNStrategy.")
        strategy_kwargs = self.strategy_config.get("kwargs", {})
        if strategy_kwargs.get("rolling_retrain", True):
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
        artifact_root = portfolio_artifact_root_from_strategy(strategy_kwargs)
        windows = pd.read_csv(artifact_root / "rolling" / "rolling_windows.csv")
        rolling_phase = _rolling_phase_by_date(report.index, windows)

        portfolio_report = report.reset_index().rename(columns={report.index.name or "index": "datetime"})
        portfolio_report["rolling_phase"] = rolling_phase.to_numpy()
        _write_csv_atomically(portfolio_report, output_dir / "portfolio_report.csv")
        holdings = _positions_to_frame(positions)
        if not holdings.empty:
            phase_map = pd.Series(rolling_phase.to_numpy(), index=pd.DatetimeIndex(report.index).normalize())
            holding_dates = pd.DatetimeIndex(holdings["datetime"]).normalize()
            holdings["rolling_phase"] = phase_map.reindex(holding_dates).to_numpy()
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

        comparison = _market_vs_dqn(
            report, float(self.backtest_config["account"]), rolling_phase=rolling_phase
        )
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
        segments = self.rolling_segments
        windows = generate_portfolio_rolling_windows(
            segments,
            trading_interval=self.rolling_config["trading_interval"],
            step=self.rolling_config["step"],
            rolling_type=self.rolling_config["rolling_type"],
            rl_split_ratio=self.rolling_config["rl_split_ratio"],
        )
        artifact_root = portfolio_artifact_root_from_strategy(strategy_kwargs)
        rolling_dir = artifact_root / "rolling"
        rolling_dir.mkdir(parents=True, exist_ok=True)
        windows_frame = rolling_windows_frame(windows)
        selected = _select_predictions(
            predictions, windows[0].score_block.start, windows[-1].score_block.end
        )
        instruments = tuple(sorted(selected.index.get_level_values("instrument").astype(str).unique()))
        market_start = windows[0].train.start - pd.Timedelta(days=90)
        market_end = windows[-1].test.end
        calendar = pd.DatetimeIndex(D.calendar(start_time=market_start, end_time=market_end, freq="day"))
        market = D.features(
            list(instruments),
            ["$close", "$volume"],
            start_time=market_start,
            end_time=market_end,
            freq="day",
        ).rename(columns={"$close": "close", "$volume": "volume"})
        market = market.reorder_levels(["datetime", "instrument"]).sort_index()
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
            turnover_penalty=float(strategy_kwargs.get("turnover_penalty", 0.0)),
        )
        action_config = PortfolioActionConfig(
            max_holdings=strategy_kwargs.get("max_holdings"),
            increase_exposure_ratio=float(strategy_kwargs.get("increase_exposure_ratio", 0.25)),
            concentrated_holdings=int(strategy_kwargs.get("concentrated_holdings", 5)),
            concentrated_equity=float(strategy_kwargs.get("concentrated_equity", 0.95)),
            rotation_count=int(strategy_kwargs.get("rotation_count", 2)),
            defensive_equity=float(strategy_kwargs.get("defensive_equity", 0.50)),
        )
        output_dir = artifact_root / "training"
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = rolling_dir / "checkpoints"
        history_path = rolling_dir / "checkpoint_history.csv"
        checkpoint_columns = [
            "window_id",
            "score_block_start",
            "score_block_end",
            "train_start",
            "train_end",
            "valid_start",
            "valid_end",
            "test_start",
            "test_end",
            "best_epoch",
            "train_transition_count",
            "valid_transition_count",
            "checkpoint_path",
            "completed_at_utc",
        ]
        checkpoint_records: list[dict[str, object]] = []
        all_training_history = []
        validation_metrics = []
        validation_transitions = []
        window_ids = [window.window_id for window in windows]
        with RollingCheckpointRun(checkpoints_dir, window_ids) as checkpoint_run:
            for window in windows:
                train_data = build_portfolio_data_split(
                    predictions=selected,
                    market=market,
                    calendar=calendar,
                    date_range=window.train,
                    instruments=instruments,
                )
                valid_data = build_portfolio_data_split(
                    predictions=selected,
                    market=market,
                    calendar=calendar,
                    date_range=window.valid,
                    instruments=instruments,
                )
                result = train_dqn(
                    train_data=train_data,
                    valid_data=valid_data,
                    dqn_config=dqn_config,
                    simulator_config=simulator_config,
                    action_config=action_config,
                )
                checkpoint_path = checkpoint_run.checkpoint_path(window.window_id)
                save_dqn_checkpoint(result.policy, checkpoint_path)
                window_dir = checkpoint_path.parent

                window_history = result.history.copy()
                window_history.insert(0, "window_id", window.window_id)
                _write_csv_atomically(window_history, window_dir / "training_history.csv")
                all_training_history.append(window_history)
                window_validation = result.validation.transitions.copy()
                window_validation.insert(0, "window_id", window.window_id)
                _write_csv_atomically(
                    window_validation, window_dir / "validation_transitions.csv"
                )
                validation_transitions.append(window_validation)
                validation_metrics.append(
                    {
                        "window_id": window.window_id,
                        "split": "valid",
                        "validation_learning_return": cumulative_learning_return(
                            result.validation.transitions
                        ),
                        **result.validation.metrics,
                    }
                )
                official_checkpoint = checkpoint_run.official_checkpoint_path(window.window_id)
                checkpoint_records.append(
                    {
                        **window.as_record(),
                        "best_epoch": result.best_epoch,
                        "train_transition_count": len(train_data.decision_dates),
                        "valid_transition_count": len(valid_data.decision_dates),
                        "checkpoint_path": str(official_checkpoint),
                        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
            checkpoint_run.publish()

        _write_csv_atomically(windows_frame, rolling_dir / "rolling_windows.csv")
        _write_csv_atomically(
            pd.DataFrame(checkpoint_records, columns=checkpoint_columns), history_path
        )
        latest_record = checkpoint_records[-1]
        latest_checkpoint = Path(str(latest_record["checkpoint_path"]))
        _promote_checkpoint(latest_checkpoint, output_dir / "best_policy.pt")
        _write_json_atomically(
            {
                "window_id": latest_record["window_id"],
                "checkpoint_path": str(latest_checkpoint),
                "completed_at_utc": latest_record["completed_at_utc"],
            },
            output_dir / "latest_checkpoint.json",
        )

        _write_csv_atomically(
            pd.concat(all_training_history, ignore_index=True), output_dir / "training_history.csv"
        )
        _write_csv_atomically(pd.DataFrame(validation_metrics), output_dir / "dqn_metrics.csv")
        _write_csv_atomically(
            pd.concat(validation_transitions, ignore_index=True),
            output_dir / "dqn_validation_transitions.csv",
        )
        training_record = {
            "dqn": config_record(dqn_config),
            "best_epoch": latest_record["best_epoch"],
            "rolling_segments": {
                name: {"start": str(segment.start.date()), "end": str(segment.end.date())}
                for name, segment in segments.items()
            },
            "latest_window_id": latest_record["window_id"],
            "checkpoint_history_path": str(history_path),
            "rolling": {
                **self.rolling_config,
                "window_count": len(windows),
                "windows_path": str(rolling_dir / "rolling_windows.csv"),
            },
            "source": "PortfolioDQNRecord",
            "simulator": simulator_config_record(simulator_config),
            "action": action_config_record(action_config),
            "validation_transitions_path": str(
                output_dir / "dqn_validation_transitions.csv"
            ),
            "checkpoint_selection_metric": "validation_learning_return",
        }
        _write_json_atomically(training_record, output_dir / "training_config.json")


def _parse_segments(values: dict | None) -> dict[str, DateRange]:
    if values is None or set(values) != {"train", "valid", "test"}:
        raise ValueError("The shared ML rolling segments must define train, valid, and test.")
    segments = {name: DateRange(bounds[0], bounds[1]) for name, bounds in values.items()}
    if not (
        segments["train"].end < segments["valid"].start
        and segments["valid"].end < segments["test"].start
    ):
        raise ValueError("RL segments must be non-overlapping and chronological.")
    return segments


def _parse_rolling_config(values: dict | None) -> dict[str, object]:
    if values is None:
        raise ValueError("PortfolioDQNRecord requires the shared task.rolling configuration.")
    try:
        trading_interval = int(values["trading_interval"])
        step = int(values["step"])
        rolling_type = str(values["rolling_type"]).upper()
        split_ratio = tuple(int(value) for value in values["rl_split_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "task.rolling must define trading_interval, step, and rolling_type."
        ) from exc
    if trading_interval <= 0:
        raise ValueError("task.rolling.trading_interval must be positive.")
    if step <= 0:
        raise ValueError("task.rolling.step must be positive.")
    if rolling_type not in {"ROLL_EX", "ROLL_SD"}:
        raise ValueError("task.rolling.rolling_type must be ROLL_EX or ROLL_SD.")
    if len(split_ratio) != 3 or any(value <= 0 for value in split_ratio):
        raise ValueError("task.rolling.rl_split_ratio must contain three positive integers.")
    return {
        "trading_interval": trading_interval,
        "step": step,
        "rolling_type": rolling_type,
        "rl_split_ratio": list(split_ratio),
    }


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


def _market_vs_dqn(
    report: pd.DataFrame, initial_value: float, rolling_phase: pd.Series | None = None
) -> pd.DataFrame:
    dqn_return = float(report["account"].iloc[-1] / initial_value - 1.0)
    market_return = float((1.0 + report["bench"].fillna(0.0)).prod() - 1.0)
    test_mask = (
        pd.Series(True, index=report.index)
        if rolling_phase is None
        else rolling_phase.set_axis(report.index).eq("dqn_test")
    )
    dqn_test_return = float((1.0 + report.loc[test_mask, "return"].fillna(0.0)).prod() - 1.0)
    market_test_return = float((1.0 + report.loc[test_mask, "bench"].fillna(0.0)).prod() - 1.0)
    return pd.DataFrame(
        [
            {
                "strategy": "dqn",
                "total_return": dqn_return,
                "dqn_test_session_return": dqn_test_return,
                "final_value": float(report["account"].iloc[-1]),
            },
            {
                "strategy": "market",
                "total_return": market_return,
                "dqn_test_session_return": market_test_return,
                "final_value": initial_value * (1.0 + market_return),
            },
        ]
    )


def _rolling_phase_by_date(dates, windows: pd.DataFrame) -> pd.Series:
    required = {
        "train_start",
        "train_end",
        "valid_start",
        "valid_end",
        "test_start",
        "test_end",
    }
    if not required.issubset(windows):
        raise ValueError("rolling_windows.csv is missing phase boundary columns.")
    normalized_dates = pd.DatetimeIndex(dates).normalize()
    phases = pd.Series("outside", index=normalized_dates, dtype="object")
    boundaries = [
        ("rl_train", "train_start", "train_end"),
        ("rl_valid", "valid_start", "valid_end"),
        ("dqn_test", "test_start", "test_end"),
    ]
    for row in windows.itertuples(index=False):
        for phase, start_column, end_column in boundaries:
            start = pd.Timestamp(getattr(row, start_column)).normalize()
            end = pd.Timestamp(getattr(row, end_column)).normalize()
            mask = (normalized_dates >= start) & (normalized_dates <= end)
            phases.loc[mask] = phase
    return phases


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


def _promote_checkpoint(source: Path, destination: Path) -> None:
    """Atomically make a completed window checkpoint the latest policy."""

    if not source.is_file():
        raise FileNotFoundError(f"Completed rolling checkpoint does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".new")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _write_json_atomically(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)
