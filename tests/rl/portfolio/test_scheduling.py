"""Decision clocks use actual exchange sessions, including at route boundaries."""

import pandas as pd
import pytest

from qlib.rl.portfolio.scheduling import scheduled_decision_dates, next_execution_session


# Synthetic exchange calendar: weekend plus a Monday closure, with no weekday fill.
SESSIONS = pd.DatetimeIndex([
    "2025-01-16", "2025-01-17", "2025-01-21", "2025-01-22",
    "2025-01-23", "2025-01-24", "2025-01-27",
])


def test_holidays_and_weekends_use_exchange_sessions():
    decisions = scheduled_decision_dates(SESSIONS, "2025-01-17", "2025-01-24")
    assert decisions.tolist() == list(pd.to_datetime(["2025-01-17", "2025-01-22", "2025-01-24"]))
    assert [next_execution_session(SESSIONS, date) for date in decisions] == list(
        pd.to_datetime(["2025-01-21", "2025-01-23", "2025-01-27"]))


def test_late_availability_preserves_original_phase():
    decisions = scheduled_decision_dates(SESSIONS, "2025-01-17", "2025-01-24", available_as_of="2025-01-21")
    assert decisions.tolist() == list(pd.to_datetime(["2025-01-22", "2025-01-24"]))
    assert scheduled_decision_dates(SESSIONS, "2025-01-17", "2025-01-24", available_as_of="2025-01-27").empty


def test_phase_is_anchored_to_each_route_and_execution_can_cross_route_end():
    first = scheduled_decision_dates(SESSIONS, "2025-01-17", "2025-01-17")
    second = scheduled_decision_dates(SESSIONS, "2025-01-21", "2025-01-24")
    assert first.tolist() == [pd.Timestamp("2025-01-17")]
    assert second.tolist() == list(pd.to_datetime(["2025-01-21", "2025-01-23"]))
    assert next_execution_session(SESSIONS, first[-1]) == second[0]


def test_unknown_next_session_does_not_invent_a_weekday():
    assert next_execution_session(SESSIONS, SESSIONS[-1]) is None
    with pytest.raises(ValueError, match="known trading session"):
        next_execution_session(SESSIONS, "2025-01-20")


@pytest.mark.parametrize("interval", [0, -1, True, 1.5])
def test_invalid_intervals_are_rejected(interval):
    with pytest.raises(ValueError, match="positive integer"):
        scheduled_decision_dates(SESSIONS, SESSIONS[0], SESSIONS[-1], interval)
