# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Trading-session scheduling shared by live inference and portfolio backtests."""

from numbers import Integral

import pandas as pd


def session_calendar(calendar) -> pd.DatetimeIndex:
    """Normalize known exchange sessions without inserting weekdays or holidays."""
    dates = pd.DatetimeIndex(calendar)
    if dates.hasnans or dates.tz is not None:
        raise ValueError("Trading calendar must contain valid timezone-naive session dates.")
    return dates.normalize().drop_duplicates().sort_values()


def scheduled_decision_dates(calendar, start, end, trading_interval=2, available_as_of=None) -> pd.DatetimeIndex:
    """Anchor decisions at the route's first session and keep that phase when late.

    Availability filters the anchored schedule; it never starts a new interval.
    The caller must provide calendar coverage from the route start, including
    sessions before a late activation. A route ends on its last decision date;
    the final decision's execution may fall beyond the route end.
    """
    if isinstance(trading_interval, bool) or not isinstance(trading_interval, Integral) or trading_interval <= 0:
        raise ValueError("trading_interval must be a positive integer.")
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("A policy route must have valid, ordered start and end dates.")
    dates = session_calendar(calendar)
    decisions = dates[(dates >= start) & (dates <= end)][::trading_interval]
    if available_as_of is not None:
        available = pd.Timestamp(available_as_of).normalize()
        if pd.isna(available):
            raise ValueError("Policy availability must be a valid date.")
        decisions = decisions[decisions >= available]
    return decisions


def next_execution_session(calendar, decision_date):
    """Return the next known exchange session, or None at an unknown boundary."""
    dates = session_calendar(calendar)
    decision = pd.Timestamp(decision_date).normalize()
    if decision not in dates:
        raise ValueError("The decision date must be a known trading session.")
    offset = dates.get_loc(decision) + 1
    return dates[offset] if offset < len(dates) else None
