"""
drawdown_metrics.py — Drawdown Analysis
========================================
All functions accept a pd.Series of portfolio values (price index)
with a DatetimeIndex.

Drawdown values are expressed as decimals in the range [-1, 0].
Duration values are in trading days.
"""

import numpy as np
import pandas as pd


def drawdown_series(portfolio_values: pd.Series) -> pd.Series:
    """
    Compute the drawdown at every point in time.

    Drawdown(t) = (Value(t) - RunningMax(t)) / RunningMax(t)

    Args:
        portfolio_values : pd.Series  price index with DatetimeIndex

    Returns:
        pd.Series  values in [-1, 0]
    """
    running_max = portfolio_values.cummax()
    return (portfolio_values - running_max) / running_max


def max_drawdown(portfolio_values: pd.Series) -> float:
    """
    Maximum drawdown over the full period (most negative value).

    Args:
        portfolio_values : pd.Series

    Returns:
        float  e.g. -0.35 means -35%
    """
    return drawdown_series(portfolio_values).min()


def drawdown_duration(portfolio_values: pd.Series) -> pd.Series:
    """
    Number of consecutive trading days spent in drawdown at each date.

    When the portfolio is at an all-time high the value is 0;
    otherwise it counts up from the day the drawdown began.

    Args:
        portfolio_values : pd.Series

    Returns:
        pd.Series  integer counts (0 = at peak, N = N days below peak)
    """
    dd = drawdown_series(portfolio_values)

    # FIX: original code used 'duration' before it was computed correctly.
    # Correct approach: mark every day in drawdown (1) vs at peak (0),
    # then group by each recovery-to-peak reset and cumsum within groups.
    in_drawdown = (dd < 0).astype(int)
    groups      = (in_drawdown == 0).cumsum()          # increments at each peak
    duration    = in_drawdown.groupby(groups).cumsum() # resets to 0 at each peak

    return duration


def max_drawdown_duration(portfolio_values: pd.Series) -> int:
    """
    Longest consecutive streak of days spent below a previous peak.

    Args:
        portfolio_values : pd.Series

    Returns:
        int  number of trading days
    """
    return int(drawdown_duration(portfolio_values).max())


def recovery_analysis(portfolio_values: pd.Series) -> dict:
    """
    Identify the date and duration of the maximum drawdown,
    and the date of full recovery (if any).

    Args:
        portfolio_values : pd.Series

    Returns:
        dict with keys:
            mdd            : float  max drawdown value (negative decimal)
            mdd_date       : pd.Timestamp  date of trough
            peak_date      : pd.Timestamp  date of prior peak
            recovery_date  : pd.Timestamp or None
            recovery_days  : int or None  calendar days peak -> recovery
    """
    dd          = drawdown_series(portfolio_values)
    mdd_date    = dd.idxmin()
    mdd_val     = dd.min()

    # Walk back to find the peak that preceded the trough
    pre_trough  = portfolio_values[:mdd_date]
    peak_date   = pre_trough.idxmax()

    # Walk forward from trough to find first full recovery
    post_trough = dd[mdd_date:]
    recovered   = post_trough[post_trough >= 0]

    if len(recovered) > 0:
        recovery_date = recovered.index[0]
        recovery_days = (recovery_date - peak_date).days
    else:
        recovery_date = None
        recovery_days = None

    return {
        "mdd":           mdd_val,
        "mdd_date":      mdd_date,
        "peak_date":     peak_date,
        "recovery_date": recovery_date,
        "recovery_days": recovery_days,
    }