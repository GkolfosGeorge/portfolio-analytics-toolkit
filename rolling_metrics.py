"""
rolling_metrics.py — Rolling Risk & Performance Metrics
=========================================================
All functions accept a pd.Series of daily returns with a DatetimeIndex.

Rolling series contain NaN for the first (window - 1) observations,
which is the expected pandas behavior.
"""

import numpy as np
import pandas as pd


def rolling_volatility(returns: pd.Series,
                        window: int = 252,
                        trading_days: int = 252) -> pd.Series:
    """
    Annualized rolling volatility.

    Args:
        returns      : pd.Series  daily returns
        window       : int        rolling window in trading days
        trading_days : int        annualization factor

    Returns:
        pd.Series  annualized volatility (decimal)
    """
    return returns.rolling(window=window).std() * np.sqrt(trading_days)


def rolling_sharpe(returns: pd.Series,
                   window: int = 252,
                   risk_free_rate: float = 0.0,
                   trading_days: int = 252) -> pd.Series:
    """
    Rolling annualized Sharpe Ratio.

    Args:
        returns        : pd.Series  daily returns
        window         : int        rolling window in trading days
        risk_free_rate : float      annual rate as decimal (e.g. 0.02)
        trading_days   : int        annualization factor

    Returns:
        pd.Series
    """
    excess_returns = returns - (risk_free_rate / trading_days)
    rolling_mean   = excess_returns.rolling(window=window).mean() * trading_days
    rolling_vol    = rolling_volatility(returns, window, trading_days)
    return rolling_mean / rolling_vol


def rolling_sortino(returns: pd.Series,
                    window: int = 252,
                    risk_free_rate: float = 0.0,
                    trading_days: int = 252) -> pd.Series:
    """
    Rolling annualized Sortino Ratio.

    Downside deviation is computed using clip(upper=0), which preserves
    zero for days with non-negative returns without inflating the count
    of negative observations.

    FIX vs original: used clip(upper=0) instead of zeroing positives via
    boolean indexing, which is the mathematically correct approach and
    avoids the copy-then-mutate anti-pattern.

    Args:
        returns        : pd.Series  daily returns
        window         : int        rolling window in trading days
        risk_free_rate : float      annual rate as decimal
        trading_days   : int        annualization factor

    Returns:
        pd.Series
    """
    # clip(upper=0): keep negatives as-is, set positives to 0.
    # This is equivalent to using only negative returns for std,
    # but compatible with rolling().std().
    downside           = returns.clip(upper=0)
    rolling_down_std   = downside.rolling(window=window).std() * np.sqrt(trading_days)
    rolling_mean       = returns.rolling(window=window).mean() * trading_days
    excess_return      = rolling_mean - risk_free_rate
    return excess_return / rolling_down_std


def rolling_max_drawdown(portfolio_values: pd.Series,
                          window: int = 252) -> pd.Series:
    """
    Rolling maximum drawdown over a trailing window.

    At each date, computes the worst peak-to-trough decline
    within the preceding `window` trading days.

    Args:
        portfolio_values : pd.Series  price index with DatetimeIndex
        window           : int        lookback in trading days

    Returns:
        pd.Series  values in [-1, 0]
    """
    roll_max = portfolio_values.rolling(window=window, min_periods=1).max()
    return (portfolio_values - roll_max) / roll_max


def rolling_cagr(portfolio_values: pd.Series,
                  window: int = 252,
                  trading_days: int = 252) -> pd.Series:
    """
    Rolling annualized CAGR over a trailing window.

    Args:
        portfolio_values : pd.Series  price index
        window           : int        rolling window in trading days
        trading_days     : int        trading days per year

    Returns:
        pd.Series
    """
    years = window / trading_days
    return (portfolio_values / portfolio_values.shift(window)) ** (1.0 / years) - 1


def rolling_beta(portfolio_returns: pd.Series,
                  benchmark_returns: pd.Series,
                  window: int = 252) -> pd.Series:
    """
    Rolling beta of the portfolio relative to a benchmark.

    Args:
        portfolio_returns  : pd.Series  daily returns
        benchmark_returns  : pd.Series  benchmark daily returns
        window             : int        rolling window in trading days

    Returns:
        pd.Series
    """
    def _beta(port, bench):
        cov = np.cov(port, bench)
        return cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else np.nan

    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    p_col   = aligned.columns[0]
    b_col   = aligned.columns[1]

    return (
        aligned[p_col]
        .rolling(window)
        .corr(aligned[b_col])
        .mul(
            aligned[p_col].rolling(window).std()
            / aligned[b_col].rolling(window).std()
        )
    )