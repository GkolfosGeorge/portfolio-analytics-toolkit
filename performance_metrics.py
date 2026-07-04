"""
performance_metrics.py — Portfolio Performance Metrics
=======================================================
All functions accept a pd.Series of portfolio values (price index)
with a DatetimeIndex, unless stated otherwise.

Return values are raw decimals (e.g. 0.12 = 12%) unless the
function name contains a hint like _pct.
"""

import numpy as np
import pandas as pd


def total_return(portfolio_values: pd.Series) -> float:
    """
    Total return over the full period.

    Args:
        portfolio_values : pd.Series  price index with DatetimeIndex

    Returns:
        float  e.g. 0.45 means +45%
    """
    return (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1


def cagr(portfolio_values: pd.Series) -> float:
    """
    Compound Annual Growth Rate (geometric annualized return).

    Args:
        portfolio_values : pd.Series  price index with DatetimeIndex

    Returns:
        float  e.g. 0.12 means 12% per year
    """
    start = portfolio_values.iloc[0]
    end   = portfolio_values.iloc[-1]
    years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25
    return (end / start) ** (1.0 / years) - 1


def volatility(portfolio_values: pd.Series, trading_days: int = 252) -> float:
    """
    Annualized volatility (standard deviation of daily returns).

    Args:
        portfolio_values : pd.Series
        trading_days     : int  default 252

    Returns:
        float  e.g. 0.15 means 15% annualized vol
    """
    returns = portfolio_values.pct_change().dropna()
    return returns.std() * np.sqrt(trading_days)


def sharpe_ratio(portfolio_values: pd.Series,
                 risk_free_rate: float = 0.0,
                 trading_days: int = 252) -> float:
    """
    Annualized Sharpe Ratio.

    Args:
        portfolio_values : pd.Series
        risk_free_rate   : float  annual rate as decimal (e.g. 0.02 for 2%)
        trading_days     : int    default 252

    Returns:
        float
    """
    returns        = portfolio_values.pct_change().dropna()
    excess_returns = returns - risk_free_rate / trading_days
    return np.sqrt(trading_days) * excess_returns.mean() / excess_returns.std()


def sortino_ratio(portfolio_values: pd.Series,
                  risk_free_rate: float = 0.0,
                  trading_days: int = 252) -> float:
    """
    Annualized Sortino Ratio.

    Uses only negative daily returns to compute downside deviation,
    which avoids artificially inflating the ratio by including zeros
    (a common implementation error).

    Args:
        portfolio_values : pd.Series
        risk_free_rate   : float  annual rate as decimal (e.g. 0.02 for 2%)
        trading_days     : int    default 252

    Returns:
        float  or 0.0 if downside std is zero
    """
    returns = portfolio_values.pct_change().dropna()

    # FIX: filter only negative observations — do NOT zero-out positives.
    # Zeroing positives inflates the observation count and understates
    # downside std, producing an artificially high Sortino ratio.
    downside_returns = returns[returns < 0]
    downside_std     = downside_returns.std() * np.sqrt(trading_days)

    if downside_std == 0 or np.isnan(downside_std):
        return 0.0

    excess_return = returns.mean() * trading_days - risk_free_rate
    return excess_return / downside_std


def calmar_ratio(portfolio_values: pd.Series) -> float:
    """
    Calmar Ratio: CAGR divided by absolute Maximum Drawdown.

    Args:
        portfolio_values : pd.Series

    Returns:
        float  or 0.0 if max drawdown is zero
    """
    from drawdown_metrics import max_drawdown  # local import to avoid circular dep
    mdd = abs(max_drawdown(portfolio_values))
    if mdd == 0:
        return 0.0
    return cagr(portfolio_values) / mdd


def mean_return(portfolio_values: pd.Series,
                trading_days: int = 252) -> float:
    """
    Arithmetic (annualized) mean return.

    Args:
        portfolio_values : pd.Series
        trading_days     : int

    Returns:
        float
    """
    returns = portfolio_values.pct_change().dropna()
    return returns.mean() * trading_days


def volatility_drag(portfolio_values: pd.Series,
                    trading_days: int = 252) -> float:
    """
    Volatility drag: difference between arithmetic mean return and CAGR.
    Represents the return lost due to compounding of volatility.

    Args:
        portfolio_values : pd.Series
        trading_days     : int

    Returns:
        float  (positive value = drag on CAGR)
    """
    return mean_return(portfolio_values, trading_days) - cagr(portfolio_values)