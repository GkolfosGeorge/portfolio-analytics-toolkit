"""
risk_metrics.py — Risk Metrics
================================
Functions for measuring portfolio risk.

Convention:
  - portfolio_values : pd.Series  price index with DatetimeIndex
  - returns          : pd.Series  daily returns (pct_change)
  - VaR / ES values are returned as negative decimals (e.g. -0.02 = -2% loss)
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
        trading_days : int        annualization factor (default 252)

    Returns:
        pd.Series  annualized volatility (decimal, e.g. 0.15 = 15%)
    """
    return returns.rolling(window=window).std() * np.sqrt(trading_days)


def var(portfolio_values: pd.Series,
        confidence: float = 0.95) -> float:
    """
    Historical Value at Risk (VaR) at the given confidence level.

    Sign convention: returns a NEGATIVE number representing the loss
    threshold (e.g. -0.0166 means -1.66%).
    At a 95% confidence level, losses exceed this value only 5% of days.

    Args:
        portfolio_values : pd.Series
        confidence       : float  e.g. 0.95 for 95% confidence

    Returns:
        float  negative decimal
    """
    returns = portfolio_values.pct_change().dropna()
    return float(np.percentile(returns, (1 - confidence) * 100))


def expected_shortfall(portfolio_values: pd.Series,
                        confidence: float = 0.95) -> float:
    """
    Expected Shortfall (CVaR) — average loss on days that breach the VaR.

    Also known as Conditional Value at Risk (CVaR).
    More conservative and informative than VaR for tail risk.

    Sign convention: returns a NEGATIVE number (e.g. -0.025 = -2.5%).

    Args:
        portfolio_values : pd.Series
        confidence       : float  e.g. 0.95

    Returns:
        float  negative decimal
    """
    returns   = portfolio_values.pct_change().dropna()
    var_level = float(np.percentile(returns, (1 - confidence) * 100))
    return float(returns[returns <= var_level].mean())


def var_scaled(portfolio_values: pd.Series,
               confidence: float = 0.95,
               horizon_days: int = 21) -> float:
    """
    VaR scaled to a multi-day horizon using the Square Root of Time rule.

    Note: assumes i.i.d. returns (appropriate for illustrative purposes;
    not suitable for fat-tailed or autocorrelated return series).

    Args:
        portfolio_values : pd.Series
        confidence       : float  e.g. 0.95
        horizon_days     : int    e.g. 21 for monthly, 252 for annual

    Returns:
        float  negative decimal
    """
    daily_var = var(portfolio_values, confidence)
    return daily_var * np.sqrt(horizon_days)


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise Pearson correlation matrix of asset returns.

    Args:
        returns : pd.DataFrame  daily returns (columns = tickers)

    Returns:
        pd.DataFrame  correlation matrix
    """
    return returns.corr()


def rolling_correlation(returns: pd.DataFrame,
                         asset1: str,
                         asset2: str,
                         window: int = 252) -> pd.Series:
    """
    Rolling pairwise correlation between two assets.

    Args:
        returns : pd.DataFrame  daily returns
        asset1  : str           column name
        asset2  : str           column name
        window  : int           rolling window in trading days

    Returns:
        pd.Series  rolling correlation in [-1, 1]
    """
    return returns[asset1].rolling(window).corr(returns[asset2])


def beta(portfolio_returns: pd.Series,
         benchmark_returns: pd.Series) -> float:
    """
    Portfolio beta relative to a benchmark, computed via OLS.

    Args:
        portfolio_returns  : pd.Series  daily returns
        benchmark_returns  : pd.Series  daily benchmark returns

    Returns:
        float
    """
    aligned      = pd.concat([portfolio_returns, benchmark_returns],
                              axis=1).dropna()
    port, bench  = aligned.iloc[:, 0], aligned.iloc[:, 1]
    covariance   = np.cov(port, bench)[0, 1]
    bench_var    = bench.var()
    return float(covariance / bench_var) if bench_var != 0 else 0.0


def alpha(portfolio_returns: pd.Series,
          benchmark_returns: pd.Series,
          risk_free_rate: float = 0.0,
          trading_days: int = 252) -> float:
    """
    Annualized Jensen's Alpha: excess return over the CAPM-predicted return.

    Args:
        portfolio_returns : pd.Series
        benchmark_returns : pd.Series
        risk_free_rate    : float  annual rate as decimal (e.g. 0.02)
        trading_days      : int

    Returns:
        float  annualized alpha as decimal
    """
    b            = beta(portfolio_returns, benchmark_returns)
    port_ret     = portfolio_returns.mean() * trading_days
    bench_ret    = benchmark_returns.mean() * trading_days
    return port_ret - (risk_free_rate + b * (bench_ret - risk_free_rate))


def information_ratio(portfolio_returns: pd.Series,
                       benchmark_returns: pd.Series,
                       trading_days: int = 252) -> float:
    """
    Information Ratio: annualized active return divided by tracking error.

    Args:
        portfolio_returns : pd.Series
        benchmark_returns : pd.Series
        trading_days      : int

    Returns:
        float
    """
    active_ret   = portfolio_returns - benchmark_returns
    tracking_err = active_ret.std() * np.sqrt(trading_days)
    if tracking_err == 0:
        return 0.0
    return float(active_ret.mean() * trading_days / tracking_err)