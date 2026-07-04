"""
dca_strategies.py — Dollar-Cost Averaging Strategy Engine
===========================================================
Three DCA strategies for backtesting systematic investment plans:

  1. Simple DCA       — fixed amount every month-end, no market timing
  2. Drawdown DCA     — increases investment when an asset falls more than
                        its rolling volatility-based threshold
  3. MA200 DCA        — tiered multipliers based on price deviation below
                        the 200-day moving average

All strategies return a (portfolio_value_series, invested_series) tuple
so they can be compared directly.

Helper functions:
  - get_monthly_dates()         — last trading day of each month
  - run_dca_backtest()          — generic equity-curve builder from returns
  - dca_summary_metrics()       — summary statistics table
  - compute_drawdown_signal()   — signal used by Drawdown DCA
  - compute_ma_multipliers()    — multiplier map used by MA200 DCA
"""

import numpy as np
import pandas as pd


# ── Date utilities ────────────────────────────────────────────────────────────

def get_monthly_dates(index: pd.DatetimeIndex) -> np.ndarray:
    """
    Return the last trading day of each calendar month in `index`.

    Args:
        index : pd.DatetimeIndex  trading day index of a price DataFrame

    Returns:
        np.ndarray of pd.Timestamps
    """
    return (
        pd.Series(index, index=index)
        .resample("ME")
        .last()
        .dropna()
        .values
    )


# ── Generic equity-curve builder (for external returns series) ────────────────

def run_dca_backtest(daily_returns: pd.Series,
                      initial_capital: float,
                      monthly_amount: float) -> pd.Series:
    """
    Build a DCA equity curve from a pre-computed daily returns series.

    Contribution is added at the last trading day of each month.
    The contribution buys into the portfolio at that day's cumulative level.

    This function is useful when the DCA target is described by a
    returns series rather than individual price history (e.g. an
    all-weather portfolio's returns).

    Args:
        daily_returns   : pd.Series  daily portfolio returns
        initial_capital : float
        monthly_amount  : float      fixed monthly investment

    Returns:
        pd.Series  portfolio value index (same index as daily_returns)
    """
    monthly_end = set(get_monthly_dates(daily_returns.index))
    value       = initial_capital
    values      = []

    for date, ret in daily_returns.items():
        if date in monthly_end:
            value += monthly_amount
        value *= (1 + ret)
        values.append(value)

    return pd.Series(values, index=daily_returns.index, name="DCA Portfolio")


# ── Strategy 1: Simple DCA ────────────────────────────────────────────────────

def simple_dca(prices: pd.DataFrame,
               weights: np.ndarray | list,
               monthly_dates: np.ndarray,
               monthly_amount: float) -> tuple[pd.Series, pd.Series]:
    """
    Invest a fixed amount each month, allocated by weights.

    Args:
        prices         : pd.DataFrame  daily adjusted close prices
        weights        : array-like    portfolio weights (sum to 1)
        monthly_dates  : np.ndarray    purchase dates (from get_monthly_dates)
        monthly_amount : float         fixed monthly investment

    Returns:
        (portfolio_values, invested_series)  both pd.Series with prices.index
    """
    weights        = np.asarray(weights, dtype=float)
    tickers        = list(prices.columns)
    shares         = {t: 0.0 for t in tickers}
    monthly_set    = set(pd.Timestamp(d) for d in monthly_dates)
    total_invested = 0.0
    port_values    = []
    inv_series     = []

    for date in prices.index:
        if date in monthly_set:
            for t, w in zip(tickers, weights):
                price       = prices.loc[date, t]
                shares[t]  += (monthly_amount * w) / price
            total_invested += monthly_amount

        value = sum(shares[t] * prices.loc[date, t] for t in tickers)
        port_values.append(value)
        inv_series.append(total_invested)

    return (
        pd.Series(port_values, index=prices.index, name="Simple DCA"),
        pd.Series(inv_series,  index=prices.index, name="Invested"),
    )


# ── Strategy 2: Drawdown DCA ──────────────────────────────────────────────────

def compute_drawdown_signal(prices: pd.DataFrame,
                              window: int = 20) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute a per-asset boolean signal that fires when the recent drawdown
    exceeds the statistically expected rolling volatility.

    Signal logic:
      rolling_std   = std of daily returns over `window` days
      threshold     = rolling_std * sqrt(window)   (approx. 1-sigma total move)
      drawdown      = (price - rolling_max) / rolling_max
      signal        = |drawdown| > threshold

    Args:
        prices : pd.DataFrame  daily prices
        window : int           lookback window in trading days

    Returns:
        (signal, drawdown, threshold)  — all pd.DataFrame, same shape as prices
    """
    ret          = prices.pct_change()
    rolling_std  = ret.rolling(window).std()
    threshold    = rolling_std * np.sqrt(window)
    rolling_max  = prices.rolling(window).max()
    drawdown     = (prices - rolling_max) / rolling_max
    signal       = drawdown.abs() > threshold
    return signal, drawdown, threshold


def drawdown_dca(prices: pd.DataFrame,
                  weights: np.ndarray | list,
                  monthly_dates: np.ndarray,
                  monthly_amount: float,
                  window: int = 20,
                  multiplier: float = 2.0) -> tuple[pd.Series, pd.Series]:
    """
    Invest more when an asset is in a statistically significant drawdown.

    On each monthly purchase date:
      - Assets WITH a drawdown signal receive  monthly_amount * multiplier * weight
      - Assets WITHOUT signal receive          monthly_amount * weight (standard)

    Args:
        prices         : pd.DataFrame  daily prices
        weights        : array-like    portfolio weights
        monthly_dates  : np.ndarray    purchase dates
        monthly_amount : float         base monthly amount
        window         : int           signal lookback window (trading days)
        multiplier     : float         investment multiplier on signal days

    Returns:
        (portfolio_values, invested_series)
    """
    weights     = np.asarray(weights, dtype=float)
    tickers     = list(prices.columns)
    signal, _, _ = compute_drawdown_signal(prices, window)
    monthly_set = set(pd.Timestamp(d) for d in monthly_dates)

    shares         = {t: 0.0 for t in tickers}
    total_invested = 0.0
    port_values    = []
    inv_series     = []

    for date in prices.index:
        if date in monthly_set:
            for t, w in zip(tickers, weights):
                price   = prices.loc[date, t]
                factor  = multiplier if signal.loc[date, t] else 1.0
                invest  = monthly_amount * w * factor
                shares[t]      += invest / price
                total_invested += invest

        value = sum(shares[t] * prices.loc[date, t] for t in tickers)
        port_values.append(value)
        inv_series.append(total_invested)

    return (
        pd.Series(port_values, index=prices.index, name="Drawdown DCA"),
        pd.Series(inv_series,  index=prices.index, name="Invested"),
    )


# ── Strategy 3: MA200 DCA ─────────────────────────────────────────────────────

def _get_ma_multiplier(deviation: float,
                        tiers: list[tuple[float, float]]) -> float:
    """
    Map a price-vs-MA deviation to a tiered investment multiplier.

    Tiers are evaluated from least negative to most negative; the first
    matching threshold returns its multiplier.

    Args:
        deviation : float   (price - MA) / MA  — negative when below MA
        tiers     : list    [(threshold, multiplier), ...]
                            sorted from least to most negative threshold

    Returns:
        float  investment multiplier (1.0 = standard DCA, no boost)
    """
    for threshold, mult in tiers:
        if deviation <= threshold:
            return mult
    return 1.0  # price above MA or within buffer zone


def compute_ma_multipliers(prices: pd.DataFrame,
                             ma_window: int,
                             tiers: list[tuple[float, float]]) -> pd.DataFrame:
    """
    Compute the per-asset, per-day investment multiplier based on
    deviation from the moving average.

    Args:
        prices    : pd.DataFrame  daily prices
        ma_window : int           moving average window (e.g. 200)
        tiers     : list          [(threshold, multiplier), ...]

    Returns:
        pd.DataFrame  same shape as prices, values = multipliers
    """
    ma        = prices.rolling(ma_window).mean()
    deviation = (prices - ma) / ma

    mult_df = deviation.copy()
    for col in deviation.columns:
        mult_df[col] = deviation[col].apply(
            lambda d: _get_ma_multiplier(d, tiers) if pd.notna(d) else 1.0
        )
    return mult_df


def ma200_dca(prices: pd.DataFrame,
               weights: np.ndarray | list,
               monthly_dates: np.ndarray,
               monthly_amount: float,
               ma_window: int = 200,
               tiers: list[tuple[float, float]] = None) -> tuple[pd.Series, pd.Series]:
    """
    Invest more when an asset trades significantly below its moving average.

    Tiers define how much extra to invest based on how far below the MA
    the price has fallen. Default tiers:
        0%  to  -5% below MA  → x1.0  (no boost)
       -5%  to -10% below MA  → x1.5
      -10%  to -20% below MA  → x2.0
      -20%+ below MA          → x2.5

    Args:
        prices         : pd.DataFrame  daily prices
        weights        : array-like    portfolio weights
        monthly_dates  : np.ndarray    purchase dates
        monthly_amount : float         base monthly amount
        ma_window      : int           moving average window
        tiers          : list          [(threshold, multiplier), ...]
                         sorted from least to most negative threshold.
                         Defaults to the standard 4-tier structure above.

    Returns:
        (portfolio_values, invested_series)
    """
    if tiers is None:
        tiers = [
            (-0.05, 1.5),
            (-0.10, 2.0),
            (-0.20, 2.5),
            (-1.00, 3.0),
        ]

    weights     = np.asarray(weights, dtype=float)
    tickers     = list(prices.columns)
    mult_df     = compute_ma_multipliers(prices, ma_window, tiers)
    monthly_set = set(pd.Timestamp(d) for d in monthly_dates)

    shares         = {t: 0.0 for t in tickers}
    total_invested = 0.0
    port_values    = []
    inv_series     = []

    for date in prices.index:
        if date in monthly_set:
            for t, w in zip(tickers, weights):
                price   = prices.loc[date, t]
                factor  = float(mult_df.loc[date, t]) if not pd.isna(mult_df.loc[date, t]) else 1.0
                invest  = monthly_amount * w * factor
                shares[t]      += invest / price
                total_invested += invest

        value = sum(shares[t] * prices.loc[date, t] for t in tickers)
        port_values.append(value)
        inv_series.append(total_invested)

    return (
        pd.Series(port_values, index=prices.index, name="MA200 DCA"),
        pd.Series(inv_series,  index=prices.index, name="Invested"),
    )


# ── Summary metrics ───────────────────────────────────────────────────────────

def dca_summary_metrics(strategy_results: dict[str, tuple[pd.Series, pd.Series]],
                          trading_days: int = 252) -> pd.DataFrame:
    """
    Compute a comparative summary table for multiple DCA strategies.

    Args:
        strategy_results : dict  {label: (portfolio_values, invested_series)}
        trading_days     : int

    Returns:
        pd.DataFrame  index=strategy labels, columns=metrics
    """
    rows = []
    for label, (port_vals, inv_series) in strategy_results.items():
        total_invested = float(inv_series.iloc[-1])
        final_value    = float(port_vals.iloc[-1])
        total_return   = (final_value / total_invested - 1) * 100 if total_invested > 0 else 0.0
        gain           = final_value - total_invested

        # Use first non-zero value as start — DCA portfolios begin at 0
        non_zero  = port_vals[port_vals > 0]
        start_val = float(non_zero.iloc[0]) if len(non_zero) > 0 else 0.0
        years     = len(port_vals) / trading_days

        if years > 0 and start_val > 0:
            cagr = (final_value / start_val) ** (1 / years) - 1
        else:
            cagr = 0.0

        # Compute returns only on non-zero portion to avoid NaN vol
        daily_ret = non_zero.pct_change().dropna()
        vol       = daily_ret.std() * np.sqrt(trading_days) * 100 if len(daily_ret) > 1 else 0.0

        dd        = (port_vals / port_vals.cummax()) - 1
        mdd       = float(dd.min()) * 100

        sharpe    = (cagr * 100 / vol) if vol != 0 else 0.0

        rows.append({
            "Strategy":          label,
            "Total Invested":    round(total_invested, 2),
            "Final Value":       round(final_value, 2),
            "Gain":              round(gain, 2),
            "Total Return (%)":  round(total_return, 2),
            "CAGR (%)":          round(cagr * 100, 2),
            "Ann. Vol (%)":      round(vol, 2),
            "Max Drawdown (%)":  round(mdd, 2),
            "Sharpe":            round(sharpe, 3),
        })

    return pd.DataFrame(rows).set_index("Strategy")
