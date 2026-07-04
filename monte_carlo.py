"""
monte_carlo.py — Portfolio Optimization & Simulation
======================================================
Monte Carlo methods for:
  1. Efficient Frontier construction (random weight sampling)
  2. Multi-regime optimization (compare optimal weights across market periods)
  3. Walk-forward (out-of-sample) validation
  4. Forward path simulation (probabilistic return forecasts)

All functions work with a pd.DataFrame of daily prices (DatetimeIndex,
columns = ticker symbols).
"""

import numpy as np
import pandas as pd


# ── Internal helpers ──────────────────────────────────────────────────────────

def _log_return_stats(prices: pd.DataFrame):
    """Return (mean, cov) of log returns, annualized."""
    log_ret  = np.log(prices / prices.shift(1)).dropna()
    mean_ret = log_ret.mean()
    cov_mat  = log_ret.cov()
    return mean_ret, cov_mat


def _portfolio_stats(weights: np.ndarray,
                     mean_ret: pd.Series,
                     cov_mat: pd.DataFrame,
                     trading_days: int = 252) -> tuple[float, float, float]:
    """
    Compute annualized (return, volatility, sharpe) for a weight vector.
    Sharpe computed without risk-free rate (raw ratio).
    """
    ret = float(np.sum(mean_ret * weights) * trading_days)
    vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_mat * trading_days, weights))))
    sharpe = ret / vol if vol != 0 else 0.0
    return ret, vol, sharpe


# ── Public API ────────────────────────────────────────────────────────────────

def efficient_frontier(prices: pd.DataFrame,
                        tickers: list[str],
                        current_weights: np.ndarray,
                        num_portfolios: int = 20_000,
                        risk_free_rate: float = 0.0,
                        trading_days: int = 252,
                        random_seed: int = 42) -> dict:
    """
    Build an Efficient Frontier via random weight sampling (Monte Carlo).

    Samples `num_portfolios` random weight vectors, computes annualized
    return, volatility and Sharpe for each, then identifies the
    Max-Sharpe and Min-Volatility frontier portfolios.

    Args:
        prices           : pd.DataFrame  daily adjusted close prices
        tickers          : list[str]     ticker order matching prices columns
        current_weights  : np.ndarray    actual portfolio weights (for overlay)
        num_portfolios   : int           number of random samples
        risk_free_rate   : float         annual risk-free rate as decimal
        trading_days     : int           annualization factor
        random_seed      : int           for reproducibility

    Returns:
        dict with keys:
            ret_arr          : np.ndarray  (num_portfolios,) annualized returns
            vol_arr          : np.ndarray  (num_portfolios,) annualized vols
            sharpe_arr       : np.ndarray  (num_portfolios,) Sharpe ratios
            all_weights      : np.ndarray  (num_portfolios, n_assets) weights
            max_sharpe_idx   : int         index of Max-Sharpe portfolio
            min_vol_idx      : int         index of Min-Volatility portfolio
            max_sharpe_weights : np.ndarray
            min_vol_weights    : np.ndarray
            current_ret      : float       current portfolio annualized return
            current_vol      : float       current portfolio annualized vol
            current_sharpe   : float
            tickers          : list[str]
    """
    np.random.seed(random_seed)
    n = len(tickers)
    mean_ret, cov_mat = _log_return_stats(prices[tickers])

    all_weights = np.zeros((num_portfolios, n))
    ret_arr     = np.zeros(num_portfolios)
    vol_arr     = np.zeros(num_portfolios)
    sharpe_arr  = np.zeros(num_portfolios)

    print(f"  Running {num_portfolios:,} Monte Carlo simulations...")
    for i in range(num_portfolios):
        w = np.random.random(n)
        w /= w.sum()
        all_weights[i] = w
        ret_arr[i], vol_arr[i], sharpe_arr[i] = _portfolio_stats(
            w, mean_ret, cov_mat, trading_days
        )

    # Adjust Sharpe for risk-free rate
    sharpe_arr_rf = (ret_arr - risk_free_rate) / vol_arr

    max_sharpe_idx = int(sharpe_arr_rf.argmax())
    min_vol_idx    = int(vol_arr.argmin())

    cur_ret, cur_vol, cur_sharpe = _portfolio_stats(
        current_weights, mean_ret, cov_mat, trading_days
    )

    print("  Simulation complete.")
    return {
        "ret_arr":             ret_arr,
        "vol_arr":             vol_arr,
        "sharpe_arr":          sharpe_arr_rf,
        "all_weights":         all_weights,
        "max_sharpe_idx":      max_sharpe_idx,
        "min_vol_idx":         min_vol_idx,
        "max_sharpe_weights":  all_weights[max_sharpe_idx],
        "min_vol_weights":     all_weights[min_vol_idx],
        "current_ret":         cur_ret,
        "current_vol":         cur_vol,
        "current_sharpe":      cur_sharpe,
        "tickers":             tickers,
    }


def optimize_for_period(prices: pd.DataFrame,
                         tickers: list[str],
                         start: str,
                         end: str,
                         label: str,
                         num_portfolios: int = 20_000,
                         trading_days: int = 252,
                         random_seed: int = 42) -> pd.Series:
    """
    Run Monte Carlo optimization for a specific date range.
    Returns the Max-Sharpe weight vector as a named pd.Series.

    Args:
        prices         : pd.DataFrame  full price history
        tickers        : list[str]
        start          : str           "YYYY-MM-DD"
        end            : str           "YYYY-MM-DD"
        label          : str           name for the result Series
        num_portfolios : int
        trading_days   : int
        random_seed    : int

    Returns:
        pd.Series  index=tickers, name=label
    """
    subset   = prices.loc[start:end, tickers]
    mean_ret, cov_mat = _log_return_stats(subset)
    np.random.seed(random_seed)
    n = len(tickers)

    all_w      = np.zeros((num_portfolios, n))
    sharpe_arr = np.zeros(num_portfolios)

    for i in range(num_portfolios):
        w = np.random.random(n)
        w /= w.sum()
        all_w[i] = w
        r = float(np.sum(mean_ret * w) * trading_days)
        v = float(np.sqrt(np.dot(w.T, np.dot(cov_mat * trading_days, w))))
        sharpe_arr[i] = r / v if v != 0 else 0.0

    best_w = all_w[sharpe_arr.argmax()]
    return pd.Series(best_w, index=tickers, name=label)


def multi_regime_optimization(prices: pd.DataFrame,
                               tickers: list[str],
                               regimes: list[dict],
                               num_portfolios: int = 20_000,
                               trading_days: int = 252) -> pd.DataFrame:
    """
    Optimize the portfolio across multiple market regimes and compute
    an All-Weather allocation as the equal-weight average.

    Args:
        prices         : pd.DataFrame  full price history
        tickers        : list[str]
        regimes        : list of dicts, each with keys:
                           label : str   display name
                           start : str   "YYYY-MM-DD"
                           end   : str   "YYYY-MM-DD"
        num_portfolios : int
        trading_days   : int

    Returns:
        pd.DataFrame  columns = regime labels + "All-Weather (Mean)",
                      index   = tickers,
                      values  = weights (decimal)

    Example regimes:
        [
            {"label": "Pandemic (2020-2021)",  "start": "2020-01-01", "end": "2021-12-31"},
            {"label": "War Stress (2022-2023)", "start": "2022-02-01", "end": "2023-05-31"},
            {"label": "Full Cycle (2015-2026)", "start": "2015-01-01", "end": "2026-03-27"},
        ]
    """
    results = {}
    for regime in regimes:
        label = regime["label"]
        print(f"  Optimizing: {label} ({regime['start']} -> {regime['end']})")
        results[label] = optimize_for_period(
            prices, tickers,
            start=regime["start"], end=regime["end"],
            label=label,
            num_portfolios=num_portfolios,
            trading_days=trading_days,
        )

    df = pd.concat(results.values(), axis=1)
    df["All-Weather (Mean)"] = df.mean(axis=1)
    return df


def walk_forward_test(prices: pd.DataFrame,
                       tickers: list[str],
                       train_start: str,
                       train_end: str,
                       test_start: str,
                       test_end: str,
                       num_portfolios: int = 20_000,
                       trading_days: int = 252) -> dict:
    """
    Out-of-sample (walk-forward) validation:
      1. Optimize on training window (in-sample).
      2. Apply the resulting weights to the test window (out-of-sample).

    Args:
        prices         : pd.DataFrame  full price history
        tickers        : list[str]
        train_start    : str  "YYYY-MM-DD"
        train_end      : str  "YYYY-MM-DD"
        test_start     : str  "YYYY-MM-DD"
        test_end       : str  "YYYY-MM-DD"
        num_portfolios : int
        trading_days   : int

    Returns:
        dict with keys:
            blind_weights       : np.ndarray   weights from training
            test_portfolio_ret  : pd.Series    daily returns in test window
            test_benchmark_ret  : pd.Series    benchmark (first ticker) daily returns
    """
    print(f"  Training: {train_start} -> {train_end}")
    blind_series  = optimize_for_period(
        prices, tickers,
        start=train_start, end=train_end,
        label="blind",
        num_portfolios=num_portfolios,
        trading_days=trading_days,
    )
    blind_weights = blind_series.values

    test_prices  = prices.loc[test_start:test_end, tickers]
    test_returns = test_prices.pct_change().dropna()

    print(f"  Testing:  {test_start} -> {test_end}")
    return {
        "blind_weights":      blind_weights,
        "test_portfolio_ret": test_returns.dot(blind_weights).rename("All-Weather (Blind)"),
        "test_benchmark_ret": test_returns.iloc[:, 0].rename(tickers[0]),
    }


def simulate_future_paths(portfolio_returns: pd.Series,
                           horizon_days: int = 252,
                           num_simulations: int = 1_000,
                           initial_value: float = 10_000.0,
                           percentiles: tuple = (5, 25, 50, 75, 95),
                           random_seed: int = 42) -> dict:
    """
    Forward Monte Carlo simulation using historical return statistics.

    Draws daily returns from a normal distribution calibrated to the
    historical mean and standard deviation of `portfolio_returns`.

    Args:
        portfolio_returns : pd.Series  historical daily returns
        horizon_days      : int        simulation length in trading days
        num_simulations   : int        number of paths
        initial_value     : float      starting portfolio value
        percentiles       : tuple      percentile bands to compute
        random_seed       : int

    Returns:
        dict with keys:
            paths       : np.ndarray  shape (horizon_days, num_simulations)
            percentiles : dict        {pct: np.ndarray of length horizon_days}
            mean_path   : np.ndarray
            final_values: np.ndarray  shape (num_simulations,)
            mu          : float       daily mean used
            sigma       : float       daily std used
    """
    np.random.seed(random_seed)
    mu    = portfolio_returns.mean()
    sigma = portfolio_returns.std()

    # Shape: (horizon_days, num_simulations)
    daily_draws = np.random.normal(mu, sigma, (horizon_days, num_simulations))
    paths       = initial_value * np.cumprod(1 + daily_draws, axis=0)

    pct_dict = {p: np.percentile(paths, p, axis=1) for p in percentiles}

    return {
        "paths":        paths,
        "percentiles":  pct_dict,
        "mean_path":    paths.mean(axis=1),
        "final_values": paths[-1],
        "mu":           mu,
        "sigma":        sigma,
    }


def weight_comparison_table(tickers: list[str],
                              current_weights: np.ndarray,
                              ef_result: dict) -> pd.DataFrame:
    """
    Build a comparison table of current, Max-Sharpe and Min-Vol weights.

    Args:
        tickers         : list[str]
        current_weights : np.ndarray
        ef_result       : dict  output of efficient_frontier()

    Returns:
        pd.DataFrame  columns: Ticker | Current (%) | Max Sharpe (%) | Min Vol (%)
    """
    return pd.DataFrame({
        "Ticker":           tickers,
        "Current (%)":      (current_weights * 100).round(2),
        "Max Sharpe (%)":   (ef_result["max_sharpe_weights"] * 100).round(2),
        "Min Vol (%)":      (ef_result["min_vol_weights"] * 100).round(2),
    })
