"""
sensitivity.py — Sensitivity & What-If Analysis
=================================================
Tools for exploring how portfolio outcomes change when key
input parameters are varied:

  1. Weight sensitivity     — shift one asset's weight, rescale the rest
  2. Capital sensitivity    — vary initial investment or monthly contribution
  3. Return sensitivity     — apply flat return shocks to all assets
  4. Volatility sensitivity — scale historical volatility up or down
  5. DCA contribution grid  — 2-D grid of final values vs. amount & horizon
"""

import numpy as np
import pandas as pd


# ── Weight sensitivity ────────────────────────────────────────────────────────

def weight_sensitivity(base_weights: np.ndarray,
                        tickers: list[str],
                        target_ticker: str,
                        weight_range: np.ndarray,
                        returns: pd.DataFrame,
                        trading_days: int = 252) -> pd.DataFrame:
    """
    Vary the weight of one asset across a range, rescale remaining weights
    proportionally, and record annualized portfolio return and volatility.

    Args:
        base_weights   : np.ndarray   starting weights (sum to 1)
        tickers        : list[str]
        target_ticker  : str          asset whose weight is varied
        weight_range   : np.ndarray   weight values to test (e.g. np.linspace(0, 0.5, 21))
        returns        : pd.DataFrame daily returns (columns = tickers)
        trading_days   : int

    Returns:
        pd.DataFrame  columns: Weight(%), Ann.Return(%), Ann.Vol(%), Sharpe
    """
    if target_ticker not in tickers:
        raise ValueError(f"'{target_ticker}' not found in tickers.")

    target_idx   = tickers.index(target_ticker)
    other_idx    = [i for i in range(len(tickers)) if i != target_idx]
    base_other   = base_weights[other_idx]
    other_sum    = base_other.sum()

    rows = []
    for w_target in weight_range:
        if w_target >= 1.0:
            continue
        # Rescale other weights to fill the remaining allocation
        remaining = 1.0 - w_target
        if other_sum > 0:
            scaled_other = base_other * (remaining / other_sum)
        else:
            scaled_other = np.full(len(other_idx), remaining / len(other_idx))

        new_weights = np.zeros(len(tickers))
        new_weights[target_idx] = w_target
        for i, idx in enumerate(other_idx):
            new_weights[idx] = scaled_other[i]

        port_ret = returns[tickers].dot(new_weights)
        ann_ret  = port_ret.mean() * trading_days * 100
        ann_vol  = port_ret.std() * np.sqrt(trading_days) * 100
        sharpe   = (ann_ret / ann_vol) if ann_vol != 0 else 0.0

        rows.append({
            f"{target_ticker} Weight (%)": round(w_target * 100, 1),
            "Ann. Return (%)":              round(ann_ret, 2),
            "Ann. Vol (%)":                 round(ann_vol, 2),
            "Sharpe":                       round(sharpe, 3),
        })

    return pd.DataFrame(rows)


# ── Capital sensitivity ───────────────────────────────────────────────────────

def initial_capital_sensitivity(portfolio_returns: pd.Series,
                                  capital_range: np.ndarray) -> pd.DataFrame:
    """
    Compute final portfolio value for a range of initial lump-sum investments.

    Args:
        portfolio_returns : pd.Series  daily returns
        capital_range     : np.ndarray array of initial capital values to test

    Returns:
        pd.DataFrame  columns: Initial Capital | Final Value | Total Return (%)
    """
    cum = (1 + portfolio_returns).cumprod().iloc[-1]
    rows = []
    for cap in capital_range:
        final = cap * cum
        rows.append({
            "Initial Capital":  round(cap, 2),
            "Final Value":      round(float(final), 2),
            "Total Return (%)": round((float(final) / cap - 1) * 100, 2),
        })
    return pd.DataFrame(rows)


def monthly_contribution_sensitivity(portfolio_returns: pd.Series,
                                       initial_capital: float,
                                       monthly_range: np.ndarray,
                                       trading_days: int = 252) -> pd.DataFrame:
    """
    Compute final DCA portfolio value for a range of monthly contributions.

    Uses a simplified model: contributions are invested at the end of each
    month (every ~21 trading days) at the prevailing cumulative return.

    Args:
        portfolio_returns : pd.Series  daily returns
        initial_capital   : float
        monthly_range     : np.ndarray  monthly amounts to test
        trading_days      : int

    Returns:
        pd.DataFrame  columns: Monthly Contribution | Total Invested | Final Value | Gain
    """
    cum_returns = (1 + portfolio_returns).cumprod()
    monthly_idx = list(range(20, len(cum_returns), 21))  # approx end-of-month

    rows = []
    for monthly_amt in monthly_range:
        # Lump sum component
        value = initial_capital * float(cum_returns.iloc[-1])

        # DCA component: each contribution grows from its purchase date
        for idx in monthly_idx:
            growth = float(cum_returns.iloc[-1] / cum_returns.iloc[idx])
            value += monthly_amt * growth

        total_invested = initial_capital + monthly_amt * len(monthly_idx)
        rows.append({
            "Monthly Contribution": round(monthly_amt, 0),
            "Total Invested":       round(total_invested, 2),
            "Final Value":          round(value, 2),
            "Gain":                 round(value - total_invested, 2),
        })

    return pd.DataFrame(rows)


# ── Return shock sensitivity ──────────────────────────────────────────────────

def return_shock_sensitivity(portfolio_returns: pd.Series,
                               shock_range: np.ndarray,
                               initial_value: float = 10_000.0) -> pd.DataFrame:
    """
    Apply a flat daily return adjustment (basis-point shift) and observe
    the effect on final portfolio value and CAGR.

    Args:
        portfolio_returns : pd.Series  historical daily returns
        shock_range       : np.ndarray  daily return adjustments to apply
                            (e.g. np.linspace(-0.001, 0.001, 21) for +/-10bp/day)
        initial_value     : float

    Returns:
        pd.DataFrame  columns: Daily Shock (bp) | Final Value | CAGR (%)
    """
    years = len(portfolio_returns) / 252
    rows  = []
    for shock in shock_range:
        adj_returns = portfolio_returns + shock
        final_val   = initial_value * float((1 + adj_returns).cumprod().iloc[-1])
        cagr        = (final_val / initial_value) ** (1 / years) - 1 if years > 0 else 0.0
        rows.append({
            "Daily Shock (bp)": round(shock * 10_000, 1),
            "Final Value":      round(final_val, 2),
            "CAGR (%)":         round(cagr * 100, 2),
        })
    return pd.DataFrame(rows)


# ── Volatility sensitivity ────────────────────────────────────────────────────

def volatility_sensitivity(portfolio_returns: pd.Series,
                             vol_multipliers: np.ndarray,
                             initial_value: float = 10_000.0,
                             random_seed: int = 42) -> pd.DataFrame:
    """
    Scale historical volatility by a multiplier and simulate the resulting
    distribution of final portfolio values via Monte Carlo.

    Args:
        portfolio_returns : pd.Series  historical daily returns
        vol_multipliers   : np.ndarray  scaling factors (e.g. [0.5, 1.0, 1.5, 2.0])
        initial_value     : float
        random_seed       : int

    Returns:
        pd.DataFrame  one row per multiplier, columns: Vol Multiplier |
                      Simulated Final (p5) | Simulated Final (p50) | Simulated Final (p95)
    """
    np.random.seed(random_seed)
    mu     = portfolio_returns.mean()
    sigma  = portfolio_returns.std()
    n      = len(portfolio_returns)

    rows = []
    for mult in vol_multipliers:
        draws  = np.random.normal(mu, sigma * mult, (n, 500))
        finals = initial_value * np.cumprod(1 + draws, axis=0)[-1]
        rows.append({
            "Vol Multiplier":         round(mult, 2),
            "Simulated Final (p5)":   round(float(np.percentile(finals, 5)),  2),
            "Simulated Final (p50)":  round(float(np.percentile(finals, 50)), 2),
            "Simulated Final (p95)":  round(float(np.percentile(finals, 95)), 2),
        })
    return pd.DataFrame(rows)


# ── DCA contribution grid ─────────────────────────────────────────────────────

def dca_contribution_grid(portfolio_returns: pd.Series,
                            initial_capital: float,
                            monthly_amounts: np.ndarray,
                            horizons_years: np.ndarray,
                            trading_days: int = 252) -> pd.DataFrame:
    """
    2-D grid: rows = monthly contributions, columns = investment horizons.
    Cell value = projected final portfolio value.

    Args:
        portfolio_returns : pd.Series  daily returns (used to estimate daily mu/sigma)
        initial_capital   : float
        monthly_amounts   : np.ndarray  e.g. np.arange(100, 1100, 100)
        horizons_years    : np.ndarray  e.g. np.array([5, 10, 15, 20, 30])
        trading_days      : int

    Returns:
        pd.DataFrame  index=monthly_amounts, columns=horizons_years
    """
    daily_mu  = portfolio_returns.mean()
    # Monthly compounded return (approximate)
    monthly_r = (1 + daily_mu) ** 21 - 1

    grid = {}
    for horizon in horizons_years:
        months = int(horizon * 12)
        col    = {}
        for monthly_amt in monthly_amounts:
            # Future value of lump sum
            fv_lump = initial_capital * (1 + monthly_r) ** months
            # Future value of annuity (DCA)
            if monthly_r != 0:
                fv_dca = monthly_amt * (((1 + monthly_r) ** months - 1) / monthly_r)
            else:
                fv_dca = monthly_amt * months
            col[monthly_amt] = round(fv_lump + fv_dca, 0)
        grid[f"{horizon}yr"] = col

    return pd.DataFrame(grid)
