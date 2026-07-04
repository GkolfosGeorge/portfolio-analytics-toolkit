"""
stress_scenarios.py — Stress Testing & Scenario Analysis
==========================================================
Tools for evaluating portfolio resilience under hypothetical
and historically-calibrated market shocks.

Two analysis modes:
  1. Instantaneous shock  — one-time percentage drop applied by asset category.
  2. Historical replay    — re-run portfolio through a specific crisis window.

Scenarios are defined by ASSET CATEGORY (e.g. "US Equity", "Commodities"),
not by individual tickers. This means they work with any portfolio
without warnings, as long as ASSET_CATEGORIES is passed from the config cell.

Config cell usage:
    import stress_scenarios as stress

    shock_table = stress.run_all_scenarios(
        weights_dict     = {"My Portfolio": weights},
        tickers          = TICKERS,
        asset_categories = ASSET_CATEGORIES,   # from config cell
    )
"""

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO LIBRARY — category-based shocks
# ══════════════════════════════════════════════════════════════════════════════
# Each scenario maps ASSET CATEGORY → shock (decimal).
# Categories must match those used in ASSET_CATEGORIES in the config cell.
# Assets whose category is not listed in a scenario are unaffected (shock = 0).
#
# Standard categories: "US Equity" | "World Equity" | "Commodities" |
#                      "Bonds" | "Cash" | "Real Estate" | "Crypto"

SCENARIOS: dict[str, dict[str, float]] = {

    "Normal Market (No Shock)": {},

    "Stagflation (Equities -20%, Commodities +15%)": {
        "US Equity":    -0.20,
        "World Equity": -0.15,
        "Bonds":        -0.08,   # bonds sell off in stagflation
        "Commodities":  +0.15,   # gold, silver, energy up
        "Real Estate":  -0.10,
    },

    "Tech Meltdown (US Equities -30%)": {
        "US Equity":    -0.30,
        "World Equity": -0.12,
        "Bonds":        +0.08,   # flight to safety
        "Commodities":  +0.03,
    },

    "EM / China Crisis (World Equity -25%)": {
        "World Equity": -0.25,
        "US Equity":    -0.10,
        "Commodities":  -0.05,
        "Bonds":        +0.05,
    },

    "Financial Crisis Redux (2008-style, Broad -40%)": {
        "US Equity":    -0.40,
        "World Equity": -0.38,
        "Bonds":        +0.12,   # treasuries rallied in 2008
        "Commodities":  -0.20,   # oil crashed, gold up modestly
        "Real Estate":  -0.45,
    },

    "Pandemic Shock (2020 Covid-style, Broad -35%)": {
        "US Equity":    -0.34,
        "World Equity": -0.32,
        "Bonds":        +0.10,
        "Commodities":  -0.15,
        "Real Estate":  -0.20,
    },

    "Rate Hike Shock (2022-style, Bonds -25%)": {
        "US Equity":    -0.20,
        "World Equity": -0.18,
        "Bonds":        -0.25,   # worst year for bonds in history
        "Commodities":  +0.10,
        "Real Estate":  -0.25,
    },

    "Geopolitical Shock (War/Sanctions, Energy +20%)": {
        "World Equity": -0.18,
        "US Equity":    -0.08,
        "Bonds":        -0.05,
        "Commodities":  +0.20,   # energy spike
    },

    "Everything Rally (Risk-on, Equities +20%)": {
        "US Equity":    +0.20,
        "World Equity": +0.18,
        "Bonds":        -0.05,
        "Commodities":  -0.03,
        "Real Estate":  +0.12,
    },

    "Deflation / Recession (Broad Risk-off)": {
        "US Equity":    -0.25,
        "World Equity": -0.22,
        "Bonds":        +0.15,   # bonds rally in deflation
        "Commodities":  -0.25,
        "Real Estate":  -0.20,
    },
}


# Historical date windows for replay analysis
HISTORICAL_WINDOWS: dict[str, dict[str, str]] = {
    "GFC 2008-2009":          {"start": "2007-10-01", "end": "2009-03-31"},
    "Covid Crash 2020":       {"start": "2020-02-01", "end": "2020-06-30"},
    "Ukraine War 2022":       {"start": "2022-02-01", "end": "2023-05-31"},
    "Dot-com Bust 2000-2002": {"start": "2000-03-01", "end": "2002-12-31"},
    "Rate Hike Cycle 2022":   {"start": "2022-01-01", "end": "2022-12-31"},
}


# ══════════════════════════════════════════════════════════════════════════════
# INSTANTANEOUS SHOCK FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def portfolio_shock_impact(weights:          np.ndarray,
                            tickers:          list[str],
                            shocks:           dict[str, float],
                            asset_categories: dict[str, str] = None) -> float:
    """
    Compute the immediate portfolio-level impact of a one-time shock.

    Supports two modes:
      - Category-based (recommended): shocks keyed by asset category
        (e.g. "US Equity": -0.20). Requires asset_categories mapping.
      - Ticker-based (legacy): shocks keyed by ticker symbol.
        Works silently — missing tickers contribute 0, no warnings.

    Args:
        weights          : np.ndarray   portfolio weights (sum to 1)
        tickers          : list[str]    ticker order matching weights
        shocks           : dict         {category_or_ticker: shock_decimal}
        asset_categories : dict         {ticker: category} from config cell
                                        required for category-based shocks

    Returns:
        float  portfolio percentage change (e.g. -12.5 means -12.5%)
    """
    impact = 0.0

    for i, ticker in enumerate(tickers):
        w = float(weights[i])

        # Category-based shock — look up the ticker's category
        if asset_categories is not None:
            category = asset_categories.get(ticker, "Other")
            shock    = shocks.get(category, 0.0)
        else:
            # Ticker-based (legacy) — silent, no warnings
            shock = shocks.get(ticker, 0.0)

        impact += w * shock

    return impact * 100.0


def run_all_scenarios(weights_dict:      dict[str, np.ndarray],
                       tickers:           list[str],
                       asset_categories:  dict[str, str] = None,
                       scenarios:         dict[str, dict[str, float]] = None
                       ) -> pd.DataFrame:
    """
    Run all scenarios for one or more portfolios and return a results table.

    Args:
        weights_dict     : dict  {portfolio_label: np.ndarray}
        tickers          : list[str]
        asset_categories : dict  {ticker: category} from config cell
                                 pass ASSET_CATEGORIES from CELL 0
        scenarios        : dict  defaults to the built-in SCENARIOS library

    Returns:
        pd.DataFrame  index=scenario names, columns=portfolio labels,
                      values=portfolio impact (%)
    """
    if scenarios is None:
        scenarios = SCENARIOS

    rows = []
    for scenario_name, shocks in scenarios.items():
        row = {"Scenario": scenario_name}
        for label, weights in weights_dict.items():
            row[label] = round(
                portfolio_shock_impact(
                    weights, tickers, shocks, asset_categories
                ), 2
            )
        rows.append(row)

    return pd.DataFrame(rows).set_index("Scenario")


def asset_contribution_breakdown(weights:          np.ndarray,
                                   tickers:          list[str],
                                   shocks:           dict[str, float],
                                   asset_categories: dict[str, str] = None,
                                   label:            str = "Portfolio"
                                   ) -> pd.DataFrame:
    """
    Show each asset's individual contribution to the total shock impact.
    Useful for identifying which positions drive portfolio losses.

    Args:
        weights          : np.ndarray
        tickers          : list[str]
        shocks           : dict  category or ticker shocks
        asset_categories : dict  {ticker: category} (optional)
        label            : str

    Returns:
        pd.DataFrame  sorted by Contribution (%) ascending (worst first)
    """
    rows = []
    for i, ticker in enumerate(tickers):
        w = float(weights[i])

        if asset_categories is not None:
            category = asset_categories.get(ticker, "Other")
            shock    = shocks.get(category, 0.0)
        else:
            shock    = shocks.get(ticker, 0.0)
            category = "—"

        if shock != 0.0:
            rows.append({
                "Asset":            ticker,
                "Category":         category,
                "Weight (%)":       round(w * 100, 2),
                "Shock (%)":        round(shock * 100, 1),
                "Contribution (%)": round(w * shock * 100, 2),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Contribution (%)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# HISTORICAL REPLAY
# ══════════════════════════════════════════════════════════════════════════════

def historical_replay(prices:           pd.DataFrame,
                       weights:          np.ndarray,
                       tickers:          list[str],
                       window_name:      str   = None,
                       start:            str   = None,
                       end:              str   = None,
                       benchmark_ticker: str   = "VOO",
                       initial_value:    float = 10_000.0) -> dict:
    """
    Re-run the portfolio through a historical crisis period and compare
    to a benchmark.

    Provide either `window_name` (from HISTORICAL_WINDOWS) or explicit
    `start` / `end` dates.

    Args:
        prices           : pd.DataFrame  full price history
        weights          : np.ndarray    portfolio weights
        tickers          : list[str]
        window_name      : str           key in HISTORICAL_WINDOWS (optional)
        start            : str           "YYYY-MM-DD" (optional)
        end              : str           "YYYY-MM-DD" (optional)
        benchmark_ticker : str           must be in prices.columns
        initial_value    : float

    Returns:
        dict with keys:
            port_values, bench_values, port_drawdown, bench_drawdown,
            port_mdd, bench_mdd, port_cagr, bench_cagr,
            port_vol, bench_vol, window_label
    """
    if window_name is not None:
        w              = HISTORICAL_WINDOWS[window_name]
        start, end     = w["start"], w["end"]
        label          = window_name
    else:
        label = f"{start} to {end}"

    subset  = prices.loc[start:end, tickers]

    if len(subset) < 5:
        raise ValueError(
            f"Insufficient data for window '{label}' "
            f"({start} to {end}). "
            f"Price history starts at {prices.index[0].date()}. "
            f"Skipping this window."
        )

    returns = subset.pct_change().dropna()

    port_ret   = returns.dot(weights)
    port_cum   = (1 + port_ret).cumprod()
    port_vals  = port_cum * initial_value

    bench_ret  = (returns[benchmark_ticker]
                  if benchmark_ticker in returns.columns else port_ret)
    bench_cum  = (1 + bench_ret).cumprod()
    bench_vals = bench_cum * initial_value

    def _dd(vals):
        return (vals / vals.cummax()) - 1

    def _cagr(rets):
        years = len(rets) / 252
        return float((1 + rets).prod() ** (1 / years) - 1) if years > 0 else 0.0

    return {
        "port_values":    port_vals,
        "bench_values":   bench_vals,
        "port_drawdown":  _dd(port_vals),
        "bench_drawdown": _dd(bench_vals),
        "port_mdd":       float(_dd(port_vals).min()),
        "bench_mdd":      float(_dd(bench_vals).min()),
        "port_cagr":      _cagr(port_ret),
        "bench_cagr":     _cagr(bench_ret),
        "port_vol":       float(port_ret.std() * np.sqrt(252)),
        "bench_vol":      float(bench_ret.std() * np.sqrt(252)),
        "window_label":   label,
    }


def worst_days_summary(returns: pd.DataFrame,
                        top_n:   int = 5) -> pd.DataFrame:
    """
    Return the N worst single-day returns per asset.

    Args:
        returns : pd.DataFrame  daily returns
        top_n   : int

    Returns:
        pd.DataFrame  index=tickers, columns=worst-day ranks
    """
    result = {}
    for ticker in returns.columns:
        result[ticker] = returns[ticker].nsmallest(top_n).values

    cols = [f"Worst Day #{i+1}" for i in range(top_n)]
    return pd.DataFrame(result, index=cols).T.round(4)