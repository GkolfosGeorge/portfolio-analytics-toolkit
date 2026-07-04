"""
correlation_regime.py — Correlation Regime Analysis
=====================================================
Reveals how asset correlations change across different market environments.

The core insight: static correlation heatmaps are misleading.
Diversification that looks solid on average can collapse exactly
when it is needed most — during a crash.

This module uses historically-defined market regimes (not arbitrary
statistical thresholds) so every result has an economic narrative
you can explain to a client.

Four analysis layers:
  1. regime_correlations()         — correlation matrix per historical regime
  2. regime_comparison_table()     — side-by-side comparison for every asset pair
  3. diversification_effectiveness()— does the defensive asset hold up in crisis?
  4. rolling_correlation_with_regimes() — time-series view with regime bands

Config cell usage:
    import correlation_regime as cr

    # Define regimes once in the config cell — customise per client
    REGIMES = cr.DEFAULT_REGIMES   # or define your own dict

    # Run analysis
    regime_corr   = cr.regime_correlations(returns, REGIMES)
    compare_tbl   = cr.regime_comparison_table(returns, REGIMES, pairs)
    div_score     = cr.diversification_effectiveness(returns, REGIMES,
                        defensive_assets=["GLD", "SLV"])
    roll_corr     = cr.rolling_correlation_with_regimes(returns, "VOO", "GLD",
                        REGIMES)
"""

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT REGIME LIBRARY  — override from config cell
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_REGIMES: dict[str, dict] = {
    "Bull Market\n(2015–2019)": {
        "start":       "2015-01-01",
        "end":         "2019-12-31",
        "description": "Steady US-led growth, low volatility, rising equities.",
        "type":        "bull",
    },
    "Covid Crash\n(Feb–May 2020)": {
        "start":       "2020-02-01",
        "end":         "2020-05-31",
        "description": "Fastest bear market in history. Liquidity crisis.",
        "type":        "bear",
    },
    "Post-Covid Rally\n(2020–2021)": {
        "start":       "2020-06-01",
        "end":         "2021-12-31",
        "description": "Zero rates, fiscal stimulus, growth-stock dominance.",
        "type":        "bull",
    },
    "Rate Hike & War\n(2022)": {
        "start":       "2022-01-01",
        "end":         "2022-12-31",
        "description": "Fed tightening, Ukraine war, tech drawdown -33%.",
        "type":        "bear",
    },
    "AI Bull\n(2023–2024)": {
        "start":       "2023-01-01",
        "end":         "2024-12-31",
        "description": "AI-driven rally, mega-cap concentration.",
        "type":        "bull",
    },
}

# Shorter labels for wide tables
_SHORT_LABELS = {
    "Bull Market\n(2015–2019)":      "Bull\n15-19",
    "Covid Crash\n(Feb–May 2020)":   "Crash\n2020",
    "Post-Covid Rally\n(2020–2021)": "Rally\n20-21",
    "Rate Hike & War\n(2022)":       "Bear\n2022",
    "AI Bull\n(2023–2024)":          "AI Bull\n23-24",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. REGIME CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════

def regime_correlations(returns: pd.DataFrame,
                         regimes: dict = None,
                         min_days: int = 20) -> dict:
    """
    Compute the Pearson correlation matrix for each historical regime.

    Args:
        returns   : pd.DataFrame  daily returns, columns = tickers,
                                  DatetimeIndex spanning all regimes
        regimes   : dict          {label: {"start": "YYYY-MM-DD",
                                           "end":   "YYYY-MM-DD",
                                           "description": str,
                                           "type": "bull"|"bear"|"neutral"}}
                                  defaults to DEFAULT_REGIMES
        min_days  : int           minimum trading days required per regime
                                  (regimes with fewer days are skipped)

    Returns:
        dict  {regime_label: pd.DataFrame (correlation matrix)}
              Only regimes with sufficient data are included.
    """
    if regimes is None:
        regimes = DEFAULT_REGIMES

    result = {}
    for label, cfg in regimes.items():
        subset = returns.loc[cfg["start"]: cfg["end"]]
        if len(subset) < min_days:
            print(f"  Skipping '{label}': only {len(subset)} days "
                  f"(min {min_days} required).")
            continue
        result[label] = subset.corr().round(3)

    return result


def full_period_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Correlation matrix for the entire price history.
    Used as a baseline to compare against regime-specific matrices.

    Args:
        returns : pd.DataFrame  daily returns

    Returns:
        pd.DataFrame  correlation matrix
    """
    return returns.corr().round(3)


# ══════════════════════════════════════════════════════════════════════════════
# 2. REGIME COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════

def regime_comparison_table(returns:       pd.DataFrame,
                              regimes:       dict = None,
                              pairs:         list[tuple[str, str]] = None,
                              include_full:  bool = True) -> pd.DataFrame:
    """
    Side-by-side correlation for selected asset pairs across all regimes.

    This is the key client-facing table: shows clearly how correlations
    change between calm and crisis periods.

    Args:
        returns      : pd.DataFrame  daily returns
        regimes      : dict          defaults to DEFAULT_REGIMES
        pairs        : list of (ticker_a, ticker_b) to include
                       If None, auto-selects all pairs from returns columns
        include_full : bool          add a "Full Period" baseline column

    Returns:
        pd.DataFrame  index = "A ↔ B" pair labels,
                      columns = regime labels (+ "Full Period" if requested)
    """
    if regimes is None:
        regimes = DEFAULT_REGIMES

    tickers = list(returns.columns)

    if pairs is None:
        pairs = [(tickers[i], tickers[j])
                 for i in range(len(tickers))
                 for j in range(i + 1, len(tickers))]

    regime_corr = regime_correlations(returns, regimes)
    full_corr   = full_period_correlation(returns) if include_full else None

    rows = {}
    for a, b in pairs:
        if a not in tickers or b not in tickers:
            continue
        pair_label = f"{a} ↔ {b}"
        rows[pair_label] = {}

        for label, corr_df in regime_corr.items():
            short = _SHORT_LABELS.get(label, label)
            rows[pair_label][short] = (
                float(corr_df.loc[a, b]) if a in corr_df.index else np.nan
            )

        if full_corr is not None:
            rows[pair_label]["Full Period"] = (
                float(full_corr.loc[a, b]) if a in full_corr.index else np.nan
            )

    df = pd.DataFrame(rows).T.round(3)
    df.index.name = "Asset Pair"
    return df


def correlation_shift_table(returns: pd.DataFrame,
                              regimes: dict = None,
                              bear_labels: list[str] = None,
                              bull_labels: list[str] = None) -> pd.DataFrame:
    """
    For each asset pair, compute the average correlation in bull vs bear
    regimes and the shift (bear - bull).

    A large negative shift for a defensive asset (GLD) confirms it truly
    diversifies in calm periods.
    A shift near zero means the diversification breaks down in crisis.

    Args:
        returns     : pd.DataFrame
        regimes     : dict
        bear_labels : list[str]  regime labels classified as "bear"
                                 (defaults to all regimes with type="bear")
        bull_labels : list[str]  regime labels classified as "bull"

    Returns:
        pd.DataFrame  columns: Bull Avg | Bear Avg | Shift (Bear-Bull) | Signal
    """
    if regimes is None:
        regimes = DEFAULT_REGIMES

    if bear_labels is None:
        bear_labels = [l for l, c in regimes.items() if c.get("type") == "bear"]
    if bull_labels is None:
        bull_labels = [l for l, c in regimes.items() if c.get("type") == "bull"]

    regime_corr = regime_correlations(returns, regimes)
    tickers     = list(returns.columns)
    pairs       = [(tickers[i], tickers[j])
                   for i in range(len(tickers))
                   for j in range(i + 1, len(tickers))]

    rows = []
    for a, b in pairs:
        bull_vals = [float(regime_corr[l].loc[a, b])
                     for l in bull_labels
                     if l in regime_corr and a in regime_corr[l].index]
        bear_vals = [float(regime_corr[l].loc[a, b])
                     for l in bear_labels
                     if l in regime_corr and a in regime_corr[l].index]

        bull_avg = float(np.mean(bull_vals)) if bull_vals else np.nan
        bear_avg = float(np.mean(bear_vals)) if bear_vals else np.nan
        shift    = bear_avg - bull_avg if not np.isnan(bear_avg + bull_avg) else np.nan

        # Signal: does correlation increase in a crisis (bad) or stay/fall (good)?
        if np.isnan(shift):
            signal = "N/A"
        elif shift > 0.15:
            signal = "CORRELATION SPIKE — diversification collapses in crisis"
        elif shift > 0.05:
            signal = "Moderate drift — partial diversification breakdown"
        elif shift < -0.05:
            signal = "Anti-correlated — diversification strengthens in crisis"
        else:
            signal = "Stable — correlation holds across regimes"

        rows.append({
            "Pair":             f"{a} ↔ {b}",
            "Bull Avg":         round(bull_avg, 3),
            "Bear Avg":         round(bear_avg, 3),
            "Shift (Bear-Bull)":round(shift, 3) if not np.isnan(shift) else None,
            "Signal":           signal,
        })

    return pd.DataFrame(rows).set_index("Pair").sort_values(
        "Shift (Bear-Bull)", ascending=False
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. DIVERSIFICATION EFFECTIVENESS
# ══════════════════════════════════════════════════════════════════════════════

def diversification_effectiveness(returns:           pd.DataFrame,
                                   regimes:           dict = None,
                                   defensive_assets:  list[str] = None,
                                   equity_benchmark:  str = "VOO") -> pd.DataFrame:
    """
    Score how well each defensive asset diversifies against the equity
    benchmark across all regimes.

    Score logic per regime:
      +2  correlation < -0.10  (strong negative — excellent diversification)
      +1  -0.10 <= corr < 0    (mild negative)
       0  0 <= corr < 0.10     (uncorrelated — neutral)
      -1  0.10 <= corr < 0.30  (mild positive — partial breakdown)
      -2  corr >= 0.30         (strong positive — diversification failed)

    Regime-type weighting:
      Bear regimes are weighted 2x because that is when
      diversification matters most.

    Args:
        returns          : pd.DataFrame  daily returns
        regimes          : dict
        defensive_assets : list[str]     tickers to evaluate
                           defaults to non-equity assets inferred from
                           low average correlation with equity_benchmark
        equity_benchmark : str           reference equity ticker

    Returns:
        pd.DataFrame  index = defensive assets,
                      columns = regime labels + weighted score + grade
    """
    if regimes is None:
        regimes = DEFAULT_REGIMES

    tickers = list(returns.columns)

    if defensive_assets is None:
        # Auto-detect: assets with full-period |corr| < 0.5 vs benchmark
        full_corr        = full_period_correlation(returns)
        defensive_assets = [
            t for t in tickers
            if t != equity_benchmark
            and abs(float(full_corr.loc[t, equity_benchmark])) < 0.5
        ] if equity_benchmark in tickers else []

    if not defensive_assets:
        return pd.DataFrame(columns=["No defensive assets identified"])

    regime_corr = regime_correlations(returns, regimes)
    rows        = {}

    for asset in defensive_assets:
        if asset not in tickers or equity_benchmark not in tickers:
            continue

        row          = {}
        total_score  = 0.0
        total_weight = 0.0

        for label, corr_df in regime_corr.items():
            if asset not in corr_df.index or equity_benchmark not in corr_df.index:
                row[label] = np.nan
                continue

            corr   = float(corr_df.loc[asset, equity_benchmark])
            weight = 2.0 if regimes[label].get("type") == "bear" else 1.0

            # Score -2 to +2
            if   corr < -0.10: pts = +2
            elif corr < 0:     pts = +1
            elif corr < 0.10:  pts =  0
            elif corr < 0.30:  pts = -1
            else:              pts = -2

            row[label]    = round(corr, 3)
            total_score  += pts * weight
            total_weight += weight

        # Weighted score (normalised to -2 .. +2 range)
        wscore = total_score / total_weight if total_weight > 0 else 0.0

        # Grade
        if   wscore >= 1.5:  grade = "A — Excellent"
        elif wscore >= 0.5:  grade = "B — Good"
        elif wscore >= -0.5: grade = "C — Neutral"
        elif wscore >= -1.0: grade = "D — Poor"
        else:                grade = "F — Fails in Crisis"

        row["Weighted Score"] = round(wscore, 2)
        row["Grade"]          = grade
        rows[asset]           = row

    df = pd.DataFrame(rows).T
    df.index.name = "Defensive Asset"
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. ROLLING CORRELATION WITH REGIME BANDS
# ══════════════════════════════════════════════════════════════════════════════

def rolling_correlation_with_regimes(returns:    pd.DataFrame,
                                      asset_a:    str,
                                      asset_b:    str,
                                      regimes:    dict = None,
                                      window:     int  = 63) -> dict:
    """
    Compute rolling pairwise correlation between two assets,
    annotated with regime start/end dates for chart shading.

    Args:
        returns : pd.DataFrame  daily returns
        asset_a : str           first ticker
        asset_b : str           second ticker
        regimes : dict          for regime band annotations
        window  : int           rolling window in trading days (default 63 = 1 quarter)

    Returns:
        dict with keys:
            rolling_corr   : pd.Series  rolling correlation
            regime_bands   : list of dicts  for chart shading:
                             [{"label", "start", "end", "type"}, ...]
            asset_a        : str
            asset_b        : str
            full_corr      : float  full-period static correlation
            min_corr       : float  lowest rolling value
            max_corr       : float  highest rolling value
            min_corr_date  : pd.Timestamp
            max_corr_date  : pd.Timestamp
    """
    if regimes is None:
        regimes = DEFAULT_REGIMES

    if asset_a not in returns.columns or asset_b not in returns.columns:
        raise ValueError(f"'{asset_a}' or '{asset_b}' not in returns columns.")

    roll_corr  = returns[asset_a].rolling(window).corr(returns[asset_b]).dropna()
    full_corr  = float(returns[asset_a].corr(returns[asset_b]))

    # Regime bands for chart shading
    bands = []
    for label, cfg in regimes.items():
        start = pd.Timestamp(cfg["start"])
        end   = pd.Timestamp(cfg["end"])
        # Only include regimes that overlap with the returns index
        if start <= returns.index[-1] and end >= returns.index[0]:
            bands.append({
                "label": label,
                "start": start,
                "end":   end,
                "type":  cfg.get("type", "neutral"),
            })

    min_val  = float(roll_corr.min())
    max_val  = float(roll_corr.max())

    return {
        "rolling_corr":    roll_corr,
        "regime_bands":    bands,
        "asset_a":         asset_a,
        "asset_b":         asset_b,
        "window":          window,
        "full_corr":       round(full_corr, 3),
        "min_corr":        round(min_val, 3),
        "max_corr":        round(max_val, 3),
        "min_corr_date":   roll_corr.idxmin(),
        "max_corr_date":   roll_corr.idxmax(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. REGIME SUMMARY  — quick one-table overview
# ══════════════════════════════════════════════════════════════════════════════

def regime_summary(returns:          pd.DataFrame,
                   regimes:          dict = None,
                   equity_benchmark: str  = "VOO") -> pd.DataFrame:
    """
    For each regime: number of days, benchmark return, volatility,
    and Sharpe — useful context before diving into correlations.

    Args:
        returns          : pd.DataFrame  daily returns
        regimes          : dict
        equity_benchmark : str

    Returns:
        pd.DataFrame  one row per regime
    """
    if regimes is None:
        regimes = DEFAULT_REGIMES

    rows = []
    for label, cfg in regimes.items():
        subset = returns.loc[cfg["start"]: cfg["end"]]
        if len(subset) == 0:
            continue

        bench_ret = subset[equity_benchmark] if equity_benchmark in subset else None
        ann_ret   = float(bench_ret.mean() * 252 * 100) if bench_ret is not None else np.nan
        ann_vol   = float(bench_ret.std() * np.sqrt(252) * 100) if bench_ret is not None else np.nan
        sharpe    = round(ann_ret / ann_vol, 2) if ann_vol and ann_vol != 0 else np.nan

        rows.append({
            "Regime":         label.replace("\n", " "),
            "Type":           cfg.get("type", "—").capitalize(),
            "Days":           len(subset),
            f"{equity_benchmark} Return (%)":  round(ann_ret, 1),
            f"{equity_benchmark} Vol (%)":     round(ann_vol, 1),
            "Sharpe":         sharpe,
            "Description":    cfg.get("description", ""),
        })

    return pd.DataFrame(rows).set_index("Regime")
