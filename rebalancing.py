"""
rebalancing.py — Portfolio Rebalancing Analysis
=================================================
Tools for analysing, backtesting and executing portfolio rebalancing.

Five analysis layers:
  1. Drift Analysis         — how actual weights diverge from targets over time
  2. Calendar Rebalancing   — rebalance on a fixed schedule (monthly/quarterly/annual)
  3. Threshold Rebalancing  — rebalance when any asset drifts beyond a tolerance band
  4. Cost/Benefit Analysis  — compare strategies net of realistic transaction costs
  5. Trade List             — exact buy/sell amounts needed to rebalance today
  6. Optimal Frequency      — automatically find the best rebalancing strategy

Transaction costs are computed via tax_costs.transaction_cost(), which models
broker commission, bid-ask spread and FX fees realistically.
Pass a tax_costs.BrokerProfile (or a broker name string) to any backtest
function to use the correct cost structure for your client's broker.

Default broker: "interactive_brokers" (lowest cost).
Available: "interactive_brokers", "degiro", "etoro", "trading212",
           "greek_bank", "custom".
"""

import numpy as np
import pandas as pd
from tax_costs import transaction_cost, BROKERS, BrokerProfile


# ── Constants ─────────────────────────────────────────────────────────────────

CALENDAR_FREQUENCIES = {
    "monthly":     21,    # approx trading days
    "quarterly":   63,
    "semi-annual": 126,
    "annual":      252,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. DRIFT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def compute_weight_drift(prices: pd.DataFrame,
                          target_weights: np.ndarray,
                          tickers: list[str],
                          initial_capital: float = 10_000.0) -> dict:
    """
    Track how actual portfolio weights drift from targets over time
    under a pure buy-and-hold strategy (no rebalancing).

    Args:
        prices          : pd.DataFrame  daily adjusted close prices
        target_weights  : np.ndarray    intended allocation (must sum to 1)
        tickers         : list[str]     ticker order matching target_weights
        initial_capital : float         starting portfolio value

    Returns:
        dict with keys:
            actual_weights  : pd.DataFrame  daily actual weights (columns=tickers)
            drift           : pd.DataFrame  actual - target per day
            abs_drift       : pd.Series     max absolute drift across all assets per day
            max_drift_asset : pd.Series     which asset has the largest drift each day
    """
    target_weights = np.asarray(target_weights, dtype=float)
    prices_sub     = prices[tickers].copy()

    # Initial share counts bought on day 0
    initial_prices = prices_sub.iloc[0]
    shares         = (initial_capital * target_weights) / initial_prices.values

    # Daily portfolio value per asset
    asset_values   = prices_sub * shares          # shape: (days, n_assets)
    total_values   = asset_values.sum(axis=1)     # shape: (days,)

    # Actual weights each day
    actual_weights = asset_values.div(total_values, axis=0)
    actual_weights.columns = tickers

    drift          = actual_weights - target_weights    # signed drift
    abs_drift      = drift.abs().max(axis=1)            # worst offender each day
    max_drift_asset = drift.abs().idxmax(axis=1)        # which asset

    return {
        "actual_weights":  actual_weights,
        "drift":           drift,
        "abs_drift":       abs_drift,
        "max_drift_asset": max_drift_asset,
    }


def drift_summary(drift_result: dict,
                   target_weights: np.ndarray,
                   tickers: list[str]) -> pd.DataFrame:
    """
    Summary table: per-asset average drift, max drift, and days outside ±5%.

    Args:
        drift_result   : dict    output of compute_weight_drift()
        target_weights : np.ndarray
        tickers        : list[str]

    Returns:
        pd.DataFrame  one row per asset
    """
    drift = drift_result["drift"]
    rows  = []
    for i, ticker in enumerate(tickers):
        col = drift[ticker]
        rows.append({
            "Ticker":              ticker,
            "Target Weight (%)":   round(target_weights[i] * 100, 2),
            "Avg Actual (%)":      round((col + target_weights[i]).mean() * 100, 2),
            "Avg Drift (pp)":      round(col.mean() * 100, 2),
            "Max Drift (pp)":      round(col.abs().max() * 100, 2),
            "Days Outside ±5%":    int((col.abs() > 0.05).sum()),
            "Days Outside ±10%":   int((col.abs() > 0.10).sum()),
        })
    return pd.DataFrame(rows).set_index("Ticker")


# ══════════════════════════════════════════════════════════════════════════════
# 2. CALENDAR REBALANCING BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def calendar_rebalance_backtest(prices: pd.DataFrame,
                                  target_weights: np.ndarray,
                                  tickers: list[str],
                                  frequency: str = "annual",
                                  initial_capital: float = 10_000.0,
                                  broker: BrokerProfile | str = "interactive_brokers",
                                  is_usd_asset: bool = True) -> dict:
    """
    Backtest a calendar-based rebalancing strategy.

    On each rebalance date, all assets are sold and repurchased at target
    weights. Transaction costs are computed via tax_costs.transaction_cost()
    using the specified broker profile (commission + spread + FX fee).

    Args:
        prices          : pd.DataFrame  daily adjusted close prices
        target_weights  : np.ndarray    target allocation
        tickers         : list[str]
        frequency       : str           "monthly" | "quarterly" |
                                        "semi-annual" | "annual"
        initial_capital : float
        broker          : BrokerProfile or str  key from tax_costs.BROKERS
        is_usd_asset    : bool          True for USD-denominated ETFs

    Returns:
        dict with keys:
            portfolio_values  : pd.Series
            rebalance_dates   : list[pd.Timestamp]
            n_rebalances      : int
            total_cost        : float   cumulative transaction costs paid
            shares_history    : pd.DataFrame  daily share counts
    """
    if frequency not in CALENDAR_FREQUENCIES:
        raise ValueError(f"frequency must be one of {list(CALENDAR_FREQUENCIES)}")

    target_weights   = np.asarray(target_weights, dtype=float)
    prices_sub       = prices[tickers].copy()
    interval         = CALENDAR_FREQUENCIES[frequency]

    shares           = np.zeros(len(tickers))
    portfolio_values = []
    rebalance_dates  = []
    total_cost       = 0.0
    day_counter      = 0
    shares_rows      = []

    for i, (date, row) in enumerate(prices_sub.iterrows()):
        current_prices = row.values

        # --- Rebalance trigger: day 0 (initial buy) or calendar interval ---
        if i == 0 or day_counter >= interval:
            current_value = (shares * current_prices).sum() if i > 0 else initial_capital

            # Transaction cost: broker-aware (commission + spread + FX)
            if i > 0:
                _broker     = BROKERS[broker] if isinstance(broker, str) else broker
                cost        = transaction_cost(current_value, _broker,
                                               is_usd_asset=is_usd_asset)["total"]
                total_cost += cost
                current_value -= cost

            # Buy at target weights
            shares      = (current_value * target_weights) / current_prices
            day_counter = 0
            rebalance_dates.append(date)

        portfolio_values.append(float((shares * current_prices).sum()))
        shares_rows.append(shares.copy())
        day_counter += 1

    return {
        "portfolio_values": pd.Series(portfolio_values,
                                       index=prices_sub.index,
                                       name=f"Calendar ({frequency})"),
        "rebalance_dates":  rebalance_dates,
        "n_rebalances":     len(rebalance_dates) - 1,
        "total_cost":       round(total_cost, 2),
        "shares_history":   pd.DataFrame(shares_rows,
                                          index=prices_sub.index,
                                          columns=tickers),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. THRESHOLD REBALANCING BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def threshold_rebalance_backtest(prices: pd.DataFrame,
                                   target_weights: np.ndarray,
                                   tickers: list[str],
                                   threshold: float = 0.05,
                                   initial_capital: float = 10_000.0,
                                   broker: BrokerProfile | str = "interactive_brokers",
                                   is_usd_asset: bool = True) -> dict:
    """
    Backtest a threshold (band) rebalancing strategy.

    Rebalancing is triggered when any single asset drifts more than
    `threshold` percentage points from its target weight. Transaction costs
    are computed via tax_costs.transaction_cost().

    Args:
        prices          : pd.DataFrame  daily prices
        target_weights  : np.ndarray    target allocation
        tickers         : list[str]
        threshold       : float         drift tolerance (e.g. 0.05 = 5pp)
        initial_capital : float
        broker          : BrokerProfile or str
        is_usd_asset    : bool

    Returns:
        dict with keys:
            portfolio_values  : pd.Series
            rebalance_dates   : list[pd.Timestamp]
            n_rebalances      : int
            total_cost        : float
            trigger_asset     : list[str]   which asset triggered each rebalance
    """
    target_weights   = np.asarray(target_weights, dtype=float)
    prices_sub       = prices[tickers].copy()

    shares           = np.zeros(len(tickers))
    portfolio_values = []
    rebalance_dates  = []
    trigger_assets   = []
    total_cost       = 0.0

    for i, (date, row) in enumerate(prices_sub.iterrows()):
        current_prices = row.values

        if i == 0:
            # Initial purchase at target weights
            shares = (initial_capital * target_weights) / current_prices
            rebalance_dates.append(date)
            trigger_assets.append("Initial Buy")
        else:
            current_value   = float((shares * current_prices).sum())
            actual_weights  = (shares * current_prices) / current_value
            drift           = np.abs(actual_weights - target_weights)
            max_drift       = drift.max()
            worst_asset     = tickers[int(drift.argmax())]

            if max_drift > threshold:
                _broker     = BROKERS[broker] if isinstance(broker, str) else broker
                cost        = transaction_cost(current_value, _broker,
                                               is_usd_asset=is_usd_asset)["total"]
                total_cost += cost
                current_value -= cost

                shares = (current_value * target_weights) / current_prices
                rebalance_dates.append(date)
                trigger_assets.append(worst_asset)

        portfolio_values.append(float((shares * current_prices).sum()))

    label = f"Threshold ({threshold*100:.0f}%)"
    return {
        "portfolio_values": pd.Series(portfolio_values,
                                       index=prices_sub.index,
                                       name=label),
        "rebalance_dates":  rebalance_dates,
        "n_rebalances":     len(rebalance_dates) - 1,
        "total_cost":       round(total_cost, 2),
        "trigger_asset":    trigger_assets,
    }


def buy_and_hold_backtest(prices: pd.DataFrame,
                           target_weights: np.ndarray,
                           tickers: list[str],
                           initial_capital: float = 10_000.0) -> pd.Series:
    """
    Pure buy-and-hold baseline (no rebalancing, no costs).

    Args:
        prices          : pd.DataFrame
        target_weights  : np.ndarray
        tickers         : list[str]
        initial_capital : float

    Returns:
        pd.Series  daily portfolio value
    """
    target_weights = np.asarray(target_weights, dtype=float)
    prices_sub     = prices[tickers].copy()
    initial_prices = prices_sub.iloc[0].values
    shares         = (initial_capital * target_weights) / initial_prices
    values         = (prices_sub * shares).sum(axis=1)
    return values.rename("Buy & Hold")


# ══════════════════════════════════════════════════════════════════════════════
# 4. COST / BENEFIT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def rebalancing_cost_benefit(prices: pd.DataFrame,
                               target_weights: np.ndarray,
                               tickers: list[str],
                               initial_capital: float = 10_000.0,
                               broker: BrokerProfile | str = "interactive_brokers",
                               is_usd_asset: bool = True,
                               frequencies: list[str] = None,
                               thresholds: list[float] = None) -> pd.DataFrame:
    """
    Run all rebalancing strategies and compare their net performance
    against a buy-and-hold baseline.

    Args:
        prices        : pd.DataFrame
        target_weights: np.ndarray
        tickers       : list[str]
        initial_capital: float
        broker        : BrokerProfile or str  broker profile for cost calculation
        is_usd_asset  : bool
        frequencies   : list[str]   calendar frequencies to test (default: all four)
        thresholds    : list[float] drift thresholds to test (default: [0.03, 0.05, 0.10])

    Returns:
        pd.DataFrame  one row per strategy, sorted by Final Value descending
    """
    if frequencies is None:
        frequencies = ["monthly", "quarterly", "semi-annual", "annual"]
    if thresholds is None:
        thresholds = [0.03, 0.05, 0.10]

    results = []

    # --- Baseline: Buy & Hold ---
    bah     = buy_and_hold_backtest(prices, target_weights, tickers, initial_capital)
    results.append(_strategy_metrics(bah, n_reb=0, cost=0.0, label="Buy & Hold"))

    # --- Calendar strategies ---
    for freq in frequencies:
        r = calendar_rebalance_backtest(
            prices, target_weights, tickers,
            frequency=freq,
            initial_capital=initial_capital,
            broker=broker,
            is_usd_asset=is_usd_asset,
        )
        results.append(_strategy_metrics(
            r["portfolio_values"],
            n_reb=r["n_rebalances"],
            cost=r["total_cost"],
            label=f"Calendar — {freq}",
        ))

    # --- Threshold strategies ---
    for thr in thresholds:
        r = threshold_rebalance_backtest(
            prices, target_weights, tickers,
            threshold=thr,
            initial_capital=initial_capital,
            broker=broker,
            is_usd_asset=is_usd_asset,
        )
        results.append(_strategy_metrics(
            r["portfolio_values"],
            n_reb=r["n_rebalances"],
            cost=r["total_cost"],
            label=f"Threshold — {thr*100:.0f}%",
        ))

    df = pd.DataFrame(results).set_index("Strategy")
    return df.sort_values("Final Value", ascending=False)


def _strategy_metrics(portfolio_values: pd.Series,
                       n_reb: int,
                       cost: float,
                       label: str) -> dict:
    """Compute summary metrics for one strategy. Internal helper."""
    years     = len(portfolio_values) / 252
    start_val = float(portfolio_values.iloc[0])
    final_val = float(portfolio_values.iloc[-1])
    cagr      = (final_val / start_val) ** (1 / years) - 1 if years > 0 else 0.0

    daily_ret = portfolio_values.pct_change().dropna()
    vol       = float(daily_ret.std() * np.sqrt(252))
    sharpe    = (cagr / vol) if vol != 0 else 0.0

    dd        = (portfolio_values / portfolio_values.cummax()) - 1
    mdd       = float(dd.min())

    return {
        "Strategy":          label,
        "Final Value":       round(final_val, 2),
        "CAGR (%)":          round(cagr * 100, 2),
        "Ann. Vol (%)":      round(vol * 100, 2),
        "Sharpe":            round(sharpe, 3),
        "Max Drawdown (%)":  round(mdd * 100, 2),
        "# Rebalances":      n_reb,
        "Total Cost (€)":    round(cost, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRADE LIST — "What do I buy/sell today?"
# ══════════════════════════════════════════════════════════════════════════════

def generate_trade_list(current_prices: pd.Series,
                         current_shares: pd.Series,
                         target_weights: np.ndarray,
                         tickers: list[str],
                         broker: BrokerProfile | str = "interactive_brokers",
                         is_usd_asset: bool = True,
                         min_trade_value: float = 50.0) -> pd.DataFrame:
    """
    Generate the exact buy/sell orders needed to rebalance to target weights
    at current market prices. Transaction costs per trade are computed via
    tax_costs.transaction_cost() using the specified broker profile.

    Args:
        current_prices  : pd.Series   latest prices (index=tickers)
        current_shares  : pd.Series   current share holdings (index=tickers)
        target_weights  : np.ndarray  desired allocation
        tickers         : list[str]
        broker          : BrokerProfile or str
        is_usd_asset    : bool
        min_trade_value : float       ignore trades smaller than this (€)

    Returns:
        pd.DataFrame  one row per asset with trade instructions,
                      plus a summary row for total portfolio
    """
    target_weights = np.asarray(target_weights, dtype=float)

    prices_arr  = np.array([current_prices[t] for t in tickers])
    shares_arr  = np.array([current_shares[t] for t in tickers])

    current_values  = shares_arr * prices_arr
    total_value     = current_values.sum()
    actual_weights  = current_values / total_value
    target_values   = total_value * target_weights

    delta_values    = target_values - current_values      # + = buy, - = sell
    delta_shares    = delta_values / prices_arr

    rows = []
    for i, ticker in enumerate(tickers):
        action     = "BUY" if delta_values[i] > 0 else ("SELL" if delta_values[i] < 0 else "HOLD")
        trade_val  = abs(delta_values[i])
        _broker    = BROKERS[broker] if isinstance(broker, str) else broker
        cost       = transaction_cost(trade_val, _broker,
                         is_usd_asset=is_usd_asset)["total"] if action != "HOLD" else 0.0

        # Skip tiny trades
        if trade_val < min_trade_value and action != "HOLD":
            action = "HOLD (below min)"

        rows.append({
            "Ticker":              ticker,
            "Current Weight (%)":  round(actual_weights[i] * 100, 2),
            "Target Weight (%)":   round(target_weights[i] * 100, 2),
            "Drift (pp)":          round((actual_weights[i] - target_weights[i]) * 100, 2),
            "Current Value (€)":   round(current_values[i], 2),
            "Target Value (€)":    round(target_values[i], 2),
            "Action":              action,
            "Trade Amount (€)":    round(delta_values[i], 2),
            "Shares to Trade":     round(delta_shares[i], 4),
            "Est. Cost (€)":       round(cost, 2),
        })

    df = pd.DataFrame(rows).set_index("Ticker")

    # Summary row
    total_trades    = df[df["Action"].str.startswith(("BUY", "SELL"))]["Trade Amount (€)"].abs().sum()
    total_cost_est  = df["Est. Cost (€)"].sum()

    summary = pd.DataFrame([{
        "Ticker":              "── TOTAL ──",
        "Current Weight (%)":  100.0,
        "Target Weight (%)":   100.0,
        "Drift (pp)":          round(df["Drift (pp)"].abs().max(), 2),
        "Current Value (€)":   round(total_value, 2),
        "Target Value (€)":    round(total_value, 2),
        "Action":              f"{len(df[df['Action'].isin(['BUY','SELL'])])} trades",
        "Trade Amount (€)":    round(total_trades, 2),
        "Shares to Trade":     None,
        "Est. Cost (€)":       round(total_cost_est, 2),
    }]).set_index("Ticker")

    return pd.concat([df, summary]).infer_objects()


def current_shares_from_prices(prices: pd.DataFrame,
                                 target_weights: np.ndarray,
                                 tickers: list[str],
                                 initial_capital: float,
                                 as_of_date: str = None) -> pd.Series:
    """
    Compute share counts for a buy-and-hold portfolio started at day 0
    (or at a specific date) with target_weights allocation.

    Useful for feeding into generate_trade_list() when you don't
    have actual brokerage data.

    Args:
        prices          : pd.DataFrame
        target_weights  : np.ndarray
        tickers         : list[str]
        initial_capital : float
        as_of_date      : str  "YYYY-MM-DD" — use prices on this date for
                               initial purchase; defaults to first available

    Returns:
        pd.Series  index=tickers, values=share counts
    """
    target_weights = np.asarray(target_weights, dtype=float)
    prices_sub     = prices[tickers]

    if as_of_date is not None:
        initial_prices = prices_sub.loc[as_of_date]
    else:
        initial_prices = prices_sub.iloc[0]

    shares = (initial_capital * target_weights) / initial_prices.values
    return pd.Series(shares, index=tickers, name="shares")


# ══════════════════════════════════════════════════════════════════════════════
# 6. REBALANCING FREQUENCY OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

def optimal_rebalancing_frequency(prices: pd.DataFrame,
                                   target_weights: np.ndarray,
                                   tickers: list[str],
                                   initial_capital: float = 10_000.0,
                                   broker: BrokerProfile | str = "interactive_brokers",
                                   is_usd_asset: bool = True) -> dict:
    """
    Find the rebalancing frequency and threshold that maximise
    risk-adjusted return (Sharpe) net of transaction costs.

    Tests all four calendar frequencies and thresholds [2%, 5%, 7%, 10%, 15%].

    Args:
        prices               : pd.DataFrame
        target_weights       : np.ndarray
        tickers              : list[str]
        initial_capital      : float
        broker               : BrokerProfile or str

    Returns:
        dict with keys:
            results_df          : pd.DataFrame  full comparison table
            best_sharpe         : str           strategy label with highest Sharpe
            best_final_value    : str           strategy label with highest final value
            recommendation      : str           plain-English recommendation
    """
    df = rebalancing_cost_benefit(
        prices, target_weights, tickers,
        initial_capital=initial_capital,
        broker=broker,
        is_usd_asset=is_usd_asset,
        frequencies=["monthly", "quarterly", "semi-annual", "annual"],
        thresholds=[0.02, 0.05, 0.07, 0.10, 0.15],
    )

    best_sharpe_label = str(df["Sharpe"].idxmax())
    best_value_label  = str(df["Final Value"].idxmax())

    # Plain-English recommendation
    best_row    = df.loc[best_sharpe_label]
    n_reb       = int(best_row["# Rebalances"])
    total_cost  = float(best_row["Total Cost (€)"])
    bah_final   = float(df.loc["Buy & Hold", "Final Value"])
    best_final  = float(best_row["Final Value"])
    bonus       = best_final - bah_final

    rec = (
        f"Optimal strategy (highest Sharpe): '{best_sharpe_label}'. "
        f"Required {n_reb} rebalance events, "
        f"total transaction costs €{total_cost:,.2f}. "
        f"Net rebalancing bonus vs Buy & Hold: €{bonus:+,.2f}."
    )

    return {
        "results_df":        df,
        "best_sharpe":       best_sharpe_label,
        "best_final_value":  best_value_label,
        "recommendation":    rec,
    }