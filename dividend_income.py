"""
dividend_income.py — Dividend & Income Analysis
=================================================
Tools for analysing the income dimension of a portfolio.

Data source:
    All dividend yields, growth rates and payment schedules come from
    cleaner.get_dividend_data() — live Yahoo Finance data.
    No hardcoded yields. No hardcoded assumptions.
    Accumulating ETFs (VUAA, IWDA, etc.) are detected automatically
    and correctly show zero cash income.

Six analysis layers:
  1. Yield Analysis          — gross/net yield per asset, weighted portfolio yield
  2. Income Projection       — forward income at historical growth rates
  3. DRIP Simulation         — compounding effect of reinvesting dividends
  4. Income Sustainability   — safe withdrawal, portfolio longevity table
  5. Dividend Growth Model   — Gordon Growth Model (DDM) where applicable
  6. Income Calendar         — which months generate cash income

Tax handling:
    Pass a tax_costs.TaxProfile (or string key) to any function that
    computes after-tax income. US withholding (15% with W-8BEN, 30% without)
    is applied automatically for USD-denominated assets.

Usage:
    from cleaner import get_dividend_data
    from tax_costs import TAX_PROFILES

    div_data   = get_dividend_data(tickers)      # one call, live data
    tax_profile = TAX_PROFILES["greece"]

    yield_df   = yield_analysis(weights_dict, portfolio_value,
                                div_data, tax_profile)
"""

import numpy as np
import pandas as pd
from tax_costs import (TaxProfile, TAX_PROFILES,
                       US_WITHHOLDING_WITH_W8BEN,
                       US_WITHHOLDING_WITHOUT_W8BEN)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _us_wht(w8ben_filed: bool) -> float:
    return US_WITHHOLDING_WITH_W8BEN if w8ben_filed else US_WITHHOLDING_WITHOUT_W8BEN


def _net_factor(tax_profile: TaxProfile, w8ben_filed: bool) -> float:
    """Combined after-tax multiplier: (1 - US withholding) × (1 - domestic tax)."""
    return (1 - _us_wht(w8ben_filed)) * (1 - tax_profile.dividend_withholding)


def _resolve_tax(tax_profile: TaxProfile | str) -> TaxProfile:
    if isinstance(tax_profile, str):
        return TAX_PROFILES[tax_profile]
    return tax_profile


def _yield(ticker: str, div_data: pd.DataFrame) -> float:
    """Trailing yield as decimal for one ticker. 0 for acc/no-dividend."""
    if ticker not in div_data.index:
        return 0.0
    row = div_data.loc[ticker]
    if row["type"] in ("acc", "no_dividend", "unknown"):
        return 0.0
    return float(row["trailing_yield_pct"]) / 100


def _growth(ticker: str, div_data: pd.DataFrame) -> float:
    """5-year dividend CAGR as decimal. 0 for acc/no-dividend."""
    if ticker not in div_data.index:
        return 0.0
    row = div_data.loc[ticker]
    if row["type"] in ("acc", "no_dividend", "unknown"):
        return 0.0
    return float(row["dividend_growth_5yr"]) / 100


def _pay_months(ticker: str, div_data: pd.DataFrame) -> list[int]:
    if ticker not in div_data.index:
        return []
    months = div_data.loc[ticker, "payment_months"]
    return months if isinstance(months, list) else []


# ══════════════════════════════════════════════════════════════════════════════
# 1. YIELD ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def yield_analysis(weights_dict:    dict[str, float],
                    portfolio_value: float,
                    div_data:        pd.DataFrame,
                    tax_profile:     TaxProfile | str = "greece",
                    w8ben_filed:     bool = True) -> pd.DataFrame:
    """
    Per-asset and portfolio-level yield analysis using live dividend data.

    Args:
        weights_dict    : dict   {ticker: weight}
        portfolio_value : float  total portfolio value in EUR
        div_data        : pd.DataFrame  output of cleaner.get_dividend_data()
        tax_profile     : TaxProfile or str
        w8ben_filed     : bool   True → 15% US withholding, False → 30%

    Returns:
        pd.DataFrame  one row per asset + "── PORTFOLIO ──" summary row
    """
    tax  = _resolve_tax(tax_profile)
    nf   = _net_factor(tax, w8ben_filed)
    wht  = _us_wht(w8ben_filed)

    rows = []
    for ticker, weight in weights_dict.items():
        asset_val    = portfolio_value * weight
        gross_yield  = _yield(ticker, div_data)
        gross_income = asset_val * gross_yield
        withholding  = gross_income * wht
        net_income   = gross_income * nf
        net_yield    = net_income / asset_val if asset_val > 0 else 0.0
        div_growth   = _growth(ticker, div_data)
        asset_type   = div_data.loc[ticker, "type"] if ticker in div_data.index else "unknown"

        rows.append({
            "Ticker":              ticker,
            "Type":                asset_type,
            "Weight (%)":          round(weight * 100, 2),
            "Asset Value (€)":     round(asset_val, 2),
            "Gross Yield (%)":     round(gross_yield * 100, 3),
            "Gross Income (€)":    round(gross_income, 2),
            "US Withholding (€)":  round(withholding, 2),
            "Domestic Tax (€)":    round(gross_income * nf * tax.dividend_withholding
                                         / (1 - tax.dividend_withholding)
                                         if tax.dividend_withholding < 1 else 0.0, 2),
            "Net Income (€)":      round(net_income, 2),
            "Net Yield (%)":       round(net_yield * 100, 3),
            "5yr Div Growth (%)":  round(div_growth * 100, 1),
        })

    df = pd.DataFrame(rows).set_index("Ticker")

    # Portfolio summary
    total_gross  = df["Gross Income (€)"].sum()
    total_net    = df["Net Income (€)"].sum()
    port_yield   = total_gross / portfolio_value * 100 if portfolio_value > 0 else 0
    port_net_yld = total_net   / portfolio_value * 100 if portfolio_value > 0 else 0

    weighted_growth = (
        (df["Gross Income (€)"] * df["5yr Div Growth (%)"]).sum() / total_gross
        if total_gross > 0 else 0.0
    )

    summary = pd.Series({
        "Type":                "portfolio",
        "Weight (%)":          100.0,
        "Asset Value (€)":     round(portfolio_value, 2),
        "Gross Yield (%)":     round(port_yield, 3),
        "Gross Income (€)":    round(total_gross, 2),
        "US Withholding (€)":  round(df["US Withholding (€)"].sum(), 2),
        "Domestic Tax (€)":    round(df["Domestic Tax (€)"].sum(), 2),
        "Net Income (€)":      round(total_net, 2),
        "Net Yield (%)":       round(port_net_yld, 3),
        "5yr Div Growth (%)":  round(weighted_growth, 1),
    }, name="── PORTFOLIO ──")

    return pd.concat([df, summary.to_frame().T])


def yield_on_cost(weights_dict:   dict[str, float],
                   purchase_value: float,
                   current_value:  float,
                   div_data:       pd.DataFrame,
                   tax_profile:    TaxProfile | str = "greece",
                   w8ben_filed:    bool = True) -> dict:
    """
    Yield-on-Cost (YoC): net dividend income as % of original amount invested.

    Args:
        weights_dict   : dict
        purchase_value : float  original investment (EUR)
        current_value  : float  current portfolio value (EUR)
        div_data       : pd.DataFrame
        tax_profile    : TaxProfile or str
        w8ben_filed    : bool

    Returns:
        dict  — gross/net YoC, current yield, annual income, capital gain
    """
    tax = _resolve_tax(tax_profile)
    nf  = _net_factor(tax, w8ben_filed)

    gross = net = 0.0
    for ticker, weight in weights_dict.items():
        g     = current_value * weight * _yield(ticker, div_data)
        gross += g
        net   += g * nf

    capital_gain = current_value - purchase_value
    total_return = (net + capital_gain) / purchase_value * 100 if purchase_value > 0 else 0.0

    return {
        "gross_yoc_pct":     round(gross / purchase_value * 100, 3),
        "net_yoc_pct":       round(net   / purchase_value * 100, 3),
        "current_yield_pct": round(net   / current_value  * 100, 3),
        "annual_net_income": round(net, 2),
        "capital_gain":      round(capital_gain, 2),
        "total_return_pct":  round(total_return, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. INCOME PROJECTION
# ══════════════════════════════════════════════════════════════════════════════

def income_projection(weights_dict:     dict[str, float],
                       portfolio_value:  float,
                       div_data:         pd.DataFrame,
                       tax_profile:      TaxProfile | str = "greece",
                       w8ben_filed:      bool  = True,
                       horizon_years:    int   = 20,
                       portfolio_growth: float = 0.08) -> pd.DataFrame:
    """
    Project annual net dividend income over a multi-year horizon.

    Portfolio value compounds at `portfolio_growth`.
    Each asset's dividend grows at its own 5-year historical CAGR
    (from div_data). Weights remain fixed.

    Args:
        weights_dict     : dict
        portfolio_value  : float  starting value (EUR)
        div_data         : pd.DataFrame  from cleaner.get_dividend_data()
        tax_profile      : TaxProfile or str
        w8ben_filed      : bool
        horizon_years    : int
        portfolio_growth : float  expected annual portfolio CAGR (decimal)

    Returns:
        pd.DataFrame  index=Year, columns: portfolio value, gross/net income,
                      net yield, monthly equivalent
    """
    tax = _resolve_tax(tax_profile)
    nf  = _net_factor(tax, w8ben_filed)

    rows = []
    for year in range(1, horizon_years + 1):
        port_val    = portfolio_value * (1 + portfolio_growth) ** year
        total_gross = total_net = 0.0

        for ticker, weight in weights_dict.items():
            asset_val   = port_val * weight
            cur_yield   = _yield(ticker, div_data) * (1 + _growth(ticker, div_data)) ** year
            gross       = asset_val * cur_yield
            total_gross += gross
            total_net   += gross * nf

        rows.append({
            "Year":                 year,
            "Portfolio Value (€)":  round(port_val, 0),
            "Gross Income (€)":     round(total_gross, 2),
            "Net Income (€)":       round(total_net, 2),
            "Net Yield (%)":        round(total_net / port_val * 100, 3),
            "Monthly Net (€)":      round(total_net / 12, 2),
        })

    return pd.DataFrame(rows).set_index("Year")


# ══════════════════════════════════════════════════════════════════════════════
# 3. DRIP SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def drip_simulation(portfolio_returns: pd.Series,
                     portfolio_values:  pd.Series,
                     weights_dict:      dict[str, float],
                     div_data:          pd.DataFrame,
                     tax_profile:       TaxProfile | str = "greece",
                     w8ben_filed:       bool = True) -> dict:
    """
    Compare cash vs DRIP (reinvested dividends) over the historical window.

    Dividends are credited quarterly at the portfolio-level net yield.

    Args:
        portfolio_returns : pd.Series  daily returns
        portfolio_values  : pd.Series  daily portfolio values
        weights_dict      : dict
        div_data          : pd.DataFrame
        tax_profile       : TaxProfile or str
        w8ben_filed       : bool

    Returns:
        dict with cash_series, drip_series, drip_bonus, drip_cagr, cash_cagr,
             cumulative_cash
    """
    tax = _resolve_tax(tax_profile)
    nf  = _net_factor(tax, w8ben_filed)

    # Portfolio-level annual net yield
    annual_net_yield = sum(
        weights_dict.get(t, 0.0) * _yield(t, div_data) * nf
        for t in weights_dict
    )
    quarterly_yield = annual_net_yield / 4

    quarter_ends = set(portfolio_values.resample("QE").last().index)

    cash_mult = drip_mult = 1.0
    total_cash_div = 0.0
    cash_vals, drip_vals = [], []
    initial_val = float(portfolio_values.iloc[0])

    for date, ret in portfolio_returns.items():
        cash_mult *= (1 + ret)
        drip_mult *= (1 + ret)

        if date in quarter_ends:
            cash_div        = cash_mult * quarterly_yield
            drip_div        = drip_mult * quarterly_yield
            total_cash_div += cash_div * initial_val
            drip_mult      += drip_div   # reinvest

        cash_vals.append(cash_mult * initial_val)
        drip_vals.append(drip_mult * initial_val)

    cash_series = pd.Series(cash_vals, index=portfolio_returns.index,
                             name="Cash Dividends")
    drip_series = pd.Series(drip_vals, index=portfolio_returns.index,
                             name="DRIP Reinvested")

    years    = len(portfolio_returns) / 252
    cash_cagr = (float(cash_series.iloc[-1]) / initial_val) ** (1/years) - 1
    drip_cagr = (float(drip_series.iloc[-1]) / initial_val) ** (1/years) - 1

    return {
        "cash_series":     cash_series,
        "drip_series":     drip_series,
        "cumulative_cash": round(total_cash_div, 2),
        "drip_bonus":      round(float(drip_series.iloc[-1])
                                  - float(cash_series.iloc[-1]), 2),
        "drip_cagr":       round(drip_cagr * 100, 3),
        "cash_cagr":       round(cash_cagr * 100, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. INCOME SUSTAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

def safe_withdrawal_analysis(portfolio_value:    float,
                               annual_withdrawal: float,
                               portfolio_returns: pd.Series,
                               weights_dict:      dict[str, float],
                               div_data:          pd.DataFrame,
                               tax_profile:       TaxProfile | str = "greece",
                               w8ben_filed:       bool = True) -> dict:
    """
    Analyse whether the portfolio can sustainably fund an annual withdrawal.

    Separates dividend income from capital drawdown.
    Projects a 30-year longevity table at the custom withdrawal rate.

    Args:
        portfolio_value   : float
        annual_withdrawal : float  desired annual income (EUR)
        portfolio_returns : pd.Series  historical daily returns
        weights_dict      : dict
        div_data          : pd.DataFrame
        tax_profile       : TaxProfile or str
        w8ben_filed       : bool

    Returns:
        dict with annual_net_dividend, income_coverage_pct, capital_draw,
             four_pct_rule_amt, sustainable_rate_pct, years_to_depletion,
             longevity_table
    """
    tax = _resolve_tax(tax_profile)
    nf  = _net_factor(tax, w8ben_filed)

    annual_net_div = sum(
        portfolio_value * w * _yield(t, div_data) * nf
        for t, w in weights_dict.items()
    )

    years     = len(portfolio_returns) / 252
    port_cum  = (1 + portfolio_returns).cumprod()
    hist_cagr = float(port_cum.iloc[-1] ** (1/years) - 1) if years > 0 else 0.08

    income_coverage  = (annual_net_div / annual_withdrawal * 100
                        if annual_withdrawal > 0 else 0.0)
    capital_draw     = max(annual_withdrawal - annual_net_div, 0.0)
    four_pct_amt     = portfolio_value * 0.04
    sustainable_rate = annual_net_div / portfolio_value * 100

    balance   = portfolio_value
    rows      = []
    depletion = None
    div_yield = annual_net_div / portfolio_value  # fixed yield ratio

    for yr in range(1, 31):
        dividend       = balance * div_yield
        capital_needed = max(annual_withdrawal - dividend, 0.0)
        balance        = balance * (1 + hist_cagr) - capital_needed

        if balance <= 0 and depletion is None:
            depletion = yr
            balance   = 0.0

        rows.append({
            "Year":                yr,
            "Portfolio Value (€)": round(balance, 0),
            "Dividend Income (€)": round(min(dividend, annual_withdrawal), 0),
            "Capital Draw (€)":    round(capital_needed, 0),
            "Withdrawal (€)":      round(annual_withdrawal, 0),
        })
        if balance == 0.0:
            break

    return {
        "annual_net_dividend":  round(annual_net_div, 2),
        "income_coverage_pct":  round(income_coverage, 1),
        "capital_draw":         round(capital_draw, 2),
        "four_pct_rule_amt":    round(four_pct_amt, 2),
        "sustainable_rate_pct": round(sustainable_rate, 3),
        "years_to_depletion":   depletion,
        "longevity_table":      pd.DataFrame(rows).set_index("Year"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. DIVIDEND GROWTH MODEL (DDM)
# ══════════════════════════════════════════════════════════════════════════════

def dividend_discount_model(current_price:    float,
                              current_dividend: float,
                              growth_rate:      float,
                              required_return:  float) -> dict:
    """
    Gordon Growth Model (constant-growth DDM) fair value estimate.

    Fair Value = D1 / (r - g)
    where D1 = next year dividend, r = required return, g = growth rate.

    Constraint: r > g. When g >= r the model does not apply — the stock
    is a high-growth compounder and a DCF model should be used instead.
    When current_dividend = 0 the model does not apply either.

    Returns:
        dict with keys: d1, fair_value, margin_of_safety_pct,
                        applicable (bool), verdict (str)
    """
    d1 = round(current_dividend * (1 + growth_rate), 4)

    if current_dividend <= 0:
        return {"d1": 0.0, "fair_value": None,
                "margin_of_safety_pct": None,
                "applicable": False,
                "verdict": "NON-DIVIDEND — DDM N/A"}

    if growth_rate >= required_return:
        return {"d1": d1, "fair_value": None,
                "margin_of_safety_pct": None,
                "applicable": False,
                "verdict": "GROWTH STOCK — DDM N/A"}

    fair_value = d1 / (required_return - growth_rate)
    mos        = (fair_value - current_price) / current_price * 100

    verdict = ("UNDERVALUED" if mos > 15
               else "OVERVALUED" if mos < -15
               else "FAIRLY VALUED")

    return {"d1": d1, "fair_value": round(fair_value, 2),
            "margin_of_safety_pct": round(mos, 1),
            "applicable": True, "verdict": verdict}


def ddm_portfolio_scan(weights_dict:    dict[str, float],
                        current_prices:  dict[str, float],
                        div_data:        pd.DataFrame,
                        required_return: float = 0.10) -> pd.DataFrame:
    """
    Run DDM valuation for all assets using live dividend data.

    Assets are automatically categorised:
      DIVIDEND STOCK  — mature dividend payer, DDM applicable
      GROWTH STOCK    — g >= r, DDM not applicable
      NON-DIVIDEND    — pays no dividend (stock)
      ACC ETF         — accumulating ETF, no cash income
      NO PRICE DATA   — price not provided

    Args:
        weights_dict    : dict   {ticker: weight}
        current_prices  : dict   {ticker: price_eur}  — provide all tickers
        div_data        : pd.DataFrame  from cleaner.get_dividend_data()
        required_return : float  annual required return (decimal)

    Returns:
        pd.DataFrame  one row per asset, all categories shown
    """
    rows = []
    for ticker, weight in weights_dict.items():
        y   = _yield(ticker, div_data)
        g   = _growth(ticker, div_data)
        p   = current_prices.get(ticker)
        typ = div_data.loc[ticker, "type"] if ticker in div_data.index else "unknown"

        if p is None:
            category = "NO PRICE DATA"
            row = {"Category": category, "Price (€)": None,
                   "Div Yield (%)": round(y*100, 2),
                   "Growth Rate (%)": round(g*100, 1),
                   "DDM Fair Value (€)": None,
                   "Margin of Safety (%)": None,
                   "Verdict": "N/A"}
        elif typ == "acc":
            category = "ACC ETF"
            row = {"Category": category, "Price (€)": round(p, 2),
                   "Div Yield (%)": 0.0, "Growth Rate (%)": 0.0,
                   "DDM Fair Value (€)": None,
                   "Margin of Safety (%)": None,
                   "Verdict": "N/A — Accumulating ETF"}
        else:
            annual_div = p * y
            result     = dividend_discount_model(p, annual_div, g, required_return)
            if result["applicable"]:
                category = "DIVIDEND STOCK"
            elif y == 0:
                category = "NON-DIVIDEND"
            else:
                category = "GROWTH STOCK (g >= r)"
            row = {"Category": category, "Price (€)": round(p, 2),
                   "Div Yield (%)": round(y*100, 2),
                   "Growth Rate (%)": round(g*100, 1),
                   "DDM Fair Value (€)": result["fair_value"],
                   "Margin of Safety (%)": result["margin_of_safety_pct"],
                   "Verdict": result["verdict"]}

        row["Ticker"]      = ticker
        row["Weight (%)"]  = round(weight * 100, 2)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return (pd.DataFrame(rows)
              .set_index("Ticker")
              [["Weight (%)", "Category", "Price (€)", "Div Yield (%)",
                "Growth Rate (%)", "DDM Fair Value (€)",
                "Margin of Safety (%)", "Verdict"]])


# ══════════════════════════════════════════════════════════════════════════════
# 6. INCOME CALENDAR
# ══════════════════════════════════════════════════════════════════════════════

def income_calendar(weights_dict:    dict[str, float],
                     portfolio_value: float,
                     div_data:        pd.DataFrame,
                     tax_profile:     TaxProfile | str = "greece",
                     w8ben_filed:     bool = True) -> pd.DataFrame:
    """
    Monthly income calendar showing expected net dividend receipts per month.

    Payment months are derived from actual ex-dividend dates in div_data
    (last 2 years of history). No hardcoded assumptions.

    Args:
        weights_dict    : dict
        portfolio_value : float  EUR
        div_data        : pd.DataFrame
        tax_profile     : TaxProfile or str
        w8ben_filed     : bool

    Returns:
        pd.DataFrame  index=month names, columns=tickers + Total + YTD
    """
    tax = _resolve_tax(tax_profile)
    nf  = _net_factor(tax, w8ben_filed)

    months     = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
    month_data = {m: {} for m in months}

    for ticker, weight in weights_dict.items():
        asset_val    = portfolio_value * weight
        annual_gross = asset_val * _yield(ticker, div_data)
        pay_months   = _pay_months(ticker, div_data)
        n_payments   = len(pay_months)

        if n_payments == 0 or annual_gross == 0:
            for m in months:
                month_data[m][ticker] = 0.0
            continue

        per_payment_net = (annual_gross / n_payments) * nf

        for m_idx, m in enumerate(months, 1):
            month_data[m][ticker] = round(per_payment_net, 2) \
                                    if m_idx in pay_months else 0.0

    df = pd.DataFrame(month_data).T
    df.index.name       = "Month"
    df["Total Net (€)"] = df.sum(axis=1).round(2)
    df["YTD Net (€)"]   = df["Total Net (€)"].cumsum().round(2)
    return df


def income_summary(weights_dict:    dict[str, float],
                    portfolio_value: float,
                    div_data:        pd.DataFrame,
                    tax_profile:     TaxProfile | str = "greece",
                    w8ben_filed:     bool = True) -> dict:
    """
    One-call income summary: key figures ready for a client report.

    Args:
        weights_dict    : dict
        portfolio_value : float
        div_data        : pd.DataFrame
        tax_profile     : TaxProfile or str
        w8ben_filed     : bool

    Returns:
        dict  — annual/monthly/weekly net income, yields, tax drag
    """
    tax = _resolve_tax(tax_profile)
    nf  = _net_factor(tax, w8ben_filed)

    gross = net = 0.0
    for ticker, weight in weights_dict.items():
        g      = portfolio_value * weight * _yield(ticker, div_data)
        gross += g
        net   += g * nf

    return {
        "annual_gross_income":  round(gross, 2),
        "annual_net_income":    round(net, 2),
        "monthly_net_income":   round(net / 12, 2),
        "weekly_net_income":    round(net / 52, 2),
        "gross_yield_pct":      round(gross / portfolio_value * 100, 3),
        "net_yield_pct":        round(net   / portfolio_value * 100, 3),
        "tax_drag_eur":         round(gross - net, 2),
        "tax_drag_pct":         round((gross - net) / gross * 100, 1)
                                if gross > 0 else 0.0,
        "w8ben_filed":          w8ben_filed,
        "tax_profile":          tax.name,
    }
