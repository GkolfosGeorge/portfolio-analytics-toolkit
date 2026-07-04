"""
tax_costs.py — Transaction Cost & Tax Model
=============================================
A realistic, broker-aware cost model for portfolio transactions.

Covers:
  1. Transaction cost engine   — commission + spread + market impact
  2. Capital gains tax         — Greek & EU tax rules, FIFO lot tracking
  3. Dividend withholding tax  — US ETFs for EU/Greek residents
  4. Broker presets            — ready-made profiles for common brokers
  5. Total cost of ownership   — annual drag from expense ratios
  6. After-tax return          — net performance adjusted for all costs

Design principle:
  Every function is self-contained and returns plain numbers or DataFrames.
  No side effects. Import and call — nothing else required.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════════════
# 1. BROKER COST PROFILES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BrokerProfile:
    """
    Complete cost profile for a single broker / account type.

    All rates are decimals (e.g. 0.004 = 0.4%).
    Fixed fees are in EUR.
    """
    name:                  str
    commission_pct:        float   # % of trade value (one-way)
    commission_min_eur:    float   # minimum per trade, EUR
    commission_max_eur:    float   # cap per trade (0 = no cap)
    spread_pct:            float   # typical bid-ask spread (one-way)
    fx_fee_pct:            float   # FX conversion fee (EUR/USD trades)
    custody_fee_annual:    float   # annual custody fee (% of AUM)
    notes:                 str     = ""


# ── Built-in broker presets ───────────────────────────────────────────────────

BROKERS: dict[str, BrokerProfile] = {

    "interactive_brokers": BrokerProfile(
        name               = "Interactive Brokers (IBKR Lite/Pro)",
        commission_pct     = 0.0005,    # 0.05% for EU stocks/ETFs
        commission_min_eur = 1.0,
        commission_max_eur = 0.0,       # no cap
        spread_pct         = 0.0001,    # very liquid ETFs ~1bp
        fx_fee_pct         = 0.0002,    # 0.02% FX conversion
        custody_fee_annual = 0.0,
        notes              = "Best for active rebalancing. Lowest per-trade cost.",
    ),

    "degiro": BrokerProfile(
        name               = "DEGIRO",
        commission_pct     = 0.0,       # flat fee model
        commission_min_eur = 3.0,       # €3 flat for US ETFs (approx)
        commission_max_eur = 3.0,
        spread_pct         = 0.0002,
        fx_fee_pct         = 0.0025,    # 0.25% FX — significant for USD ETFs
        custody_fee_annual = 0.0,
        notes              = "Flat fee good for small trades. FX fee hurts USD ETFs.",
    ),

    "etoro": BrokerProfile(
        name               = "eToro",
        commission_pct     = 0.0,       # zero commission but wide spreads
        commission_min_eur = 0.0,
        commission_max_eur = 0.0,
        spread_pct         = 0.0015,    # ~0.15% effective spread
        fx_fee_pct         = 0.005,     # 0.5% FX on deposits/withdrawals
        custody_fee_annual = 0.0,
        notes              = "Zero commission but spread & FX costs are high.",
    ),

    "greek_bank": BrokerProfile(
        name               = "Greek Bank Brokerage (avg)",
        commission_pct     = 0.004,     # 0.4% typical
        commission_min_eur = 10.0,
        commission_max_eur = 0.0,
        spread_pct         = 0.0005,
        fx_fee_pct         = 0.005,
        custody_fee_annual = 0.001,     # ~0.1% custody
        notes              = "Highest cost option. Avoid for frequent rebalancing.",
    ),

    "trading212": BrokerProfile(
        name               = "Trading 212",
        commission_pct     = 0.0,
        commission_min_eur = 0.0,
        commission_max_eur = 0.0,
        spread_pct         = 0.0003,
        fx_fee_pct         = 0.0015,    # 0.15% FX
        custody_fee_annual = 0.0,
        notes              = "Zero commission. FX fee applies for USD ETFs.",
    ),

    "custom": BrokerProfile(
        name               = "Custom Profile",
        commission_pct     = 0.001,
        commission_min_eur = 1.0,
        commission_max_eur = 0.0,
        spread_pct         = 0.0002,
        fx_fee_pct         = 0.001,
        custody_fee_annual = 0.0,
        notes              = "Edit this profile for your specific broker.",
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRANSACTION COST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def transaction_cost(trade_value_eur:   float,
                      broker:            BrokerProfile | str = "interactive_brokers",
                      is_usd_asset:      bool  = True,
                      include_spread:    bool  = True) -> dict:
    """
    Compute the full one-way transaction cost for a single trade.

    Args:
        trade_value_eur : float   gross trade value in EUR
        broker          : BrokerProfile or str key from BROKERS dict
        is_usd_asset    : bool    True for US-listed ETFs (triggers FX fee)
        include_spread  : bool    include bid-ask spread cost

    Returns:
        dict with keys:
            commission    : float  EUR
            spread        : float  EUR
            fx_fee        : float  EUR
            total         : float  EUR
            total_pct     : float  % of trade value
    """
    if isinstance(broker, str):
        if broker not in BROKERS:
            raise ValueError(f"Unknown broker '{broker}'. "
                             f"Choose from: {list(BROKERS)}")
        broker = BROKERS[broker]

    # Commission
    commission = trade_value_eur * broker.commission_pct
    commission = max(commission, broker.commission_min_eur)
    if broker.commission_max_eur > 0:
        commission = min(commission, broker.commission_max_eur)

    # Spread
    spread = trade_value_eur * broker.spread_pct if include_spread else 0.0

    # FX fee (EUR investor buying USD-denominated asset)
    fx_fee = trade_value_eur * broker.fx_fee_pct if is_usd_asset else 0.0

    total     = commission + spread + fx_fee
    total_pct = (total / trade_value_eur * 100) if trade_value_eur > 0 else 0.0

    return {
        "commission":  round(commission, 4),
        "spread":      round(spread, 4),
        "fx_fee":      round(fx_fee, 4),
        "total":       round(total, 4),
        "total_pct":   round(total_pct, 4),
    }


def rebalance_trade_costs(trade_list:    pd.DataFrame,
                           broker:        BrokerProfile | str = "interactive_brokers",
                           usd_tickers:   list[str] = None) -> pd.DataFrame:
    """
    Apply transaction costs to a full rebalancing trade list
    (output of rebalancing.generate_trade_list()).

    Args:
        trade_list   : pd.DataFrame   from rebalancing.generate_trade_list()
        broker       : BrokerProfile or str
        usd_tickers  : list[str]  tickers denominated in USD (triggers FX fee)
                       defaults to all tickers if None

    Returns:
        pd.DataFrame  trade_list enriched with cost columns
    """
    if isinstance(broker, str):
        broker = BROKERS[broker]

    result = trade_list.copy()
    result["Commission (€)"] = 0.0
    result["Spread (€)"]     = 0.0
    result["FX Fee (€)"]     = 0.0
    result["Total Cost (€)"] = 0.0

    for idx in result.index:
        if idx == "── TOTAL ──":
            continue
        raw_action = str(result.loc[idx, "Action"])
        if not raw_action.startswith(("BUY", "SELL")):
            continue

        trade_val  = abs(float(result.loc[idx, "Trade Amount (€)"]))
        is_usd     = (usd_tickers is None) or (idx in usd_tickers)
        costs      = transaction_cost(trade_val, broker, is_usd_asset=is_usd)

        result.loc[idx, "Commission (€)"] = costs["commission"]
        result.loc[idx, "Spread (€)"]     = costs["spread"]
        result.loc[idx, "FX Fee (€)"]     = costs["fx_fee"]
        result.loc[idx, "Total Cost (€)"] = costs["total"]

    # Update totals row
    cost_cols = ["Commission (€)", "Spread (€)", "FX Fee (€)", "Total Cost (€)"]
    data_rows = result.index[result.index != "── TOTAL ──"]
    for col in cost_cols:
        result.loc["── TOTAL ──", col] = result.loc[data_rows, col].sum().round(4)

    return result


def broker_comparison(trade_value_eur: float,
                       is_usd_asset:   bool = True) -> pd.DataFrame:
    """
    Compare total one-way transaction cost across all built-in broker profiles
    for a given trade size.

    Args:
        trade_value_eur : float  trade size in EUR
        is_usd_asset    : bool

    Returns:
        pd.DataFrame  sorted by Total Cost ascending
    """
    rows = []
    for key, profile in BROKERS.items():
        costs = transaction_cost(trade_value_eur, profile, is_usd_asset)
        rows.append({
            "Broker":           profile.name,
            "Commission (€)":   costs["commission"],
            "Spread (€)":       costs["spread"],
            "FX Fee (€)":       costs["fx_fee"],
            "Total Cost (€)":   costs["total"],
            "Total Cost (%)":   costs["total_pct"],
            "Notes":            profile.notes,
        })
    return (pd.DataFrame(rows)
              .set_index("Broker")
              .sort_values("Total Cost (€)"))


# ══════════════════════════════════════════════════════════════════════════════
# 3. CAPITAL GAINS TAX — FIFO LOT TRACKING
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaxProfile:
    """Tax rules for a specific investor jurisdiction."""
    name:                    str
    capital_gains_rate:      float    # flat rate on realized gains (decimal)
    dividend_withholding:    float    # withholding on foreign dividends
    annual_exempt_amount:    float    # tax-free gains per year (EUR)
    loss_carry_forward:      bool     # can losses offset future gains?
    notes:                   str = ""


TAX_PROFILES: dict[str, TaxProfile] = {

    "greece": TaxProfile(
        name                 = "Greece",
        capital_gains_rate   = 0.15,    # 15% flat on stock/ETF gains
        dividend_withholding = 0.05,    # 5% domestic; US ETFs subject to treaty
        annual_exempt_amount = 0.0,     # no exemption
        loss_carry_forward   = False,   # losses cannot offset gains in GR
        notes                = "15% CGT. US dividends: 15% withholding with W-8BEN, 30% without.",
    ),

    "germany": TaxProfile(
        name                 = "Germany",
        capital_gains_rate   = 0.26375, # 25% + 5.5% solidarity + church tax avg
        dividend_withholding = 0.26375,
        annual_exempt_amount = 1000.0,  # €1,000 Sparerpauschbetrag (2023+)
        loss_carry_forward   = True,
        notes                = "Abgeltungssteuer. €1k annual exemption per person.",
    ),

    "uk": TaxProfile(
        name                 = "United Kingdom",
        capital_gains_rate   = 0.20,    # higher rate taxpayer; 10% basic rate
        dividend_withholding = 0.0,     # no additional UK withholding on ETF divs
        annual_exempt_amount = 3000.0,  # CGT annual exempt amount (2024/25)
        loss_carry_forward   = True,
        notes                = "Annual exempt amount reduced from £12,300 to £3,000 (2024).",
    ),

    "us_resident": TaxProfile(
        name                 = "US Resident",
        capital_gains_rate   = 0.15,    # long-term CGT (assume middle bracket)
        dividend_withholding = 0.0,
        annual_exempt_amount = 0.0,
        loss_carry_forward   = True,
        notes                = "Long-term rate 0/15/20% depending on income. Short-term = ordinary income.",
    ),

    "zero_tax": TaxProfile(
        name                 = "Zero Tax (UAE / comparison baseline)",
        capital_gains_rate   = 0.0,
        dividend_withholding = 0.0,
        annual_exempt_amount = 0.0,
        loss_carry_forward   = False,
        notes                = "Useful as a baseline to measure tax drag.",
    ),
}


def capital_gains_tax(sale_value:       float,
                       cost_basis:       float,
                       tax_profile:      TaxProfile | str = "greece",
                       ytd_gains:        float = 0.0) -> dict:
    """
    Compute capital gains tax on a single sale.

    Uses a simplified flat-rate model (no progressive brackets).
    Respects annual exempt amount if defined in the tax profile.

    Args:
        sale_value   : float   gross proceeds from the sale (EUR)
        cost_basis   : float   original purchase price (EUR)
        tax_profile  : TaxProfile or str key from TAX_PROFILES
        ytd_gains    : float   gains already realized this tax year (EUR)
                               used to check if exemption still available

    Returns:
        dict with keys:
            gross_gain      : float  sale_value - cost_basis
            taxable_gain    : float  after applying annual exemption
            tax_due         : float  EUR
            effective_rate  : float  tax / gross_gain
            net_proceeds    : float  sale_value - tax_due
    """
    if isinstance(tax_profile, str):
        if tax_profile not in TAX_PROFILES:
            raise ValueError(f"Unknown tax profile '{tax_profile}'. "
                             f"Choose from: {list(TAX_PROFILES)}")
        tax_profile = TAX_PROFILES[tax_profile]

    gross_gain  = sale_value - cost_basis

    if gross_gain <= 0:
        return {
            "gross_gain":     round(gross_gain, 2),
            "taxable_gain":   0.0,
            "tax_due":        0.0,
            "effective_rate": 0.0,
            "net_proceeds":   round(sale_value, 2),
        }

    # Apply annual exemption (only the portion not yet used this year)
    exemption_remaining = max(tax_profile.annual_exempt_amount - ytd_gains, 0.0)
    taxable_gain        = max(gross_gain - exemption_remaining, 0.0)
    tax_due             = taxable_gain * tax_profile.capital_gains_rate
    effective_rate      = (tax_due / gross_gain) if gross_gain > 0 else 0.0

    return {
        "gross_gain":     round(gross_gain, 2),
        "taxable_gain":   round(taxable_gain, 2),
        "tax_due":        round(tax_due, 2),
        "effective_rate": round(effective_rate, 4),
        "net_proceeds":   round(sale_value - tax_due, 2),
    }


def fifo_cost_basis(purchases: list[dict]) -> "FIFOLedger":
    """
    Create a FIFO lot ledger from a list of purchase transactions.

    Args:
        purchases : list of dicts, each with:
                      date      : str   "YYYY-MM-DD"
                      shares    : float
                      price_eur : float  price per share at purchase

    Returns:
        FIFOLedger object with a .sell() method
    """
    return FIFOLedger(purchases)


class FIFOLedger:
    """
    Tracks purchase lots using First-In-First-Out accounting.
    Used to determine cost basis when selling shares.
    """

    def __init__(self, purchases: list[dict]):
        self.lots = [
            {"date": p["date"],
             "shares": float(p["shares"]),
             "price_eur": float(p["price_eur"])}
            for p in purchases
        ]

    def sell(self, shares_to_sell: float,
              sale_price_eur:   float,
              tax_profile:      TaxProfile | str = "greece",
              ytd_gains:        float = 0.0) -> dict:
        """
        Sell a number of shares using FIFO lot matching.

        Args:
            shares_to_sell : float
            sale_price_eur : float  current price per share
            tax_profile    : TaxProfile or str
            ytd_gains      : float  gains already realized this year

        Returns:
            dict with keys:
                proceeds        : float
                total_cost_basis: float
                gross_gain      : float
                tax_due         : float
                net_proceeds    : float
                lots_consumed   : list  detail of which lots were used
        """
        remaining      = shares_to_sell
        total_basis    = 0.0
        lots_consumed  = []

        for lot in self.lots:
            if remaining <= 0:
                break
            used          = min(lot["shares"], remaining)
            basis         = used * lot["price_eur"]
            total_basis  += basis
            lots_consumed.append({
                "date":       lot["date"],
                "shares":     round(used, 6),
                "price_eur":  lot["price_eur"],
                "basis":      round(basis, 2),
            })
            lot["shares"] -= used
            remaining     -= used

        # Remove exhausted lots
        self.lots = [l for l in self.lots if l["shares"] > 1e-8]

        proceeds = shares_to_sell * sale_price_eur
        tax_info = capital_gains_tax(proceeds, total_basis, tax_profile, ytd_gains)

        return {
            "proceeds":         round(proceeds, 2),
            "total_cost_basis": round(total_basis, 2),
            "gross_gain":       tax_info["gross_gain"],
            "tax_due":          tax_info["tax_due"],
            "net_proceeds":     tax_info["net_proceeds"],
            "lots_consumed":    lots_consumed,
        }

    @property
    def remaining_lots(self) -> pd.DataFrame:
        """View current lot inventory."""
        return pd.DataFrame(self.lots)


# ══════════════════════════════════════════════════════════════════════════════
# 4. DIVIDEND WITHHOLDING TAX
# ══════════════════════════════════════════════════════════════════════════════

# Standard US withholding rates for EU residents
# With W-8BEN form filed: 15% treaty rate
# Without W-8BEN:         30% statutory rate
US_WITHHOLDING_WITH_W8BEN    = 0.15
US_WITHHOLDING_WITHOUT_W8BEN = 0.30

# Approximate dividend yields for common ETFs (updated ~2024)
ETF_DIVIDEND_YIELDS: dict[str, float] = {
    "VOO":  0.0132,   # S&P 500 — ~1.3%
    "VT":   0.0200,   # Total World — ~2.0%
    "VEA":  0.0310,   # Developed ex-US — ~3.1%
    "GLD":  0.0000,   # Gold — no dividend
    "SLV":  0.0000,   # Silver — no dividend
    "V":    0.0076,   # Visa — ~0.76%
    "AMZN": 0.0000,   # Amazon — no dividend
    "MSFT": 0.0075,   # Microsoft — ~0.75%
    "UNH":  0.0150,   # UnitedHealth — ~1.5%
    "BABA": 0.0000,   # Alibaba — no dividend
    "XLE":  0.0340,   # Energy Select — ~3.4%
}


def dividend_tax_analysis(weights_dict:    dict[str, float],
                            portfolio_value: float,
                            tax_profile:     TaxProfile | str = "greece",
                            w8ben_filed:     bool = True,
                            custom_yields:   dict[str, float] = None) -> pd.DataFrame:
    """
    Estimate annual dividend income and withholding tax per asset.

    Args:
        weights_dict    : dict   {ticker: weight}
        portfolio_value : float  total portfolio value in EUR
        tax_profile     : TaxProfile or str
        w8ben_filed     : bool   True if W-8BEN is on file (15% vs 30%)
        custom_yields   : dict   override default ETF_DIVIDEND_YIELDS

    Returns:
        pd.DataFrame  one row per asset + summary row
    """
    if isinstance(tax_profile, str):
        tax_profile = TAX_PROFILES[tax_profile]

    yields = {**ETF_DIVIDEND_YIELDS, **(custom_yields or {})}
    us_rate = US_WITHHOLDING_WITH_W8BEN if w8ben_filed else US_WITHHOLDING_WITHOUT_W8BEN

    rows = []
    for ticker, weight in weights_dict.items():
        asset_value   = portfolio_value * weight
        div_yield     = yields.get(ticker, 0.0)
        gross_div     = asset_value * div_yield
        withholding   = gross_div * us_rate          # US withholding at source
        # Additional domestic tax on net dividend (Greece: 5% on dividends)
        domestic_tax  = (gross_div - withholding) * tax_profile.dividend_withholding
        net_div       = gross_div - withholding - domestic_tax

        rows.append({
            "Ticker":             ticker,
            "Weight (%)":         round(weight * 100, 2),
            "Asset Value (€)":    round(asset_value, 2),
            "Div Yield (%)":      round(div_yield * 100, 2),
            "Gross Dividend (€)": round(gross_div, 2),
            "US Withholding (€)": round(withholding, 2),
            "Domestic Tax (€)":   round(domestic_tax, 2),
            "Net Dividend (€)":   round(net_div, 2),
        })

    df = pd.DataFrame(rows).set_index("Ticker")

    # Summary row
    summary_cols = ["Gross Dividend (€)", "US Withholding (€)",
                    "Domestic Tax (€)", "Net Dividend (€)"]
    summary = df[summary_cols].sum().to_frame().T
    summary.index = ["── TOTAL ──"]
    summary["Weight (%)"]      = 100.0
    summary["Asset Value (€)"] = round(portfolio_value, 2)
    summary["Div Yield (%)"]   = round(
        df["Gross Dividend (€)"].sum() / portfolio_value * 100, 2
    )

    return pd.concat([df, summary[df.columns]])


# ══════════════════════════════════════════════════════════════════════════════
# 5. TOTAL COST OF OWNERSHIP (TCO)
# ══════════════════════════════════════════════════════════════════════════════

ETF_EXPENSE_RATIOS: dict[str, float] = {
    "VOO":  0.0003,   # 0.03%
    "VT":   0.0007,   # 0.07%
    "VEA":  0.0005,   # 0.05%
    "GLD":  0.0040,   # 0.40%
    "SLV":  0.0050,   # 0.50%
    "V":    0.0,      # stock — no ER
    "AMZN": 0.0,
    "MSFT": 0.0,
    "UNH":  0.0,
    "BABA": 0.0,
    "XLE":  0.0009,   # 0.09%
}


def total_cost_of_ownership(weights_dict:      dict[str, float],
                              portfolio_value:   float,
                              broker:            BrokerProfile | str = "interactive_brokers",
                              tax_profile:       TaxProfile | str = "greece",
                              rebalances_per_yr: int   = 1,
                              avg_trade_pct:     float = 0.20,
                              w8ben_filed:       bool  = True,
                              custom_er:         dict[str, float] = None,
                              custom_yields:     dict[str, float] = None) -> dict:
    """
    Estimate total annual cost of ownership: expense ratios + transaction
    costs + dividend tax drag.

    Args:
        weights_dict      : dict   {ticker: weight}
        portfolio_value   : float  EUR
        broker            : BrokerProfile or str
        tax_profile       : TaxProfile or str
        rebalances_per_yr : int    expected rebalance events per year
        avg_trade_pct     : float  fraction of portfolio traded per rebalance
                                   (e.g. 0.20 = 20% of portfolio turns over)
        w8ben_filed       : bool
        custom_er         : dict   override ETF_EXPENSE_RATIOS
        custom_yields     : dict   override ETF_DIVIDEND_YIELDS

    Returns:
        dict with keys:
            expense_ratio_drag_eur   : float  annual cost of ETF TERs
            transaction_cost_eur     : float  annual trading costs
            dividend_tax_drag_eur    : float  annual dividend withholding
            total_annual_cost_eur    : float
            total_annual_cost_pct    : float  as % of AUM
            breakdown                : pd.DataFrame  per-asset ER drag
    """
    if isinstance(broker, str):
        broker = BROKERS[broker]
    if isinstance(tax_profile, str):
        tax_profile = TAX_PROFILES[tax_profile]

    er_rates = {**ETF_EXPENSE_RATIOS, **(custom_er or {})}

    # --- 1. Expense ratio drag ---
    er_rows = []
    total_er_drag = 0.0
    for ticker, weight in weights_dict.items():
        er     = er_rates.get(ticker, 0.0)
        drag   = portfolio_value * weight * er
        total_er_drag += drag
        er_rows.append({
            "Ticker":       ticker,
            "Weight (%)":   round(weight * 100, 2),
            "ER (%)":       round(er * 100, 3),
            "Annual Drag (€)": round(drag, 2),
        })
    er_breakdown = pd.DataFrame(er_rows).set_index("Ticker")

    # --- 2. Transaction costs (rebalancing) ---
    avg_trade_value = portfolio_value * avg_trade_pct
    single_trade_cost = transaction_cost(
        avg_trade_value, broker, is_usd_asset=True
    )["total"]
    # Each rebalance involves buys and sells (approx 2 trades per event)
    annual_trade_cost = single_trade_cost * 2 * rebalances_per_yr

    # --- 3. Dividend withholding drag ---
    div_df  = dividend_tax_analysis(
        weights_dict, portfolio_value, tax_profile, w8ben_filed, custom_yields
    )
    div_tax = float(div_df.loc["── TOTAL ──", "US Withholding (€)"])
    div_tax += float(div_df.loc["── TOTAL ──", "Domestic Tax (€)"])

    total_annual = total_er_drag + annual_trade_cost + div_tax

    return {
        "expense_ratio_drag_eur":  round(total_er_drag, 2),
        "transaction_cost_eur":    round(annual_trade_cost, 2),
        "dividend_tax_drag_eur":   round(div_tax, 2),
        "total_annual_cost_eur":   round(total_annual, 2),
        "total_annual_cost_pct":   round(total_annual / portfolio_value * 100, 3),
        "breakdown":               er_breakdown,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. AFTER-TAX RETURN
# ══════════════════════════════════════════════════════════════════════════════

def after_tax_return(gross_cagr:         float,
                      portfolio_value:    float,
                      holding_years:      float,
                      tax_profile:        TaxProfile | str = "greece",
                      tco_pct:            float = 0.002) -> dict:
    """
    Estimate after-tax, after-cost net return for a lump sum investment.

    Assumes the entire gain is realized at the end of the holding period
    (single liquidation event). Does not model intermediate dividend taxes
    (use dividend_tax_analysis() for that separately).

    Args:
        gross_cagr      : float   pre-tax CAGR as decimal (e.g. 0.10 for 10%)
        portfolio_value : float   initial investment (EUR)
        holding_years   : float   investment horizon
        tax_profile     : TaxProfile or str
        tco_pct         : float   total annual cost % (from total_cost_of_ownership)

    Returns:
        dict with keys:
            gross_final_value  : float
            net_cagr_pct       : float  after tax & costs
            after_tax_value    : float
            total_tax_paid     : float
            total_costs_paid   : float
            tax_drag_pct       : float  annual return reduction from tax
            cost_drag_pct      : float  annual return reduction from costs
    """
    if isinstance(tax_profile, str):
        tax_profile = TAX_PROFILES[tax_profile]

    # Gross growth (before costs and tax)
    gross_final    = portfolio_value * (1 + gross_cagr) ** holding_years

    # After annual costs (compound drag)
    net_cagr_costs = gross_cagr - tco_pct
    value_after_costs = portfolio_value * (1 + net_cagr_costs) ** holding_years

    # Capital gains tax on realized gain at end
    tax_info    = capital_gains_tax(
        value_after_costs, portfolio_value, tax_profile
    )
    after_tax   = tax_info["net_proceeds"]
    tax_paid    = tax_info["tax_due"]
    costs_paid  = gross_final - value_after_costs

    net_cagr    = (after_tax / portfolio_value) ** (1 / holding_years) - 1 \
                  if holding_years > 0 else 0.0

    return {
        "gross_final_value":  round(gross_final, 2),
        "after_tax_value":    round(after_tax, 2),
        "net_cagr_pct":       round(net_cagr * 100, 3),
        "total_tax_paid":     round(tax_paid, 2),
        "total_costs_paid":   round(costs_paid, 2),
        "tax_drag_pct":       round((gross_cagr - net_cagr) * 100 * 0.4, 3),
        "cost_drag_pct":      round(tco_pct * 100, 3),
    }
