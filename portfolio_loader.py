"""
portfolio_loader.py — Real Portfolio Loader
=============================================
Loads actual client transaction history from broker exports or manual
entry, converts all values to a single base currency using historical
FX rates from Yahoo Finance, and produces a clean holdings DataFrame
ready for analysis by all other modules.

Supported input formats:
  - DEGIRO   : CSV export from DEGIRO platform
  - IBKR     : CSV export from Interactive Brokers Activity Statement
  - Manual   : Standardised CSV/Excel template (see MANUAL_TEMPLATE below)

Two operation modes in the notebook config cell:
  MODE = "hypothetical"  →  use WEIGHTS + INITIAL_CAPITAL (existing flow)
  MODE = "real"          →  use this module to load actual transactions

Config cell usage:
    from portfolio_loader import (load_transactions, build_real_portfolio,
                                  AVAILABLE_CURRENCIES, MANUAL_TEMPLATE)

    # Required config
    MODE              = "real"
    TRANSACTIONS_FILE = "clients/smith_degiro.csv"
    BROKER_FORMAT     = "degiro"       # "degiro" | "ibkr" | "manual"
    BASE_CURRENCY     = "EUR"          # change freely per client

    # Load
    transactions = load_transactions(TRANSACTIONS_FILE, BROKER_FORMAT)
    holdings     = build_real_portfolio(transactions, BASE_CURRENCY)

    # Drop-in replacements for hypothetical-mode variables
    weights           = holdings["weights"]
    portfolio_values  = holdings["portfolio_values"]
    portfolio_returns = holdings["portfolio_returns"]
    total_cost_basis  = holdings["total_cost_basis"]

FX data is fetched once from Yahoo Finance and cached in memory.
All monetary values in the output are in BASE_CURRENCY.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path


# ── Supported currencies ──────────────────────────────────────────────────────

AVAILABLE_CURRENCIES: dict[str, str] = {
    "EUR": "Euro (default)",
    "USD": "US Dollar",
    "GBP": "British Pound",
    "CHF": "Swiss Franc",
    "SEK": "Swedish Krona",
    "NOK": "Norwegian Krone",
    "DKK": "Danish Krone",
    "PLN": "Polish Zloty",
    "CZK": "Czech Koruna",
    "HUF": "Hungarian Forint",
}

# Yahoo Finance FX ticker pattern: "USDEUR=X" means USD → EUR
_FX_CACHE: dict[str, pd.Series] = {}


# ── Manual entry template ─────────────────────────────────────────────────────

MANUAL_TEMPLATE: str = """
MANUAL TRANSACTION TEMPLATE
============================
Save as CSV with the following columns (header row required):

  date          : YYYY-MM-DD          e.g. 2023-06-15
  ticker        : Yahoo Finance symbol e.g. VOO, IWDA, AAPL
  action        : BUY or SELL
  quantity      : number of shares    e.g. 10
  price         : price per share     e.g. 449.80
  currency      : 3-letter ISO code   e.g. USD, EUR, GBP
  fee           : transaction fee     e.g. 3.00  (use 0 if none)
  broker        : optional label      e.g. DEGIRO

Example rows:
  date,ticker,action,quantity,price,currency,fee,broker
  2022-03-01,VOO,BUY,5,410.20,USD,3.00,Manual
  2022-06-15,GLD,BUY,3,172.50,USD,3.00,Manual
  2023-01-10,VOO,BUY,2,388.45,USD,2.00,Manual
  2023-09-20,VT,BUY,10,98.30,USD,3.00,Manual
"""


# ══════════════════════════════════════════════════════════════════════════════
# 1. FX CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def _fx_ticker(from_ccy: str, to_ccy: str) -> str:
    """Yahoo Finance FX ticker e.g. USD→EUR = 'USDEUR=X'."""
    return f"{from_ccy}{to_ccy}=X"


def get_fx_rate(from_ccy:   str,
                to_ccy:     str,
                date:       str | pd.Timestamp,
                cache:      bool = True) -> float:
    """
    Get the historical FX rate for a specific date.

    Uses Yahoo Finance data. Falls back to the nearest available
    trading day if the exact date has no data (weekends, holidays).

    Args:
        from_ccy : str   source currency (e.g. "USD")
        to_ccy   : str   target currency (e.g. "EUR")
        date     : str or Timestamp
        cache    : bool  cache FX series in memory to avoid re-downloading

    Returns:
        float  exchange rate (multiply from_ccy amount by this to get to_ccy)
    """
    if from_ccy == to_ccy:
        return 1.0

    key = _fx_ticker(from_ccy, to_ccy)

    if cache and key in _FX_CACHE:
        series = _FX_CACHE[key]
    else:
        print(f"  Fetching FX: {from_ccy} → {to_ccy}...")
        raw    = yf.download(key, period="max", progress=False, auto_adjust=True)
        if raw.empty:
            print(f"  Warning: no FX data for {key}, using rate 1.0")
            return 1.0
        col    = "Close" if "Close" in raw.columns else raw.columns[0]
        # Flatten MultiIndex if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        series = raw["Close"].dropna()
        if cache:
            _FX_CACHE[key] = series

    # Find nearest available rate to requested date
    target = pd.Timestamp(date)
    try:
        # Look backwards up to 5 business days
        window = series[:target].tail(5)
        if len(window) == 0:
            window = series.head(1)
        rate = float(window.iloc[-1])
    except Exception:
        rate = 1.0

    return rate


def fx_convert(amount:    float,
               from_ccy:  str,
               to_ccy:    str,
               date:      str | pd.Timestamp) -> float:
    """
    Convert a monetary amount from one currency to another
    using the historical rate on a given date.

    Args:
        amount   : float  value in from_ccy
        from_ccy : str
        to_ccy   : str
        date     : str or Timestamp

    Returns:
        float  value in to_ccy
    """
    if from_ccy == to_ccy:
        return amount
    return amount * get_fx_rate(from_ccy, to_ccy, date)


def get_fx_series(from_ccy: str,
                   to_ccy:   str,
                   start:    str,
                   end:      str = None) -> pd.Series:
    """
    Get a full daily FX rate series between two dates.
    Used to convert portfolio value time-series.

    Args:
        from_ccy : str
        to_ccy   : str
        start    : str  "YYYY-MM-DD"
        end      : str  "YYYY-MM-DD" (defaults to today)

    Returns:
        pd.Series  daily rates, forward-filled for non-trading days
    """
    if from_ccy == to_ccy:
        return None

    key = _fx_ticker(from_ccy, to_ccy)
    raw = yf.download(key, start=start, end=end,
                       progress=False, auto_adjust=True)
    if raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    series = raw["Close"].dropna().ffill()
    return series


# ══════════════════════════════════════════════════════════════════════════════
# 2. BROKER PARSERS
# ══════════════════════════════════════════════════════════════════════════════

# Standard internal column names used by all parsers
_STANDARD_COLS = [
    "date",       # pd.Timestamp
    "ticker",     # str  Yahoo Finance symbol
    "action",     # "BUY" | "SELL"
    "quantity",   # float  (positive for both BUY and SELL)
    "price",      # float  per share in original currency
    "currency",   # str  ISO 3-letter
    "fee",        # float  in original currency
    "broker",     # str  source label
]


def _parse_degiro(filepath: str) -> pd.DataFrame:
    """
    Parse a DEGIRO transaction export CSV.

    DEGIRO CSV columns (may vary slightly by region):
      Datum, Product, ISIN, Omschrijving/Description, Aantal/Quantity,
      Koers/Price, Waarde/Value, Transactiekosten/Costs, Totaal/Total,
      Valuta/Currency

    Handles both Dutch and English column headers.
    """
    raw = pd.read_csv(filepath, sep=",", encoding="utf-8-sig")
    raw.columns = raw.columns.str.strip()

    # Map Dutch/English column names to standard
    col_map = {
        # Dutch
        "Datum":              "raw_date",
        "Product":            "raw_product",
        "Aantal":             "raw_quantity",
        "Koers":              "raw_price",
        "Valuta":             "raw_currency",
        "Transactiekosten":   "raw_fee",
        "Omschrijving":       "raw_description",
        # English
        "Date":               "raw_date",
        "Description":        "raw_description",
        "Quantity":           "raw_quantity",
        "Price":              "raw_price",
        "Currency":           "raw_currency",
        "Transaction costs":  "raw_fee",
    }
    raw = raw.rename(columns={k: v for k, v in col_map.items()
                               if k in raw.columns})

    rows = []
    for _, r in raw.iterrows():
        # Skip non-trade rows (deposits, dividends, fees without product)
        if "raw_description" in raw.columns:
            desc = str(r.get("raw_description", "")).upper()
            if not any(w in desc for w in ["KOOP", "BUY", "VERKOOP", "SELL"]):
                continue
            action = "BUY" if any(w in desc for w in ["KOOP", "BUY"]) else "SELL"
        else:
            continue

        try:
            qty      = abs(float(str(r.get("raw_quantity", 0)).replace(",", ".")))
            price    = abs(float(str(r.get("raw_price",    0)).replace(",", ".")))
            fee      = abs(float(str(r.get("raw_fee",      0)).replace(",", ".")))
            currency = str(r.get("raw_currency", "EUR")).strip().upper()
            product  = str(r.get("raw_product",  "")).strip()

            # DEGIRO doesn't store Yahoo tickers — user must map ISIN→ticker
            # We store the product name; reconciliation happens in build_real_portfolio
            rows.append({
                "date":     pd.Timestamp(str(r.get("raw_date", ""))),
                "ticker":   product,          # will need manual mapping if not Yahoo-compatible
                "action":   action,
                "quantity": qty,
                "price":    price,
                "currency": currency,
                "fee":      fee,
                "broker":   "DEGIRO",
            })
        except (ValueError, TypeError):
            continue

    return pd.DataFrame(rows, columns=_STANDARD_COLS)


def _parse_ibkr(filepath: str) -> pd.DataFrame:
    """
    Parse an Interactive Brokers Activity Statement CSV.

    IBKR exports contain multiple sections; we extract the Trades section.
    Key columns: TradeDate, Symbol, Quantity, TradePrice, TradeMoney,
                 Commission, CurrencyPrimary, Buy/Sell
    """
    # IBKR CSV has section headers — find the Trades section
    with open(filepath, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    # Find lines belonging to the Trades section
    trade_lines = []
    in_trades   = False
    header_line = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Trades,Header"):
            header_line = stripped
            in_trades   = True
            continue
        if in_trades:
            if stripped.startswith("Trades,Data"):
                trade_lines.append(stripped)
            elif stripped.startswith("Trades,") and "Subtotal" in stripped:
                continue
            elif not stripped.startswith("Trades,"):
                in_trades = False

    if not header_line or not trade_lines:
        # Fallback: try reading as a simple CSV (some IBKR exports)
        try:
            raw = pd.read_csv(filepath, encoding="utf-8-sig")
            raw.columns = raw.columns.str.strip()
        except Exception:
            return pd.DataFrame(columns=_STANDARD_COLS)
    else:
        # Parse header + data lines
        header  = header_line.split(",")[2:]   # skip "Trades,Header"
        parsed  = [l.split(",")[2:] for l in trade_lines]
        raw     = pd.DataFrame(parsed, columns=header)

    col_map = {
        "TradeDate":        "raw_date",
        "Symbol":           "raw_ticker",
        "Quantity":         "raw_quantity",
        "TradePrice":       "raw_price",
        "Commission":       "raw_fee",
        "CurrencyPrimary":  "raw_currency",
        "Buy/Sell":         "raw_action",
        "Asset Category":   "raw_asset",
    }
    raw = raw.rename(columns={k: v for k, v in col_map.items()
                               if k in raw.columns})

    rows = []
    for _, r in raw.iterrows():
        try:
            action_raw = str(r.get("raw_action", "")).strip().upper()
            if action_raw not in ("BUY", "SELL"):
                continue

            qty      = abs(float(str(r.get("raw_quantity", 0)).replace(",", "")))
            price    = abs(float(str(r.get("raw_price",    0)).replace(",", "")))
            fee      = abs(float(str(r.get("raw_fee",      0)).replace(",", "")))
            currency = str(r.get("raw_currency", "USD")).strip().upper()
            ticker   = str(r.get("raw_ticker",   "")).strip()

            rows.append({
                "date":     pd.Timestamp(str(r.get("raw_date", ""))),
                "ticker":   ticker,
                "action":   action_raw,
                "quantity": qty,
                "price":    price,
                "currency": currency,
                "fee":      fee,
                "broker":   "IBKR",
            })
        except (ValueError, TypeError):
            continue

    return pd.DataFrame(rows, columns=_STANDARD_COLS)


def _parse_manual(filepath: str) -> pd.DataFrame:
    """
    Parse the standardised manual transaction CSV/Excel template.

    Expected columns (see MANUAL_TEMPLATE for details):
      date, ticker, action, quantity, price, currency, fee, broker
    """
    path = Path(filepath)
    if path.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(filepath)
    else:
        raw = pd.read_csv(filepath, encoding="utf-8-sig")

    raw.columns = raw.columns.str.strip().str.lower()

    # Validate required columns
    required = {"date", "ticker", "action", "quantity", "price", "currency"}
    missing  = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"Manual file missing required columns: {missing}. "
            f"See MANUAL_TEMPLATE for the expected format."
        )

    rows = []
    for _, r in raw.iterrows():
        try:
            action = str(r["action"]).strip().upper()
            if action not in ("BUY", "SELL"):
                continue

            rows.append({
                "date":     pd.Timestamp(str(r["date"])),
                "ticker":   str(r["ticker"]).strip().upper(),
                "action":   action,
                "quantity": abs(float(r["quantity"])),
                "price":    abs(float(r["price"])),
                "currency": str(r["currency"]).strip().upper(),
                "fee":      abs(float(r.get("fee", 0) or 0)),
                "broker":   str(r.get("broker", "Manual")).strip(),
            })
        except (ValueError, TypeError):
            continue

    return pd.DataFrame(rows, columns=_STANDARD_COLS)


# ══════════════════════════════════════════════════════════════════════════════
# 3. PUBLIC LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_transactions(filepath:      str,
                       broker_format: str = "manual") -> pd.DataFrame:
    """
    Load and parse a broker transaction file into the standard format.

    Args:
        filepath      : str  path to the CSV/Excel file
        broker_format : str  "degiro" | "ibkr" | "manual"

    Returns:
        pd.DataFrame  standard columns: date, ticker, action, quantity,
                      price, currency, fee, broker
                      Sorted ascending by date.

    Raises:
        ValueError  if broker_format is not recognised
        FileNotFoundError  if filepath does not exist
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Transaction file not found: '{filepath}'")

    fmt = broker_format.strip().lower()

    if fmt == "degiro":
        df = _parse_degiro(filepath)
    elif fmt in ("ibkr", "interactive_brokers"):
        df = _parse_ibkr(filepath)
    elif fmt == "manual":
        df = _parse_manual(filepath)
    else:
        raise ValueError(
            f"Unknown broker format '{broker_format}'. "
            f"Choose from: 'degiro', 'ibkr', 'manual'"
        )

    df = df.dropna(subset=["date", "ticker", "quantity", "price"])
    df = df[df["quantity"] > 0]
    df = df.sort_values("date").reset_index(drop=True)

    print(f"  Loaded {len(df)} transactions "
          f"({fmt.upper()}) from {df['date'].min().date()} "
          f"to {df['date'].max().date()}")
    print(f"  Tickers: {sorted(df['ticker'].unique().tolist())}")

    return df


def merge_transactions(*dataframes: pd.DataFrame) -> pd.DataFrame:
    """
    Merge transaction DataFrames from multiple brokers or files
    into a single sorted DataFrame.

    Usage:
        degiro_tx = load_transactions("degiro.csv",  "degiro")
        ibkr_tx   = load_transactions("ibkr.csv",    "ibkr")
        manual_tx = load_transactions("manual.csv",  "manual")
        all_tx    = merge_transactions(degiro_tx, ibkr_tx, manual_tx)

    Args:
        *dataframes : pd.DataFrame  any number of standard-format DataFrames

    Returns:
        pd.DataFrame  merged and sorted by date
    """
    merged = pd.concat(dataframes, ignore_index=True)
    merged = merged.sort_values("date").reset_index(drop=True)
    print(f"  Merged {len(merged)} total transactions "
          f"across {merged['broker'].nunique()} broker(s).")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# 4. REAL PORTFOLIO BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_real_portfolio(transactions:   pd.DataFrame,
                          base_currency:  str = "EUR",
                          end_date:       str = None) -> dict:
    """
    Convert a transaction log into actual portfolio holdings,
    converting all values to base_currency using historical FX rates.

    Steps:
      1. Convert each trade value to base_currency on trade date
      2. Track shares held per ticker (FIFO not required — just quantity)
      3. Download current prices from Yahoo Finance
      4. Compute current market value, weights, P&L, and cost basis

    Args:
        transactions  : pd.DataFrame  output of load_transactions()
        base_currency : str           target currency for all outputs (default "EUR")
                                      Change freely in config cell — FX is re-fetched.
        end_date      : str           "YYYY-MM-DD" — compute holdings as of this date
                                      (defaults to today / latest available price)

    Returns:
        dict with keys:
            holdings          : pd.DataFrame  one row per ticker
            weights           : dict          {ticker: weight}  (sums to 1.0)
            total_value       : float         total portfolio value in base_currency
            total_cost_basis  : float         total invested (net of sells) in base_currency
            total_pnl         : float         unrealised P&L in base_currency
            total_pnl_pct     : float         P&L as % of cost basis
            portfolio_values  : pd.Series     daily portfolio value (for charts)
            portfolio_returns : pd.Series     daily returns
            base_currency     : str
            fx_note           : str           which FX pairs were used
    """
    base_currency = base_currency.strip().upper()
    tickers       = sorted(transactions["ticker"].unique().tolist())

    # ── Step 1: Compute shares held per ticker ────────────────────────────────
    shares_held  = {t: 0.0 for t in tickers}
    cost_basis   = {t: 0.0 for t in tickers}   # in base_currency
    fx_pairs_used = set()

    for _, tx in transactions.iterrows():
        ticker   = tx["ticker"]
        qty      = float(tx["quantity"])
        price    = float(tx["price"])
        fee      = float(tx["fee"])
        ccy      = str(tx["currency"]).strip().upper()
        date     = tx["date"]
        action   = tx["action"]

        # Convert trade value to base_currency
        gross_native = qty * price
        fee_native   = fee
        if ccy != base_currency:
            fx_pairs_used.add(f"{ccy}→{base_currency}")
            rate         = get_fx_rate(ccy, base_currency, date)
            gross_base   = gross_native * rate
            fee_base     = fee_native   * rate
        else:
            gross_base   = gross_native
            fee_base     = fee_native

        if action == "BUY":
            shares_held[ticker] += qty
            cost_basis[ticker]  += gross_base + fee_base
        elif action == "SELL":
            # Reduce cost basis proportionally
            if shares_held[ticker] > 0:
                avg_cost         = cost_basis[ticker] / shares_held[ticker]
                cost_basis[ticker] -= avg_cost * qty
            shares_held[ticker]  = max(shares_held[ticker] - qty, 0.0)

    # ── Step 2: Get current prices from Yahoo Finance ─────────────────────────
    active_tickers = [t for t, s in shares_held.items() if s > 0.001]
    if not active_tickers:
        raise ValueError("No active holdings found after processing transactions.")

    print(f"  Fetching current prices for: {active_tickers}")
    price_data = {}
    for ticker in active_tickers:
        try:
            info   = yf.Ticker(ticker).fast_info
            last_p = float(info.get("last_price", 0) or
                           info.get("regularMarketPrice", 0) or 0)
            ccy    = str(info.get("currency", "USD")).upper()
            if last_p == 0:
                # Fallback: download last 5 days
                hist   = yf.download(ticker, period="5d",
                                      progress=False, auto_adjust=True)
                if not hist.empty:
                    last_p = float(hist["Close"].iloc[-1])
            price_data[ticker] = {"price_native": last_p, "currency": ccy}
        except Exception as e:
            print(f"  Warning: could not fetch price for {ticker}: {e}")
            price_data[ticker] = {"price_native": 0.0, "currency": "USD"}

    # ── Step 3: Build holdings table ─────────────────────────────────────────
    rows = []
    total_value = 0.0

    for ticker in active_tickers:
        shares  = shares_held[ticker]
        p_info  = price_data[ticker]
        p_nat   = p_info["price_native"]
        p_ccy   = p_info["currency"]

        if p_ccy != base_currency:
            fx_pairs_used.add(f"{p_ccy}→{base_currency}")
            today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
            rate      = get_fx_rate(p_ccy, base_currency, today_str)
            p_base    = p_nat * rate
        else:
            p_base = p_nat

        mkt_value   = shares * p_base
        cb          = cost_basis.get(ticker, 0.0)
        pnl         = mkt_value - cb
        pnl_pct     = pnl / cb * 100 if cb > 0 else 0.0

        total_value += mkt_value
        rows.append({
            "Ticker":              ticker,
            "Shares":              round(shares, 6),
            f"Price ({base_currency})": round(p_base, 4),
            f"Market Value ({base_currency})": round(mkt_value, 2),
            f"Cost Basis ({base_currency})":   round(cb, 2),
            f"Unrealised P&L ({base_currency})": round(pnl, 2),
            "P&L (%)":             round(pnl_pct, 2),
        })

    holdings_df = pd.DataFrame(rows).set_index("Ticker")

    # ── Step 4: Weights ───────────────────────────────────────────────────────
    weights = {
        r["Ticker"]: round(
            r[f"Market Value ({base_currency})"] / total_value, 6
        )
        for r in rows
    } if total_value > 0 else {}

    # ── Step 5: Historical portfolio value series ─────────────────────────────
    # Download price history for all active tickers
    start_date = transactions["date"].min().strftime("%Y-%m-%d")
    end_dt     = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")

    print(f"  Building historical value series ({start_date} → {end_dt})...")
    try:
        hist_prices = yf.download(
            active_tickers, start=start_date, end=end_dt,
            progress=False, auto_adjust=True
        )["Close"]

        if isinstance(hist_prices, pd.Series):
            hist_prices = hist_prices.to_frame(name=active_tickers[0])

        # Convert each ticker's price series to base_currency
        for ticker in active_tickers:
            if ticker not in hist_prices.columns:
                continue
            p_ccy = price_data[ticker]["currency"]
            if p_ccy != base_currency:
                fx_series = get_fx_series(p_ccy, base_currency,
                                           start_date, end_dt)
                if fx_series is not None:
                    fx_aligned = fx_series.reindex(
                        hist_prices.index, method="ffill"
                    )
                    hist_prices[ticker] = hist_prices[ticker] * fx_aligned

        # Weight each ticker by its current share count
        port_values = pd.Series(0.0, index=hist_prices.index)
        for ticker in active_tickers:
            if ticker in hist_prices.columns:
                port_values += hist_prices[ticker] * shares_held[ticker]

        port_values  = port_values.dropna()
        port_returns = port_values.pct_change().dropna()

    except Exception as e:
        print(f"  Warning: could not build historical series: {e}")
        port_values  = pd.Series(dtype=float)
        port_returns = pd.Series(dtype=float)

    total_cb  = sum(cost_basis[t] for t in active_tickers)
    total_pnl = total_value - total_cb

    fx_note = (f"FX conversions applied: {', '.join(sorted(fx_pairs_used))}"
               if fx_pairs_used else "No FX conversion needed.")

    print(f"\n  ── Portfolio Summary ({base_currency}) ──")
    print(f"  Holdings:    {len(active_tickers)} assets")
    print(f"  Total value: {base_currency} {total_value:,.2f}")
    print(f"  Cost basis:  {base_currency} {total_cb:,.2f}")
    print(f"  P&L:         {base_currency} {total_pnl:+,.2f} "
          f"({total_pnl / total_cb * 100:+.1f}%)" if total_cb > 0 else "")
    print(f"  {fx_note}")

    return {
        "holdings":          holdings_df,
        "weights":           weights,
        "total_value":       round(total_value, 2),
        "total_cost_basis":  round(total_cb, 2),
        "total_pnl":         round(total_pnl, 2),
        "total_pnl_pct":     round(total_pnl / total_cb * 100, 2) if total_cb > 0 else 0.0,
        "portfolio_values":  port_values,
        "portfolio_returns": port_returns,
        "base_currency":     base_currency,
        "fx_note":           fx_note,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRANSACTION SUMMARY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def transaction_summary(transactions: pd.DataFrame,
                          base_currency: str = "EUR") -> pd.DataFrame:
    """
    Summary table of total invested, number of trades,
    and average purchase price per ticker.

    Args:
        transactions  : pd.DataFrame  from load_transactions()
        base_currency : str

    Returns:
        pd.DataFrame  one row per ticker
    """
    rows = []
    for ticker, grp in transactions.groupby("ticker"):
        buys  = grp[grp["action"] == "BUY"]
        sells = grp[grp["action"] == "SELL"]

        total_bought = float(buys["quantity"].sum())
        total_sold   = float(sells["quantity"].sum())
        net_shares   = total_bought - total_sold

        avg_buy_price = float(
            (buys["price"] * buys["quantity"]).sum() / total_bought
        ) if total_bought > 0 else 0.0

        rows.append({
            "Ticker":          ticker,
            "Buy Trades":      len(buys),
            "Sell Trades":     len(sells),
            "Shares Bought":   round(total_bought, 4),
            "Shares Sold":     round(total_sold, 4),
            "Net Shares":      round(net_shares, 4),
            "Avg Buy Price":   round(avg_buy_price, 4),
            "Currency":        grp["currency"].mode()[0] if len(grp) > 0 else "—",
            "First Trade":     grp["date"].min().date(),
            "Last Trade":      grp["date"].max().date(),
        })

    return pd.DataFrame(rows).set_index("Ticker")


def validate_transactions(transactions: pd.DataFrame) -> list[str]:
    """
    Run basic sanity checks on a transaction DataFrame.
    Returns a list of warning strings (empty list = all clear).

    Checks:
      - No negative quantities or prices
      - No sells that exceed cumulative buys
      - No unrecognised currencies
      - No future dates
    """
    warnings = []
    today    = pd.Timestamp.today()

    # Future dates
    future = transactions[transactions["date"] > today]
    if len(future) > 0:
        warnings.append(
            f"{len(future)} transaction(s) have future dates: "
            f"{future['date'].dt.date.tolist()}"
        )

    # Negative values
    if (transactions["quantity"] <= 0).any():
        warnings.append("Some rows have quantity <= 0.")
    if (transactions["price"] <= 0).any():
        warnings.append("Some rows have price <= 0.")

    # Sells exceeding buys
    for ticker, grp in transactions.groupby("ticker"):
        cum_shares = 0.0
        for _, tx in grp.sort_values("date").iterrows():
            if tx["action"] == "BUY":
                cum_shares += tx["quantity"]
            elif tx["action"] == "SELL":
                cum_shares -= tx["quantity"]
                if cum_shares < -0.001:
                    warnings.append(
                        f"{ticker}: SELL quantity exceeds cumulative holdings "
                        f"on {tx['date'].date()}."
                    )

    # Unknown currencies
    known = set(AVAILABLE_CURRENCIES.keys()) | {"USD", "GBP", "JPY", "CAD",
                                                  "AUD", "CNY", "HKD", "SGD"}
    unknown_ccy = set(transactions["currency"].str.upper()) - known
    if unknown_ccy:
        warnings.append(f"Unrecognised currencies: {unknown_ccy}")

    if not warnings:
        print("  Validation passed — no issues found.")
    else:
        for w in warnings:
            print(f"  Warning: {w}")

    return warnings
