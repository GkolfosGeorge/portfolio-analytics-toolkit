"""
cleaner.py — Data Acquisition & Cleaning
==========================================
Downloads adjusted close prices from Yahoo Finance and returns
a clean, aligned DataFrame ready for portfolio analysis.
"""

import pandas as pd
import yfinance as yf


# Maximum fraction of missing values allowed per asset before warning.
_MISSING_THRESHOLD = 0.02   # 2%


def get_clean_data(ticker: str,
                   start_date: str = None,
                   end_date:   str = None,
                   period:     str = "max") -> pd.Series | None:
    """
    Download and clean adjusted close prices for a single ticker.

    Args:
        ticker     : str  Yahoo Finance ticker symbol (e.g. "VOO")
        start_date : str  "YYYY-MM-DD"  (optional)
        end_date   : str  "YYYY-MM-DD"  (optional)
        period     : str  yfinance period string, used only if dates are None

    Returns:
        pd.Series with DatetimeIndex, or None on failure.
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date,
                         period=period if start_date is None else None,
                         progress=False, auto_adjust=True)

        if df.empty:
            print(f"  Warning: No data returned for {ticker}.")
            return None

        # Flatten MultiIndex columns produced by some yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Prefer "Close" when auto_adjust=True (already adjusted);
        # fall back to "Adj Close" for older yfinance behaviour.
        col = "Close" if "Close" in df.columns else "Adj Close"
        series = df[col].dropna()

        return series

    except Exception as exc:
        print(f"  Error downloading {ticker}: {exc}")
        return None


def build_price_dataframe(tickers: list[str],
                           start_date: str = None,
                           end_date:   str = None,
                           period:     str = "max") -> pd.DataFrame:
    """
    Download and align adjusted close prices for a list of tickers.

    Steps:
      1. Download each ticker individually.
      2. Concatenate into a single DataFrame (outer join on dates).
      3. Warn about assets with high missing-data fractions.
      4. Forward-fill gaps of up to 5 consecutive days (e.g. holidays
         where markets trade but yfinance has no record).
      5. Drop any rows still containing NaN after forward-fill.

    Args:
        tickers    : list[str]  list of Yahoo Finance ticker symbols
        start_date : str        "YYYY-MM-DD"  (optional)
        end_date   : str        "YYYY-MM-DD"  (optional)
        period     : str        yfinance period string (default "max")

    Returns:
        pd.DataFrame  shape (trading_days, n_assets), sorted ascending by date

    Raises:
        ValueError  if no data could be downloaded for any ticker
    """
    all_series = []

    for ticker in tickers:
        data = get_clean_data(ticker, start_date, end_date, period)
        if data is None:
            print(f"  Warning: {ticker} skipped — no data.")
            continue
        data.name = ticker
        all_series.append(data)

    if not all_series:
        raise ValueError(
            "No data downloaded. Check your ticker symbols and date range."
        )

    df = pd.concat(all_series, axis=1).sort_index()

    # FIX: warn about assets with significant missing data BEFORE filling,
    # so the caller is aware that some prices are synthetic.
    pct_missing = df.isnull().mean()
    noisy = pct_missing[pct_missing > _MISSING_THRESHOLD]
    if not noisy.empty:
        print("\n  Warning: high missing-data fraction detected:")
        for ticker, frac in noisy.items():
            print(f"    {ticker}: {frac:.1%} missing — consider removing.")
        print()

    # Forward-fill short gaps (limit=5 keeps us from propagating stale
    # prices across long weekends or data outages indefinitely).
    df = df.ffill(limit=5).dropna()

    n_loaded = len(df.columns)
    n_req    = len(tickers)
    print(f"  Loaded {n_loaded}/{n_req} assets | "
          f"{len(df)} trading days | "
          f"{df.index[0].date()} -> {df.index[-1].date()}")

    return df


def get_dividend_data(tickers: list[str],
                       years_back: int = 5) -> pd.DataFrame:
    """
    Download live dividend data from Yahoo Finance for a list of tickers.

    Automatically detects Accumulating vs Distributing ETFs:
      - Distributing (Dist): has paid dividends in the past → live yield data
      - Accumulating (Acc):  no dividend history → internally reinvests

    This replaces hardcoded yield dictionaries. Always reflects current data.

    Args:
        tickers    : list[str]  Yahoo Finance ticker symbols
        years_back : int        how many years of dividend history to fetch
                                (used to compute trailing growth rate)

    Returns:
        pd.DataFrame  index=tickers, columns:
            type                : str    "dist" | "acc" | "no_dividend" | "stock"
            trailing_yield_pct  : float  trailing 12-month yield (%)
            annual_dividend     : float  trailing annual dividend per share
            dividend_growth_5yr : float  5-year dividend CAGR (%)
            ex_dividend_date    : str    next ex-dividend date or ""
            payment_months      : list   months that typically pay (e.g. [3,6,9,12])
            currency            : str    dividend currency
            note                : str    short human-readable explanation
    """
    import datetime
    import numpy as np

    cutoff = datetime.date.today() - datetime.timedelta(days=years_back * 365)
    rows   = {}

    for ticker in tickers:
        print(f"  Fetching dividend data: {ticker}...", end=" ")
        try:
            t    = yf.Ticker(ticker)
            info = t.info or {}
            divs = t.dividends  # pd.Series, index=DatetimeIndex, values=amount

            # ── Classify ─────────────────────────────────────────────────────
            trailing_yield = info.get("trailingAnnualDividendYield") or 0.0
            annual_div     = info.get("trailingAnnualDividendRate")  or 0.0
            ex_div_ts      = info.get("exDividendDate")              or None
            currency       = info.get("currency", "USD")

            # Ex-dividend date — convert UNIX timestamp if needed
            if ex_div_ts and isinstance(ex_div_ts, (int, float)):
                ex_div_str = datetime.date.fromtimestamp(ex_div_ts).isoformat()
            elif ex_div_ts:
                ex_div_str = str(ex_div_ts)[:10]
            else:
                ex_div_str = ""

            has_divs = (divs is not None and len(divs) > 0)

            if not has_divs and annual_div == 0:
                # Accumulating ETF or non-dividend stock — check quote type
                quote_type = info.get("quoteType", "").upper()
                if quote_type == "ETF":
                    asset_type = "acc"
                    note = ("Accumulating ETF — dividends reinvested internally. "
                            "No cash income paid to investor.")
                else:
                    asset_type = "no_dividend"
                    note = "Stock pays no dividend. Total return is capital gain only."

                rows[ticker] = {
                    "type":               asset_type,
                    "trailing_yield_pct": 0.0,
                    "annual_dividend":    0.0,
                    "dividend_growth_5yr":0.0,
                    "ex_dividend_date":   "",
                    "payment_months":     [],
                    "currency":           currency,
                    "note":               note,
                }
                print(f"{'ACC ETF' if asset_type == 'acc' else 'NO DIV'}")
                continue

            # ── Distributing — compute growth & payment pattern ───────────────
            # 5-year dividend CAGR: compare last 12 months vs 12 months
            # ending 5 years ago. Both windows must have actual payments.
            div_growth = 0.0
            if has_divs and len(divs) >= 4:
                today      = datetime.date.today()
                # Recent 12 months
                recent_start = today - datetime.timedelta(days=365)
                recent_divs  = divs[divs.index.date >= recent_start]
                recent_annual = float(recent_divs.sum()) if len(recent_divs) > 0 else 0.0

                # 12 months ending ~5 years ago
                old_end   = today - datetime.timedelta(days=years_back * 365)
                old_start = old_end - datetime.timedelta(days=365)
                older_divs = divs[(divs.index.date >= old_start) &
                                   (divs.index.date <  old_end)]
                older_annual = float(older_divs.sum()) if len(older_divs) > 0 else 0.0

                # CAGR only if both periods have data and older > 0
                if older_annual > 0 and recent_annual > 0:
                    div_growth = (recent_annual / older_annual) ** (1 / years_back) - 1
                    # Clip to realistic range: -30% to +30% annual growth
                    div_growth = float(np.clip(div_growth, -0.30, 0.30))

            # Payment months — from actual ex-dividend dates in last 2 years
            pay_months: list[int] = []
            if has_divs:
                two_yr_ago = datetime.date.today() - datetime.timedelta(days=730)
                recent_divs = divs[divs.index.date >= two_yr_ago]
                if len(recent_divs) > 0:
                    raw_months = sorted(set(recent_divs.index.month.tolist()))
                    pay_months = raw_months

            # Determine type
            if trailing_yield > 0 or annual_div > 0:
                quote_type = info.get("quoteType", "").upper()
                asset_type = "dist" if quote_type == "ETF" else "stock"
                note = (f"{'Distributing ETF' if asset_type == 'dist' else 'Dividend stock'} "
                        f"— pays ~{trailing_yield*100:.2f}% trailing yield. "
                        f"Payments typically in months: {pay_months}.")
            else:
                asset_type = "dist"
                note = "Distributing asset — low or irregular recent dividends."

            rows[ticker] = {
                "type":                asset_type,
                "trailing_yield_pct":  round(float(trailing_yield) * 100, 4),
                "annual_dividend":     round(float(annual_div), 4),
                "dividend_growth_5yr": round(div_growth * 100, 2),
                "ex_dividend_date":    ex_div_str,
                "payment_months":      pay_months,
                "currency":            currency,
                "note":                note,
            }
            print(f"DIST  yield={trailing_yield*100:.2f}%  "
                  f"growth={div_growth*100:.1f}%/yr  "
                  f"months={pay_months}")

        except Exception as exc:
            print(f"ERROR — {exc}")
            rows[ticker] = {
                "type":                "unknown",
                "trailing_yield_pct":  0.0,
                "annual_dividend":     0.0,
                "dividend_growth_5yr": 0.0,
                "ex_dividend_date":    "",
                "payment_months":      [],
                "currency":            "USD",
                "note":                f"Could not fetch data: {exc}",
            }

    df = pd.DataFrame(rows).T
    df.index.name = "Ticker"
    return df


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Available benchmarks — set BENCHMARK in the notebook config cell.
# The value is a Yahoo Finance ticker downloaded alongside portfolio assets.
AVAILABLE_BENCHMARKS: dict[str, str] = {
    "S&P 500 (default)":  "VOO",    # Vanguard S&P 500 ETF
    "MSCI World":         "IWDA",   # iShares MSCI World — global developed
    "MSCI ACWI":          "VWCE",   # Vanguard FTSE All-World — global all-cap
    "Nasdaq 100":         "QQQ",    # Invesco — tech-heavy US
    "Euro Stoxx 50":      "FEZ",    # SPDR — European large-cap
    "MSCI Emerging":      "VWO",    # Vanguard — emerging markets
    "MSCI Europe":        "VGK",    # Vanguard — European equities
    "Dow Jones":          "DIA",    # SPDR — 30 blue-chip US stocks
    "Russell 2000":       "IWM",    # iShares — US small-cap
    "FTSE 100":           "ISF.L",  # iShares — UK large-cap
}


def get_benchmark_returns(prices:          pd.DataFrame,
                           benchmark:       str = "VOO") -> pd.Series:
    """
    Extract the benchmark return series from the already-downloaded
    price DataFrame.

    This is called in the notebook after build_price_dataframe() —
    the benchmark ticker must be included in the tickers list so it
    is downloaded together with the portfolio assets.

    Config cell usage:
        BENCHMARK = "VOO"           # change to any ticker or key below
        # --- or use a named preset ---
        BENCHMARK = AVAILABLE_BENCHMARKS["MSCI World"]  # → "IWDA"

        benchmark_returns = get_benchmark_returns(prices, BENCHMARK)

    Args:
        prices    : pd.DataFrame  output of build_price_dataframe()
                                  must contain the benchmark ticker as a column
        benchmark : str           ticker symbol (e.g. "VOO", "IWDA", "QQQ")

    Returns:
        pd.Series  daily returns for the benchmark, same DatetimeIndex
                   as the portfolio returns

    Raises:
        KeyError  if benchmark ticker is not in prices.columns
    """
    if benchmark not in prices.columns:
        available = list(prices.columns)
        raise KeyError(
            f"Benchmark '{benchmark}' not found in price data. "
            f"Add it to your TICKERS list in the config cell. "
            f"Available columns: {available}"
        )

    return prices[benchmark].pct_change().dropna().rename(f"Benchmark ({benchmark})")


def benchmark_info(benchmark: str = "VOO") -> str:
    """
    Return a human-readable description of the selected benchmark.

    Args:
        benchmark : str  ticker symbol

    Returns:
        str  "Ticker — Full Name" or just ticker if not in registry
    """
    reverse = {v: k for k, v in AVAILABLE_BENCHMARKS.items()}
    name    = reverse.get(benchmark, "Custom benchmark")
    return f"{benchmark} — {name}"
