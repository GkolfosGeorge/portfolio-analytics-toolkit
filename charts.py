"""
charts.py — Portfolio Analyzer Visualization Module
=====================================================
All charts for the Portfolio Analyzer in a single module.

Features:
  - White background throughout (Word / PDF report-ready)
  - save_fig=True  →  saves PNG to output_dir (next to the notebook)
  - plt.show()     →  renders inline during notebook execution
  - Returns (fig, ax) or (fig, axes) for further customization

Usage in notebook:
  import charts as ch

  SAVE_CHARTS = True
  OUTPUT_DIR  = "./charts_output"   # folder created next to the notebook

  ch.plot_equity_curve(portfolio_cumulative, save_fig=SAVE_CHARTS, output_dir=OUTPUT_DIR)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Global Style ──────────────────────────────────────────────────────────────
# White background everywhere — Word / PDF ready
matplotlib.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
})

# ── Color Palette ─────────────────────────────────────────────────────────────
BLUE  = "#2E5A88"
GREY  = "#7F8C8D"
GREEN = "#27AE60"
RED   = "#C0392B"
GOLD  = "#D4AF37"
DARK  = "#1A1A2E"

# ── Internal Helpers ──────────────────────────────────────────────────────────

def _style_ax(ax, title, ylabel="", xlabel=""):
    """Apply consistent axis formatting."""
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    ax.tick_params(labelsize=10)
    return ax


def _save(fig, filename, output_dir):
    """Save figure to output_dir as PNG at 150 dpi."""
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved -> {path}")


def _show_and_save(fig, filename, save_fig, output_dir):
    """Render inline and optionally save to disk."""
    plt.tight_layout()
    plt.show()
    if save_fig:
        _save(fig, filename, output_dir)


# ══════════════════════════════════════════════════════════════════════════════
# A. PORTFOLIO STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

def plot_donut(weights_dict,
               title="Portfolio Asset Allocation",
               save_fig=False, output_dir="."):
    """
    A1. Donut chart of portfolio asset weights.

    Args:
        weights_dict : dict   {ticker: weight}  — must sum to 1.0
        title        : str    chart title
        save_fig     : bool   save PNG to output_dir
        output_dir   : str    destination folder

    Returns:
        fig, ax
    """
    labels = list(weights_dict.keys())
    sizes  = list(weights_dict.values())
    colors = plt.cm.tab20(range(len(labels)))

    fig, ax = plt.subplots(figsize=(10, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        startangle=140, colors=colors,
        wedgeprops=dict(width=0.65, edgecolor="white"),
        textprops=dict(color="black")
    )
    plt.setp(texts,     size=11, weight="bold")
    plt.setp(autotexts, size=10, weight="bold", color="white")
    ax.text(0, 0, "Asset\nAllocation", ha="center", va="center",
            fontsize=15, weight="bold", color=DARK)
    ax.set_title(title, fontsize=17, weight="bold", pad=20)
    ax.axis("equal")

    _show_and_save(fig, "A1_donut.png", save_fig, output_dir)
    return fig, ax


def plot_treemap(weights_dict, asset_categories, category_colors=None,
                 title="Portfolio Structure by Asset Class",
                 save_fig=False, output_dir="."):
    """
    A2. Treemap of portfolio weights grouped by asset class.

    Args:
        weights_dict     : dict  {ticker: weight}
        asset_categories : dict  {ticker: category_name}
        category_colors  : dict  {category_name: hex_color}  (optional)
        title            : str
        save_fig         : bool
        output_dir       : str

    Returns:
        fig, ax
    """
    try:
        import squarify
    except ImportError:
        print("Required: pip install squarify")
        return None, None

    if category_colors is None:
        category_colors = {
            "US Equity":    "#2E5A88",
            "World Equity": "#4F9D69",
            "Commodities":  "#D4AF37",
            "Bonds":        "#8E44AD",
            "Other":        "#7F8C8D",
        }

    sorted_w = dict(sorted(weights_dict.items(),
                            key=lambda x: x[1], reverse=True))
    labels = [f"{k}\n({v*100:.0f}%)" for k, v in sorted_w.items()]
    sizes  = list(sorted_w.values())
    colors = [category_colors.get(asset_categories.get(t, "Other"), GREY)
              for t in sorted_w]

    fig, ax = plt.subplots(figsize=(14, 8))
    squarify.plot(sizes=sizes, label=labels, color=colors,
                  alpha=0.85, edgecolor="white", linewidth=3, ax=ax)
    ax.set_title(title, fontsize=19, weight="bold", pad=22)
    ax.axis("off")

    legend_handles = [mpatches.Patch(color=c, label=l)
                      for l, c in category_colors.items()
                      if l in set(asset_categories.values())]
    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1, 1), title="Asset Classes", frameon=True)

    _show_and_save(fig, "A2_treemap.png", save_fig, output_dir)
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# B. PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

def plot_equity_curve(portfolio_cumulative,
                      title="Portfolio Growth (Equity Curve)",
                      save_fig=False, output_dir="."):
    """
    B2. Cumulative portfolio growth starting from 1.0.

    Args:
        portfolio_cumulative : pd.Series  cumprod series (base 1.0)
        title                : str
        save_fig             : bool
        output_dir           : str

    Returns:
        fig, ax
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(portfolio_cumulative, color=BLUE, lw=2, label="Portfolio")
    ax.fill_between(portfolio_cumulative.index,
                    1, portfolio_cumulative,
                    color=BLUE, alpha=0.10)
    ax.axhline(1.0, color=GREY, lw=0.8, linestyle="--")
    _style_ax(ax, title, ylabel="Growth of $1 Invested")
    ax.legend(fontsize=10)

    _show_and_save(fig, "B2_equity_curve.png", save_fig, output_dir)
    return fig, ax


def plot_benchmark_comparison(port_cum, bench_cum,
                               bench_label="Benchmark (VOO)",
                               title="Portfolio vs Benchmark",
                               save_fig=False, output_dir="."):
    """
    B3. Portfolio cumulative value vs benchmark with filled outperformance area.

    Args:
        port_cum    : pd.Series  portfolio cumulative value (currency or index)
        bench_cum   : pd.Series  benchmark cumulative value
        bench_label : str
        title       : str
        save_fig    : bool
        output_dir  : str

    Returns:
        fig, ax
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(port_cum,  color=BLUE, lw=2,   label="My Portfolio")
    ax.plot(bench_cum, color=GREY, lw=1.5, linestyle="--",
            alpha=0.75, label=bench_label)
    ax.fill_between(port_cum.index, bench_cum, port_cum,
                    where=(port_cum >= bench_cum),
                    color=GREEN, alpha=0.12, label="Outperformance")
    ax.fill_between(port_cum.index, bench_cum, port_cum,
                    where=(port_cum < bench_cum),
                    color=RED, alpha=0.10, label="Underperformance")
    _style_ax(ax, title, ylabel="Portfolio Value")
    ax.legend(fontsize=10)

    _show_and_save(fig, "B3_benchmark.png", save_fig, output_dir)
    return fig, ax


def plot_annual_returns(portfolio_returns,
                        title="Annual Portfolio Returns (%)",
                        save_fig=False, output_dir="."):
    """
    B4. Bar chart of annualized portfolio returns per calendar year.

    Args:
        portfolio_returns : pd.Series  daily returns
        title             : str
        save_fig          : bool
        output_dir        : str

    Returns:
        fig, ax
    """
    annual = (
        portfolio_returns
        .groupby(portfolio_returns.index.year)
        .apply(lambda x: (1 + x).prod() - 1) * 100
    )
    colors = [GREEN if v >= 0 else RED for v in annual]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(annual.index, annual.values, color=colors,
                  edgecolor="white", width=0.6)
    ax.axhline(0, color=DARK, lw=1)

    for bar, val in zip(bars, annual.values):
        ypos = bar.get_height() + 0.4 if val >= 0 else bar.get_height() - 1.5
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold",
                color=GREEN if val >= 0 else RED)

    _style_ax(ax, title, ylabel="Return (%)", xlabel="Year")
    ax.set_xticks(annual.index)

    _show_and_save(fig, "B4_annual_returns.png", save_fig, output_dir)
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# C. RISK & VOLATILITY
# ══════════════════════════════════════════════════════════════════════════════

def plot_rolling_volatility(portfolio_returns, bench_returns=None,
                             static_vol=None, bench_label="VOO",
                             windows=(21, 252),
                             title="Rolling Volatility",
                             save_fig=False, output_dir="."):
    """
    C1. Short and long rolling volatility with optional benchmark overlay.

    Args:
        portfolio_returns : pd.Series  daily returns
        bench_returns     : pd.Series  benchmark daily returns (optional)
        static_vol        : float      annualized static volatility % (optional)
        bench_label       : str
        windows           : tuple      (short_window, long_window) in trading days
        title             : str
        save_fig          : bool
        output_dir        : str

    Returns:
        fig, ax
    """
    w_short, w_long = windows
    roll_short = portfolio_returns.rolling(w_short).std() * np.sqrt(252) * 100
    roll_long  = portfolio_returns.rolling(w_long ).std() * np.sqrt(252) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(roll_short, color=BLUE, alpha=0.45, lw=1.2,
            label=f"Portfolio Vol ({w_short}d)")
    ax.plot(roll_long,  color=DARK, lw=2,
            label=f"Portfolio Vol ({w_long}d)")

    if bench_returns is not None:
        bench_roll = bench_returns.rolling(w_short).std() * np.sqrt(252) * 100
        ax.plot(bench_roll, color=GREEN, alpha=0.6, lw=1.2,
                linestyle="--", label=f"{bench_label} Vol ({w_short}d)")

    if static_vol is not None:
        ax.axhline(static_vol, color=GREY, linestyle=":",
                   lw=1.5, label=f"Static Avg ({static_vol:.1f}%)")

    _style_ax(ax, title, ylabel="Annualized Volatility (%)")
    ax.legend(fontsize=10)

    _show_and_save(fig, "C1_rolling_vol.png", save_fig, output_dir)
    return fig, ax


def plot_correlation_map(returns,
                          title="Assets Correlation Matrix",
                          save_fig=False, output_dir="."):
    """
    C2. Seaborn heatmap of pairwise asset correlations.

    Args:
        returns    : pd.DataFrame  daily returns for all assets
        title      : str
        save_fig   : bool
        output_dir : str

    Returns:
        fig, ax
    """
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax,
                annot_kws={"size": 9},
                vmin=-1, vmax=1)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)

    _show_and_save(fig, "C2_correlation.png", save_fig, output_dir)
    return fig, ax


def plot_drawdown_underwater(drawdown, bench_drawdown=None,
                              bench_label="VOO",
                              title="Maximum Drawdown — Underwater Plot",
                              save_fig=False, output_dir="."):
    """
    C3. Underwater plot of portfolio drawdown vs benchmark.

    Args:
        drawdown       : pd.Series  drawdown series (values in range [−1, 0])
        bench_drawdown : pd.Series  benchmark drawdown (optional)
        bench_label    : str
        title          : str
        save_fig       : bool
        output_dir     : str

    Returns:
        fig, ax
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(drawdown.index, drawdown * 100, 0,
                    color=BLUE, alpha=0.25, label="Portfolio Drawdown")
    ax.plot(drawdown * 100, color=BLUE, lw=1)

    if bench_drawdown is not None:
        ax.plot(bench_drawdown * 100, color=RED, lw=1.2, alpha=0.7,
                linestyle="--", label=f"{bench_label} Drawdown")

    ax.axhline(0, color=DARK, lw=0.8)

    mdd_date = drawdown.idxmin()
    ax.annotate(
        f"MDD {drawdown.min()*100:.1f}%\n{mdd_date.date()}",
        xy=(mdd_date, drawdown.min() * 100),
        xytext=(20, -20), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=DARK),
        fontsize=9, color=DARK
    )

    _style_ax(ax, title, ylabel="Drawdown (%)")
    ax.legend(fontsize=10)

    _show_and_save(fig, "C3_drawdown.png", save_fig, output_dir)
    return fig, ax


def plot_drawdown_distribution(drawdown, bench_drawdown=None,
                                thresholds=(-0.03, -0.08, -0.15),
                                bench_label="VOO",
                                title="Drawdown Distribution (Stress Test)",
                                save_fig=False, output_dir="."):
    """
    C4. Histogram of drawdown distribution with threshold lines.
    Also prints a frequency table to stdout.

    Args:
        drawdown       : pd.Series  portfolio drawdown series
        bench_drawdown : pd.Series  benchmark drawdown (optional)
        thresholds     : tuple      drawdown thresholds for frequency table
        bench_label    : str
        title          : str
        save_fig       : bool
        output_dir     : str

    Returns:
        fig, ax
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(drawdown * 100, bins=50, alpha=0.55,
            color=BLUE, edgecolor="white", label="Portfolio")

    if bench_drawdown is not None:
        ax.hist(bench_drawdown * 100, bins=50, alpha=0.45,
                color=RED, edgecolor="white", label=bench_label)

    for ts in thresholds:
        ax.axvline(ts * 100, color=GREY, linestyle="--", lw=1.2,
                   label=f"{abs(ts)*100:.0f}% threshold")

    _style_ax(ax, title, ylabel="Frequency (Days)", xlabel="Drawdown (%)")
    ax.legend(fontsize=9)

    # Frequency table printed to stdout
    print(f"\n{'─'*50}")
    print("DRAWDOWN FREQUENCY ANALYSIS")
    print(f"{'─'*50}")
    for ts in thresholds:
        pf   = (drawdown < ts).mean() * 100
        line = f"  > {abs(ts)*100:.0f}%:  Portfolio {pf:.2f}% of the time"
        if bench_drawdown is not None:
            pb   = (bench_drawdown < ts).mean() * 100
            line += f"  |  {bench_label} {pb:.2f}% of the time"
        print(line)
    print(f"{'─'*50}\n")

    _show_and_save(fig, "C4_dd_distribution.png", save_fig, output_dir)
    return fig, ax


def plot_var_summary(portfolio_returns, bench_returns=None,
                     confidence_levels=(0.95, 0.99),
                     bench_label="VOO",
                     title="Value at Risk — Daily VaR",
                     save_fig=False, output_dir="."):
    """
    C5. Bar chart of Historical VaR at 95% and 99% confidence levels.

    Args:
        portfolio_returns : pd.Series  daily returns
        bench_returns     : pd.Series  benchmark daily returns (optional)
        confidence_levels : tuple      e.g. (0.95, 0.99)
        bench_label       : str
        title             : str
        save_fig          : bool
        output_dir        : str

    Returns:
        fig, ax
    """
    labels    = [f"{int(c*100)}% CI" for c in confidence_levels]
    port_vars = [abs(np.percentile(portfolio_returns, (1 - c) * 100)) * 100
                 for c in confidence_levels]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, port_vars, width, color=BLUE, alpha=0.85,
           label="Portfolio", edgecolor="white")

    if bench_returns is not None:
        bench_vars = [abs(np.percentile(bench_returns, (1 - c) * 100)) * 100
                      for c in confidence_levels]
        ax.bar(x + width / 2, bench_vars, width, color=GREY, alpha=0.75,
               label=bench_label, edgecolor="white")

    for i, v in enumerate(port_vars):
        ax.text(i - width / 2, v + 0.05, f"{v:.2f}%",
                ha="center", fontsize=10, fontweight="bold", color=BLUE)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    _style_ax(ax, title, ylabel="Daily Loss (%)")
    ax.legend(fontsize=10)

    _show_and_save(fig, "C5_var.png", save_fig, output_dir)
    return fig, ax


def plot_rolling_sharpe_sortino(portfolio_returns,
                                 risk_free_rate=0.0,
                                 window=252,
                                 title="Rolling Sharpe & Sortino (1-Year Window)",
                                 save_fig=False, output_dir="."):
    """
    C8. Dual-subplot rolling Sharpe and Sortino ratios.

    Args:
        portfolio_returns : pd.Series  daily returns
        risk_free_rate    : float      annual rate as decimal (e.g. 0.02 for 2%)
        window            : int        rolling window in trading days
        title             : str
        save_fig          : bool
        output_dir        : str

    Returns:
        fig, axes
    """
    excess   = portfolio_returns - risk_free_rate / 252
    r_mean   = excess.rolling(window).mean() * 252
    r_vol    = portfolio_returns.rolling(window).std() * np.sqrt(252)
    r_sharpe = r_mean / r_vol

    downside   = portfolio_returns.clip(upper=0)
    r_down_vol = downside.rolling(window).std() * np.sqrt(252)
    r_sortino  = r_mean / r_down_vol

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    for ax, series, label, color in zip(
        axes,
        [r_sharpe, r_sortino],
        ["Sharpe Ratio", "Sortino Ratio"],
        [BLUE, GREEN]
    ):
        ax.plot(series, color=color, lw=1.5, label=label)
        ax.axhline(0, color=DARK,  lw=0.8, linestyle="--")
        ax.axhline(1, color=GREEN, lw=0.8, linestyle=":", alpha=0.6,
                   label="Target = 1.0")
        ax.fill_between(series.index, series, 0,
                        where=(series >= 0), color=color, alpha=0.08)
        ax.fill_between(series.index, series, 0,
                        where=(series <  0), color=RED,   alpha=0.08)
        _style_ax(ax, label, ylabel=label)
        ax.legend(fontsize=9)

    _show_and_save(fig, "C8_rolling_ratios.png", save_fig, output_dir)
    return fig, axes


def plot_risk_return_scatter(asset_returns, portfolio_returns,
                              risk_free_rate=2.0,
                              title="Risk vs Return: Assets vs Portfolio",
                              save_fig=False, output_dir="."):
    """
    C9. Scatter plot of annualized risk vs return per asset, with portfolio star.

    Args:
        asset_returns     : pd.DataFrame  daily returns for individual assets
        portfolio_returns : pd.Series     daily portfolio returns
        risk_free_rate    : float         annual risk-free rate in % (e.g. 2.0)
        title             : str
        save_fig          : bool
        output_dir        : str

    Returns:
        fig, ax
    """
    means  = asset_returns.mean() * 252 * 100
    vols   = asset_returns.std()  * np.sqrt(252) * 100
    sharpe = (means - risk_free_rate) / vols

    port_mean = portfolio_returns.mean() * 252 * 100
    port_vol  = portfolio_returns.std()  * np.sqrt(252) * 100

    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(vols, means, c=sharpe, cmap="RdYlGn",
                    s=120, alpha=0.75, edgecolors="white", zorder=3)
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

    for ticker in means.index:
        ax.annotate(ticker, (vols[ticker] + 0.3, means[ticker]),
                    fontsize=9, color=DARK)

    ax.scatter(port_vol, port_mean, color=BLUE, marker="*",
               s=350, edgecolor="white", zorder=5, label="PORTFOLIO")
    ax.annotate("PORTFOLIO", (port_vol + 0.3, port_mean + 0.3),
                fontsize=11, fontweight="bold", color=BLUE)

    _style_ax(ax, title,
              ylabel="Avg Annual Return (%)",
              xlabel="Annualized Volatility (%) — Risk")
    ax.legend(fontsize=10)

    _show_and_save(fig, "C9_risk_return.png", save_fig, output_dir)
    return fig, ax


def plot_alpha_beta_table(asset_returns, portfolio_returns,
                           benchmark_returns,
                           risk_free_rate=0.02,
                           save_fig=False, output_dir="."):
    """
    C10. Table of Alpha, Beta, Information Ratio and Tracking Error
         computed via OLS regression of portfolio vs benchmark returns.

    Args:
        asset_returns     : pd.DataFrame  (not used in computation, kept for API consistency)
        portfolio_returns : pd.Series     daily portfolio returns
        benchmark_returns : pd.Series     daily benchmark returns
        risk_free_rate    : float         annual rate as decimal (e.g. 0.02)
        save_fig          : bool
        output_dir        : str

    Returns:
        fig, ax
    """
    from numpy.linalg import lstsq

    bench = benchmark_returns.values
    port  = portfolio_returns.values
    X     = np.vstack([np.ones_like(bench), bench]).T
    (intercept, beta), _, _, _ = lstsq(X, port, rcond=None)
    alpha = intercept * 252

    active_ret   = portfolio_returns - benchmark_returns
    tracking_err = active_ret.std() * np.sqrt(252)
    ir           = active_ret.mean() * 252 / tracking_err if tracking_err != 0 else 0

    metrics = {
        "Beta":                      f"{beta:.3f}",
        "Alpha (annualized)":        f"{alpha*100:.2f}%",
        "Information Ratio":         f"{ir:.3f}",
        "Tracking Error (ann.)":     f"{tracking_err*100:.2f}%",
    }

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    rows  = [[k, v] for k, v in metrics.items()]
    table = ax.table(cellText=rows,
                     colLabels=["Metric", "Value"],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.4, 1.8)
    ax.set_title("Alpha, Beta & Information Ratio",
                 fontsize=13, fontweight="bold", pad=14)

    _show_and_save(fig, "C10_alpha_beta.png", save_fig, output_dir)
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# D. ENTRY STRATEGY / DCA
# ══════════════════════════════════════════════════════════════════════════════

def plot_lump_vs_dca(lump_sum_series, dca_series,
                     title="Lump Sum vs Systematic DCA",
                     save_fig=False, output_dir="."):
    """
    D1. Equity curve comparison between Lump Sum and DCA strategies.

    Args:
        lump_sum_series : pd.Series  portfolio value under lump sum
        dca_series      : pd.Series  portfolio value under DCA
        title           : str
        save_fig        : bool
        output_dir      : str

    Returns:
        fig, ax
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(lump_sum_series, color=BLUE,  lw=2, label="Lump Sum")
    ax.plot(dca_series,      color=GREEN, lw=2, linestyle="--",
            label="Systematic DCA")
    ax.fill_between(lump_sum_series.index,
                    lump_sum_series, dca_series,
                    where=(lump_sum_series >= dca_series),
                    color=BLUE,  alpha=0.08)
    ax.fill_between(lump_sum_series.index,
                    lump_sum_series, dca_series,
                    where=(lump_sum_series < dca_series),
                    color=GREEN, alpha=0.10)
    _style_ax(ax, title, ylabel="Portfolio Value")
    ax.legend(fontsize=10)

    _show_and_save(fig, "D1_lump_vs_dca.png", save_fig, output_dir)
    return fig, ax


def plot_cost_basis(dca_dates, cost_basis_series, price_series,
                    title="DCA Cost Basis vs Market Price",
                    save_fig=False, output_dir="."):
    """
    D2. Average DCA cost basis overlaid on market price,
    with filled regions for unrealized gain / loss.

    Args:
        dca_dates         : DatetimeIndex  purchase dates
        cost_basis_series : pd.Series      running average purchase price
        price_series      : pd.Series      market price of benchmark asset
        title             : str
        save_fig          : bool
        output_dir        : str

    Returns:
        fig, ax
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(price_series,      color=GREY, lw=1.5,
            alpha=0.7, label="Market Price")
    ax.plot(cost_basis_series, color=GOLD, lw=2,
            linestyle="--", label="Avg Cost Basis (DCA)")
    ax.fill_between(price_series.index,
                    cost_basis_series, price_series,
                    where=(price_series >= cost_basis_series),
                    color=GREEN, alpha=0.12, label="Unrealized Gain")
    ax.fill_between(price_series.index,
                    cost_basis_series, price_series,
                    where=(price_series < cost_basis_series),
                    color=RED, alpha=0.12, label="Unrealized Loss")
    _style_ax(ax, title, ylabel="Price")
    ax.legend(fontsize=10)

    _show_and_save(fig, "D2_cost_basis.png", save_fig, output_dir)
    return fig, ax

# ══════════════════════════════════════════════════════════════════════════════
# G. Fan chart of Monte Carlo forward simulation paths.
# ══════════════════════════════════════════════════════════════════════════════

def plot_monte_carlo_paths(sim_result:    dict,
                            initial_value: float,
                            goal_target:   float = None,
                            horizon_years: float = None,
                            title:         str   = "Monte Carlo — Forward Portfolio Simulation",
                            save_fig:      bool  = False,
                            output_dir:    str   = "."):
    """
    G3a. Clean fan chart — percentile bands only, no individual paths.
    Report-ready, less visually noisy than plot_monte_carlo_fan().

    Args:
        sim_result    : dict   output of monte_carlo.simulate_future_paths()
                               must contain keys: percentiles, mean_path
        initial_value : float  starting portfolio value (reference line)
        goal_target   : float  optional goal target line (e.g. GOAL_TARGET)
        horizon_years : float  optional — if provided x-axis shows years,
                               otherwise shows trading days
        title         : str
        save_fig      : bool
        output_dir    : str

    Returns:
        fig, ax
    """
    pcts = sim_result["percentiles"]
    n    = len(sim_result["mean_path"])

    # X-axis: years if provided, otherwise trading days
    if horizon_years is not None:
        x_axis  = np.linspace(0, horizon_years, n)
        x_label = "Years"
    else:
        x_axis  = np.linspace(0, n / 252, n)
        x_label = "Trading Days (252 ≈ 1 year)"

    fig, ax = plt.subplots(figsize=(12, 6))

    # Shaded bands — outer to inner
    ax.fill_between(x_axis, pcts[5],  pcts[95],
                    color=BLUE, alpha=0.10, label="p5 – p95")
    ax.fill_between(x_axis, pcts[25], pcts[75],
                    color=BLUE, alpha=0.18, label="p25 – p75")

    # Median path
    ax.plot(x_axis, pcts[50], color=BLUE, lw=2,
            label=f"Median (p50): €{pcts[50][-1]:,.0f}")

    # Mean path
    ax.plot(x_axis, sim_result["mean_path"],
            color=DARK, lw=1.2, linestyle="--", alpha=0.6, label="Mean path")

    # Outer percentile lines
    ax.plot(x_axis, pcts[5],  color=BLUE, lw=1.0,
            linestyle="--", alpha=0.6,
            label=f"P5  (worst 5%): €{pcts[5][-1]:,.0f}")
    ax.plot(x_axis, pcts[95], color=BLUE, lw=1.0,
            linestyle="--", alpha=0.6,
            label=f"P95 (best 5%): €{pcts[95][-1]:,.0f}")

    # Reference lines
    ax.axhline(initial_value, color=GREY, lw=0.8,
               linestyle=":", label=f"Initial: €{initial_value:,.0f}")

    if goal_target is not None:
        ax.axhline(goal_target, color=GREEN, lw=1.5,
                   linestyle="--", label=f"Goal: €{goal_target:,.0f}")

    _style_ax(ax, title, ylabel="Portfolio Value (€)", xlabel=x_label)
    ax.legend(fontsize=10)

    _show_and_save(fig, "G3_monte_carlo_paths.png", save_fig, output_dir)
    return fig, ax

def plot_monte_carlo_fan(sim_result:     dict,
                          initial_value:  float,
                          goal_target:    float = None,
                          horizon_years:  float = None,
                          title:          str   = "Monte Carlo — Forward Portfolio Simulation",
                          save_fig:       bool  = False,
                          output_dir:     str   = "."):
    """
    G3. Fan chart + path cloud for Monte Carlo forward simulation.
 
    Combines two visualisation layers:
      1. Individual paths (faint lines) — shows the full chaos of outcomes
      2. Percentile bands (p5/p25/median/p75/p95) — shows the structure
 
    Accepts the output dict of monte_carlo.simulate_future_paths() directly.
 
    Args:
        sim_result    : dict   output of monte_carlo.simulate_future_paths()
                               must contain keys: paths, percentiles, mean_path
        initial_value : float  starting portfolio value (for reference line)
        goal_target   : float  optional goal line (e.g. GOAL_TARGET)
        horizon_years : float  optional — used for x-axis label in years
                               if None, x-axis shows trading days
        title         : str
        save_fig      : bool
        output_dir    : str
 
    Returns:
        fig, ax
    """
    paths = sim_result["paths"]          # shape: (n_days, n_simulations)
    pcts  = sim_result["percentiles"]    # dict {pct: np.ndarray}
    mean  = sim_result["mean_path"]      # np.ndarray
 
    n_days = paths.shape[0]
 
    # X-axis: trading days or years
    if horizon_years is not None:
        x_axis  = np.linspace(0, horizon_years, n_days)
        x_label = "Years"
    else:
        x_axis  = np.arange(n_days)
        x_label = "Trading Days (252 ≈ 1 year)"
 
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
 
    # ── Layer 1: Individual paths (cloud effect) ──────────────────────
    n_show = min(150, paths.shape[1])   # show up to 150 paths
    ax.plot(x_axis, paths[:, :n_show],
            alpha=0.04, color=BLUE, lw=0.4)
 
    # ── Layer 2: Percentile bands ─────────────────────────────────────
    # Outer band: p5 – p95
    if 5 in pcts and 95 in pcts:
        ax.fill_between(x_axis, pcts[5], pcts[95],
                        alpha=0.08, color=BLUE, label="P5 – P95")
        ax.plot(x_axis, pcts[5],  color=BLUE, lw=1.0,
                linestyle="--", alpha=0.7,
                label=f"P5  (worst 5%):  €{pcts[5][-1]:,.0f}")
        ax.plot(x_axis, pcts[95], color=BLUE, lw=1.0,
                linestyle="--", alpha=0.7,
                label=f"P95 (best 5%):   €{pcts[95][-1]:,.0f}")
 
    # Inner band: p25 – p75
    if 25 in pcts and 75 in pcts:
        ax.fill_between(x_axis, pcts[25], pcts[75],
                        alpha=0.18, color=BLUE, label="P25 – P75")
 
    # ── Layer 3: Median & Mean ────────────────────────────────────────
    if 50 in pcts:
        ax.plot(x_axis, pcts[50], color=BLUE, lw=2.5,
                label=f"Median:          €{pcts[50][-1]:,.0f}")
 
    ax.plot(x_axis, mean, color=DARK, lw=1.2,
            linestyle="--", alpha=0.5, label="Mean path")
 
    # ── Reference lines ───────────────────────────────────────────────
    ax.axhline(initial_value, color=GREY, lw=1.0, linestyle=":",
               label=f"Starting capital: €{initial_value:,.0f}")
 
    if goal_target is not None:
        ax.axhline(goal_target, color=GREEN, lw=1.5, linestyle="--",
                   label=f"Goal target:      €{goal_target:,.0f}")
 
    # ── Annotation box ────────────────────────────────────────────────
    median_final = float(pcts[50][-1]) if 50 in pcts else float(mean[-1])
    p5_final     = float(pcts[5][-1])  if 5  in pcts else 0.0
    p95_final    = float(pcts[95][-1]) if 95 in pcts else 0.0
    prob_loss    = float((paths[-1] < initial_value).mean() * 100)
 
    textstr = (f"Median : €{median_final:>10,.0f}\n"
               f"P5     : €{p5_final:>10,.0f}\n"
               f"P95    : €{p95_final:>10,.0f}\n"
               f"P(Loss):  {prob_loss:>8.1f}%")
 
    ax.text(0.02, 0.97, textstr,
            transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            family="monospace", color=DARK,
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="white",
                      edgecolor="#CCCCCC",
                      alpha=0.9))
 
    # ── Formatting ────────────────────────────────────────────────────
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"€{v/1_000:.0f}k")
    )
    _style_ax(ax, title, ylabel="Portfolio Value", xlabel=x_label)
    ax.legend(fontsize=9, framealpha=0.9,
              facecolor="white", edgecolor="#CCCCCC")
 
    _show_and_save(fig, "G3_monte_carlo_fan.png", save_fig, output_dir)
    return fig, ax
 

# ══════════════════════════════════════════════════════════════════════════════
# D. REBALANCING CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_weight_drift(drift_result:    dict,
                       target_weights:  "np.ndarray",
                       tickers:         list,
                       threshold:       float = 0.05,
                       title:           str   = "Portfolio Weight Drift — Current vs Target",
                       save_fig:        bool  = False,
                       output_dir:      str   = "."):
    """
    D5. Horizontal bar chart showing actual vs target weight per asset,
    color-coded by drift direction and magnitude.

    Green  = underweight (should buy)
    Red    = overweight  (should sell)
    Grey   = within threshold band

    Args:
        drift_result   : dict        output of rebalancing.compute_weight_drift()
        target_weights : np.ndarray  target weights (same order as tickers)
        tickers        : list[str]
        threshold      : float       drift band — assets within ±threshold
                                     are shown in grey (default 5%)
        title          : str
        save_fig       : bool
        output_dir     : str

    Returns:
        fig, ax
    """
    import numpy as np

    actual_w = drift_result["actual_weights"].iloc[-1].values  # latest day
    target_w = np.asarray(target_weights)
    drift    = actual_w - target_w

    # Sort by drift magnitude for readability
    order   = np.argsort(drift)
    tickers_sorted = [tickers[i] for i in order]
    drift_sorted   = drift[order]
    actual_sorted  = actual_w[order] * 100
    target_sorted  = target_w[order] * 100

    # Color by drift direction and magnitude
    colors = []
    for d in drift_sorted:
        if d > threshold:
            colors.append(RED)        # overweight → sell
        elif d < -threshold:
            colors.append(GREEN)      # underweight → buy
        else:
            colors.append(GREY)       # within band → hold

    fig, ax = plt.subplots(figsize=(10, max(6, len(tickers) * 0.55)))

    y = range(len(tickers_sorted))

    # Actual weight bars
    ax.barh(y, actual_sorted, color=colors, alpha=0.75,
            height=0.5, label="Actual Weight")

    # Target weight markers
    ax.scatter(target_sorted, y, color=DARK, zorder=5,
               marker="|", s=200, linewidths=2.5, label="Target Weight")

    # Drift labels
    for i, (d, a) in enumerate(zip(drift_sorted, actual_sorted)):
        sign  = "+" if d >= 0 else ""
        color = RED if d > threshold else GREEN if d < -threshold else GREY
        ax.text(a + 0.3, i, f"{sign}{d*100:.1f}pp",
                va="center", fontsize=8.5, color=color, fontweight="bold")

    ax.set_yticks(list(y))
    ax.set_yticklabels(tickers_sorted, fontsize=10)
    ax.set_xlabel("Weight (%)", fontsize=11)

    # Legend patches
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color=RED,   label=f"Overweight  (drift > +{threshold*100:.0f}pp) — SELL"),
        mpatches.Patch(color=GREEN, label=f"Underweight (drift < -{threshold*100:.0f}pp) — BUY"),
        mpatches.Patch(color=GREY,  label="Within band — HOLD"),
        plt.Line2D([0], [0], color=DARK, marker="|", linestyle="None",
                   markersize=12, markeredgewidth=2.5, label="Target Weight"),
    ]
    ax.legend(handles=legend_handles, fontsize=9,
              loc="lower right", frameon=True,
              facecolor="white", edgecolor="#CCCCCC")

    _style_ax(ax, title)
    _show_and_save(fig, "D5_weight_drift.png", save_fig, output_dir)
    return fig, ax


def plot_trade_list(trade_list:   "pd.DataFrame",
                    title:        str  = "Rebalancing Trade List — Today",
                    save_fig:     bool = False,
                    output_dir:   str  = "."):
    """
    D6. Waterfall bar chart of rebalancing trades.

    Green bars = BUY  (positive trade amount)
    Red bars   = SELL (negative trade amount)
    Sorted by trade size for impact clarity.

    Args:
        trade_list : pd.DataFrame  output of rebalancing.generate_trade_list()
                                   must have columns: Action, Trade Amount (€)
        title      : str
        save_fig   : bool
        output_dir : str

    Returns:
        fig, ax
    """
    import numpy as np

    # Filter out TOTAL row and HOLD actions
    df = trade_list[trade_list.index != "── TOTAL ──"].copy()
    df = df[df["Action"].isin(["BUY", "SELL"])].copy()

    if df.empty:
        print("  No trades to plot — portfolio is already at target weights.")
        return None, None

    amounts = df["Trade Amount (€)"].astype(float)
    tickers = df.index.tolist()

    # Sort by absolute amount descending
    order   = amounts.abs().argsort()[::-1]
    amounts = amounts.iloc[order]
    tickers = [tickers[i] for i in order]

    colors = [GREEN if v > 0 else RED for v in amounts]

    fig, ax = plt.subplots(figsize=(10, max(5, len(tickers) * 0.55)))

    y = range(len(tickers))
    ax.barh(y, amounts, color=colors, alpha=0.80, height=0.55)

    # Value labels
    for i, v in enumerate(amounts):
        sign  = "+" if v >= 0 else ""
        xpos  = v + (max(abs(amounts)) * 0.01) if v >= 0 else v - (max(abs(amounts)) * 0.01)
        ha    = "left" if v >= 0 else "right"
        ax.text(xpos, i, f"{sign}€{abs(v):,.0f}",
                va="center", ha=ha, fontsize=9, fontweight="bold",
                color=GREEN if v > 0 else RED)

    ax.set_yticks(list(y))
    ax.set_yticklabels(tickers, fontsize=10)
    ax.axvline(0, color=DARK, lw=1.2)
    ax.set_xlabel("Trade Amount (€)", fontsize=11)

    # Total cost annotation
    if "Est. Cost (€)" in df.columns:
        total_cost = df["Est. Cost (€)"].sum()
        ax.text(0.99, 0.02,
                f"Est. total cost: €{total_cost:.2f}",
                transform=ax.transAxes,
                fontsize=9, ha="right", va="bottom",
                color=GREY, style="italic")

    _style_ax(ax, title)
    _show_and_save(fig, "D6_trade_list.png", save_fig, output_dir)
    return fig, ax


def plot_efficient_frontier(ef_result:  dict,
                             title:      str  = "Efficient Frontier & Portfolio Optimization",
                             save_fig:   bool = False,
                             output_dir: str  = "."):
    """
    G1. Efficient Frontier scatter plot.

    Plots all Monte Carlo simulated portfolios colored by Sharpe Ratio,
    with markers for Max Sharpe, Min Volatility and the Current Portfolio.

    Args:
        ef_result  : dict  output of monte_carlo.efficient_frontier()
                           required keys: ret_arr, vol_arr, sharpe_arr,
                           max_sharpe_idx, min_vol_idx,
                           current_ret, current_vol, current_sharpe
        title      : str
        save_fig   : bool
        output_dir : str

    Returns:
        fig, ax
    """
    ret_arr        = ef_result["ret_arr"]
    vol_arr        = ef_result["vol_arr"]
    sharpe_arr     = ef_result["sharpe_arr"]
    max_sharpe_idx = ef_result["max_sharpe_idx"]
    min_vol_idx    = ef_result["min_vol_idx"]
    current_ret    = ef_result["current_ret"]
    current_vol    = ef_result["current_vol"]
    current_sharpe = ef_result["current_sharpe"]

    fig, ax = plt.subplots(figsize=(12, 8))

    # All simulated portfolios — colored by Sharpe Ratio
    sc = ax.scatter(
        vol_arr * 100, ret_arr * 100,
        c=sharpe_arr, cmap="viridis",
        alpha=0.25, s=8
    )
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

    # Max Sharpe portfolio — red star
    ax.scatter(
        vol_arr[max_sharpe_idx] * 100,
        ret_arr[max_sharpe_idx] * 100,
        color=RED, marker="*", s=350, zorder=5,
        label=(f"Max Sharpe  "
               f"ret={ret_arr[max_sharpe_idx]*100:.1f}%  "
               f"vol={vol_arr[max_sharpe_idx]*100:.1f}%")
    )

    # Min Volatility portfolio — blue star
    ax.scatter(
        vol_arr[min_vol_idx] * 100,
        ret_arr[min_vol_idx] * 100,
        color=BLUE, marker="*", s=350, zorder=5,
        label=(f"Min Volatility  "
               f"ret={ret_arr[min_vol_idx]*100:.1f}%  "
               f"vol={vol_arr[min_vol_idx]*100:.1f}%")
    )

    # Current portfolio — black X
    ax.scatter(
        current_vol * 100, current_ret * 100,
        color=DARK, marker="X", s=250, zorder=5,
        label=(f"Current Portfolio  "
               f"ret={current_ret*100:.1f}%  "
               f"vol={current_vol*100:.1f}%  "
               f"Sharpe={current_sharpe:.2f}")
    )

    _style_ax(ax, title,
              ylabel="Annualized Return (%)",
              xlabel="Annualized Volatility / Risk (%)")
    ax.legend(loc="upper left", fontsize=10,
              frameon=True, facecolor="white", edgecolor="#CCCCCC")

    _show_and_save(fig, "G1_efficient_frontier.png", save_fig, output_dir)
    return fig, ax
