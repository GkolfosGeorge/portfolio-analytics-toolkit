# Portfolio Analytics Toolkit

A comprehensive Python framework for investment portfolio analysis, built for financial advisory workflows. It combines performance and risk metrics, drawdown and rolling analytics, Monte Carlo simulation, stress scenario testing, dividend and tax modeling, rebalancing logic, and automated chart generation into a tiered reporting system — producing polished, client-ready reports from raw portfolio data.

## Features

- **Performance metrics** — returns, CAGR, Sharpe/Sortino ratios, benchmark comparison
- **Risk metrics** — volatility, VaR, drawdown analysis, rolling risk statistics
- **Monte Carlo simulation** — forward-looking portfolio projections
- **Stress scenario testing** — asset-category-based shock modeling
- **Dividend income analysis** — live dividend data via Yahoo Finance
- **DCA (Dollar-Cost Averaging) strategy modeling**
- **Rebalancing logic** with realistic broker-based transaction costs
- **Tax cost estimation**
- **Correlation & regime analysis**
- **Goal planning** tools
- **Sensitivity analysis**
- **Automated chart generation** (Word/PDF-compatible, white background)
- **Tiered Jupyter notebook** (Basic / Standard / Premium) with a single configuration cell controlling which sections run

## Project Structure

```
portfolio-analytics-toolkit/
│
├── portfolio_analyzer_v2.ipynb   # Main tiered analysis notebook
├── portfolio_loader.py           # Portfolio data loading
├── cleaner.py                    # Data cleaning utilities
├── performance_metrics.py        # Return & performance calculations
├── risk_metrics.py               # Risk statistics
├── drawdown_metrics.py           # Drawdown analysis
├── rolling_metrics.py            # Rolling window statistics
├── correlation_regime.py         # Correlation & regime detection
├── monte_carlo.py                # Monte Carlo simulation
├── stress_scenarios.py           # Stress testing
├── dividend_income.py            # Dividend analysis
├── dca_strategies.py             # DCA strategy modeling
├── rebalancing.py                # Rebalancing logic & costs
├── tax_costs.py                  # Tax cost estimation
├── sensitivity.py                # Sensitivity analysis
├── goal_planning.py              # Goal-based planning tools
├── charts.py                     # Chart generation
└── __init__.py
```

## Getting Started

### Requirements

- Python 3.10+
- Jupyter Notebook / JupyterLab
- Key libraries: `yfinance`, `pandas`, `numpy`, `matplotlib` (see notebook imports for full list)

### Installation

```bash
git clone https://github.com/GkolfosGeorge/portfolio-analytics-toolkit.git
cd portfolio-analytics-toolkit
pip install -r requirements.txt
```

### Usage

1. Open `portfolio_analyzer_v2.ipynb` in Jupyter
2. Set your portfolio holdings and configuration in the config cell
3. Choose the desired tier (Basic / Standard / Premium)
4. Toggle `RUN_THIS_CELL` flags for the sections you want
5. Run the notebook to generate analysis and charts

> **Note:** EU-listed UCITS ETFs (e.g. VUAA, VWCE) are substituted with their US-listed equivalents (e.g. VOO, VT) to access longer historical price data from Yahoo Finance for backtesting purposes.

## License

All rights reserved. This repository is public for portfolio and demonstration purposes only — viewing it does not grant any license to use, copy, modify, or distribute the code. See [LICENSE](LICENSE) for details.

---
## Author
**George Gkolfos**  
Quantitative Investment Systems | Macro-Driven Frameworks  
[LinkedIn](https://linkedin.com/in/giorgos-gkolfos-243122119/) · [GitHub](https://github.com/GkolfosGeorge)
Email: [georgegolfos@yahoo.gr]
---

*Built for research and educational purposes. Not financial advice.*
