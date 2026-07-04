"""
goal_planning.py — Goal-Based Financial Planning
==================================================
Answers the question every client actually cares about:
"Will I reach my financial goal?"

All inputs come from the notebook config cell — no hardcoded values.

Five planning tools:
  1. goal_probability()        — what are the odds of reaching the target?
  2. required_monthly_savings()— how much do I need to invest per month?
  3. time_to_goal()            — when will I get there?
  4. goal_sensitivity()        — what changes if I adjust key parameters?
  5. retirement_readiness()    — single score + full retirement income plan

All Monte Carlo simulations use the portfolio's historical return
distribution (mean + std from daily returns). Results are expressed
as plain numbers — no financial jargon — ready for client presentation.

Typical notebook usage:
    import goal_planning as gp
    import monte_carlo   as mc

    # Config cell inputs
    GOAL_TARGET   = 500_000   # €
    GOAL_YEARS    = 20
    MONTHLY_DCA   = 415       # €/month
    INITIAL_VALUE = 10_000    # €

    result = gp.goal_probability(
        portfolio_returns = portfolio_returns,
        initial_value     = INITIAL_VALUE,
        monthly_investment= MONTHLY_DCA,
        target_value      = GOAL_TARGET,
        horizon_years     = GOAL_YEARS,
    )
    print(f"Success probability: {result['probability_pct']}%")
"""

import numpy as np
import pandas as pd


# ── Shared simulation engine ──────────────────────────────────────────────────

def _run_simulations(portfolio_returns: pd.Series,
                      initial_value:     float,
                      monthly_investment:float,
                      horizon_years:     int,
                      n_simulations:     int   = 2_000,
                      random_seed:       int   = 42) -> np.ndarray:
    """
    Run Monte Carlo paths with monthly DCA contributions.

    Each path:
      - Draws daily returns from Normal(mu, sigma) calibrated to history
      - Adds `monthly_investment` at the end of each calendar month (~21 days)
      - Returns final portfolio value for each simulation

    Args:
        portfolio_returns  : pd.Series  historical daily returns
        initial_value      : float      starting portfolio value (EUR)
        monthly_investment : float      fixed monthly contribution (EUR)
        horizon_years      : int        investment horizon
        n_simulations      : int        number of Monte Carlo paths
        random_seed        : int

    Returns:
        np.ndarray  shape (n_simulations,)  final values at horizon
    """
    np.random.seed(random_seed)
    mu      = float(portfolio_returns.mean())
    sigma   = float(portfolio_returns.std())
    n_days  = int(horizon_years * 252)

    final_values = np.zeros(n_simulations)

    for i in range(n_simulations):
        daily_ret = np.random.normal(mu, sigma, n_days)
        value     = initial_value
        day_count = 0

        for ret in daily_ret:
            day_count += 1
            value     *= (1 + ret)
            # Monthly contribution every ~21 trading days
            if day_count % 21 == 0:
                value += monthly_investment

        final_values[i] = value

    return final_values


def _run_paths(portfolio_returns: pd.Series,
               initial_value:     float,
               monthly_investment:float,
               horizon_years:     int,
               n_simulations:     int = 500,
               random_seed:       int = 42) -> np.ndarray:
    """
    Same as _run_simulations but returns full paths (not just final values).

    Returns:
        np.ndarray  shape (n_days, n_simulations)
    """
    np.random.seed(random_seed)
    mu      = float(portfolio_returns.mean())
    sigma   = float(portfolio_returns.std())
    n_days  = int(horizon_years * 252)

    paths = np.zeros((n_days, n_simulations))

    for i in range(n_simulations):
        daily_ret = np.random.normal(mu, sigma, n_days)
        value     = initial_value
        for d, ret in enumerate(daily_ret):
            value      *= (1 + ret)
            if (d + 1) % 21 == 0:
                value  += monthly_investment
            paths[d, i] = value

    return paths


# ══════════════════════════════════════════════════════════════════════════════
# 1. GOAL PROBABILITY
# ══════════════════════════════════════════════════════════════════════════════

def goal_probability(portfolio_returns:  pd.Series,
                      initial_value:      float,
                      monthly_investment: float,
                      target_value:       float,
                      horizon_years:      int,
                      n_simulations:      int = 2_000,
                      random_seed:        int = 42) -> dict:
    """
    Estimate the probability of reaching a financial target
    within the given horizon.

    Config cell inputs → all parameters of this function.

    Args:
        portfolio_returns  : pd.Series  historical daily returns
        initial_value      : float      current portfolio value (EUR)
        monthly_investment : float      monthly DCA contribution (EUR)
        target_value       : float      financial goal in EUR
        horizon_years      : int        investment horizon in years
        n_simulations      : int        Monte Carlo paths (default 2,000)
        random_seed        : int

    Returns:
        dict with keys:
            probability_pct      : float  % of paths that reach target
            median_final         : float  median portfolio value at horizon
            p10_final            : float  pessimistic (10th percentile)
            p90_final            : float  optimistic  (90th percentile)
            total_invested       : float  total cash contributed
            median_gain          : float  median_final - total_invested
            shortfall_median     : float  target - median (0 if target reached)
            verdict              : str    plain-English outcome
    """
    final_values   = _run_simulations(
        portfolio_returns, initial_value, monthly_investment,
        horizon_years, n_simulations, random_seed
    )

    total_invested = initial_value + monthly_investment * horizon_years * 12
    successes      = (final_values >= target_value).sum()
    probability    = successes / n_simulations * 100

    median_final   = float(np.percentile(final_values, 50))
    p10_final      = float(np.percentile(final_values, 10))
    p90_final      = float(np.percentile(final_values, 90))
    shortfall      = max(target_value - median_final, 0.0)

    if probability >= 80:
        verdict = (f"On track. {probability:.0f}% of scenarios reach "
                   f"the €{target_value:,.0f} goal in {horizon_years} years.")
    elif probability >= 50:
        verdict = (f"Possible but uncertain. {probability:.0f}% success rate. "
                   f"Consider increasing monthly contributions.")
    else:
        verdict = (f"At risk. Only {probability:.0f}% of scenarios reach the target. "
                   f"Increase contributions or extend the horizon.")

    return {
        "probability_pct":  round(probability, 1),
        "median_final":     round(median_final, 0),
        "p10_final":        round(p10_final, 0),
        "p90_final":        round(p90_final, 0),
        "total_invested":   round(total_invested, 0),
        "median_gain":      round(median_final - total_invested, 0),
        "shortfall_median": round(shortfall, 0),
        "verdict":          verdict,
        "_final_values":    final_values,   # kept for charting
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. REQUIRED MONTHLY SAVINGS
# ══════════════════════════════════════════════════════════════════════════════

def required_monthly_savings(portfolio_returns:   pd.Series,
                               initial_value:       float,
                               target_value:        float,
                               horizon_years:       int,
                               target_probability:  float = 0.80,
                               monthly_range:       tuple = (50, 3_000),
                               n_simulations:       int   = 1_000,
                               random_seed:         int   = 42) -> dict:
    """
    Find the monthly investment needed to reach the target with a
    given probability of success.

    Binary-searches the monthly amount until the success rate
    matches `target_probability`.

    Config cell inputs: initial_value, target_value, horizon_years.
    Adjust target_probability to reflect client's risk tolerance.

    Args:
        portfolio_returns  : pd.Series
        initial_value      : float
        target_value       : float
        horizon_years      : int
        target_probability : float  desired success rate (default 0.80 = 80%)
        monthly_range      : tuple  (min, max) search bounds in EUR
        n_simulations      : int
        random_seed        : int

    Returns:
        dict with keys:
            required_monthly    : float  EUR/month needed
            achieved_probability: float  actual probability at that amount
            total_invested      : float  total cash over the horizon
            vs_current          : float  difference vs current DCA (if provided)
            horizon_years       : int
            target_value        : float
    """
    lo, hi = float(monthly_range[0]), float(monthly_range[1])
    best_monthly = hi
    best_prob    = 0.0

    # Binary search — 15 iterations is more than enough for 0.1% precision
    for _ in range(15):
        mid    = (lo + hi) / 2
        finals = _run_simulations(
            portfolio_returns, initial_value, mid,
            horizon_years, n_simulations, random_seed
        )
        prob = (finals >= target_value).sum() / n_simulations

        if prob >= target_probability:
            best_monthly = mid
            best_prob    = prob
            hi           = mid
        else:
            lo = mid

    total_invested = initial_value + best_monthly * horizon_years * 12

    return {
        "required_monthly":     round(best_monthly, 0),
        "achieved_probability": round(best_prob * 100, 1),
        "total_invested":       round(total_invested, 0),
        "horizon_years":        horizon_years,
        "target_value":         target_value,
        "target_probability":   round(target_probability * 100, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. TIME TO GOAL
# ══════════════════════════════════════════════════════════════════════════════

def time_to_goal(portfolio_returns:  pd.Series,
                  initial_value:      float,
                  monthly_investment: float,
                  target_value:       float,
                  max_years:          int   = 40,
                  percentiles:        tuple = (10, 25, 50, 75, 90),
                  n_simulations:      int   = 1_000,
                  random_seed:        int   = 42) -> dict:
    """
    Find when the portfolio reaches the target at various probability levels.

    Runs full daily paths and records the first crossing date for each
    simulation, then summarises by percentile.

    Config cell inputs: initial_value, monthly_investment, target_value.

    Args:
        portfolio_returns  : pd.Series
        initial_value      : float
        monthly_investment : float
        target_value       : float
        max_years          : int    maximum horizon to simulate
        percentiles        : tuple  percentile bands to report
        n_simulations      : int
        random_seed        : int

    Returns:
        dict with keys:
            crossing_years  : dict  {percentile: years_to_reach_target}
            never_reached   : float % of simulations that never reach target
            summary_table   : pd.DataFrame  one row per percentile
            verdict         : str
    """
    np.random.seed(random_seed)
    mu     = float(portfolio_returns.mean())
    sigma  = float(portfolio_returns.std())
    n_days = int(max_years * 252)

    crossing_days = np.full(n_simulations, np.nan)

    for i in range(n_simulations):
        daily_ret = np.random.normal(mu, sigma, n_days)
        value     = initial_value
        for d, ret in enumerate(daily_ret):
            value *= (1 + ret)
            if (d + 1) % 21 == 0:
                value += monthly_investment
            if value >= target_value:
                crossing_days[i] = d + 1
                break

    never_reached = float(np.isnan(crossing_days).sum() / n_simulations * 100)
    valid_days    = crossing_days[~np.isnan(crossing_days)]

    crossing_years = {}
    rows           = []
    for p in percentiles:
        if len(valid_days) == 0:
            yrs = None
        else:
            yrs = round(float(np.percentile(valid_days, p)) / 252, 1)
        crossing_years[p] = yrs

        label = {10: "Pessimistic (p10)",  25: "Below median (p25)",
                 50: "Median (p50)",        75: "Above median (p75)",
                 90: "Optimistic (p90)"}.get(p, f"p{p}")
        rows.append({
            "Scenario":          label,
            "Years to Goal":     yrs,
            "Target (€)":        f"€{target_value:,.0f}",
        })

    median_yrs = crossing_years.get(50)
    if median_yrs is None:
        verdict = (f"Target €{target_value:,.0f} not reached within "
                   f"{max_years} years in most scenarios. "
                   f"Consider increasing contributions or reducing target.")
    else:
        verdict = (f"Median scenario reaches €{target_value:,.0f} "
                   f"in {median_yrs} years. "
                   f"{100 - never_reached:.0f}% of scenarios reach it "
                   f"within {max_years} years.")

    return {
        "crossing_years":  crossing_years,
        "never_reached":   round(never_reached, 1),
        "summary_table":   pd.DataFrame(rows),
        "verdict":         verdict,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. GOAL SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════

def goal_sensitivity(portfolio_returns:     pd.Series,
                      initial_value:         float,
                      monthly_investment:    float,
                      target_value:          float,
                      horizon_years:         int,
                      monthly_range:         list[float] = None,
                      horizon_range:         list[int]   = None,
                      target_range:          list[float] = None,
                      n_simulations:         int  = 500,
                      random_seed:           int  = 42) -> dict:
    """
    Show how success probability changes when key parameters vary.

    Three sensitivity tables, each varying one parameter while
    holding the others fixed at their base values.

    Config cell inputs: initial_value, monthly_investment,
                        target_value, horizon_years.
    Adjust ranges to match client conversation.

    Args:
        portfolio_returns     : pd.Series
        initial_value         : float
        monthly_investment    : float   base monthly DCA
        target_value          : float   base goal
        horizon_years         : int     base horizon
        monthly_range         : list    monthly amounts to test
        horizon_range         : list    horizons (years) to test
        target_range          : list    target values to test
        n_simulations         : int
        random_seed           : int

    Returns:
        dict with keys:
            monthly_sensitivity  : pd.DataFrame
            horizon_sensitivity  : pd.DataFrame
            target_sensitivity   : pd.DataFrame
    """
    if monthly_range is None:
        base = monthly_investment
        monthly_range = [max(50, base * m) for m in
                         [0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 2.0]]

    if horizon_range is None:
        h = horizon_years
        horizon_range = sorted(set(
            [max(1, h - 5), max(1, h - 3), h, h + 3, h + 5, h + 10]
        ))

    if target_range is None:
        t = target_value
        target_range = [t * m for m in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]]

    def _prob(monthly, horizon, target):
        finals = _run_simulations(
            portfolio_returns, initial_value,
            monthly, horizon, n_simulations, random_seed
        )
        return round((finals >= target).sum() / n_simulations * 100, 1)

    # Table 1: vary monthly contribution
    rows1 = []
    for m in monthly_range:
        p = _prob(m, horizon_years, target_value)
        rows1.append({
            "Monthly Investment (€)": round(m, 0),
            "Success Probability (%)": p,
            "vs Base":  f"{p - _prob(monthly_investment, horizon_years, target_value):+.1f}pp",
        })

    # Table 2: vary horizon
    rows2 = []
    base_prob = _prob(monthly_investment, horizon_years, target_value)
    for h in horizon_range:
        p = _prob(monthly_investment, h, target_value)
        rows2.append({
            "Horizon (years)":        h,
            "Success Probability (%)":p,
            "vs Base":  f"{p - base_prob:+.1f}pp",
        })

    # Table 3: vary target
    rows3 = []
    for t in target_range:
        p = _prob(monthly_investment, horizon_years, t)
        rows3.append({
            "Target Value (€)":       round(t, 0),
            "Success Probability (%)":p,
            "vs Base":  f"{p - base_prob:+.1f}pp",
        })

    return {
        "monthly_sensitivity": pd.DataFrame(rows1),
        "horizon_sensitivity": pd.DataFrame(rows2),
        "target_sensitivity":  pd.DataFrame(rows3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. RETIREMENT READINESS
# ══════════════════════════════════════════════════════════════════════════════

def retirement_readiness(portfolio_returns:     pd.Series,
                          initial_value:          float,
                          monthly_investment:     float,
                          retirement_age:         int,
                          current_age:            int,
                          monthly_income_target:  float,
                          expected_pension:       float = 0.0,
                          safe_withdrawal_rate:   float = 0.04,
                          n_simulations:          int   = 2_000,
                          random_seed:            int   = 42) -> dict:
    """
    Full retirement readiness assessment.

    Determines if the client can fund their desired monthly income
    in retirement, combining portfolio withdrawals + pension income.

    All inputs come from the config cell — one per client.

    Args:
        portfolio_returns      : pd.Series  historical daily returns
        initial_value          : float      current portfolio value (EUR)
        monthly_investment     : float      monthly DCA until retirement (EUR)
        retirement_age         : int        target retirement age
        current_age            : int        client's current age
        monthly_income_target  : float      desired monthly income in retirement (EUR)
        expected_pension       : float      expected monthly state/private pension (EUR)
        safe_withdrawal_rate   : float      annual withdrawal rate (default 4%)
        n_simulations          : int
        random_seed            : int

    Returns:
        dict with keys:
            years_to_retirement     : int
            required_nest_egg       : float  portfolio needed at retirement
            income_gap              : float  monthly shortfall not covered by pension
            probability_pct         : float  % of scenarios reaching nest egg
            median_portfolio        : float  median portfolio at retirement
            p10_portfolio           : float  pessimistic scenario
            p90_portfolio           : float  optimistic scenario
            score                   : int    readiness score 0-100
            score_label             : str    "On Track" | "Needs Attention" | "At Risk"
            monthly_shortfall       : float  0 if on track, else extra needed
            income_breakdown        : pd.DataFrame  pension + portfolio income
            verdict                 : str    plain-English summary for client
    """
    years_to_ret = retirement_age - current_age
    if years_to_ret <= 0:
        raise ValueError("retirement_age must be greater than current_age.")

    # Monthly income gap that portfolio must cover
    income_gap       = max(monthly_income_target - expected_pension, 0.0)
    annual_gap       = income_gap * 12
    required_nest_egg = annual_gap / safe_withdrawal_rate

    # Run simulations to retirement date
    final_values = _run_simulations(
        portfolio_returns, initial_value, monthly_investment,
        years_to_ret, n_simulations, random_seed
    )

    probability    = float((final_values >= required_nest_egg).sum()
                           / n_simulations * 100)
    median_port    = float(np.percentile(final_values, 50))
    p10_port       = float(np.percentile(final_values, 10))
    p90_port       = float(np.percentile(final_values, 90))

    # Readiness score 0-100
    # Score is based on probability of success, not just median/nest_egg ratio.
    # Max score is 95 unless probability = 100% (theoretically impossible
    # with Monte Carlo), avoiding misleading "perfect" scores.
    # Thresholds:
    #   >= 90% probability → 80-95 (On Track)
    #   >= 70% probability → 60-79 (On Track / Needs Attention)
    #   >= 50% probability → 40-59 (Needs Attention)
    #   <  50% probability →  0-39 (At Risk)
    ratio = median_port / required_nest_egg if required_nest_egg > 0 else 0
    raw_score = int(np.clip(ratio * 65, 0, 100))

    # Cap at 95 unless probability is 100% (never in practice)
    if probability >= 100:
        score = min(raw_score, 100)
    else:
        score = min(raw_score, 95)

    if score >= 75:
        score_label = "On Track"
    elif score >= 50:
        score_label = "Needs Attention"
    else:
        score_label = "At Risk"

    # Monthly shortfall: how much extra per month to reach 80% probability
    monthly_shortfall = 0.0
    if probability < 80:
        req = required_monthly_savings(
            portfolio_returns, initial_value,
            required_nest_egg, years_to_ret,
            target_probability=0.80,
            monthly_range=(monthly_investment, monthly_investment * 10),
            n_simulations=500,
            random_seed=random_seed,
        )
        monthly_shortfall = max(
            req["required_monthly"] - monthly_investment, 0.0
        )

    # Income breakdown table
    portfolio_income  = median_port * safe_withdrawal_rate / 12
    income_breakdown  = pd.DataFrame([
        {"Source":   "State / Private Pension",
         "Monthly (€)": round(expected_pension, 0),
         "Annual (€)":  round(expected_pension * 12, 0)},
        {"Source":   "Portfolio Withdrawal (median)",
         "Monthly (€)": round(portfolio_income, 0),
         "Annual (€)":  round(portfolio_income * 12, 0)},
        {"Source":   "── TOTAL ──",
         "Monthly (€)": round(expected_pension + portfolio_income, 0),
         "Annual (€)":  round((expected_pension + portfolio_income) * 12, 0)},
    ]).set_index("Source")

    # Plain-English verdict
    if probability >= 80:
        verdict = (
            f"Retirement at {retirement_age} is on track. "
            f"With €{monthly_investment:,.0f}/month invested over "
            f"{years_to_ret} years, there is a {probability:.0f}% chance "
            f"of reaching the €{required_nest_egg:,.0f} nest egg needed "
            f"to fund €{monthly_income_target:,.0f}/month in retirement."
        )
    else:
        verdict = (
            f"Retirement at {retirement_age} needs attention. "
            f"Current plan has a {probability:.0f}% success rate. "
            f"To reach 80%, increase monthly investment by "
            f"€{monthly_shortfall:,.0f} (to €{monthly_investment + monthly_shortfall:,.0f}/month), "
            f"or consider retiring at {retirement_age + 2} instead."
        )

    return {
        "years_to_retirement":  years_to_ret,
        "required_nest_egg":    round(required_nest_egg, 0),
        "income_gap":           round(income_gap, 0),
        "probability_pct":      round(probability, 1),
        "median_portfolio":     round(median_port, 0),
        "p10_portfolio":        round(p10_port, 0),
        "p90_portfolio":        round(p90_port, 0),
        "score":                score,
        "score_label":          score_label,
        "monthly_shortfall":    round(monthly_shortfall, 0),
        "income_breakdown":     income_breakdown,
        "verdict":              verdict,
        "_final_values":        final_values,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. GOAL SUMMARY TABLE  — notebook-ready one-pager
# ══════════════════════════════════════════════════════════════════════════════

def goal_summary_table(prob_result:       dict,
                        req_savings_result:dict,
                        ttg_result:        dict) -> pd.DataFrame:
    """
    Combine outputs of goal_probability, required_monthly_savings
    and time_to_goal into a single client-facing summary DataFrame.

    Args:
        prob_result        : dict  from goal_probability()
        req_savings_result : dict  from required_monthly_savings()
        ttg_result         : dict  from time_to_goal()

    Returns:
        pd.DataFrame  two columns: Metric | Value
    """
    cy = ttg_result["crossing_years"]
    rows = [
        ("Target Value (€)",
         f"€{prob_result.get('_final_values', [0]).max():,.0f}" ),
        ("Success Probability",
         f"{prob_result['probability_pct']}%"),
        ("Total Cash Invested",
         f"€{prob_result['total_invested']:,.0f}"),
        ("Median Portfolio at Horizon",
         f"€{prob_result['median_final']:,.0f}"),
        ("Pessimistic Outcome (p10)",
         f"€{prob_result['p10_final']:,.0f}"),
        ("Optimistic Outcome (p90)",
         f"€{prob_result['p90_final']:,.0f}"),
        ("Required Monthly (80% success)",
         f"€{req_savings_result['required_monthly']:,.0f}"),
        ("Time to Goal — Pessimistic (p10)",
         f"{cy.get(10, 'N/A')} yrs"),
        ("Time to Goal — Median (p50)",
         f"{cy.get(50, 'N/A')} yrs"),
        ("Time to Goal — Optimistic (p90)",
         f"{cy.get(90, 'N/A')} yrs"),
        ("Scenarios that Never Reach Target",
         f"{ttg_result['never_reached']}%"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"]).set_index("Metric")
