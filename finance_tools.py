"""
finance_tools.py
=================
Pure financial-planning logic: glide-path allocation, ETF dollar mapping,
and Monte Carlo simulation. These functions are wrapped as "tools" the LLM
agent can call in app.py — this file has zero knowledge of the agent or
Streamlit, on purpose, so it's easy to unit-test and easy to explain.
"""

import numpy as np

ETF_MAP = {
    "equity": "VTI (US Total Equity)",
    "bonds": "BND (US Total Bond Market)",
    "cash": "Cash / money market",
}

ANNUAL_ASSUMPTIONS = {
    "equity": {"return": 0.10, "vol": 0.18},
    "bonds": {"return": 0.045, "vol": 0.06},
    "cash": {"return": 0.03, "vol": 0.01},
}

# Extension exercise (see 04_guided_practice_class_24.md Step 10c): the equity return/vol
# above comes from U.S. stock market history. Anarkulova, Cederburg, and
# O'Doherty (2022, Journal of Financial Economics) studied 39 developed
# countries from 1841-2019 and found a broader-sample equity return/vol
# that annualizes to roughly 9.6% return / 22.9% volatility — very similar
# average return, but meaningfully more volatile than the U.S.-only figure.
# To try it: change equity's "return" above to 0.096 and "vol" to 0.229,
# then redeploy and compare the resulting probability of success.


def compute_allocation(age: int, risk_tolerance: str) -> dict:
    """Base equity % = 110 - age, +/-10pp for risk tolerance, remainder 80/20 bonds/cash."""
    risk_adjustments = {"aggressive": 10, "moderate": 0, "conservative": -10}
    if risk_tolerance not in risk_adjustments:
        raise ValueError(f"risk_tolerance must be one of {list(risk_adjustments)}")

    equity_pct = (110 - age) + risk_adjustments[risk_tolerance]
    equity_pct = min(max(equity_pct, 0), 100)
    remainder_pct = 100 - equity_pct

    return {
        "equity": equity_pct / 100,
        "bonds": (remainder_pct * 0.8) / 100,
        "cash": (remainder_pct * 0.2) / 100,
    }


def dollar_allocation(current_savings: float, weights: dict) -> dict:
    return {asset: current_savings * w for asset, w in weights.items()}


def _annual_to_monthly(annual_return: float, annual_vol: float) -> tuple:
    monthly_mean = (1 + annual_return) ** (1 / 12) - 1
    monthly_vol = annual_vol / np.sqrt(12)
    return monthly_mean, monthly_vol


def simulate_portfolio(
    current_age: int,
    target_age: int,
    current_savings: float,
    monthly_contribution: float,
    weights: dict,
    n_trials: int = 500,
    seed: int = None,
) -> np.ndarray:
    """
    Monte Carlo simulate portfolio value monthly from current_age to target_age.
    Simplifying assumptions: monthly rebalancing to target weights, and
    independent (uncorrelated) normal monthly returns per asset class.
    """
    n_months = (target_age - current_age) * 12
    if n_months <= 0:
        return np.full(n_trials, current_savings, dtype=float)

    rng = np.random.default_rng(seed)

    me, ve = _annual_to_monthly(ANNUAL_ASSUMPTIONS["equity"]["return"], ANNUAL_ASSUMPTIONS["equity"]["vol"])
    mb, vb = _annual_to_monthly(ANNUAL_ASSUMPTIONS["bonds"]["return"], ANNUAL_ASSUMPTIONS["bonds"]["vol"])
    mc, vc = _annual_to_monthly(ANNUAL_ASSUMPTIONS["cash"]["return"], ANNUAL_ASSUMPTIONS["cash"]["vol"])

    r_equity = rng.normal(me, ve, size=(n_trials, n_months))
    r_bonds = rng.normal(mb, vb, size=(n_trials, n_months))
    r_cash = rng.normal(mc, vc, size=(n_trials, n_months))

    portfolio_returns = (
        weights["equity"] * r_equity + weights["bonds"] * r_bonds + weights["cash"] * r_cash
    )

    values = np.full(n_trials, current_savings, dtype=float)
    for m in range(n_months):
        values = values * (1 + portfolio_returns[:, m]) + monthly_contribution

    return values


def probability_of_success(ending_values: np.ndarray, target_amount: float) -> float:
    return float(np.mean(ending_values >= target_amount))


def run_full_simulation(
    current_age: int,
    risk_tolerance: str,
    target_age: int,
    target_amount: float,
    current_savings: float,
    monthly_contribution: float,
    n_trials: int = 500,
    seed: int = None,
) -> dict:
    """
    Convenience wrapper: allocation + simulation + diagnostics in one call.
    This is the function exposed to the LLM agent as a tool.
    """
    weights = compute_allocation(current_age, risk_tolerance)
    ending_values = simulate_portfolio(
        current_age, target_age, current_savings, monthly_contribution, weights,
        n_trials=n_trials, seed=seed,
    )
    prob = probability_of_success(ending_values, target_amount)

    return {
        "allocation": {k: round(v * 100, 1) for k, v in weights.items()},
        "monthly_contribution": monthly_contribution,
        "retirement_age": target_age,
        "years_until_retirement": round((target_age - current_age), 1),
        "probability_of_success_pct": round(prob * 100, 1),
        "median_ending_value": round(float(np.median(ending_values)), 2),
        "min_ending_value": round(float(np.min(ending_values)), 2),
        "max_ending_value": round(float(np.max(ending_values)), 2),
    }
