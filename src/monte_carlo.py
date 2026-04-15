"""Monte Carlo simulation of future price paths using Geometric Brownian Motion (GBM).

Used by the Streamlit app to display fan charts of possible future price trajectories.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def estimate_gbm_params(returns: np.ndarray) -> tuple[float, float]:
    """Estimate daily drift (mu) and volatility (sigma) from historical log-returns."""
    log_rets = np.log1p(returns)
    log_rets = log_rets[np.isfinite(log_rets)]
    mu = float(np.mean(log_rets))
    sigma = float(np.std(log_rets))
    return mu, sigma


def simulate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    n_days: int = 30,
    n_paths: int = 500,
    dt: float = 1.0,
    random_state: int = 42,
) -> np.ndarray:
    """
    Simulate GBM paths: dS = S*(mu*dt + sigma*sqrt(dt)*Z).

    Returns:
        paths: shape (n_paths, n_days + 1) — includes S0 at index 0.
    """
    rng = np.random.default_rng(random_state)
    Z = rng.standard_normal((n_paths, n_days))
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    cum = np.cumsum(log_returns, axis=1)
    paths = S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), cum]))
    return paths


def monte_carlo_summary(
    paths: np.ndarray,
    percentiles: tuple[int, ...] = (5, 25, 50, 75, 95),
) -> dict[str, np.ndarray]:
    """Compute percentile fan-chart bands across simulated paths."""
    result = {}
    for p in percentiles:
        result[f"p{p}"] = np.percentile(paths, p, axis=0)
    result["mean"] = np.mean(paths, axis=0)
    return result


def run_monte_carlo(
    close_series: pd.Series | np.ndarray,
    n_days: int = 30,
    n_paths: int = 1000,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Full Monte Carlo pipeline for a close-price series.

    Returns a dict with:
        - 'paths': (n_paths, n_days+1) array
        - 'summary': dict of p5/p25/p50/p75/p95/mean arrays
        - 'S0': float last price
        - 'mu': float daily drift
        - 'sigma': float daily vol
        - 'annualized_vol': float
        - 'n_days': int
        - 'n_paths': int
    """
    arr = np.asarray(close_series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 20:
        raise ValueError("Need at least 20 close prices for Monte Carlo simulation.")

    returns = np.diff(arr) / np.maximum(arr[:-1], 1e-12)
    mu, sigma = estimate_gbm_params(returns)
    S0 = float(arr[-1])

    paths = simulate_gbm_paths(S0, mu, sigma, n_days=n_days, n_paths=n_paths,
                                random_state=random_state)
    summary = monte_carlo_summary(paths)

    return {
        "paths": paths,
        "summary": summary,
        "S0": S0,
        "mu": mu,
        "sigma": sigma,
        "annualized_vol": sigma * np.sqrt(252),
        "n_days": n_days,
        "n_paths": n_paths,
        "var_95": float(np.percentile(paths[:, -1] / S0 - 1, 5)),  # 95% VaR
        "expected_return": float(np.mean(paths[:, -1] / S0 - 1)),
        "prob_gain": float(np.mean(paths[:, -1] > S0)),
    }
