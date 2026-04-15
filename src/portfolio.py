"""Portfolio-level analytics: correlation matrix, sector performance, diversification.

Used by the Streamlit Portfolio & Correlation tab.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_returns_matrix(
    price_dict: dict[str, np.ndarray | pd.Series],
) -> pd.DataFrame:
    """
    Given a dict of {ticker: close_price_array}, compute daily returns DataFrame.
    Aligns on common index using the longer series.
    """
    data = {}
    for ticker, prices in price_dict.items():
        arr = np.asarray(prices, dtype=float)
        rets = np.diff(arr) / np.maximum(arr[:-1], 1e-12)
        data[ticker] = rets
    # Align to shortest series
    min_len = min(len(v) for v in data.values()) if data else 0
    aligned = {k: v[-min_len:] for k, v in data.items()} if min_len else data
    return pd.DataFrame(aligned)


def correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation matrix from a returns DataFrame."""
    return returns_df.corr(method="pearson")


def portfolio_sharpe(
    returns_df: pd.DataFrame,
    weights: np.ndarray | None = None,
    risk_free_daily: float = 0.0,
) -> float:
    """Equal-weight (or custom-weight) portfolio annualized Sharpe ratio."""
    if returns_df.empty:
        return float("nan")
    n = returns_df.shape[1]
    w = weights if weights is not None else np.ones(n) / n
    port_rets = (returns_df * w).sum(axis=1)
    mu = port_rets.mean()
    sig = port_rets.std()
    return float((mu - risk_free_daily) / max(sig, 1e-12) * np.sqrt(252))


def diversification_ratio(returns_df: pd.DataFrame) -> float:
    """
    Diversification Ratio = (weighted avg vol) / (portfolio vol).
    Higher = more benefit from diversification. Equal weights assumed.
    """
    if returns_df.empty or returns_df.shape[1] < 2:
        return float("nan")
    n = returns_df.shape[1]
    w = np.ones(n) / n
    vols = returns_df.std().values
    weighted_avg_vol = float(np.dot(w, vols))
    cov = returns_df.cov().values
    port_var = float(w @ cov @ w)
    port_vol = np.sqrt(max(port_var, 1e-16))
    return weighted_avg_vol / port_vol if port_vol > 1e-12 else float("nan")


def sector_performance(
    price_dict: dict[str, np.ndarray],
    sector_map: dict[str, str],
    lookback: int = 252,
) -> pd.DataFrame:
    """
    Compute average annualized return and volatility per sector.

    Returns DataFrame with columns: sector, tickers, avg_annual_return, avg_annual_vol.
    """
    rows = []
    for ticker, prices in price_dict.items():
        arr = np.asarray(prices, dtype=float)[-lookback:]
        if len(arr) < 5:
            continue
        rets = np.diff(arr) / np.maximum(arr[:-1], 1e-12)
        ann_ret = (1 + rets.mean()) ** 252 - 1
        ann_vol = rets.std() * np.sqrt(252)
        sector = sector_map.get(ticker, "Unknown")
        rows.append({"ticker": ticker, "sector": sector,
                     "ann_return": ann_ret, "ann_vol": ann_vol,
                     "total_return": (arr[-1] / arr[0] - 1) if arr[0] > 0 else float("nan")})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df


def max_drawdown_series(prices: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (max_drawdown, drawdown_series) for a price array."""
    arr = np.asarray(prices, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / np.maximum(peak, 1e-12)
    return float(np.min(dd)), dd


def rolling_sharpe(
    returns: np.ndarray,
    window: int = 63,  # ~quarterly
    risk_free_daily: float = 0.0,
) -> np.ndarray:
    """Rolling Sharpe ratio over a returns array."""
    s = pd.Series(returns, dtype=float)
    mu = s.rolling(window).mean() - risk_free_daily
    sig = s.rolling(window).std()
    sharpe = (mu / sig.clip(lower=1e-12)) * np.sqrt(252)
    return sharpe.values


def portfolio_analytics(
    price_dict: dict[str, np.ndarray],
    sector_map: dict[str, str],
) -> dict[str, Any]:
    """Master function: returns all portfolio-level analytics in one call."""
    ret_df = compute_returns_matrix(price_dict)
    corr = correlation_matrix(ret_df)
    sharpe = portfolio_sharpe(ret_df)
    div_ratio = diversification_ratio(ret_df)
    sector_df = sector_performance(price_dict, sector_map)

    ticker_stats = {}
    for ticker, prices in price_dict.items():
        arr = np.asarray(prices, dtype=float)
        dd, _ = max_drawdown_series(arr)
        rets = np.diff(arr) / np.maximum(arr[:-1], 1e-12)
        ticker_stats[ticker] = {
            "max_drawdown": dd,
            "ann_vol": float(np.std(rets) * np.sqrt(252)),
            "total_return": float(arr[-1] / arr[0] - 1) if len(arr) > 1 and arr[0] > 0 else float("nan"),
        }

    return {
        "correlation_matrix": corr,
        "portfolio_sharpe": sharpe,
        "diversification_ratio": div_ratio,
        "sector_performance": sector_df,
        "ticker_stats": ticker_stats,
        "returns_df": ret_df,
    }
