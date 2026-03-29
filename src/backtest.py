"""Simple directional backtest with transaction costs (industry-style sanity check)."""

from __future__ import annotations

import numpy as np


def run_directional_backtest(
    close: np.ndarray,
    pred_next_close: np.ndarray,
    crash_proba: np.ndarray | None = None,
    cost_bps: float = 10.0,
    long_if_pred_return_positive: bool = True,
    avoid_crash: bool = False,
    crash_threshold: float = 0.35,
) -> dict[str, float]:
    """
    Long-only or cash: at day i, go long for i→i+1 if predicted return > 0 (optional: skip if crash_proba high).

    Costs: pay `cost_bps` per notional when position changes (round-trip style on flip).

    Returns summary metrics: cumulative_return, annualized_sharpe, max_drawdown, hit_rate, n_days.
    """
    close = np.asarray(close, dtype=float).ravel()
    pred_next_close = np.asarray(pred_next_close, dtype=float).ravel()
    n = min(len(close), len(pred_next_close))
    close = close[:n]
    pred_next_close = pred_next_close[:n]

    # One-day-ahead return realized
    actual_ret = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-12)
    pred_ret = (pred_next_close[:-1] - close[:-1]) / np.maximum(close[:-1], 1e-12)

    position = np.ones(len(actual_ret), dtype=float)
    if long_if_pred_return_positive:
        position = (pred_ret > 0).astype(float)
    if avoid_crash and crash_proba is not None:
        cp = np.asarray(crash_proba[: len(pred_ret)], dtype=float)
        position = np.where(cp < crash_threshold, position, 0.0)

    strat_ret = position * actual_ret

    # Cost when position changes (same length as strat_ret)
    pos_ext = np.concatenate([[0.0], position])
    turnover = np.abs(np.diff(pos_ext))
    costs = turnover * (cost_bps / 10000.0)
    strat_ret = strat_ret - costs

    equity = np.cumprod(1.0 + strat_ret)
    total_ret = float(equity[-1] - 1.0) if len(equity) else 0.0

    mu = float(np.mean(strat_ret)) if len(strat_ret) else 0.0
    sig = float(np.std(strat_ret)) if len(strat_ret) > 1 else 1e-12
    sharpe = (mu / max(sig, 1e-12)) * np.sqrt(252.0)

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-12)
    max_dd = float(np.min(dd)) if len(dd) else 0.0

    hit = float(np.mean((pred_ret > 0) == (actual_ret > 0))) if len(actual_ret) else 0.0

    buy_hold = float(np.prod(1.0 + actual_ret) - 1.0) if len(actual_ret) else 0.0

    return {
        "strategy_total_return": total_ret,
        "buy_hold_total_return": buy_hold,
        "annualized_sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "directional_hit_rate": hit,
        "mean_daily_return": mu,
        "n_trading_days": float(len(strat_ret)),
        "cost_bps_assumed": float(cost_bps),
    }
