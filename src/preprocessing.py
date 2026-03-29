"""Stock data cleaning, sorting, and technical-indicator primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sort_and_fill_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by ticker and date; forward-fill then back-fill OHLCV; drop remaining NaNs in price."""
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    out = out.sort_values(["ticker", "Date"])
    price_cols = ["Open", "High", "Low", "Close"]
    for c in price_cols + ["Volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    filled: list[pd.DataFrame] = []
    for _, g in out.groupby("ticker"):
        g = g.copy()
        for c in price_cols + ["Volume"]:
            g[c] = g[c].ffill().bfill()
        filled.append(g)
    out = pd.concat(filled, ignore_index=True)
    out = out.dropna(subset=["Close"])
    return out.reset_index(drop=True)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    line = ema_fast - ema_slow
    sig = _ema(line, signal)
    hist = line - sig
    return line, sig, hist


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def add_technical_indicators(g: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker group: SMA, EMA, RSI, MACD, Bollinger."""
    g = g.copy()
    close = g["Close"]
    g["sma_20"] = close.rolling(20).mean()
    g["sma_50"] = close.rolling(50).mean()
    g["ema_12"] = _ema(close, 12)
    g["ema_26"] = _ema(close, 26)
    g["rsi_14"] = rsi(close, 14)
    macd_line, macd_sig, macd_hist = macd(close)
    g["macd"] = macd_line
    g["macd_signal"] = macd_sig
    g["macd_hist"] = macd_hist
    bb_u, bb_m, bb_l = bollinger_bands(close, 20, 2.0)
    g["bb_upper"] = bb_u
    g["bb_mid"] = bb_m
    g["bb_lower"] = bb_l
    return g


def normalize_features_per_ticker(
    df: pd.DataFrame,
    feature_cols: list[str],
    method: str = "standard",
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, float]]]]:
    """
    Normalize selected columns per ticker. Returns frame and fitted stats for inverse transform if needed.
    method: 'standard' (mean/std) or 'minmax'
    """
    out = df.copy()
    stats: dict[str, dict[str, dict[str, float]]] = {}
    for t, grp in out.groupby("ticker"):
        stats[t] = {}
        for c in feature_cols:
            if c not in grp.columns:
                continue
            s = grp[c].astype(float)
            if method == "standard":
                mu = float(s.mean())
                sig = float(s.std()) or 1.0
                out.loc[grp.index, c] = (s - mu) / sig
                stats[t][c] = {"mean": mu, "std": sig}
            elif method == "minmax":
                lo = float(s.min())
                hi = float(s.max())
                rng = (hi - lo) or 1.0
                out.loc[grp.index, c] = (s - lo) / rng
                stats[t][c] = {"min": lo, "max": hi}
            else:
                raise ValueError(method)
    return out, stats


def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


def rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(252)
