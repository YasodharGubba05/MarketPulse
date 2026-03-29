"""Lags, rolling stats, volatility, targets, and sentiment merge."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    CLASS_LABEL,
    CRASH_THRESHOLD,
    LAG_DAYS,
    REGRESSION_TARGET,
    ROLL_WINDOWS,
    VOL_WINDOW,
)
from src.preprocessing import add_technical_indicators, daily_returns, rolling_volatility


def build_panel_features(
    df: pd.DataFrame,
    sentiment_daily: pd.DataFrame | None,
    crash_threshold: float = CRASH_THRESHOLD,
) -> pd.DataFrame:
    """
    df: columns Date, ticker, OHLCV + technicals already on df.
    sentiment_daily: columns Date, ticker, sentiment_mean, sentiment_count (optional)
    """
    parts: list[pd.DataFrame] = []
    for t, g in df.groupby("ticker"):
        g = g.sort_values("Date").copy()
        g = add_technical_indicators(g)
        close = g["Close"]
        g["return_1d"] = daily_returns(close)
        g["volatility_roll"] = rolling_volatility(g["return_1d"], VOL_WINDOW)

        for w in ROLL_WINDOWS:
            g[f"roll_mean_ret_{w}"] = g["return_1d"].rolling(w).mean()
            g[f"roll_std_ret_{w}"] = g["return_1d"].rolling(w).std()

        for lag in range(1, LAG_DAYS + 1):
            g[f"lag_close_{lag}"] = close.shift(lag)
            g[f"lag_ret_{lag}"] = g["return_1d"].shift(lag)

        g[REGRESSION_TARGET] = close.shift(-1)
        fut_ret = close.pct_change().shift(-1)
        g[CLASS_LABEL] = (fut_ret < -crash_threshold).astype(int)

        if sentiment_daily is not None and not sentiment_daily.empty:
            ssub = sentiment_daily[sentiment_daily["ticker"] == t][
                ["Date", "sentiment_mean", "sentiment_count"]
            ].drop_duplicates("Date")
            g = g.merge(ssub, on="Date", how="left")
        else:
            g["sentiment_mean"] = np.nan
            g["sentiment_count"] = 0.0

        g["sentiment_mean"] = g["sentiment_mean"].fillna(0.0)
        g["sentiment_count"] = g["sentiment_count"].fillna(0.0)
        g["sentiment_momentum"] = g["sentiment_mean"].diff().fillna(0.0)

        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["ticker", "Date"]).reset_index(drop=True)
    return out


def feature_columns(
    include_sentiment: bool = True,
    include_ticker_id: bool = False,
    ticker_encoder: dict[str, int] | None = None,
) -> list[str]:
    base = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "sma_20",
        "sma_50",
        "ema_12",
        "ema_26",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_mid",
        "bb_lower",
        "return_1d",
        "volatility_roll",
    ]
    for w in ROLL_WINDOWS:
        base.extend([f"roll_mean_ret_{w}", f"roll_std_ret_{w}"])
    for lag in range(1, LAG_DAYS + 1):
        base.extend([f"lag_close_{lag}", f"lag_ret_{lag}"])
    if include_sentiment:
        base.extend(["sentiment_mean", "sentiment_momentum", "sentiment_count"])
    if include_ticker_id and ticker_encoder:
        base.append("ticker_id")
    return base


def add_ticker_id_column(df: pd.DataFrame, encoder: dict[str, int]) -> pd.DataFrame:
    out = df.copy()
    out["ticker_id"] = out["ticker"].map(encoder).astype(float)
    return out


def drop_na_targets_and_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    cols = [c for c in feature_cols if c in df.columns]
    sub = df.dropna(subset=cols + [REGRESSION_TARGET, CLASS_LABEL])
    return sub.reset_index(drop=True)


def make_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
    target_col: str = REGRESSION_TARGET,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (N, seq_len, F) and (N,) for LSTM from a single-ticker sorted frame."""
    data = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)
    xs, ys = [], []
    for i in range(seq_len, len(data)):
        xs.append(data[i - seq_len : i])
        ys.append(y[i - 1])
    if not xs:
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0,))
    return np.stack(xs), np.array(ys)
