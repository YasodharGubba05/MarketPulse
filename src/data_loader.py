"""Download OHLCV for multiple tickers via yfinance."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import pandas as pd
import yfinance as yf

from src.config import DATA_DIR, DEFAULT_LOOKBACK_YEARS, ensure_dirs

logger = logging.getLogger(__name__)


def download_ohlcv(
    tickers: Sequence[str],
    start: datetime | None = None,
    end: datetime | None = None,
    lookback_years: int = DEFAULT_LOOKBACK_YEARS,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch Open, High, Low, Close, Volume for multiple symbols.
    Returns long DataFrame with columns: Date, ticker, Open, High, Low, Close, Volume.
    """
    ensure_dirs()
    end = end or datetime.utcnow()
    start = start or (end - timedelta(days=365 * lookback_years))
    tickers = [t.upper().strip() for t in tickers if t and str(t).strip()]
    if not tickers:
        raise ValueError("No tickers provided")

    raw = yf.download(
        tickers=list(tickers),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        group_by="ticker",
        threads=True,
        auto_adjust=False,
        progress=False,
    )

    frames: list[pd.DataFrame] = []
    if len(tickers) == 1:
        sym = tickers[0]
        if isinstance(raw.columns, pd.MultiIndex) and sym in raw.columns.get_level_values(0):
            sub = raw[sym].copy()
        else:
            sub = raw.copy()
        if sub.empty:
            logger.warning("No data for %s", sym)
        else:
            sub = sub.rename_axis("Date").reset_index()
            sub["ticker"] = sym
            frames.append(sub)
    else:
        for sym in tickers:
            if sym not in raw.columns.get_level_values(0):
                logger.warning("Missing column group for %s", sym)
                continue
            sub = raw[sym].copy()
            if sub.empty or sub.dropna(how="all").empty:
                logger.warning("No data for %s", sym)
                continue
            sub = sub.rename_axis("Date").reset_index()
            sub["ticker"] = sym
            frames.append(sub)

    if not frames:
        raise RuntimeError("yfinance returned no rows for the given tickers/date range.")

    out = pd.concat(frames, ignore_index=True)
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in out.columns:
            raise KeyError(f"Expected column {col} in download result")
    out = out.drop(columns=[c for c in out.columns if c not in ("Date", "ticker", "Open", "High", "Low", "Close", "Volume")], errors="ignore")
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    out = out.sort_values(["ticker", "Date"]).reset_index(drop=True)
    return out


def save_raw_prices(df: pd.DataFrame, name: str = "raw_prices.parquet") -> Path:
    ensure_dirs()
    path = DATA_DIR / name
    df.to_parquet(path, index=False)
    return path


def load_raw_prices(name: str = "raw_prices.parquet") -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)
