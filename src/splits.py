"""Time-based train/test split (no dependency on training pipeline or models)."""

from __future__ import annotations

import pandas as pd

from src.config import TEST_SIZE


def time_based_split(df: pd.DataFrame, test_ratio: float = TEST_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("Date")
    n = len(df)
    k = int(n * (1 - test_ratio))
    return df.iloc[:k].copy(), df.iloc[k:].copy()
