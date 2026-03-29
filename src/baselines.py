"""Naive forecasting baselines for comparison against ML models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def naive_persistence_prediction(df: pd.DataFrame) -> np.ndarray:
    """
    Predict next close = today's close (random-walk level).
    For each row t, forecast of target_next_close is Close[t].
    """
    return df["Close"].astype(float).values


def mean_train_prediction(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """Constant forecast: mean of training targets (length n_test)."""
    m = float(np.mean(y_train))
    return np.full(n_test, m, dtype=float)


def baseline_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from src.evaluation import regression_metrics

    return regression_metrics(y_true, y_pred)
