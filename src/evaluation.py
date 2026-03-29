"""Metrics, cross-validation helpers, optional SHAP."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception as e:
            logger.debug("ROC-AUC skipped: %s", e)
    return out


def time_series_split_indices(n: int, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding window indices for time-series CV."""
    folds = []
    chunk = n // (n_splits + 1)
    for k in range(1, n_splits + 1):
        train_end = chunk * k
        test_start = train_end
        test_end = min(train_end + chunk, n)
        if test_end <= test_start:
            break
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        folds.append((train_idx, test_idx))
    return folds


def shap_summary(
    model: Any,
    X_sample: pd.DataFrame | np.ndarray,
    feature_names: list[str] | None = None,
    max_samples: int = 200,
) -> dict[str, Any] | None:
    try:
        import shap

        X = X_sample
        if isinstance(X, pd.DataFrame):
            X = X.values[:max_samples]
        else:
            X = np.asarray(X)[:max_samples]
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        mean_abs = np.mean(np.abs(sv), axis=0)
        names = feature_names or [f"f{i}" for i in range(mean_abs.shape[0])]
        importance = pd.Series(mean_abs, index=names[: len(mean_abs)]).sort_values(ascending=False)
        return {"mean_abs_shap": importance.to_dict()}
    except Exception as e:
        logger.warning("SHAP not available or failed: %s", e)
        return None
