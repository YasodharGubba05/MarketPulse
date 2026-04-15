"""Metrics, cross-validation helpers, leaderboard, optional SHAP."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
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
    # MAPE — skip zero-targets to avoid division by zero
    nz = y_true != 0
    mape = float(mean_absolute_percentage_error(y_true[nz], y_pred[nz]) * 100) if nz.any() else float("nan")
    # Directional accuracy — did pred direction match actual?
    if len(y_true) > 1:
        actual_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(y_pred[1:] - y_true[:-1])
        dir_acc = float(np.mean(actual_dir == pred_dir))
    else:
        dir_acc = float("nan")
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE_%": mape, "Dir_Acc": dir_acc}


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


def shap_values_for_plot(
    model: Any,
    X_sample: np.ndarray,
    feature_names: list[str],
    max_samples: int = 300,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Returns (shap_values, X_sample) arrays for interactive plotting.
    Returns None when SHAP is unavailable.
    """
    try:
        import shap
        X = np.asarray(X_sample)[:max_samples]
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        return sv, X
    except Exception as e:
        logger.warning("SHAP values failed: %s", e)
        return None


def leaderboard_df(
    per_ticker_metrics: dict[str, dict],
    ticker: str,
    metric: str = "RMSE",
) -> pd.DataFrame:
    """
    Build a ranked leaderboard DataFrame for all regression models for a given ticker.

    per_ticker_metrics: the metrics["per_ticker"] dict from metrics.json
    Returns DataFrame with columns: model, RMSE, MAE, R2, MAPE_%, Dir_Acc
    and sorted ascending by `metric`.
    """
    pt = per_ticker_metrics.get(ticker, {})
    reg = pt.get("regression", {})
    rows = []
    for model_name, m in reg.items():
        if model_name.endswith("_shap"):
            continue
        if not isinstance(m, dict):
            continue
        rows.append({
            "Model": model_name.replace("_", " ").title(),
            "RMSE": m.get("RMSE"),
            "MAE": m.get("MAE"),
            "R2": m.get("R2"),
            "MAPE_%": m.get("MAPE_%"),
            "Dir_Acc": m.get("Dir_Acc"),
        })
    # Add baselines
    bh = pt.get("baselines_holdout", {})
    for bl_name, bm in bh.items():
        if isinstance(bm, dict):
            rows.append({
                "Model": f"[Baseline] {bl_name.replace('_', ' ').title()}",
                "RMSE": bm.get("RMSE"),
                "MAE": bm.get("MAE"),
                "R2": bm.get("R2"),
                "MAPE_%": bm.get("MAPE_%"),
                "Dir_Acc": bm.get("Dir_Acc"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if metric in df.columns:
        df = df.sort_values(metric, na_position="last").reset_index(drop=True)
    return df


def classification_leaderboard_df(per_ticker_metrics: dict, ticker: str) -> pd.DataFrame:
    """Build classification model leaderboard for crash detection."""
    pt = per_ticker_metrics.get(ticker, {})
    clf = pt.get("classification", {})
    rows = []
    for model_name, m in clf.items():
        if not isinstance(m, dict):
            continue
        rows.append({
            "Model": model_name.replace("_", " ").title(),
            "Accuracy": m.get("accuracy"),
            "Precision": m.get("precision"),
            "Recall": m.get("recall"),
            "F1": m.get("f1"),
            "ROC-AUC": m.get("roc_auc"),
        })
    df = pd.DataFrame(rows)
    if not df.empty and "F1" in df.columns:
        df = df.sort_values("F1", ascending=False, na_position="last").reset_index(drop=True)
    return df
