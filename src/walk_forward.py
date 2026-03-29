"""Expanding-window (walk-forward) evaluation with sklearn TimeSeriesSplit."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from src.baselines import mean_train_prediction, naive_persistence_prediction
from src.config import CLASS_LABEL, REGRESSION_TARGET
from src.evaluation import classification_metrics, regression_metrics
from src.models import (
    scale_fit,
    train_linear_regression,
    train_logistic,
    train_xgb_classifier,
    train_xgb_regressor,
)

logger = logging.getLogger(__name__)


def walk_forward_evaluate_ticker(
    sub: pd.DataFrame,
    feat_cols: list[str],
    n_splits: int = 5,
) -> dict[str, Any]:
    """
    Expanding-window CV: for each fold, fit scaler + models on train, score on test.
    Aggregates mean/std of RMSE for regression; mean F1/ROC for classification if applicable.
    """
    sub = sub.sort_values("Date").reset_index(drop=True)
    X = sub[feat_cols].values.astype(float)
    y_r = sub[REGRESSION_TARGET].values.astype(float)
    y_c = sub[CLASS_LABEL].values.astype(int)

    n = len(sub)
    if n < 80:
        return {"error": "insufficient_rows", "n_rows": n}

    tsc = TimeSeriesSplit(n_splits=n_splits)
    fold_regs: dict[str, list[float]] = {
        "linear_regression": [],
        "xgboost": [],
        "naive_persistence": [],
        "mean_train": [],
    }
    fold_clf: dict[str, list[float]] = {"xgboost": [], "logistic": []}

    for fold_idx, (train_idx, test_idx) in enumerate(tsc.split(X)):
        if len(test_idx) < 5 or len(train_idx) < 40:
            continue
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr_r, y_te_r = y_r[train_idx], y_r[test_idx]
        y_tr_c, y_te_c = y_c[train_idx], y_c[test_idx]

        sc, X_tr_s, X_te_s = scale_fit(X_tr, X_te)

        # Baselines (same test indices)
        test_df = sub.iloc[test_idx]
        naive = naive_persistence_prediction(test_df)
        mean_b = mean_train_prediction(y_tr_r, len(test_idx))
        m_naive = regression_metrics(y_te_r, naive)["RMSE"]
        m_mean = regression_metrics(y_te_r, mean_b)["RMSE"]
        fold_regs["naive_persistence"].append(m_naive)
        fold_regs["mean_train"].append(m_mean)

        lr = train_linear_regression(X_tr_s, y_tr_r)
        pred_lr = lr.predict(X_te_s)
        fold_regs["linear_regression"].append(regression_metrics(y_te_r, pred_lr)["RMSE"])

        xgb_r = train_xgb_regressor(X_tr_s, y_tr_r)
        pred_x = xgb_r.predict(X_te_s)
        fold_regs["xgboost"].append(regression_metrics(y_te_r, pred_x)["RMSE"])

        log_m = train_logistic(X_tr_s, y_tr_c)
        pred_lc = log_m.predict(X_te_s)
        fold_clf["logistic"].append(classification_metrics(y_te_c, pred_lc).get("f1", 0.0))

        xgb_c = train_xgb_classifier(X_tr_s, y_tr_c)
        pred_xc = xgb_c.predict(X_te_s)
        proba = xgb_c.predict_proba(X_te_s)[:, 1]
        fold_clf["xgboost"].append(classification_metrics(y_te_c, pred_xc, proba).get("f1", 0.0))

    def summarize(name: str, vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"mean_rmse": float("nan"), "std_rmse": float("nan"), "n_folds": 0}
        a = np.array(vals, dtype=float)
        return {"mean_rmse": float(np.mean(a)), "std_rmse": float(np.std(a)), "n_folds": len(vals)}

    def summarize_f1(name: str, vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"mean_f1": float("nan"), "std_f1": float("nan"), "n_folds": 0}
        a = np.array(vals, dtype=float)
        return {"mean_f1": float(np.mean(a)), "std_f1": float(np.std(a)), "n_folds": len(vals)}

    out: dict[str, Any] = {
        "n_splits_requested": n_splits,
        "regression_rmse_by_fold": {
            k: summarize(k, v) for k, v in fold_regs.items()
        },
        "classification_f1_by_fold": {
            "logistic": summarize_f1("logistic", fold_clf["logistic"]),
            "xgboost": summarize_f1("xgboost", fold_clf["xgboost"]),
        },
    }
    return out
