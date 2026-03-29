"""End-to-end training: download → sentiment → features → models → metrics."""

from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.config import (
    CLASS_LABEL,
    LSTM_SEQUENCE_LENGTH,
    MODELS_DIR,
    REGRESSION_TARGET,
    TEST_SIZE,
    ensure_dirs,
    load_ticker_universe,
    symbols_from_universe,
)
from src.backtest import run_directional_backtest
from src.baselines import mean_train_prediction, naive_persistence_prediction
from src.data_loader import download_ohlcv, save_raw_prices
from src.evaluation import classification_metrics, regression_metrics, shap_summary
from src.feature_engineering import (
    add_ticker_id_column,
    build_panel_features,
    drop_na_targets_and_features,
    feature_columns,
    make_sequences,
)
from src.models import (
    fit_garch_volatility,
    predict_lstm_torch,
    preferred_lstm_backend,
    save_keras_model,
    save_sklearn_model,
    save_torch_lstm,
    train_lstm,
    train_lstm_torch,
    train_linear_regression,
    train_logistic,
    train_rf_classifier,
    train_rf_regressor,
    train_volatility_xgb,
    train_xgb_classifier,
    train_xgb_regressor,
    scale_fit,
)
from src.preprocessing import sort_and_fill_ohlcv
from src.sentiment import build_daily_sentiment
from src.splits import time_based_split
from src.mlflow_utils import log_nested_json_artifact, mlflow_run
from src.walk_forward import walk_forward_evaluate_ticker

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)


def train_for_ticker(
    panel: pd.DataFrame,
    ticker: str,
    include_sentiment: bool,
    crash_threshold: float,
    run_walk_forward: bool = True,
) -> dict[str, Any]:
    from src.config import CRASH_THRESHOLD

    ct = crash_threshold if crash_threshold is not None else CRASH_THRESHOLD
    sub = panel[panel["ticker"] == ticker].copy()
    sub = sub.sort_values("Date")
    feat_cols = feature_columns(include_sentiment=include_sentiment, include_ticker_id=False)
    sub = drop_na_targets_and_features(sub, feat_cols)

    if len(sub) < 80:
        logger.warning("Insufficient rows for %s: %s", ticker, len(sub))
        return {}

    train_df, test_df = time_based_split(sub)
    X_train = train_df[feat_cols].values
    X_test = test_df[feat_cols].values
    y_train_r = train_df[REGRESSION_TARGET].values
    y_test_r = test_df[REGRESSION_TARGET].values
    y_train_c = train_df[CLASS_LABEL].values
    y_test_c = test_df[CLASS_LABEL].values

    sc, X_train_s, X_test_s = scale_fit(X_train, X_test)

    out: dict[str, Any] = {"ticker": ticker, "regression": {}, "classification": {}, "lstm": {}, "garch": {}}

    garch_res = fit_garch_volatility(sub.sort_values("Date")["return_1d"])
    if garch_res is not None:
        out["garch"] = {"aic": float(garch_res.aic), "bic": float(garch_res.bic)}

    lr = train_linear_regression(X_train_s, y_train_r)
    pred_lr = lr.predict(X_test_s)
    out["regression"]["linear_regression"] = regression_metrics(y_test_r, pred_lr)

    rf = train_rf_regressor(X_train_s, y_train_r)
    pred_rf = rf.predict(X_test_s)
    out["regression"]["random_forest"] = regression_metrics(y_test_r, pred_rf)

    xgb_r = train_xgb_regressor(X_train_s, y_train_r)
    pred_xgb = xgb_r.predict(X_test_s)
    out["regression"]["xgboost"] = regression_metrics(y_test_r, pred_xgb)

    naive_pred = naive_persistence_prediction(test_df)
    mean_pred = mean_train_prediction(y_train_r, len(test_df))
    out["baselines_holdout"] = {
        "naive_persistence": regression_metrics(y_test_r, naive_pred),
        "mean_train_target": regression_metrics(y_test_r, mean_pred),
    }

    shap_rf = shap_summary(rf, test_df[feat_cols], feat_cols)
    if shap_rf:
        out["regression"]["random_forest_shap"] = shap_rf

    vol_y_train = np.abs(train_df["return_1d"].shift(-1)).values
    vol_y_test = np.abs(test_df["return_1d"].shift(-1)).values
    vtrain = np.isfinite(vol_y_train) & np.isfinite(X_train_s).all(axis=1)
    vtest = np.isfinite(vol_y_test) & np.isfinite(X_test_s).all(axis=1)
    vol_model = train_volatility_xgb(X_train_s[vtrain], vol_y_train[vtrain])
    y_vol_te = vol_y_test[vtest]
    pred_vol = vol_model.predict(X_test_s[vtest])
    out["volatility_xgb_rmse"] = (
        float(np.sqrt(np.mean((pred_vol - y_vol_te) ** 2))) if len(y_vol_te) else float("nan")
    )

    log_m = train_logistic(X_train_s, y_train_c)
    pred_log = log_m.predict(X_test_s)
    proba_log = log_m.predict_proba(X_test_s)[:, 1]
    out["classification"]["logistic"] = classification_metrics(y_test_c, pred_log, proba_log)

    rfc = train_rf_classifier(X_train_s, y_train_c)
    pred_rfc = rfc.predict(X_test_s)
    proba_rfc = rfc.predict_proba(X_test_s)[:, 1]
    out["classification"]["random_forest"] = classification_metrics(y_test_c, pred_rfc, proba_rfc)

    xgb_c = train_xgb_classifier(X_train_s, y_train_c)
    pred_xc = xgb_c.predict(X_test_s)
    proba_xc = xgb_c.predict_proba(X_test_s)[:, 1]
    out["classification"]["xgboost"] = classification_metrics(y_test_c, pred_xc, proba_xc)

    try:
        out["backtest_holdout"] = {
            "long_on_positive_pred_return": run_directional_backtest(
                test_df["Close"].values,
                pred_xgb,
                crash_proba=proba_xc,
                cost_bps=10.0,
                avoid_crash=False,
            ),
            "long_positive_avoid_high_crash_risk": run_directional_backtest(
                test_df["Close"].values,
                pred_xgb,
                crash_proba=proba_xc,
                cost_bps=10.0,
                avoid_crash=True,
                crash_threshold=0.35,
            ),
        }
    except Exception as e:
        logger.warning("Backtest failed for %s: %s", ticker, e)
        out["backtest_holdout"] = {"error": str(e)}

    # LSTM: Keras (TensorFlow) when available; else PyTorch (works on Python 3.13+ without TF)
    seq_len = LSTM_SEQUENCE_LENGTH
    X_seq, y_seq = make_sequences(train_df, feat_cols, seq_len)
    if len(X_seq) > 30:
        try:
            ns, nt, nf = X_seq.shape
            from sklearn.preprocessing import StandardScaler

            flat_train = X_seq.reshape(-1, nf)
            sc_lstm = StandardScaler().fit(flat_train)
            X_seq_s = sc_lstm.transform(flat_train).reshape(ns, nt, nf)
            epochs = min(50, 10 + len(X_seq) // 50)
            X_test_seq, y_test_seq = make_sequences(test_df, feat_cols, seq_len)
            if len(X_test_seq) > 0:
                ft = X_test_seq.reshape(-1, nf)
                X_test_seq_s = sc_lstm.transform(ft).reshape(X_test_seq.shape[0], nt, nf)
                backend = preferred_lstm_backend()
                if backend == "keras":
                    lstm = train_lstm(X_seq_s, y_seq, epochs=epochs)
                    pred_lstm = lstm.predict(X_test_seq_s, verbose=0).ravel()
                    save_keras_model(lstm, MODELS_DIR / f"lstm_{ticker}.keras")
                elif backend == "torch":
                    lstm, device, nfeat_m = train_lstm_torch(X_seq_s, y_seq, epochs=epochs)
                    pred_lstm = predict_lstm_torch(lstm, X_test_seq_s, device)
                    save_torch_lstm(lstm, MODELS_DIR / f"lstm_{ticker}.pt", nfeat_m)
                else:
                    pred_lstm = None
                if pred_lstm is not None:
                    out["lstm"] = {**regression_metrics(y_test_seq, pred_lstm), "backend": backend}
                else:
                    out["lstm"] = {"error": "Install PyTorch or TensorFlow for LSTM"}
                import joblib

                joblib.dump(sc_lstm, MODELS_DIR / f"lstm_scaler_{ticker}.joblib")
            else:
                out["lstm"] = {}
        except Exception as e:
            logger.warning("LSTM failed for %s: %s", ticker, e)
            out["lstm"] = {"error": str(e)}
    else:
        out["lstm"] = {}

    if run_walk_forward:
        try:
            out["walk_forward_cv"] = walk_forward_evaluate_ticker(sub, feat_cols, n_splits=5)
        except Exception as e:
            logger.warning("Walk-forward CV failed for %s: %s", ticker, e)
            out["walk_forward_cv"] = {"error": str(e)}

    save_sklearn_model(sc, MODELS_DIR / f"scaler_reg_{ticker}.joblib")
    save_sklearn_model(lr, MODELS_DIR / f"lr_{ticker}.joblib")
    save_sklearn_model(rf, MODELS_DIR / f"rf_reg_{ticker}.joblib")
    save_sklearn_model(xgb_r, MODELS_DIR / f"xgb_reg_{ticker}.joblib")
    save_sklearn_model(vol_model, MODELS_DIR / f"xgb_vol_{ticker}.joblib")
    save_sklearn_model(log_m, MODELS_DIR / f"log_clf_{ticker}.joblib")
    save_sklearn_model(rfc, MODELS_DIR / f"rf_clf_{ticker}.joblib")
    save_sklearn_model(xgb_c, MODELS_DIR / f"xgb_clf_{ticker}.joblib")

    return out


def train_combined_model(
    panel: pd.DataFrame,
    tickers: list[str],
    include_sentiment: bool,
) -> dict[str, Any]:
    enc = {t: i for i, t in enumerate(sorted(tickers))}
    df = add_ticker_id_column(panel, enc)
    feat_cols = feature_columns(include_sentiment=include_sentiment, include_ticker_id=True, ticker_encoder=enc)
    df = drop_na_targets_and_features(df, feat_cols)
    train_df, test_df = time_based_split(df)
    X_train = train_df[feat_cols].values
    X_test = test_df[feat_cols].values
    y_train_r = train_df[REGRESSION_TARGET].values
    y_test_r = test_df[REGRESSION_TARGET].values
    y_train_c = train_df[CLASS_LABEL].values
    y_test_c = test_df[CLASS_LABEL].values
    sc, X_train_s, X_test_s = scale_fit(X_train, X_test)
    out: dict[str, Any] = {"combined_regression": {}, "combined_classification": {}}

    xgb_r = train_xgb_regressor(X_train_s, y_train_r)
    pred = xgb_r.predict(X_test_s)
    out["combined_regression"]["xgboost"] = regression_metrics(y_test_r, pred)

    xgb_c = train_xgb_classifier(X_train_s, y_train_c)
    pred_c = xgb_c.predict(X_test_s)
    proba = xgb_c.predict_proba(X_test_s)[:, 1]
    out["combined_classification"]["xgboost"] = classification_metrics(y_test_c, pred_c, proba)

    save_sklearn_model(sc, MODELS_DIR / "scaler_combined.joblib")
    save_sklearn_model(xgb_r, MODELS_DIR / "xgb_reg_combined.joblib")
    save_sklearn_model(xgb_c, MODELS_DIR / "xgb_clf_combined.joblib")
    with open(MODELS_DIR / "ticker_encoder.json", "w", encoding="utf-8") as f:
        json.dump(enc, f)
    return out


def ablation_without_sentiment(panel: pd.DataFrame, ticker: str) -> dict[str, Any]:
    """Train XGBoost regression with sentiment features zeroed vs actual."""
    sub = panel[panel["ticker"] == ticker].copy().sort_values("Date")
    feat_with = feature_columns(include_sentiment=True, include_ticker_id=False)
    feat_without = feature_columns(include_sentiment=False, include_ticker_id=False)
    sub = drop_na_targets_and_features(sub, feat_with)
    train_df, test_df = time_based_split(sub)
    sc_w, X_train_w, X_test_w = scale_fit(train_df[feat_with].values, test_df[feat_with].values)
    xgb_w = train_xgb_regressor(X_train_w, train_df[REGRESSION_TARGET].values)
    pred_w = xgb_w.predict(X_test_w)
    m_with = regression_metrics(test_df[REGRESSION_TARGET].values, pred_w)

    sub2 = panel[panel["ticker"] == ticker].copy().sort_values("Date")
    sub2 = drop_na_targets_and_features(sub2, feat_with)
    train_df2, test_df2 = time_based_split(sub2)
    X_tr = train_df2[feat_without].values
    X_te = test_df2[feat_without].values
    sc_wo, X_train_wo, X_test_wo = scale_fit(X_tr, X_te)
    xgb_wo = train_xgb_regressor(X_train_wo, train_df2[REGRESSION_TARGET].values)
    pred_wo = xgb_wo.predict(X_test_wo)
    m_without = regression_metrics(test_df2[REGRESSION_TARGET].values, pred_wo)
    return {"with_sentiment": m_with, "without_sentiment": m_without}


def run_training(
    combined_model: bool = True,
    per_ticker: bool = True,
    include_sentiment: bool = True,
    lookback_years: int | None = None,
    crash_threshold: float | None = None,
    max_tickers: int | None = None,
    use_mlflow: bool = True,
    walk_forward: bool = True,
) -> TrainResult:
    ensure_dirs()
    lb = lookback_years or config.DEFAULT_LOOKBACK_YEARS
    ct = crash_threshold if crash_threshold is not None else config.CRASH_THRESHOLD

    universe = load_ticker_universe()
    if max_tickers is not None:
        universe = universe[:max_tickers]
    syms = symbols_from_universe(universe)
    name_map = {e["symbol"].upper(): e.get("company_name", e["symbol"]) for e in universe}

    mlflow_ctx = mlflow_run(experiment_name="multi-stock-ads", tags={"pipeline": "train"}) if use_mlflow else nullcontext(None)

    with mlflow_ctx as active_run:
        if active_run is not None:
            try:
                import mlflow

                mlflow.log_param("lookback_years", lb)
                mlflow.log_param("crash_threshold", ct)
                mlflow.log_param("n_tickers", len(syms))
                mlflow.log_param("include_sentiment", include_sentiment)
                mlflow.log_param("walk_forward", walk_forward)
            except Exception as e:
                logger.debug("mlflow params: %s", e)

        logger.info("Downloading OHLCV for %s", syms)
        raw = download_ohlcv(syms, lookback_years=lb)
        raw = sort_and_fill_ohlcv(raw)
        save_raw_prices(raw)

        sentiment_parts: list[pd.DataFrame] = []
        if include_sentiment:
            for sym in syms:
                comp = name_map.get(sym, sym)
                daily = build_daily_sentiment(sym, comp, price_df=raw)
                sentiment_parts.append(daily)
            sentiment_df = pd.concat(sentiment_parts, ignore_index=True) if sentiment_parts else pd.DataFrame()
        else:
            sentiment_df = pd.DataFrame()

        panel = build_panel_features(raw, sentiment_df if not sentiment_df.empty else None, crash_threshold=ct)

        metrics: dict[str, Any] = {"per_ticker": {}, "combined": {}, "ablation": {}}

        if per_ticker:
            for sym in syms:
                metrics["per_ticker"][sym] = train_for_ticker(
                    panel, sym, include_sentiment, ct, run_walk_forward=walk_forward
                )
                try:
                    metrics["ablation"][sym] = ablation_without_sentiment(panel, sym)
                except Exception as e:
                    logger.warning("Ablation failed for %s: %s", sym, e)

        if combined_model and len(syms) > 1:
            metrics["combined"] = train_combined_model(panel, syms, include_sentiment)

        out_path = MODELS_DIR / "metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)

        if active_run is not None:
            log_nested_json_artifact(metrics)

    artifacts: dict[str, str] = {"metrics_json": str(out_path)}
    if use_mlflow:
        artifacts["mlflow_note"] = "file:./data/mlruns — run: mlflow ui --backend-store-uri file:./data/mlruns"
    return TrainResult(metrics=metrics, artifacts=artifacts)


def load_trained_artifacts(ticker: str) -> dict[str, Any]:
    """Paths for app inference."""
    return {
        "scaler": MODELS_DIR / f"scaler_reg_{ticker}.joblib",
        "xgb_reg": MODELS_DIR / f"xgb_reg_{ticker}.joblib",
        "xgb_clf": MODELS_DIR / f"xgb_clf_{ticker}.joblib",
        "lstm": MODELS_DIR / f"lstm_{ticker}.keras",
        "lstm_scaler": MODELS_DIR / f"lstm_scaler_{ticker}.joblib",
    }
