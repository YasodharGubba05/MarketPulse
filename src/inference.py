"""Load artifacts and produce predictions + plots for the Streamlit app.

Does not import src.models at module load (avoids XGBoost/torch); LSTM helpers load lazily.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import LSTM_SEQUENCE_LENGTH, MODELS_DIR, REGRESSION_TARGET, load_ticker_universe
from src.data_loader import download_ohlcv
from src.feature_engineering import build_panel_features, drop_na_targets_and_features, feature_columns, make_sequences
from src.preprocessing import sort_and_fill_ohlcv
from src.sentiment import build_daily_sentiment

logger = logging.getLogger(__name__)


def load_metrics() -> dict[str, Any] | None:
    p = MODELS_DIR / "metrics.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def predict_for_ticker(
    ticker: str,
    include_sentiment: bool = True,
    lookback_years: int = 5,
    *,
    include_lstm: bool = False,
    use_transformer_sentiment: bool = False,
) -> dict[str, Any]:
    """
    Download fresh data, rebuild features, load saved regressors, return series for plots.

    include_lstm: if True, loads TensorFlow or PyTorch LSTM (heavy; can crash some Streamlit setups).
    use_transformer_sentiment: FinBERT path; keep False in the app for speed/stability (VADER only).
    """
    ticker = ticker.upper().strip()
    universe = load_ticker_universe()
    name_map = {e["symbol"].upper(): e.get("company_name", e["symbol"]) for e in universe}
    company = name_map.get(ticker, ticker)

    raw = download_ohlcv([ticker], lookback_years=lookback_years)
    raw = sort_and_fill_ohlcv(raw)
    sent = build_daily_sentiment(
        ticker,
        company,
        price_df=raw,
        use_transformer=use_transformer_sentiment,
    )
    panel = build_panel_features(raw, sent if not sent.empty else None)

    feat_cols = feature_columns(include_sentiment=include_sentiment, include_ticker_id=False)
    sub = panel[panel["ticker"] == ticker].sort_values("Date")
    sub = drop_na_targets_and_features(sub, feat_cols)
    if sub.empty or len(sub) < 30:
        raise ValueError(f"Not enough data for {ticker}")

    scaler_path = MODELS_DIR / f"scaler_reg_{ticker}.joblib"
    xgb_r_path = MODELS_DIR / f"xgb_reg_{ticker}.joblib"
    xgb_c_path = MODELS_DIR / f"xgb_clf_{ticker}.joblib"
    lstm_keras_path = MODELS_DIR / f"lstm_{ticker}.keras"
    lstm_torch_path = MODELS_DIR / f"lstm_{ticker}.pt"
    lstm_sc_path = MODELS_DIR / f"lstm_scaler_{ticker}.joblib"

    if not scaler_path.exists() or not xgb_r_path.exists():
        raise FileNotFoundError(
            f"Train models first (missing {scaler_path} or {xgb_r_path}). Run: python train.py"
        )

    sc = joblib.load(scaler_path)
    xgb_r = joblib.load(xgb_r_path)
    xgb_c = joblib.load(xgb_c_path) if xgb_c_path.exists() else None

    X_all = sub[feat_cols].values
    X_s = sc.transform(X_all)
    pred_price = xgb_r.predict(X_s)
    actual_next = sub[REGRESSION_TARGET].values.astype(float)

    crash_proba = None
    if xgb_c is not None:
        crash_proba = xgb_c.predict_proba(X_s)[:, 1]

    lstm_pred = None
    if include_lstm and lstm_sc_path.exists() and (lstm_keras_path.exists() or lstm_torch_path.exists()):
        seq_len = LSTM_SEQUENCE_LENGTH
        X_seq, _y_seq = make_sequences(sub, feat_cols, seq_len)
        if len(X_seq) > 0:
            sc_l = joblib.load(lstm_sc_path)
            ns, nt, nf = X_seq.shape
            flat = X_seq.reshape(-1, nf)
            X_seq_s = sc_l.transform(flat).reshape(ns, nt, nf)
            try:
                if lstm_keras_path.exists():
                    from src.models import load_keras_model

                    lstm = load_keras_model(lstm_keras_path)
                    lstm_pred = lstm.predict(X_seq_s, verbose=0).ravel()
                else:
                    from src.models import load_torch_lstm, predict_lstm_torch

                    lstm, device, _ = load_torch_lstm(lstm_torch_path)
                    lstm_pred = predict_lstm_torch(lstm, X_seq_s, device)
            except Exception as e:
                logger.warning("LSTM inference skipped: %s", e)

    dates = sub["Date"].values
    actual = sub["Close"].values
    out = {
        "dates": pd.to_datetime(dates),
        "actual_close": actual,
        "actual_next_close": actual_next,
        "predicted_next_close_xgb": pred_price,
        "volatility_roll": sub["volatility_roll"].values,
        "sentiment_mean": sub["sentiment_mean"].values,
        "crash_probability": crash_proba,
        "lstm_predicted": lstm_pred,
        "last_date": sub["Date"].iloc[-1],
        "forecast_next_close_xgb": float(pred_price[-1]),
        "crash_risk_now": float(crash_proba[-1]) if crash_proba is not None else None,
    }
    return out
