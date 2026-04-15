#!/usr/bin/env python3
"""Train all models and write metrics + artifacts under data/artifacts/models/.

Models trained:
  Regression:     Linear, RandomForest, XGBoost, LightGBM, Stacking Ensemble
  Classification: Logistic, RandomForest, XGBoost, LightGBM, Stacking Ensemble
  Volatility:     XGBoost-vol, GARCH(1,1)
  Deep learning:  LSTM (Keras/TF → PyTorch fallback, stacked 2-layer)
  Extras:         Walk-forward 5-fold CV, SHAP, Sentiment ablation, MLflow
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train")


def main() -> None:
    p = argparse.ArgumentParser(description="Train MarketPulse multi-stock ML pipeline")
    p.add_argument("--no-sentiment", action="store_true", help="Disable sentiment features (ablation)")
    p.add_argument("--no-combined", action="store_true", help="Skip pooled multi-stock model")
    p.add_argument("--no-per-ticker", action="store_true", help="Skip per-ticker models")
    p.add_argument("--lookback-years", type=int, default=None)
    p.add_argument("--crash-threshold", type=float, default=None, help="Next-day drop threshold, e.g. 0.05")
    p.add_argument(
        "--max-tickers", type=int, default=None,
        help="Limit number of tickers (for quick smoke tests, e.g. --max-tickers 2)",
    )
    p.add_argument("--no-mlflow", action="store_true", help="Disable MLflow experiment logging")
    p.add_argument("--no-walk-forward", action="store_true", help="Skip walk-forward CV (faster)")
    p.add_argument("--optuna", action="store_true", help="Enable Optuna hyperparameter tuning (slower)")
    args = p.parse_args()

    logger.info("=" * 60)
    logger.info("MarketPulse — ML Training Pipeline")
    logger.info("  Tickers: %s | Lookback: %s yrs | Sentiment: %s",
                args.max_tickers or "all", args.lookback_years or "default",
                not args.no_sentiment)
    logger.info("  Optuna tuning: %s | Walk-forward CV: %s",
                args.optuna, not args.no_walk_forward)
    logger.info("=" * 60)

    result = run_training(
        combined_model=not args.no_combined,
        per_ticker=not args.no_per_ticker,
        include_sentiment=not args.no_sentiment,
        lookback_years=args.lookback_years,
        crash_threshold=args.crash_threshold,
        max_tickers=args.max_tickers,
        use_mlflow=not args.no_mlflow,
        walk_forward=not args.no_walk_forward,
        run_optuna=args.optuna,
    )
    logger.info("=" * 60)
    logger.info("✅ Training complete!")
    logger.info("   Metrics: %s", result.artifacts.get("metrics_json"))
    if result.artifacts.get("mlflow_note"):
        logger.info("   MLflow:  %s", result.artifacts["mlflow_note"])
    logger.info("   Run app: streamlit run app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
