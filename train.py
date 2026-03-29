#!/usr/bin/env python3
"""Train all models and write metrics + artifacts under data/artifacts/models/."""

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
    p = argparse.ArgumentParser(description="Train multi-stock ML pipeline")
    p.add_argument("--no-sentiment", action="store_true", help="Disable sentiment features (ablation)")
    p.add_argument("--no-combined", action="store_true", help="Skip pooled multi-stock model")
    p.add_argument("--no-per-ticker", action="store_true", help="Skip per-ticker models")
    p.add_argument("--lookback-years", type=int, default=None)
    p.add_argument("--crash-threshold", type=float, default=None, help="Next-day drop threshold, e.g. 0.05")
    p.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Limit number of tickers (for quick smoke tests)",
    )
    p.add_argument("--no-mlflow", action="store_true", help="Disable MLflow experiment logging")
    p.add_argument("--no-walk-forward", action="store_true", help="Skip walk-forward CV (faster)")
    args = p.parse_args()

    result = run_training(
        combined_model=not args.no_combined,
        per_ticker=not args.no_per_ticker,
        include_sentiment=not args.no_sentiment,
        lookback_years=args.lookback_years,
        crash_threshold=args.crash_threshold,
        max_tickers=args.max_tickers,
        use_mlflow=not args.no_mlflow,
        walk_forward=not args.no_walk_forward,
    )
    logger.info("Done. Metrics: %s", result.artifacts.get("metrics_json"))


if __name__ == "__main__":
    main()
