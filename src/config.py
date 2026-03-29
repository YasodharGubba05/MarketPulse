"""Paths, defaults, and ticker universe loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
SCALERS_DIR = ARTIFACTS_DIR / "scalers"
TICKERS_FILE = DATA_DIR / "tickers.yaml"

# Training defaults
DEFAULT_LOOKBACK_YEARS = 5
CRASH_THRESHOLD = float(os.environ.get("CRASH_THRESHOLD", "0.05"))
REGRESSION_TARGET = "target_next_close"
CLASS_LABEL = "crash_next_day"
LAG_DAYS = 5
ROLL_WINDOWS = (5, 10, 20)
VOL_WINDOW = 20
LSTM_SEQUENCE_LENGTH = 20
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Sentiment
USE_FINBERT = os.environ.get("USE_FINBERT", "1") == "1"
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")


def ensure_dirs() -> None:
    for d in (DATA_DIR, ARTIFACTS_DIR, MODELS_DIR, SCALERS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_ticker_universe(path: Path | None = None) -> list[dict[str, Any]]:
    """Load ticker list from YAML: each item has symbol, company_name, sector."""
    p = path or TICKERS_FILE
    if not p.exists():
        raise FileNotFoundError(f"Ticker config not found: {p}")
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return list(raw.get("tickers", []))


def symbols_from_universe(entries: list[dict[str, Any]]) -> list[str]:
    return [e["symbol"].upper() for e in entries]
