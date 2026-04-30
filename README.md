# Multi-Stock Price Prediction, Volatility Modeling, and Crash Detection

Applied data science system that combines **time-series ML**, **optional Twitter + news sentiment** (VADER or FinBERT), and **multi-ticker** training for regression (next-day close), classification (crash risk), rolling volatility, **GARCH** diagnostics, and an **LSTM** sequence model.

## Problem statement

Institutional and retail workflows need scalable pipelines that:

- Pull OHLCV for **arbitrary tickers** (not hardcoded in code).
- Engineer technical and sentiment features, align them by date, and train models **per ticker** and/or **pooled across tickers** (with a ticker id feature).
- Quantify **downside risk** (next-day large drop) and **volatility** alongside point forecasts.

This project demonstrates that end-to-end design with a small **Streamlit** front end focused on ML outputs.

## Multi-stock approach

- **Universe**: Edit `data/tickers.yaml` — each row has `symbol`, `company_name`, and `sector`. The training script and app load this file; no ticker lists are embedded in Python source.
- **Per-ticker models**: Separate scalers, regressors, classifiers, LSTM, and volatility XGBoost per symbol under `data/artifacts/models/`.
- **Combined model**: Optional pooled **XGBoost** regressor/classifier with a numeric **`ticker_id`** feature (see `src/pipeline.py`).

## Data sources

| Source | Usage |
|--------|--------|
| **yfinance** | Daily OHLCV (default ~5 years, configurable). |
| **Twitter API v2** | Optional: set `TWITTER_BEARER_TOKEN`; queries like `"AAPL stock"`, `"Tesla stock"`. |
| **yfinance news** | Headlines used when Twitter is unavailable or as extra text. |

If no text is collected for a symbol, sentiment defaults to **0**; you can also compare **with vs without** sentiment via the ablation block in training metrics.

## ML + NLP integration

- **Text**: Cleaning (URLs, mentions, hashtags), tokenization, stopwords, lemmatization (`nltk`).
- **Sentiment**: **VADER** (fast baseline); **FinBERT** (`ProsusAI/finbert`) when `USE_FINBERT=1` (default) and `SKIP_FINBERT` is not set.
- **Aggregation**: Per-day mean sentiment and tweet count, merged on `(Date, ticker)` with prices.

## Feature engineering

Implemented in `src/preprocessing.py` and `src/feature_engineering.py`:

- SMA/EMA, RSI, MACD, Bollinger Bands, daily returns, rolling volatility (annualized).
- Lags of close and returns, rolling mean/std of returns.
- Sentiment level and **sentiment momentum** (day-over-day change).

## Models

| Task | Models |
|------|--------|
| Regression (next close) | Linear Regression, Random Forest, **XGBoost**, **LSTM** |
| Crash (next-day return &lt; −threshold) | Logistic Regression, Random Forest, **XGBoost** |
| Volatility | Rolling std; **GARCH(1,1)** (AIC/BIC in metrics); XGBoost on absolute next-day return |
| Evaluation | RMSE, MAE, R²; accuracy, precision, recall, F1, ROC-AUC; optional **SHAP** for tree regressors |

**Advanced / interview hooks**: time-based holdout, ablation JSON (`with_sentiment` vs `without_sentiment`), hyperparameters centralized in `src/models.py` (extend with `GridSearchCV` or Optuna as needed).

## Project layout

```
data/
  tickers.yaml          # Universe (editable)
  artifacts/models/     # Created by training (gitignored recommended)
notebooks/              # Exploratory work
src/
  config.py
  data_loader.py
  preprocessing.py
  feature_engineering.py
  sentiment.py
  models.py
  evaluation.py
  pipeline.py
  inference.py
app.py
train.py
requirements.txt
```

## Setup

```bash
cd ads-project
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Use this same environment for `streamlit run app.py`. If you use **Anaconda** instead, install deps there too: `pip install -r requirements.txt` (includes optional `arch` for GARCH during training).

Optional:

- **macOS + XGBoost:** if you see `libomp.dylib` / OpenMP errors, run `brew install libomp`, then restart the terminal and try `python -c "import xgboost"` again.
- `export TWITTER_BEARER_TOKEN=...` for Twitter recent search.
- `export SKIP_FINBERT=1` for faster training using VADER only.
- `export CRASH_THRESHOLD=0.05` (default) for crash labels.

## Train

```bash
python train.py
# Quick smoke test (fewer tickers, shorter history):
SKIP_FINBERT=1 python train.py --max-tickers 3 --lookback-years 3
```

Artifacts: `data/artifacts/models/metrics.json` plus joblib models per ticker. **LSTM** is saved as `lstm_<TICKER>.keras` when TensorFlow is installed (Python 3.10–3.12), or as `lstm_<TICKER>.pt` when using **PyTorch** (typical on Python 3.13+ where TensorFlow may be unavailable).

## Streamlit app

```bash
cd ads-project
source .venv/bin/activate
streamlit run app.py
```

Always use the **project venv** (not a bare Anaconda base env) so dependencies match `requirements.txt`.

The UI keeps **LSTM off by default** (sidebar checkbox). Loading PyTorch/TensorFlow inside Streamlit can **segfault** on some macOS setups; XGBoost is also **lazy-loaded** during training only so the app does not import it at startup.

Select a ticker, inspect forecast next close, volatility, sentiment (VADER in the app for stability), crash probability, and optional multi-ticker comparison. Enable **Include LSTM** only if you need the green LSTM line.

## Results across stocks

After training, open `metrics.json` to compare **per-ticker** regression/classification metrics and **combined** XGBoost scores. Use ablation entries to discuss whether sentiment helped for each symbol.

## Key insights 

- **Leakage control**: Time-based split; targets are **next-day** close and forward returns.
- **Scalability**: New ticker = YAML row + retrain; combined model uses **ticker_id** for cross-sectional learning.
- **NLP pragmatics**: Real APIs are rate-limited — the pipeline degrades gracefully (news-only or zero sentiment).


