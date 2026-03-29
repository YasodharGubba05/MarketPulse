"""Text cleaning, VADER / FinBERT sentiment, Twitter + news ingestion, daily aggregation."""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yfinance as yf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import TWITTER_BEARER_TOKEN, USE_FINBERT

logger = logging.getLogger(__name__)

_lemma: WordNetLemmatizer | None = None
_stop: set[str] | None = None
_vader: SentimentIntensityAnalyzer | None = None
_finbert_pipe: Any = None


def _ensure_nltk() -> None:
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger", quiet=True)


def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text.lower()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"#(\w+)", r"\1", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize_lemmatize(text: str) -> str:
    global _lemma, _stop
    _ensure_nltk()
    if _lemma is None:
        _lemma = WordNetLemmatizer()
    if _stop is None:
        _stop = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    out = [_lemma.lemmatize(w) for w in tokens if w not in _stop and len(w) > 1]
    return " ".join(out)


def vader_compound(text: str) -> float:
    global _vader
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()
    cleaned = clean_text(text)
    if not cleaned:
        return 0.0
    return float(_vader.polarity_scores(cleaned)["compound"])


def get_finbert_pipeline():
    global _finbert_pipe
    if _finbert_pipe is not None:
        return _finbert_pipe
    from transformers import pipeline

    # Financial sentiment (ProsusAI/finbert uses pos/neg/neu labels)
    _finbert_pipe = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        device=-1,
    )
    return _finbert_pipe


def finbert_score(text: str) -> float:
    """Map FinBERT label to [-1, 1]."""
    cleaned = clean_text(text)
    if not cleaned:
        return 0.0
    try:
        pipe = get_finbert_pipeline()
        r = pipe(cleaned[:512], truncation=True)[0]
        label = str(r["label"]).lower()
        conf = float(r["score"])
        if label.startswith("pos") or "positive" in label:
            return conf
        if label.startswith("neg") or "negative" in label:
            return -conf
        return 0.0
    except Exception as e:
        logger.debug("FinBERT fallback to VADER: %s", e)
        return vader_compound(text)


def score_text(text: str, use_transformer: bool = USE_FINBERT) -> float:
    if use_transformer and os.environ.get("SKIP_FINBERT", "0") != "1":
        return finbert_score(text)
    return vader_compound(text)


def fetch_yfinance_news_headlines(symbol: str, max_items: int = 50) -> list[tuple[datetime | None, str]]:
    """Return (published, title) from yfinance Ticker.news."""
    try:
        t = yf.Ticker(symbol)
        news = t.news or []
    except Exception as e:
        logger.warning("yfinance news failed for %s: %s", symbol, e)
        return []
    out: list[tuple[datetime | None, str]] = []
    for item in news[:max_items]:
        title = item.get("title") or ""
        ts = item.get("providerPublishTime")
        dt = datetime.utcfromtimestamp(ts) if ts else None
        if title:
            out.append((dt, title))
    return out


def fetch_twitter_recent(
    query: str,
    max_results: int = 100,
) -> list[tuple[datetime | None, str]]:
    """Twitter API v2 recent search; requires TWITTER_BEARER_TOKEN."""
    if not TWITTER_BEARER_TOKEN:
        return []
    try:
        import tweepy

        client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
        resp = client.search_recent_tweets(
            query=query,
            max_results=min(max_results, 100),
            tweet_fields=["created_at", "text"],
        )
        if not resp.data:
            return []
        rows: list[tuple[datetime | None, str]] = []
        for tw in resp.data:
            rows.append((tw.created_at, tw.text or ""))
        return rows
    except Exception as e:
        logger.warning("Twitter fetch failed for %s: %s", query, e)
        return []


def collect_texts_for_ticker(
    symbol: str,
    company_name: str,
    max_twitter: int = 100,
) -> list[tuple[datetime | None, str]]:
    """Keywords: '<SYMBOL> stock', '<Company> stock'; merge Twitter + news."""
    sym = symbol.upper()
    texts: list[tuple[datetime | None, str]] = []
    for q in (f"{sym} stock", f"{company_name} stock"):
        texts.extend(fetch_twitter_recent(q, max_results=max_twitter // 2))
    for dt, title in fetch_yfinance_news_headlines(sym):
        texts.append((dt, title))
    return texts


def texts_to_daily_dataframe(
    symbol: str,
    texts: Sequence[tuple[datetime | None, str]],
    use_transformer: bool = USE_FINBERT,
) -> pd.DataFrame:
    """Aggregate sentiment by calendar day."""
    if not texts:
        return pd.DataFrame(columns=["Date", "ticker", "sentiment_mean", "sentiment_count"])

    by_day: dict[str, list[float]] = defaultdict(list)
    for dt, raw in texts:
        day = (dt or datetime.utcnow()).strftime("%Y-%m-%d") if dt else datetime.utcnow().strftime("%Y-%m-%d")
        proc = tokenize_lemmatize(clean_text(raw)) or clean_text(raw)
        if not proc:
            continue
        by_day[day].append(score_text(proc, use_transformer=use_transformer))

    rows = []
    for day, scores in sorted(by_day.items()):
        arr = np.array(scores, dtype=float)
        rows.append(
            {
                "Date": pd.Timestamp(day).normalize(),
                "ticker": symbol.upper(),
                "sentiment_mean": float(arr.mean()),
                "sentiment_count": float(len(arr)),
            }
        )
    return pd.DataFrame(rows)


def sentiment_from_price_proxy(df_ticker: pd.DataFrame) -> pd.DataFrame:
    """
    When no text is available, derive a smooth proxy from past returns (demo / ablation).
    Not used as default if real texts exist.
    """
    g = df_ticker.sort_values("Date").copy()
    r = g["Close"].pct_change().fillna(0.0)
    proxy = np.tanh(r.rolling(5).mean() * 50)
    out = pd.DataFrame(
        {
            "Date": g["Date"].values,
            "ticker": g["ticker"].iloc[0],
            "sentiment_mean": proxy.fillna(0.0).values,
            "sentiment_count": 0.0,
        }
    )
    return out


def build_daily_sentiment(
    symbol: str,
    company_name: str,
    price_df: pd.DataFrame | None = None,
    use_transformer: bool = USE_FINBERT,
) -> pd.DataFrame:
    """
    Full pipeline: collect texts, score, aggregate by day.
    If no texts, fall back to price proxy when price_df is provided.
    """
    texts = collect_texts_for_ticker(symbol, company_name)
    daily = texts_to_daily_dataframe(symbol, texts, use_transformer=use_transformer)
    if daily.empty and price_df is not None:
        sub = price_df[price_df["ticker"] == symbol.upper()]
        if not sub.empty:
            daily = sentiment_from_price_proxy(sub)
    return daily


def merge_sentiment_to_prices(
    prices: pd.DataFrame,
    sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """Left join prices with daily sentiment on Date + ticker."""
    if sentiment.empty:
        p = prices.copy()
        p["sentiment_mean"] = 0.0
        p["sentiment_count"] = 0.0
        return p
    s = sentiment.copy()
    s["Date"] = pd.to_datetime(s["Date"]).dt.normalize()
    p = prices.copy()
    p["Date"] = pd.to_datetime(p["Date"]).dt.normalize()
    return p.merge(s, on=["Date", "ticker"], how="left")
