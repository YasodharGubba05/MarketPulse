"""
MarketPulse — Multi-Stock ML Lab
Industry-grade Applied Data Science Dashboard

Run:  streamlit run app.py  (use project .venv)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.config import load_ticker_universe, symbols_from_universe
from src.inference import load_metrics, predict_for_ticker

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MarketPulse — ML Lab",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Premium CSS — dark gradient theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
  }
  section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stMultiSelect label,
  section[data-testid="stSidebar"] .stCheckbox label { color: #94a3b8 !important; font-size: 0.8rem; }

  /* ── Main area ── */
  .main { background: #0f172a; }
  .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

  /* ── Metric cards ── */
  div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e293b 0%, #0f1f3a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(14,165,233,0.15);
  }
  div[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.78rem !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.05em; }
  div[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.5rem !important; font-weight: 700 !important; }
  div[data-testid="stMetricDelta"] { font-size: 0.85rem !important; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #1e293b;
    border-radius: 12px;
    padding: 6px;
    border: 1px solid #334155;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #64748b !important;
    font-weight: 500;
    font-size: 0.85rem;
    padding: 6px 14px;
    background: transparent !important;
    transition: all 0.2s;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0ea5e9, #8b5cf6) !important;
    color: #fff !important;
    font-weight: 600 !important;
  }

  /* ── Cards ── */
  .mp-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f1f3a 100%);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
  }
  .mp-card-accent { border-left: 4px solid #0ea5e9; }
  .mp-card-warning { border-left: 4px solid #f59e0b; }
  .mp-card-danger  { border-left: 4px solid #ef4444; }
  .mp-card-success { border-left: 4px solid #22c55e; }
  .mp-card-purple  { border-left: 4px solid #8b5cf6; }

  /* ── Hero title ── */
  .mp-hero {
    background: linear-gradient(135deg, #0ea5e9 0%, #8b5cf6 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.1;
  }
  .mp-sub { color: #64748b; font-size: 0.95rem; font-weight: 400; margin-top: 0.3rem; }

  /* ── Section headers ── */
  .mp-section {
    color: #e2e8f0;
    font-size: 1.1rem;
    font-weight: 700;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 0.5rem;
    margin: 1.25rem 0 0.75rem 0;
    letter-spacing: -0.01em;
  }

  /* ── Badge / chip ── */
  .mp-badge {
    display: inline-block;
    background: rgba(14,165,233,0.15);
    color: #38bdf8;
    border: 1px solid rgba(14,165,233,0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .mp-badge-green  { background:rgba(34,197,94,.15);  color:#4ade80; border-color:rgba(34,197,94,.3); }
  .mp-badge-red    { background:rgba(239,68,68,.15);   color:#f87171; border-color:rgba(239,68,68,.3); }
  .mp-badge-purple { background:rgba(139,92,246,.15);  color:#a78bfa; border-color:rgba(139,92,246,.3); }
  .mp-badge-amber  { background:rgba(245,158,11,.15);  color:#fbbf24; border-color:rgba(245,158,11,.3); }

  /* ── Table ── */
  .stDataFrame { border-radius: 10px; overflow: hidden; }

  /* ── Divider ── */
  hr { border-color: #1e293b !important; }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #8b5cf6);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.25rem;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.85; }

  /* ── Progress ── */
  .stProgress > div > div { background: linear-gradient(90deg, #0ea5e9, #8b5cf6) !important; border-radius: 99px; }

  /* Hide Streamlit chrome ── */
  #MainMenu, footer { visibility: hidden; }
  header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(15,23,42,0)",
    plot_bgcolor="rgba(30,41,59,0.4)",
    font=dict(family="Inter", color="#cbd5e1"),
    xaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
)

COLOR_PALETTE = {
    "blue":   "#0ea5e9",
    "purple": "#8b5cf6",
    "green":  "#22c55e",
    "amber":  "#f59e0b",
    "red":    "#ef4444",
    "pink":   "#ec4899",
    "teal":   "#14b8a6",
    "indigo": "#6366f1",
}

def pct_fmt(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v*100:.{decimals}f}%"

def fmt(v, decimals=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.{decimals}f}"


@st.cache_data(ttl=600, show_spinner="Fetching predictions…")
def _cached_predict(sym: str, lstm: bool):
    return predict_for_ticker(sym, include_lstm=lstm)


# ──────────────────────────────────────────────────────────────────────────────
# Load universe & metrics
# ──────────────────────────────────────────────────────────────────────────────
universe = load_ticker_universe()
syms = symbols_from_universe(universe)
sector_map = {e["symbol"].upper(): e.get("sector", "—") for e in universe}
company_map = {e["symbol"].upper(): e.get("company_name", e["symbol"]) for e in universe}
mets = load_metrics()

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-size:1.3rem;font-weight:800;color:#f1f5f9;margin-bottom:0">📡 MarketPulse</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.75rem;color:#475569;margin-top:0">Multi-Stock ML Lab · ADS Final Project</p>', unsafe_allow_html=True)
    st.markdown("---")

    ticker = st.selectbox("🔍 Primary Ticker", syms, index=0)
    compare = st.multiselect("🔀 Compare (z-scored)", syms, max_selections=5,
                              default=[], help="Up to 5 tickers overlaid")
    st.markdown("---")
    include_lstm = st.checkbox("⚡ Include LSTM curve", value=False,
                                help="Heavy — disable if Streamlit crashes on macOS.")
    st.markdown("---")

    st.markdown("**🏋️ Training Controls**")
    run_training_btn = st.button("▶ Run Training Pipeline", use_container_width=True)
    with st.expander("Advanced options"):
        max_tickers = st.slider("Max tickers", 1, len(syms), len(syms))
        lookback = st.slider("Lookback years", 1, 10, 5)
        run_optuna = st.checkbox("Optuna tuning (slow)", value=False)
        run_wf = st.checkbox("Walk-forward CV", value=True)
        use_sentiment = st.checkbox("Sentiment features", value=True)

    if st.button("🔄 Clear cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown('<p style="color:#334155;font-size:0.7rem">Research / educational use only.<br>Not investment advice.</p>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Inline training
# ──────────────────────────────────────────────────────────────────────────────
if run_training_btn:
    from src.pipeline import run_training

    st.markdown("---")
    st.markdown("### 🏋️ Training Pipeline")
    prog_bar = st.progress(0, text="Initialising…")
    status_box = st.empty()
    _ticker_count = [0]
    _total = max_tickers

    def _progress_callback(msg: str, pct: float):
        overall = (_ticker_count[0] + pct) / max(_total, 1)
        prog_bar.progress(min(overall, 1.0), text=f"[{ticker}] {msg}")
        status_box.markdown(f'<p style="color:#64748b;font-size:0.8rem">{msg}</p>', unsafe_allow_html=True)

    try:
        result = run_training(
            combined_model=True,
            per_ticker=True,
            include_sentiment=use_sentiment,
            lookback_years=lookback,
            max_tickers=max_tickers,
            use_mlflow=True,
            walk_forward=run_wf,
            run_optuna=run_optuna,
            progress_callback=_progress_callback,
        )
        prog_bar.progress(1.0, text="✅ Training complete!")
        mets = result.metrics  # refresh in-session
        st.cache_data.clear()
        st.success(f"Training complete! Metrics saved to `{result.artifacts.get('metrics_json')}`")
        if result.artifacts.get("mlflow_note"):
            st.info(f"MLflow: `{result.artifacts['mlflow_note']}`")
    except Exception as e:
        st.exception(e)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Hero
# ──────────────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(f'<div class="mp-hero">MarketPulse ML Lab</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="mp-sub">Next-day price · crash risk · volatility · sentiment · '
        f'walk-forward CV · backtesting · SHAP · Monte Carlo</div>',
        unsafe_allow_html=True
    )
with col_h2:
    st.markdown(
        f'<div style="text-align:right;padding-top:0.5rem">'
        f'<span class="mp-badge">{sector_map.get(ticker, "—")}</span>&nbsp;'
        f'<span class="mp-badge mp-badge-purple">{company_map.get(ticker, ticker)}</span>'
        f'</div>', unsafe_allow_html=True
    )

# ──────────────────────────────────────────────────────────────────────────────
# Load predictions
# ──────────────────────────────────────────────────────────────────────────────
try:
    out = _cached_predict(ticker, include_lstm)
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Run `python train.py` first, then click **Refresh cache**.")
    st.stop()
except Exception as e:
    st.exception(e)
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "📈 Forecasting",
    "💥 Crash & Volatility",
    "🏆 Model Leaderboard",
    "🔁 Backtesting & Risk",
    "🧠 Explainability",
    "🌐 Portfolio & Correlation",
    "🔬 Training Lab",
    "ℹ️ About",
])

tab_overview, tab_forecast, tab_crash, tab_leader, tab_backtest, tab_shap, tab_portfolio, tab_train, tab_about = tabs


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    pt = (mets or {}).get("per_ticker", {}).get(ticker, {}) if mets else {}

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Latest Date", str(pd.Timestamp(out["last_date"]).date()))
    with c2:
        st.metric("Forecast Next Close", f"${out['forecast_next_close_xgb']:.2f}")
    with c3:
        last_close = float(out["actual_close"][-1]) if len(out["actual_close"]) else 0
        delta = out['forecast_next_close_xgb'] - last_close
        st.metric("Predicted Δ", f"${delta:+.2f}", delta_color="normal")
    with c4:
        cr = out["crash_risk_now"]
        cr_label = "🔴 HIGH" if (cr or 0) > 0.4 else ("🟡 MED" if (cr or 0) > 0.2 else "🟢 LOW")
        st.metric("Crash Risk", f"{cr:.3f} {cr_label}" if cr is not None else "n/a")
    with c5:
        vol_now = out["volatility_roll"][-1] if len(out["volatility_roll"]) else 0
        st.metric("Rolling Vol (ann.)", f"{vol_now*100:.1f}%" if np.isfinite(vol_now) else "n/a")

    st.markdown("---")

    # Mini model health summary
    if pt:
        st.markdown('<div class="mp-section">Model Health Summary</div>', unsafe_allow_html=True)
        reg = pt.get("regression", {})
        clf = pt.get("classification", {})
        mcols = st.columns(4)
        model_health = [
            ("XGBoost Reg", reg.get("xgboost", {}).get("RMSE"), "RMSE"),
            ("LightGBM Reg", reg.get("lightgbm", {}).get("RMSE"), "RMSE"),
            ("Ensemble Stacking", reg.get("ensemble_stacking", {}).get("RMSE"), "RMSE"),
            ("XGBoost Crash F1", clf.get("xgboost", {}).get("f1"), "F1"),
        ]
        for col, (name, val, label) in zip(mcols, model_health):
            with col:
                st.metric(name, fmt(val, 4) if val is not None else "n/a", help=f"Holdout {label}")

    # Ablation card
    abl = (mets or {}).get("ablation", {}).get(ticker, {}) if mets else {}
    if abl:
        st.markdown('<div class="mp-section">Sentiment Ablation Study</div>', unsafe_allow_html=True)
        ac1, ac2 = st.columns(2)
        with_sent = abl.get("with_sentiment", {})
        without_sent = abl.get("without_sentiment", {})
        with ac1:
            st.markdown(
                f'<div class="mp-card mp-card-accent">'
                f'<b style="color:#0ea5e9">With Sentiment Features</b><br>'
                f'RMSE: <b style="color:#f1f5f9">{fmt(with_sent.get("RMSE"), 4)}</b> &nbsp; '
                f'R²: <b style="color:#f1f5f9">{fmt(with_sent.get("R2"), 4)}</b> &nbsp; '
                f'Dir Acc: <b style="color:#f1f5f9">{fmt(with_sent.get("Dir_Acc"), 3)}</b>'
                f'</div>', unsafe_allow_html=True
            )
        with ac2:
            st.markdown(
                f'<div class="mp-card mp-card-warning">'
                f'<b style="color:#f59e0b">Without Sentiment Features</b><br>'
                f'RMSE: <b style="color:#f1f5f9">{fmt(without_sent.get("RMSE"), 4)}</b> &nbsp; '
                f'R²: <b style="color:#f1f5f9">{fmt(without_sent.get("R2"), 4)}</b> &nbsp; '
                f'Dir Acc: <b style="color:#f1f5f9">{fmt(without_sent.get("Dir_Acc"), 3)}</b>'
                f'</div>', unsafe_allow_html=True
            )

    # GARCH info
    garch = pt.get("garch", {}) if pt else {}
    if garch:
        st.markdown(
            f'<div class="mp-card mp-card-purple" style="margin-top:1rem">'
            f'⚡ <b style="color:#a78bfa">GARCH(1,1)</b>&nbsp;&nbsp;'
            f'AIC: <b style="color:#f1f5f9">{garch.get("aic", "n/a"):.4f}</b>&nbsp;&nbsp;'
            f'BIC: <b style="color:#f1f5f9">{garch.get("bic", "n/a"):.4f}</b>'
            f'</div>', unsafe_allow_html=True
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════
with tab_forecast:
    st.markdown('<div class="mp-section">Price vs Forecast</div>', unsafe_allow_html=True)

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=out["dates"], y=out["actual_close"], name="Close (actual)",
        line=dict(color=COLOR_PALETTE["blue"], width=2)
    ))
    fig_price.add_trace(go.Scatter(
        x=out["dates"], y=out["actual_next_close"], name="Next close (actual)",
        line=dict(color="#bae6fd", width=1, dash="dot")
    ))
    fig_price.add_trace(go.Scatter(
        x=out["dates"], y=out["predicted_next_close_xgb"], name="XGB Forecast",
        line=dict(color=COLOR_PALETTE["amber"], dash="dot", width=2)
    ))
    if out.get("lstm_predicted") is not None:
        lstm_dates = out["dates"].values[-len(out["lstm_predicted"]):]
        fig_price.add_trace(go.Scatter(
            x=lstm_dates, y=out["lstm_predicted"], name="LSTM Forecast",
            line=dict(color=COLOR_PALETTE["green"], width=2)
        ))
    fig_price.update_layout(
        **PLOTLY_DARK,
        title=f"{ticker} — Historical Close & Next-Day Forecasts",
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, l=60, r=20, b=40),
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # Residuals
    actual = np.array(out["actual_next_close"])
    pred = np.array(out["predicted_next_close_xgb"])
    residuals = actual - pred
    valid = np.isfinite(residuals)

    r1, r2 = st.columns(2)
    with r1:
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=out["dates"][valid], y=residuals[valid], name="Residuals",
            mode="markers", marker=dict(color=COLOR_PALETTE["purple"], size=4, opacity=0.7)
        ))
        fig_res.add_hline(y=0, line_dash="dash", line_color="#ef4444")
        fig_res.update_layout(**PLOTLY_DARK, title="Prediction Residuals (Actual − Pred)",
                               height=320, margin=dict(t=50))
        st.plotly_chart(fig_res, use_container_width=True)
    with r2:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=residuals[valid], nbinsx=40, name="Residual dist.",
            marker_color=COLOR_PALETTE["indigo"], opacity=0.8
        ))
        fig_hist.update_layout(**PLOTLY_DARK, title="Residual Distribution",
                                height=320, margin=dict(t=50))
        st.plotly_chart(fig_hist, use_container_width=True)

    # Comparison
    if compare:
        st.markdown('<div class="mp-section">Normalized Price Comparison</div>', unsafe_allow_html=True)
        fig_cmp = go.Figure()
        colors = list(COLOR_PALETTE.values())
        for i, t in enumerate([ticker] + list(compare)):
            try:
                o = _cached_predict(t, include_lstm)
                s = pd.Series(o["actual_close"], dtype=float)
                normed = (s - s.mean()) / (s.std() or 1.0)
                fig_cmp.add_trace(go.Scatter(
                    x=o["dates"], y=normed, name=t,
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
            except Exception:
                continue
        fig_cmp.update_layout(**PLOTLY_DARK, title="Z-Scored Close Prices",
                               height=380, margin=dict(t=50))
        st.plotly_chart(fig_cmp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CRASH & VOLATILITY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_crash:
    v1, v2 = st.columns(2)

    with v1:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=out["dates"], y=out["volatility_roll"],
            name="Rolling Vol (ann.)", fill="tozeroy",
            line=dict(color=COLOR_PALETTE["purple"], width=2),
            fillcolor="rgba(139,92,246,0.12)"
        ))
        vol_mean = np.nanmean(out["volatility_roll"])
        fig_vol.add_hline(y=vol_mean, line_dash="dash", line_color="#f59e0b",
                           annotation_text=f"avg {vol_mean:.3f}", annotation_position="top right")
        fig_vol.update_layout(**PLOTLY_DARK, title="Annualized Rolling Volatility (20-day)",
                               height=350, margin=dict(t=50))
        st.plotly_chart(fig_vol, use_container_width=True)

    with v2:
        fig_sent = go.Figure()
        sent = np.array(out["sentiment_mean"])
        colors_sent = [COLOR_PALETTE["green"] if s >= 0 else COLOR_PALETTE["red"] for s in sent]
        fig_sent.add_trace(go.Bar(
            x=out["dates"], y=sent, name="Sentiment",
            marker_color=colors_sent, opacity=0.8
        ))
        fig_sent.add_hline(y=0, line_color="#475569")
        fig_sent.update_layout(**PLOTLY_DARK, title="Daily Sentiment Score (VADER)",
                                height=350, margin=dict(t=50))
        st.plotly_chart(fig_sent, use_container_width=True)

    if out["crash_probability"] is not None:
        fig_cr = go.Figure()
        cp = np.array(out["crash_probability"])
        fig_cr.add_trace(go.Scatter(
            x=out["dates"], y=cp, name="Crash Probability", fill="tozeroy",
            line=dict(color=COLOR_PALETTE["red"], width=2),
            fillcolor="rgba(239,68,68,0.10)"
        ))
        fig_cr.add_hrect(y0=0.35, y1=1.0, fillcolor="rgba(239,68,68,0.05)",
                          line_width=0, annotation_text="High-risk zone", annotation_position="top left")
        fig_cr.add_hline(y=0.35, line_dash="dash", line_color="#ef4444",
                          annotation_text="Threshold 0.35")
        fig_cr.update_layout(**PLOTLY_DARK, title="Crash Model Probability Over Time",
                              height=320, margin=dict(t=50))
        fig_cr.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_cr, use_container_width=True)

    # Returns histogram
    returns = pd.Series(out["actual_close"]).pct_change().dropna()
    fig_ret = go.Figure()
    fig_ret.add_trace(go.Histogram(
        x=returns * 100, nbinsx=60, name="Daily returns",
        marker_color=COLOR_PALETTE["teal"], opacity=0.8
    ))
    fig_ret.add_vline(x=0, line_color="#64748b", line_dash="dash")
    fig_ret.add_vline(x=float(returns.mean() * 100), line_color="#22c55e",
                       line_dash="dot", annotation_text="mean")
    q5 = float(returns.quantile(0.05) * 100)
    fig_ret.add_vline(x=q5, line_color="#ef4444",
                       annotation_text=f"5% VaR: {q5:.2f}%")
    fig_ret.update_layout(**PLOTLY_DARK, title="Daily Return Distribution",
                           height=300, xaxis_title="Return (%)", margin=dict(t=50))
    st.plotly_chart(fig_ret, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL LEADERBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_leader:
    if not mets:
        st.info("Run `python train.py` to populate metrics.")
    else:
        from src.evaluation import leaderboard_df, classification_leaderboard_df

        st.markdown('<div class="mp-section">Regression Model Leaderboard</div>', unsafe_allow_html=True)
        lb_df = leaderboard_df(mets.get("per_ticker", {}), ticker, metric="RMSE")

        if not lb_df.empty:
            # Highlight best ML model vs baselines
            is_baseline = lb_df["Model"].str.startswith("[Baseline]")
            fig_lb = go.Figure()
            fig_lb.add_trace(go.Bar(
                x=lb_df[~is_baseline]["Model"],
                y=lb_df[~is_baseline]["RMSE"],
                name="ML Models",
                marker_color=COLOR_PALETTE["blue"],
                opacity=0.9
            ))
            fig_lb.add_trace(go.Bar(
                x=lb_df[is_baseline]["Model"].str.replace("[Baseline] ", ""),
                y=lb_df[is_baseline]["RMSE"],
                name="Baselines",
                marker_color=COLOR_PALETTE["amber"],
                opacity=0.7
            ))
            fig_lb.update_layout(
                **PLOTLY_DARK,
                title="Holdout RMSE — Lower is Better",
                height=360,
                barmode="group",
                xaxis_tickangle=-20,
                margin=dict(t=50),
            )
            st.plotly_chart(fig_lb, use_container_width=True)

            # Full table
            styled_df = lb_df.copy()
            st.dataframe(
                styled_df.style.background_gradient(subset=["RMSE", "MAE"], cmap="RdYlGn_r")
                               .background_gradient(subset=["R2"], cmap="RdYlGn")
                               .format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "R2": "{:.4f}",
                                        "MAPE_%": "{:.2f}", "Dir_Acc": "{:.3f}"}, na_rep="n/a"),
                use_container_width=True, hide_index=True
            )

        # R² comparison
        r2_rows = [(r["Model"], r["R2"]) for _, r in lb_df.iterrows() if r["R2"] is not None and not np.isnan(r["R2"] or float("nan"))]
        if r2_rows:
            r2_df = pd.DataFrame(r2_rows, columns=["Model", "R2"]).sort_values("R2", ascending=True)
            fig_r2 = go.Figure(go.Bar(
                x=r2_df["R2"], y=r2_df["Model"], orientation="h",
                marker_color=[COLOR_PALETTE["green"] if v > 0 else COLOR_PALETTE["red"] for v in r2_df["R2"]],
            ))
            fig_r2.add_vline(x=0, line_color="#475569")
            fig_r2.update_layout(**PLOTLY_DARK, title="R² Score (Higher is Better)",
                                  height=max(280, len(r2_df) * 40), margin=dict(t=50))
            st.plotly_chart(fig_r2, use_container_width=True)

        st.markdown('<div class="mp-section">Classification Leaderboard (Crash Detection)</div>', unsafe_allow_html=True)
        clf_df = classification_leaderboard_df(mets.get("per_ticker", {}), ticker)
        if not clf_df.empty:
            fig_clf = go.Figure()
            metrics_show = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
            colors_clf = [COLOR_PALETTE["blue"], COLOR_PALETTE["purple"], COLOR_PALETTE["amber"],
                          COLOR_PALETTE["green"], COLOR_PALETTE["pink"]]
            for met_name, col in zip(metrics_show, colors_clf):
                if met_name in clf_df.columns:
                    fig_clf.add_trace(go.Bar(
                        name=met_name,
                        x=clf_df["Model"],
                        y=clf_df[met_name],
                        marker_color=col, opacity=0.85
                    ))
            fig_clf.update_layout(**PLOTLY_DARK, title="Classification Metrics (Holdout)",
                                   barmode="group", height=380,
                                   xaxis_tickangle=-15, margin=dict(t=50))
            st.plotly_chart(fig_clf, use_container_width=True)
            st.dataframe(clf_df.style.format(
                {c: "{:.4f}" for c in ["Accuracy","Precision","Recall","F1","ROC-AUC"] if c in clf_df.columns},
                na_rep="n/a"
            ), use_container_width=True, hide_index=True)

        # Walk-forward CV
        pt = mets.get("per_ticker", {}).get(ticker, {})
        wfv = pt.get("walk_forward_cv", {})
        if wfv and "error" not in wfv:
            st.markdown('<div class="mp-section">Walk-Forward Cross-Validation (5 Folds)</div>', unsafe_allow_html=True)
            reg_folds = wfv.get("regression_rmse_by_fold", {})
            wf_rows = []
            for name, d in reg_folds.items():
                if isinstance(d, dict) and d.get("n_folds", 0):
                    wf_rows.append({
                        "Model": name.replace("_", " ").title(),
                        "Mean RMSE": d.get("mean_rmse"),
                        "Std RMSE":  d.get("std_rmse"),
                        "Folds": d.get("n_folds"),
                    })
            if wf_rows:
                wf_df = pd.DataFrame(wf_rows).sort_values("Mean RMSE")
                st.dataframe(wf_df.style.format({"Mean RMSE": "{:.4f}", "Std RMSE": "{:.4f}"}, na_rep="n/a"),
                             use_container_width=True, hide_index=True)

                fig_wf = go.Figure()
                fig_wf.add_trace(go.Bar(
                    x=wf_df["Model"], y=wf_df["Mean RMSE"], name="Mean RMSE",
                    error_y=dict(type="data", array=wf_df["Std RMSE"].tolist()),
                    marker_color=COLOR_PALETTE["indigo"], opacity=0.85
                ))
                fig_wf.update_layout(**PLOTLY_DARK, title="Walk-Forward Mean RMSE ± Std (5-fold)",
                                      height=340, margin=dict(t=50))
                st.plotly_chart(fig_wf, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — BACKTESTING & RISK
# ═══════════════════════════════════════════════════════════════════════════════
with tab_backtest:
    pt = (mets or {}).get("per_ticker", {}).get(ticker, {}) if mets else {}
    bt = pt.get("backtest_holdout", {}) if pt else {}

    if bt and "error" not in bt:
        st.markdown('<div class="mp-section">Strategy vs Buy-and-Hold</div>', unsafe_allow_html=True)
        b1 = bt.get("long_on_positive_pred_return", {})
        b2 = bt.get("long_positive_avoid_high_crash_risk", {})

        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            st.metric("Strategy Return (XGB long)", pct_fmt(b1.get("strategy_total_return")),
                       delta=pct_fmt(b1.get("strategy_total_return", 0) - b1.get("buy_hold_total_return", 0)))
        with bc2:
            st.metric("Buy & Hold Return", pct_fmt(b1.get("buy_hold_total_return")))
        with bc3:
            st.metric("Sharpe (strategy)", fmt(b1.get("annualized_sharpe"), 2))
        with bc4:
            st.metric("Max Drawdown", pct_fmt(b1.get("max_drawdown")))

        bc5, bc6 = st.columns(2)
        with bc5:
            st.metric("Dir. Hit Rate", pct_fmt(b1.get("directional_hit_rate")))
        with bc6:
            st.metric("+ Avoid Crash Strategy", pct_fmt(b2.get("strategy_total_return")))

        st.caption("⚠️ Educational backtest only — 10 bps cost on turns. Not investment advice.")

    # Monte Carlo
    st.markdown('<div class="mp-section">Monte Carlo Simulation (GBM, 30-day Horizon)</div>', unsafe_allow_html=True)
    try:
        from src.monte_carlo import run_monte_carlo
        mc = run_monte_carlo(out["actual_close"], n_days=30, n_paths=1000)
        summary = mc["summary"]
        last_date = pd.Timestamp(out["last_date"])
        future_dates = pd.date_range(last_date, periods=31, freq="B")

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scatter(
            x=future_dates, y=summary["p5"], name="P5 (bear)",
            line=dict(color=COLOR_PALETTE["red"], dash="dot", width=1),
            showlegend=True
        ))
        fig_mc.add_trace(go.Scatter(
            x=future_dates, y=summary["p95"], name="P95 (bull)",
            fill="tonexty", fillcolor="rgba(14,165,233,0.07)",
            line=dict(color=COLOR_PALETTE["blue"], dash="dot", width=1),
        ))
        fig_mc.add_trace(go.Scatter(
            x=future_dates, y=summary["p25"], name="P25–P75 (IQR)",
            line=dict(color=COLOR_PALETTE["purple"], width=0), showlegend=True
        ))
        fig_mc.add_trace(go.Scatter(
            x=future_dates, y=summary["p75"],
            fill="tonexty", fillcolor="rgba(139,92,246,0.15)",
            line=dict(color=COLOR_PALETTE["purple"], width=0), showlegend=False
        ))
        fig_mc.add_trace(go.Scatter(
            x=future_dates, y=summary["p50"], name="Median (P50)",
            line=dict(color=COLOR_PALETTE["green"], width=2.5)
        ))
        fig_mc.add_trace(go.Scatter(
            x=future_dates, y=summary["mean"], name="Mean path",
            line=dict(color=COLOR_PALETTE["amber"], dash="dash", width=2)
        ))
        fig_mc.update_layout(
            **PLOTLY_DARK,
            title=f"{ticker} — 30-Day GBM Paths (1 000 simulations)",
            height=400, margin=dict(t=55)
        )
        st.plotly_chart(fig_mc, use_container_width=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Current Price (S₀)", f"${mc['S0']:.2f}")
        with mc2:
            st.metric("Expected Return (30d)", pct_fmt(mc["expected_return"]))
        with mc3:
            st.metric("95% VaR (30d)", pct_fmt(mc["var_95"]))
        with mc4:
            st.metric("P(Gain) in 30d", pct_fmt(mc["prob_gain"]))
    except Exception as e:
        st.warning(f"Monte Carlo unavailable: {e}")

    # Equity curve (if backtest data available)
    if bt and "error" not in bt:
        st.markdown('<div class="mp-section">Equity Curve Reconstruction</div>', unsafe_allow_html=True)

        close_arr = np.array(out["actual_close"])
        pred_arr = np.array(out["predicted_next_close_xgb"])
        n_bt = min(len(close_arr), len(pred_arr))
        prices_bt = close_arr[:n_bt]
        preds_bt = pred_arr[:n_bt]
        actual_ret_bk = np.diff(prices_bt) / np.maximum(prices_bt[:-1], 1e-12)
        pred_ret_bk = (preds_bt[:-1] - prices_bt[:-1]) / np.maximum(prices_bt[:-1], 1e-12)
        position = (pred_ret_bk > 0).astype(float)
        strat_ret = position * actual_ret_bk
        equity_ml = np.cumprod(1 + strat_ret)
        equity_bh = np.cumprod(1 + actual_ret_bk)
        equity_dates = out["dates"].values[1:n_bt]

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=equity_dates, y=equity_ml, name="ML Strategy",
                                     line=dict(color=COLOR_PALETTE["green"], width=2)))
        fig_eq.add_trace(go.Scatter(x=equity_dates, y=equity_bh, name="Buy & Hold",
                                     line=dict(color=COLOR_PALETTE["blue"], width=2, dash="dot")))
        fig_eq.add_hline(y=1.0, line_dash="dash", line_color="#475569")
        fig_eq.update_layout(**PLOTLY_DARK, title="Equity Curve (normalized to 1.0)",
                              height=340, margin=dict(t=50))
        st.plotly_chart(fig_eq, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — EXPLAINABILITY (SHAP)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_shap:
    st.markdown('<div class="mp-section">SHAP Feature Importance — XGBoost Regressor</div>', unsafe_allow_html=True)
    pt = (mets or {}).get("per_ticker", {}).get(ticker, {}) if mets else {}
    xgb_shap = pt.get("regression", {}).get("xgboost_shap", {}) if pt else {}
    rf_shap = pt.get("regression", {}).get("random_forest_shap", {}) if pt else {}

    def render_shap_bar(shap_dict: dict, title: str, color: str):
        if not shap_dict or not shap_dict.get("mean_abs_shap"):
            st.info(f"No SHAP data for {title}. Run training first.")
            return
        items = list(shap_dict["mean_abs_shap"].items())
        items.sort(key=lambda x: x[1], reverse=True)
        top_n = items[:20]
        feat_names = [i[0] for i in top_n][::-1]
        shap_vals = [i[1] for i in top_n][::-1]
        fig = go.Figure(go.Bar(
            x=shap_vals, y=feat_names, orientation="h",
            marker=dict(
                color=shap_vals,
                colorscale=[[0, "#1e293b"], [0.5, color], [1.0, "#f1f5f9"]],
                showscale=True,
                colorbar=dict(title="SHAP", thickness=12)
            )
        ))
        fig.update_layout(**PLOTLY_DARK, title=f"{title} — Top 20 Features",
                           height=max(400, len(feat_names) * 22), xaxis_title="Mean |SHAP value|",
                           margin=dict(t=50, l=160))
        st.plotly_chart(fig, use_container_width=True)

    sc1, sc2 = st.columns(2)
    with sc1:
        render_shap_bar(xgb_shap, "XGBoost Reg", COLOR_PALETTE["blue"])
    with sc2:
        render_shap_bar(rf_shap, "Random Forest Reg", COLOR_PALETTE["purple"])

    if not xgb_shap and not rf_shap and not mets:
        st.info("Run `python train.py` to generate SHAP importance values.")

    st.markdown("---")
    st.markdown(
        '<div class="mp-card mp-card-accent">'
        '<b style="color:#0ea5e9">About SHAP</b><br>'
        '<span style="color:#94a3b8;font-size:0.9rem">'
        'SHAP (SHapley Additive exPlanations) decomposes each prediction into feature contributions. '
        'The bar chart shows mean(|SHAP|) across the hold-out test set — features at the top '
        'drive the model\'s dollar-value forecasts the most. '
        'This provides model transparency beyond accuracy metrics alone.'
        '</span></div>',
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — PORTFOLIO & CORRELATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_portfolio:
    st.markdown('<div class="mp-section">Multi-Ticker Portfolio Analytics</div>', unsafe_allow_html=True)

    all_tickers_sel = st.multiselect(
        "Select tickers for portfolio analysis",
        syms, default=syms[:min(6, len(syms))],
        help="Up to all tickers supported."
    )

    if len(all_tickers_sel) < 2:
        st.warning("Select at least 2 tickers for portfolio analysis.")
    else:
        price_dict = {}
        with st.spinner("Loading price data for portfolio analysis…"):
            for t_sym in all_tickers_sel:
                try:
                    o = _cached_predict(t_sym, False)
                    price_dict[t_sym] = o["actual_close"]
                except Exception:
                    continue

        if len(price_dict) >= 2:
            from src.portfolio import portfolio_analytics

            try:
                pa = portfolio_analytics(price_dict, sector_map)
                corr = pa["correlation_matrix"]
                ret_df = pa["returns_df"]

                # Correlation heatmap
                fig_corr = go.Figure(go.Heatmap(
                    z=corr.values,
                    x=corr.columns.tolist(),
                    y=corr.index.tolist(),
                    colorscale=[[0, "#ef4444"], [0.5, "#0f172a"], [1, "#22c55e"]],
                    zmin=-1, zmax=1,
                    text=np.round(corr.values, 2).astype(str),
                    texttemplate="%{text}",
                    colorbar=dict(title="ρ", thickness=14)
                ))
                fig_corr.update_layout(
                    **PLOTLY_DARK,
                    title="Pearson Correlation Matrix (Daily Returns)",
                    height=max(400, len(corr) * 60),
                    margin=dict(t=55)
                )
                st.plotly_chart(fig_corr, use_container_width=True)

                # Portfolio KPIs
                pk1, pk2, pk3 = st.columns(3)
                with pk1:
                    st.metric("Equal-Weight Portfolio Sharpe", fmt(pa["portfolio_sharpe"], 3))
                with pk2:
                    st.metric("Diversification Ratio", fmt(pa["diversification_ratio"], 3),
                               help="Ratio of weighted-avg vol to portfolio vol. > 1 = diversification benefit.")
                with pk3:
                    max_dd_port = min(v["max_drawdown"] for v in pa["ticker_stats"].values()) if pa["ticker_stats"] else float("nan")
                    st.metric("Worst Individual Max DD", pct_fmt(max_dd_port))

                # Sector performance
                sect_df = pa["sector_performance"]
                if not sect_df.empty:
                    st.markdown('<div class="mp-section">Sector Performance Breakdown</div>', unsafe_allow_html=True)
                    fig_sect = px.scatter(
                        sect_df, x="ann_vol", y="ann_return",
                        text="ticker", color="sector",
                        size="total_return",
                        size_max=30,
                        labels={"ann_vol": "Ann. Volatility", "ann_return": "Ann. Return"},
                        title="Risk-Return by Ticker & Sector",
                        template="plotly_dark",
                    )
                    fig_sect.update_layout(**PLOTLY_DARK, height=420, margin=dict(t=55))
                    fig_sect.update_traces(textposition="top center")
                    st.plotly_chart(fig_sect, use_container_width=True)

                # Per-ticker stats table
                stats_df = pd.DataFrame(pa["ticker_stats"]).T.reset_index().rename(columns={"index": "Ticker"})
                stats_df.columns = ["Ticker", "Max Drawdown", "Ann. Vol", "Total Return"]
                st.dataframe(
                    stats_df.style.format({"Max Drawdown": "{:.2%}", "Ann. Vol": "{:.2%}", "Total Return": "{:.2%}"}, na_rep="n/a"),
                    use_container_width=True, hide_index=True
                )
            except Exception as e:
                st.error(f"Portfolio analytics error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 — TRAINING LAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.markdown('<div class="mp-section">ML Training Lab — Detailed Per-Ticker Metrics</div>', unsafe_allow_html=True)

    if not mets:
        st.info("No metrics found. Use the **▶ Run Training Pipeline** button in the sidebar.")
    else:
        per_ticker = mets.get("per_ticker", {})
        combined = mets.get("combined", {})

        # Ticker selector
        trained_tickers = [t for t, v in per_ticker.items() if isinstance(v, dict) and v.get("regression")]
        sel_ticker_train = st.selectbox("Inspect ticker", trained_tickers, key="train_sel")
        pt_data = per_ticker.get(sel_ticker_train, {})

        if pt_data:
            ti1, ti2, ti3 = st.columns(3)
            with ti1:
                st.markdown(f'<div class="mp-card mp-card-accent"><b style="color:#38bdf8">Train rows</b><br>'
                            f'<span style="font-size:2rem;font-weight:700;color:#f1f5f9">{pt_data.get("train_rows","—")}</span></div>',
                            unsafe_allow_html=True)
            with ti2:
                st.markdown(f'<div class="mp-card mp-card-purple"><b style="color:#a78bfa">Test rows</b><br>'
                            f'<span style="font-size:2rem;font-weight:700;color:#f1f5f9">{pt_data.get("test_rows","—")}</span></div>',
                            unsafe_allow_html=True)
            with ti3:
                st.markdown(f'<div class="mp-card mp-card-success"><b style="color:#4ade80">Features</b><br>'
                            f'<span style="font-size:2rem;font-weight:700;color:#f1f5f9">{pt_data.get("n_features","—")}</span></div>',
                            unsafe_allow_html=True)

            # Regression detailed view
            reg = pt_data.get("regression", {})
            if reg:
                st.markdown('<div class="mp-section">Regression Models — Holdout Metrics</div>', unsafe_allow_html=True)
                model_colors = {
                    "linear_regression": COLOR_PALETTE["teal"],
                    "random_forest": COLOR_PALETTE["purple"],
                    "xgboost": COLOR_PALETTE["blue"],
                    "lightgbm": COLOR_PALETTE["green"],
                    "ensemble_stacking": COLOR_PALETTE["pink"],
                }
                reg_cols = st.columns(min(len(model_colors), 5))
                for col, (model_name, color) in zip(reg_cols, model_colors.items()):
                    m = reg.get(model_name, {})
                    if not isinstance(m, dict):
                        continue
                    with col:
                        st.markdown(
                            f'<div class="mp-card" style="border-left:4px solid {color}">'
                            f'<b style="color:{color};font-size:0.8rem">{model_name.replace("_"," ").upper()}</b><br>'
                            f'<span style="color:#64748b;font-size:0.75rem">RMSE</span> '
                            f'<span style="color:#f1f5f9;font-weight:700">{fmt(m.get("RMSE"), 4)}</span><br>'
                            f'<span style="color:#64748b;font-size:0.75rem">MAE</span> '
                            f'<span style="color:#f1f5f9;font-weight:700">{fmt(m.get("MAE"), 4)}</span><br>'
                            f'<span style="color:#64748b;font-size:0.75rem">R²</span> '
                            f'<span style="color:#f1f5f9;font-weight:700">{fmt(m.get("R2"), 4)}</span><br>'
                            f'<span style="color:#64748b;font-size:0.75rem">Dir Acc</span> '
                            f'<span style="color:#f1f5f9;font-weight:700">{fmt(m.get("Dir_Acc"), 3)}</span>'
                            f'</div>', unsafe_allow_html=True
                        )

            # Classification metrics table
            clf = pt_data.get("classification", {})
            if clf:
                st.markdown('<div class="mp-section">Classification Models — Crash Detection</div>', unsafe_allow_html=True)
                clf_rows = []
                for name, m in clf.items():
                    if not isinstance(m, dict): continue
                    clf_rows.append({
                        "Model": name.replace("_", " ").title(),
                        "Accuracy": m.get("accuracy"), "F1": m.get("f1"),
                        "Precision": m.get("precision"), "Recall": m.get("recall"),
                        "ROC-AUC": m.get("roc_auc"),
                    })
                if clf_rows:
                    st.dataframe(pd.DataFrame(clf_rows).style.format(
                        {c: "{:.4f}" for c in ["Accuracy","F1","Precision","Recall","ROC-AUC"]}, na_rep="n/a"
                    ), use_container_width=True, hide_index=True)

            # LSTM
            lstm_data = pt_data.get("lstm", {})
            if lstm_data and isinstance(lstm_data, dict) and "RMSE" in lstm_data:
                backend = lstm_data.get("backend", "?")
                st.markdown(
                    f'<div class="mp-card mp-card-success">'
                    f'🔁 <b style="color:#4ade80">LSTM</b> ({backend})&nbsp;&nbsp;'
                    f'RMSE: <b style="color:#f1f5f9">{fmt(lstm_data.get("RMSE"), 4)}</b>&nbsp;&nbsp;'
                    f'R²: <b style="color:#f1f5f9">{fmt(lstm_data.get("R2"), 4)}</b>'
                    f'</div>', unsafe_allow_html=True
                )
            elif lstm_data and isinstance(lstm_data, dict) and "error" in lstm_data:
                st.markdown(
                    f'<div class="mp-card mp-card-warning">⚠️ LSTM: {lstm_data["error"]}</div>',
                    unsafe_allow_html=True
                )

        # Combined model
        if combined:
            st.markdown('<div class="mp-section">Combined (Pooled Multi-Stock) Model</div>', unsafe_allow_html=True)
            comb_reg = combined.get("combined_regression", {})
            comb_clf = combined.get("combined_classification", {})
            co1, co2 = st.columns(2)
            with co1:
                xgb_c_m = comb_reg.get("xgboost", {})
                st.markdown(
                    f'<div class="mp-card mp-card-accent">'
                    f'<b style="color:#38bdf8">Combined XGB Reg</b><br>'
                    f'RMSE: {fmt(xgb_c_m.get("RMSE"),4)} | R²: {fmt(xgb_c_m.get("R2"),4)}'
                    f'</div>', unsafe_allow_html=True
                )
            with co2:
                xgb_c_c = comb_clf.get("xgboost", {})
                st.markdown(
                    f'<div class="mp-card mp-card-purple">'
                    f'<b style="color:#a78bfa">Combined XGB Clf</b><br>'
                    f'F1: {fmt(xgb_c_c.get("f1"),4)} | ROC-AUC: {fmt(xgb_c_c.get("roc_auc"),4)}'
                    f'</div>', unsafe_allow_html=True
                )

        # Raw JSON
        with st.expander("📄 Raw metrics JSON"):
            st.json(mets)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 9 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
<div style="max-width:860px">

<div class="mp-hero" style="font-size:1.8rem;margin-bottom:0.5rem">About MarketPulse</div>
<div class="mp-sub" style="margin-bottom:1.5rem">Applied Data Science Final Project — Multi-Stock ML Forecasting & Risk Platform</div>

<div class="mp-card mp-card-accent">
<b style="color:#38bdf8;font-size:1rem">🎯 Project Objective</b><br>
<p style="color:#94a3b8;margin:0.5rem 0 0">
Build an industry-grade, end-to-end ML pipeline for multi-stock price forecasting, crash risk detection,
volatility modelling, and portfolio analytics — combining classical ML, deep learning, NLP sentiment,
and rigorous statistical validation.
</p>
</div>

<div class="mp-section">Architecture Overview</div>
<div class="mp-card">
<pre style="color:#94a3b8;font-size:0.82rem;line-height:1.6">
  yfinance OHLCV ──▶ Data Loader ──▶ Preprocessing ──▶ Feature Engineering
                                                              │
                          News/Twitter ──▶ Sentiment ────────┤ (30+ features)
                                          VADER / FinBERT     │
                                                              ▼
                                                         Panel DataFrame
                                                              │
                              ┌───────────────────────────────┤
                              │                               │
                     Regression Models              Classification Models
                     ─────────────────              ─────────────────────
                     · Linear Regression            · Logistic Regression
                     · Random Forest                · Random Forest
                     · XGBoost ✓ Optuna             · XGBoost
                     · LightGBM                     · LightGBM
                     · Stacking Ensemble            · Stacking Ensemble
                     · LSTM (Keras/PyTorch)
                     · GARCH(1,1) volatility
                              │
              ┌───────────────┼──────────────────────┐
              ▼               ▼                      ▼
        Walk-Forward     SHAP Explainability    Directional Backtest
        CV (5-fold)      (RF + XGBoost)        + Monte Carlo GBM
              │
              ▼
          MLflow Experiment Tracking → metrics.json → Streamlit Dashboard
</pre>
</div>
""", unsafe_allow_html=True)

    col_feat1, col_feat2 = st.columns(2)
    with col_feat1:
        st.markdown("""
<div class="mp-card mp-card-purple">
<b style="color:#a78bfa">📐 Feature Engineering</b>
<ul style="color:#94a3b8;margin-top:0.5rem;font-size:0.87rem">
  <li>SMA(20/50), EMA(12/26)</li>
  <li>RSI(14), MACD, Bollinger Bands</li>
  <li>ATR(14), OBV, Stochastic %K, Williams %R</li>
  <li>Price-to-SMA ratio, Volume z-score</li>
  <li>5-day lag returns & closes</li>
  <li>Rolling stats (5/10/20-day)</li>
  <li>VADER / FinBERT sentiment</li>
  <li>Sentiment momentum</li>
</ul>
</div>
""", unsafe_allow_html=True)

    with col_feat2:
        st.markdown("""
<div class="mp-card mp-card-success">
<b style="color:#4ade80">✅ Validation & Evaluation</b>
<ul style="color:#94a3b8;margin-top:0.5rem;font-size:0.87rem">
  <li>Holdout test set (time-based split)</li>
  <li>Walk-forward 5-fold CV (sklearn TSS)</li>
  <li>Naïve persistence & mean-train baselines</li>
  <li>RMSE, MAE, R², MAPE, Directional Accuracy</li>
  <li>F1, Precision, Recall, ROC-AUC (crash)</li>
  <li>Directional backtest w/ 10 bps cost</li>
  <li>Monte Carlo GBM (1 000 paths)</li>
  <li>SHAP mean(|SHAP|) feature importance</li>
  <li>Sentiment ablation study</li>
  <li>MLflow experiment tracking</li>
</ul>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="mp-card mp-card-warning" style="margin-top:1rem">
<b style="color:#fbbf24">📦 Tech Stack</b>
<p style="color:#94a3b8;font-size:0.87rem;margin:0.5rem 0 0">
Python 3.10–3.12 · scikit-learn · XGBoost · LightGBM · TensorFlow/Keras or PyTorch (LSTM) ·
GARCH via arch · VADER Sentiment · FinBERT (Hugging Face Transformers) · yfinance · SHAP ·
Optuna (hyperparameter tuning) · MLflow · Streamlit · Plotly · Pandas · NumPy
</p>
</div>

<div style="color:#475569;font-size:0.75rem;margin-top:1.5rem;text-align:center">
  MarketPulse — Applied Data Science Final Project &nbsp;|&nbsp; Research / educational use only &nbsp;|&nbsp; Past performance ≠ future results
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="color:#1e293b;font-size:0.72rem;text-align:center">'
    'MarketPulse ADS Project · Research / educational use only · Past performance ≠ future results'
    '</p>',
    unsafe_allow_html=True
)
