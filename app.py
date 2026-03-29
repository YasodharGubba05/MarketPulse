"""
Streamlit UI: forecasts, volatility, crash risk, and training-quality panels.
Run: streamlit run app.py  (use project .venv)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import load_ticker_universe, symbols_from_universe
from src.inference import load_metrics, predict_for_ticker

st.set_page_config(
    page_title="Multi-Stock ML Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container { padding-top: 1.2rem; max-width: 1280px; }
    div[data-testid="stMetricValue"] { font-size: 1.35rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; }
    h1 { font-weight: 700; letter-spacing: -0.02em; }
    .caption-muted { color: #64748b; font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Multi-stock ML lab")
st.markdown(
    '<p class="caption-muted">Next-day price · crash risk · volatility · sentiment · '
    "walk-forward CV & backtests from <code>python train.py</code></p>",
    unsafe_allow_html=True,
)

universe = load_ticker_universe()
syms = symbols_from_universe(universe)
sector_map = {e["symbol"].upper(): e.get("sector", "") for e in universe}

include_lstm = st.sidebar.checkbox(
    "Include LSTM curve (PyTorch / TensorFlow)",
    value=False,
    help="Off by default: heavy native libs can destabilize Streamlit on some Macs.",
)

mets = load_metrics()

col1, col2 = st.columns([1, 1])
with col1:
    ticker = st.selectbox("Ticker", syms, index=0)
with col2:
    compare = st.multiselect("Compare tickers (z-scored closes)", syms, max_selections=4)

if st.button("Refresh cache"):
    st.cache_data.clear()

tab_main, tab_quality, tab_raw = st.tabs(["Charts", "Model quality", "Raw metrics JSON"])


@st.cache_data(ttl=600, show_spinner="Computing predictions…")
def _cached_predict(sym: str, lstm: bool):
    return predict_for_ticker(sym, include_lstm=lstm)


try:
    out = _cached_predict(ticker, include_lstm)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.exception(e)
    st.stop()

with tab_main:
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Latest date", str(pd.Timestamp(out["last_date"]).date()))
    with kpi2:
        st.metric("Forecast next close (GBM/XGB)", f"${out['forecast_next_close_xgb']:.2f}")
    with kpi3:
        cr = out["crash_risk_now"]
        st.metric("Crash risk (P)", f"{cr:.3f}" if cr is not None else "n/a")
    with kpi4:
        st.metric("Sector", sector_map.get(ticker, "—"))

    fig_price = go.Figure()
    fig_price.add_trace(
        go.Scatter(
            x=out["dates"],
            y=out["actual_close"],
            name="Close",
            line=dict(color="#0ea5e9", width=2),
        )
    )
    fig_price.add_trace(
        go.Scatter(
            x=out["dates"],
            y=out["actual_next_close"],
            name="Next close (actual)",
            line=dict(color="#bae6fd"),
        )
    )
    fig_price.add_trace(
        go.Scatter(
            x=out["dates"],
            y=out["predicted_next_close_xgb"],
            name="Pred next close",
            line=dict(color="#f97316", dash="dot", width=2),
        )
    )
    if out.get("lstm_predicted") is not None:
        lstm_dates = out["dates"].values[-len(out["lstm_predicted"]) :]
        fig_price.add_trace(
            go.Scatter(
                x=lstm_dates,
                y=out["lstm_predicted"],
                name="LSTM",
                line=dict(color="#22c55e"),
            )
        )
    fig_price.update_layout(
        title=f"{ticker} — price vs forecasts",
        height=420,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60),
    )
    st.plotly_chart(fig_price, use_container_width=True)

    fc1, fc2 = st.columns(2)
    with fc1:
        fig_vol = go.Figure()
        fig_vol.add_trace(
            go.Scatter(
                x=out["dates"],
                y=out["volatility_roll"],
                name="Rolling vol",
                fill="tozeroy",
                line=dict(color="#8b5cf6"),
            )
        )
        fig_vol.update_layout(
            title="Annualized rolling volatility",
            height=340,
            template="plotly_white",
        )
        st.plotly_chart(fig_vol, use_container_width=True)
    with fc2:
        fig_sent = go.Figure()
        fig_sent.add_trace(
            go.Scatter(
                x=out["dates"],
                y=out["sentiment_mean"],
                name="Sentiment",
                line=dict(color="#f59e0b"),
            )
        )
        fig_sent.update_layout(title="Daily sentiment (VADER in app)", height=340, template="plotly_white")
        st.plotly_chart(fig_sent, use_container_width=True)

    if out["crash_probability"] is not None:
        fig_cr = go.Figure()
        fig_cr.add_trace(
            go.Scatter(
                x=out["dates"],
                y=out["crash_probability"],
                name="Crash probability",
                fill="tozeroy",
                line=dict(color="#dc2626"),
            )
        )
        fig_cr.update_layout(title="Crash model probability over time", height=300, template="plotly_white")
        st.plotly_chart(fig_cr, use_container_width=True)

    if compare:

        def norm(s: pd.Series) -> pd.Series:
            return (s - s.mean()) / (s.std() or 1.0)

        fig_cmp = go.Figure()
        for t in compare:
            try:
                o = _cached_predict(t, include_lstm)
                fig_cmp.add_trace(
                    go.Scatter(
                        x=o["dates"],
                        y=norm(pd.Series(o["actual_close"])),
                        name=f"{t}",
                    )
                )
            except Exception:
                continue
        fig_cmp.update_layout(title="Normalized closes — comparison", height=380, template="plotly_white")
        st.plotly_chart(fig_cmp, use_container_width=True)

with tab_quality:
    st.subheader("Holdout vs baselines & backtest")
    pt = (mets or {}).get("per_ticker", {}).get(ticker)
    if not pt:
        st.info("Run `python train.py` to populate metrics, then refresh.")
    else:
        bh = pt.get("baselines_holdout")
        bt = pt.get("backtest_holdout")
        wfv = pt.get("walk_forward_cv")

        if bh:
            c1, c2, c3 = st.columns(3)
            xgb_rmse = pt.get("regression", {}).get("xgboost", {}).get("RMSE")
            naive_rmse = bh.get("naive_persistence", {}).get("RMSE")
            mean_rmse = bh.get("mean_train_target", {}).get("RMSE")
            with c1:
                st.metric("XGBoost RMSE (holdout)", f"{xgb_rmse:.4f}" if xgb_rmse else "n/a")
            with c2:
                st.metric("Naive (persist) RMSE", f"{naive_rmse:.4f}" if naive_rmse else "n/a")
            with c3:
                st.metric("Mean-train RMSE", f"{mean_rmse:.4f}" if mean_rmse else "n/a")
            if xgb_rmse and naive_rmse:
                st.caption(
                    "ML should beat naive persistence on RMSE when the signal is useful."
                )

        if bt and "error" not in bt:
            st.markdown("**Backtest (holdout, ~10 bps cost on turns)**")
            b1 = bt.get("long_on_positive_pred_return", {})
            b2 = bt.get("long_positive_avoid_high_crash_risk", {})
            bcol1, bcol2 = st.columns(2)
            with bcol1:
                st.metric("Strategy return (long if pred>0)", f"{b1.get('strategy_total_return', 0)*100:.2f}%")
                st.caption(f"Sharpe ~ {b1.get('annualized_sharpe', 0):.2f} · Max DD {b1.get('max_drawdown', 0)*100:.1f}%")
            with bcol2:
                st.metric("+ avoid high crash risk", f"{b2.get('strategy_total_return', 0)*100:.2f}%")
                st.caption("Educational only — not investment advice.")

        if wfv and "error" not in wfv:
            st.markdown("**Walk-forward CV (mean RMSE by fold)**")
            reg = wfv.get("regression_rmse_by_fold", {})
            rows = []
            for name, d in reg.items():
                if isinstance(d, dict) and d.get("n_folds", 0):
                    rows.append(
                        {
                            "model": name,
                            "mean_rmse": d.get("mean_rmse"),
                            "std_rmse": d.get("std_rmse"),
                            "folds": d.get("n_folds"),
                        }
                    )
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab_raw:
    if mets:
        st.json(mets)
    else:
        st.caption("No metrics.json yet.")

st.markdown("---")
st.caption("Research / educational use only. Past performance ≠ future results.")
