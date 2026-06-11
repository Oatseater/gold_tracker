import time
from datetime import datetime, timezone

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser

import gold_models as gm
import gold_ui as ui

st.set_page_config(page_title="XAU // TERMINAL", page_icon="◆", layout="wide",
                    initial_sidebar_state="expanded")
st.markdown(f"<style>{ui.CSS}</style>", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# Sidebar config
# ------------------------------------------------------------------ #
st.sidebar.markdown(ui.section_label("// CONFIG"), unsafe_allow_html=True)
interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"], index=0)
period = st.sidebar.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo"], index=1)
horizon = st.sidebar.slider("Forecast horizon (days)", 1, 30, 7)
n_sims = st.sidebar.slider("Monte Carlo paths", 200, 5000, 1500, step=100)
ml_model = st.sidebar.radio("ML model", ["gbr", "linear"], index=0)
run_bt = st.sidebar.checkbox("Run walk-forward backtest", value=False)
refresh = st.sidebar.checkbox("Auto-refresh (60s)", value=True)
st.sidebar.markdown(ui.section_label("// KARATS (AED)"), unsafe_allow_html=True)
karats = st.sidebar.multiselect("Karats to display", [24, 22, 21, 18, 14], default=[24, 22, 18])

if "balance" not in st.session_state:
    st.session_state.balance = 10000.0
    st.session_state.position = 0.0
    st.session_state.history = []

TICKERS = {"Gold": "GC=F", "DXY": "DX-Y.NYB", "US10Y": "^TNX",
           "VIX": "^VIX", "Oil": "CL=F", "SP500": "^GSPC", "Silver": "SI=F"}


@st.cache_data(ttl=60)
def get_data(period, interval):
    raw = yf.download(list(TICKERS.values()), period=period, interval=interval, group_by="ticker")
    data = {}
    for name, tk in TICKERS.items():
        try:
            data[name] = raw[tk]["Close"].dropna()
        except Exception:
            pass
    return pd.DataFrame(data).dropna()


@st.cache_data(ttl=300)
def get_daily(period="1y"):
    return yf.download("GC=F", period=period, interval="1d")["Close"].dropna()


@st.cache_data(ttl=300)
def get_fx():
    try:
        d = yf.download("AED=X", period="5d", interval="1d")["Close"].dropna()
        if len(d):
            return float(d.iloc[-1])
    except Exception:
        pass
    return None


@st.cache_data(ttl=300)
def get_news_sentiment():
    feeds = [
        "https://news.google.com/rss/search?q=gold+price+war+OR+conflict+OR+sanctions&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=gold+central+bank+OR+fed+rate&hl=en-US&gl=US&ceid=US:en",
    ]
    pos = ["safe haven", "rally", "surge", "demand", "buying", "rate cut", "dovish"]
    neg = ["ceasefire", "peace", "deal", "agreement", "easing", "hawkish", "selloff"]
    war = ["war", "conflict", "attack", "sanctions", "tension", "strike", "invasion"]
    headlines, score = [], 0
    for f in feeds:
        try:
            d = feedparser.parse(f)
            for e in d.entries[:10]:
                t = e.title.lower()
                headlines.append(e.title)
                score += sum(w in t for w in war) * 1.5
                score += sum(w in t for w in pos)
                score -= sum(w in t for w in neg)
        except Exception:
            pass
    return score, headlines[:8]


# ------------------------------------------------------------------ #
# Data
# ------------------------------------------------------------------ #
df = get_data(period, interval)
if df.empty or "Gold" not in df:
    st.error("No data returned. Try a different period/interval.")
    st.stop()

df = gm.build_features(df)
price = float(df["Gold"].iloc[-1])
prev_price = float(df["Gold"].iloc[-2]) if len(df) > 1 else price
gold = df["Gold"]
geo_score, headlines = get_news_sentiment()
fx = gm.usd_to_aed_rate(get_fx())

# physics: Kalman
kal_series, kal_state = gm.kalman_filter(gold)
kal_price, kal_vel = float(kal_state[0]), float(kal_state[1])

# physics: GARCH vol from daily history
daily = get_daily()
dlog = np.log(daily / daily.shift(1)).dropna()
mu = float(dlog.mean())
sigma = gm.garch_sigma(dlog)
geo_drift = geo_score * 0.0003

# physics: Monte Carlo with GARCH sigma
mc = gm.monte_carlo_gbm(price, mu, sigma, horizon, n_sims, geo_drift)
pred_horizon = float(mc["mean"][-1])
grid, centers = gm.mc_density_terrain(mc["paths"], price_bins=36)

# ML: multi-factor next-step
feats = [c for c in ["t", "vel", "acc", "DXY", "US10Y", "VIX", "Oil", "SP500", "Silver", "real_yield"]
         if c in df.columns]
model, scaler = gm.train_predict(df, feats, model=ml_model)
next_row = df.iloc[-1].copy()
next_row["t"] = len(df)
next_row["vel"] = df["vel"].iloc[-1] + df["acc"].iloc[-1]
pred_next = gm.predict_next(model, scaler, next_row, feats) + geo_score * (price * 0.0005)

# karat table
kt = gm.karat_price_table(price, fx, karats=tuple(sorted(karats, reverse=True)) or (24,))

# ------------------------------------------------------------------ #
# Ticker bar
# ------------------------------------------------------------------ #
sync_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
price_aed_24k = float(kt[kt["karat"] == 24]["aed_per_gram"].iloc[0]) if 24 in kt["karat"].values \
    else float(kt["aed_per_gram"].iloc[0])
st.markdown(ui.ticker_html(price, price - prev_price, price_aed_24k, fx, sync_time, refresh),
            unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# Hero: 3D probability terrain + side stats
# ------------------------------------------------------------------ #
col_terrain, col_side = st.columns([2, 1])
with col_terrain:
    st.plotly_chart(ui.terrain_figure(grid, centers), use_container_width=True,
                     config={"displayModeBar": False})
with col_side:
    st.markdown(ui.section_label("// XAU IN AED"), unsafe_allow_html=True)
    st.markdown(ui.karat_table_html(kt), unsafe_allow_html=True)
    st.markdown(ui.stat_card_html(
        f"{horizon}D MC FORECAST",
        f"${ui.fmt(pred_horizon)}",
        f"P10 ${ui.fmt(float(mc['p10'][-1]))} &middot; P90 ${ui.fmt(float(mc['p90'][-1]))}"
    ), unsafe_allow_html=True)
    st.markdown(ui.stat_card_html(
        "GEO / NEWS SCORE",
        f"{geo_score:+.1f}",
        f"GARCH σ {sigma:.5f}/d &middot; drift μ {mu:+.5f}"
    ), unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# Forecast chart
# ------------------------------------------------------------------ #
st.markdown(ui.section_label("// PRICE — ACTUAL · KALMAN · MC FORECAST"), unsafe_allow_html=True)
future_idx = pd.date_range(gold.index[-1], periods=horizon + 1, freq="D")
st.plotly_chart(
    ui.forecast_figure(gold.index, gold.values, kal_series, future_idx, mc["mean"], mc["p10"], mc["p90"]),
    use_container_width=True, config={"displayModeBar": False}
)

# ------------------------------------------------------------------ #
# Factors + News
# ------------------------------------------------------------------ #
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(ui.section_label("// MODEL FACTORS"), unsafe_allow_html=True)
    weights = model.coef_ if ml_model == "linear" else model.feature_importances_
    st.markdown(ui.factor_table_html(
        feats, [float(df[f].iloc[-1]) for f in feats], [float(w) for w in weights]
    ), unsafe_allow_html=True)
    st.markdown(ui.stat_card_html("ML NEXT-STEP", f"${ui.fmt(pred_next)}",
                                   f"Δ {pred_next - price:+.2f} vs spot"), unsafe_allow_html=True)
with col_b:
    st.markdown(ui.section_label("// NEWS LOG"), unsafe_allow_html=True)
    st.markdown(ui.news_log_html(headlines), unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# CEEMDAN decomposition
# ------------------------------------------------------------------ #
with st.expander("// SIGNAL DECOMPOSITION (CEEMDAN)"):
    imfs = gm.ceemdan_decompose(gold.values, max_imfs=6)
    st.plotly_chart(ui.imf_figure(imfs), use_container_width=True, config={"displayModeBar": False})

# ------------------------------------------------------------------ #
# Backtest
# ------------------------------------------------------------------ #
if run_bt:
    st.markdown(ui.section_label("// WALK-FORWARD BACKTEST vs RANDOM WALK"), unsafe_allow_html=True)
    with st.spinner("backtesting..."):
        bt = gm.walk_forward_backtest(df, feats, model=ml_model, min_train=max(60, len(df) // 3))
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.markdown(ui.stat_card_html("MODEL RMSE", f"{bt['model']['RMSE']:.2f}",
                                       f"DirAcc {bt['model']['DirAcc%']:.1f}%"), unsafe_allow_html=True)
    with bcol2:
        st.markdown(ui.stat_card_html("RANDOM WALK RMSE", f"{bt['random_walk']['RMSE']:.2f}",
                                       f"DirAcc {bt['random_walk']['DirAcc%']:.1f}%"), unsafe_allow_html=True)
    win = bt["model"]["RMSE"] < bt["random_walk"]["RMSE"]
    if win:
        st.success(f"MODEL BEATS RANDOM WALK  (n={bt['n']})")
    else:
        st.warning(f"MODEL DOES NOT BEAT RANDOM WALK  (n={bt['n']}) — trust the benchmark.")

# ------------------------------------------------------------------ #
# Paper trading
# ------------------------------------------------------------------ #
st.markdown(ui.section_label("// ORDER ENTRY (PAPER)"), unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(ui.stat_card_html("CASH", f"${ui.fmt(st.session_state.balance)}"), unsafe_allow_html=True)
with m2:
    st.markdown(ui.stat_card_html("POSITION", f"{st.session_state.position:.4f} oz"), unsafe_allow_html=True)
with m3:
    st.markdown(ui.stat_card_html("POS VALUE", f"${ui.fmt(st.session_state.position*price)}"), unsafe_allow_html=True)
with m4:
    nw = st.session_state.balance + st.session_state.position * price
    st.markdown(ui.stat_card_html("NET WORTH", f"${ui.fmt(nw)}"), unsafe_allow_html=True)

amt = st.number_input("USD amount", min_value=10.0, value=1000.0, step=100.0)
b1, b2 = st.columns(2)
if b1.button("BUY") and amt <= st.session_state.balance:
    st.session_state.position += amt / price
    st.session_state.balance -= amt
    st.session_state.history.append(("BUY", round(price, 2), amt))
if b2.button("SELL") and amt / price <= st.session_state.position:
    st.session_state.position -= amt / price
    st.session_state.balance += amt
    st.session_state.history.append(("SELL", round(price, 2), amt))
st.markdown(ui.history_table_html(st.session_state.history), unsafe_allow_html=True)

st.markdown(
    "<div class='t-sub' style='margin-top:18px;'>"
    "AED figures are spot-equivalent metal value (no making charges). "
    "FX uses live rate or 3.6725 peg fallback. "
    "No model reliably predicts gold — forecasts are probabilistic, not advice."
    "</div>", unsafe_allow_html=True)

if refresh:
    time.sleep(60)
    st.rerun()
