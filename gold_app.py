import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import plotly.graph_objects as go
import time
import gold_models as gm

st.set_page_config(page_title="Gold Trader Pro", layout="wide")
st.title("🥇 Gold — Physics + ML Forecasting (GARCH · CEEMDAN · Monte Carlo)")

interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"], index=0)
period = st.sidebar.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo"], index=1)
horizon = st.sidebar.slider("Forecast horizon (days)", 1, 30, 7)
n_sims = st.sidebar.slider("Monte Carlo paths", 200, 5000, 1500, step=100)
ml_model = st.sidebar.radio("ML model", ["gbr", "linear"], index=0)
run_bt = st.sidebar.checkbox("Run walk-forward backtest", value=False)
refresh = st.sidebar.checkbox("Auto-refresh (60s real-time)", value=True)

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

df = get_data(period, interval)
if df.empty or "Gold" not in df:
    st.error("No data returned. Try a different period/interval.")
    st.stop()

df = gm.build_features(df)
price = float(df["Gold"].iloc[-1])
gold = df["Gold"]
geo_score, headlines = get_news_sentiment()

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

# ML: gradient boosting multi-factor next-step
feats = [c for c in ["t","vel","acc","DXY","US10Y","VIX","Oil","SP500","Silver","real_yield"] if c in df.columns]
model, scaler = gm.train_predict(df, feats, model=ml_model)
next_row = df.iloc[-1].copy()
next_row["t"] = len(df)
next_row["vel"] = df["vel"].iloc[-1] + df["acc"].iloc[-1]
pred_next = gm.predict_next(model, scaler, next_row, feats) + geo_score * (price * 0.0005)

trend = "📈 Bullish" if pred_horizon > price else "📉 Bearish"

c = st.columns(5)
c[0].metric("Gold (USD/oz)", f"${price:,.2f}")
c[1].metric("Next-step (ML)", f"${pred_next:,.2f}", f"{pred_next-price:+.2f}")
c[2].metric(f"{horizon}d MC", f"${pred_horizon:,.2f}", f"{pred_horizon-price:+.2f}")
c[3].metric("Kalman", f"${kal_price:,.2f}", f"vel {kal_vel:+.2f}")
c[4].metric("Geo/News Score", f"{geo_score:.1f}")

st.caption(f"GARCH σ={sigma:.5f}/day · drift μ={mu:.5f} · geo-drift {geo_drift:+.5f} · real-yield proxy {df['real_yield'].iloc[-1]:.2f}" if 'real_yield' in df else "")

# chart
future_idx = pd.date_range(gold.index[-1], periods=horizon+1, freq="D")
fig = go.Figure()
fig.add_trace(go.Scatter(x=gold.index, y=gold.values, name="Gold", line=dict(color="gold")))
fig.add_trace(go.Scatter(x=gold.index, y=kal_series, name="Kalman", line=dict(color="orange", dash="dot")))
fig.add_trace(go.Scatter(x=future_idx, y=mc["mean"], name="MC mean", line=dict(color="cyan")))
fig.add_trace(go.Scatter(x=future_idx, y=mc["p90"], line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=future_idx, y=mc["p10"], fill="tonexty",
                         fillcolor="rgba(0,255,255,0.15)", line=dict(width=0), name="P10–P90"))
fig.update_layout(template="plotly_dark", height=460)
st.plotly_chart(fig, use_container_width=True)

# CEEMDAN decomposition view
with st.expander("CEEMDAN signal decomposition"):
    imfs = gm.ceemdan_decompose(gold.values, max_imfs=6)
    fig2 = go.Figure()
    for i, imf in enumerate(imfs):
        fig2.add_trace(go.Scatter(y=imf, name=f"IMF {i+1}" if i < len(imfs)-1 else "Residual/trend"))
    fig2.update_layout(template="plotly_dark", height=350, title=f"{len(imfs)} components")
    st.plotly_chart(fig2, use_container_width=True)

colA, colB = st.columns(2)
with colA:
    st.subheader("Factors")
    if ml_model == "linear":
        weights = model.coef_
    else:
        weights = model.feature_importances_
    st.dataframe(pd.DataFrame({
        "Factor": feats,
        "Value": [round(float(df[f].iloc[-1]), 3) for f in feats],
        "Weight/Imp": [round(float(w), 4) for w in weights],
    }), use_container_width=True, hide_index=True)
with colB:
    st.subheader("Live Headlines")
    for h in headlines:
        st.write(f"• {h}")

# backtest panel
if run_bt:
    st.subheader("Walk-forward backtest vs Random Walk")
    with st.spinner("Backtesting..."):
        bt = gm.walk_forward_backtest(df, feats, model=ml_model, min_train=max(60, len(df)//3))
    bcols = st.columns(2)
    bcols[0].write("**Model**"); bcols[0].json(bt["model"])
    bcols[1].write("**Random Walk**"); bcols[1].json(bt["random_walk"])
    win = bt["model"]["RMSE"] < bt["random_walk"]["RMSE"]
    st.success(f"Model BEATS random walk (n={bt['n']})") if win else st.warning(
        f"Model does NOT beat random walk (n={bt['n']}). Trust the benchmark, not the forecast.")

# paper trading
st.subheader("Paper Trading")
m = st.columns(4)
m[0].metric("Cash", f"${st.session_state.balance:,.2f}")
m[1].metric("Position (oz)", f"{st.session_state.position:.4f}")
m[2].metric("Pos Value", f"${st.session_state.position*price:,.2f}")
m[3].metric("Net Worth", f"${st.session_state.balance + st.session_state.position*price:,.2f}")
amt = st.number_input("USD amount", min_value=10.0, value=1000.0, step=100.0)
b1, b2 = st.columns(2)
if b1.button("Buy") and amt <= st.session_state.balance:
    st.session_state.position += amt/price; st.session_state.balance -= amt
    st.session_state.history.append(("BUY", round(price,2), amt))
if b2.button("Sell") and amt/price <= st.session_state.position:
    st.session_state.position -= amt/price; st.session_state.balance += amt
    st.session_state.history.append(("SELL", round(price,2), amt))
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history, columns=["Action","Price","USD"]),
                 use_container_width=True, hide_index=True)

st.caption("⚠️ Educational paper trading. No model reliably predicts gold; treat forecasts as probabilistic.")

if refresh:
    time.sleep(60)
    st.rerun()
