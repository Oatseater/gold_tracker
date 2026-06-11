# Gold Tracker — Physics + ML Forecasting

Real-time gold price dashboard with paper trading.

## Models
- Kalman filter (real-time price/velocity smoothing)
- GARCH(1,1) volatility -> Monte Carlo (GBM) forecast with P10/P90 bands
- CEEMDAN signal decomposition
- Gradient Boosting / Linear multi-factor model (DXY, yields, VIX, oil, S&P, silver, real-yield proxy)
- News/geopolitical sentiment score (war/conflict headlines)
- Walk-forward backtest vs random-walk benchmark

## Run locally
```bash
pip install -r requirements.txt
python test_models.py     # sanity check
streamlit run gold_app.py
```

## Deploy free (Streamlit Community Cloud)
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io -> New app -> pick this repo -> main file `gold_app.py`.
3. Done — auto-redeploys on every push.

⚠️ Educational only. No model reliably predicts gold prices.
