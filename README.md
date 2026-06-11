# XAU // TERMINAL — Gold Forecasting Dashboard

Real-time gold trading terminal: physics + ML forecasting, AED karat
pricing (24K/22K/18K), and a custom dark/mono "data terminal" UI with a
3D Monte-Carlo probability-terrain hero visual.

## Modules
- `gold_models.py` — pure model logic: Kalman filter, GARCH(1,1) vol,
  Monte Carlo (GBM) forecast + density terrain, CEEMDAN decomposition,
  Gradient Boosting / Linear multi-factor model, real-yield proxy,
  karat -> AED conversion, walk-forward backtest vs random walk.
- `gold_ui.py` — pure UI builders: terminal CSS theme, HTML cards/tables,
  3D terrain figure, forecast figure, IMF figure. No Streamlit calls.
- `gold_app.py` — thin Streamlit orchestration layer.

## Tests
```bash
pip install -r requirements.txt
python test_models.py   # math: Kalman/GARCH/MC/CEEMDAN/GBR/backtest/karat
python test_ui.py        # every HTML/figure builder
python test_smoke.py     # execs the real app with mocked st/yfinance/feedparser
streamlit run gold_app.py
```

## Karat / AED pricing
`karat_price_table()` converts XAU/USD spot -> AED per gram for any
karat (purity = K/24), using live USD/AED FX (yfinance `AED=X`) with a
3.6725 peg fallback. This is **spot-equivalent metal value**, not a
retail quote (no making charges/VAT).

## Deploy free (Streamlit Community Cloud)
Push to GitHub -> share.streamlit.io -> New app -> main file `gold_app.py`.

⚠️ Educational only. No model reliably predicts gold prices.
