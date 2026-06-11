"""Tests for gold_ui — pure HTML/figure builders. Feeds them real
gold_models outputs to make sure shapes/types line up end-to-end."""
import numpy as np
import pandas as pd
import gold_models as gm
import gold_ui as ui

np.random.seed(7)
N = 120
gold = 2700 + np.cumsum(np.random.normal(0.3, 4, N))
idx = pd.date_range("2026-01-01", periods=N, freq="D")
series = pd.Series(gold, index=idx)

print("=" * 55)

# ---- formatting ----
assert ui.fmt(1234.5) == "1,234.50"
assert ui.fmt(1234.567, 4) == "1,234.5670"
print("[PASS] fmt")

# ---- ticker ----
html = ui.ticker_html(2750.12, +3.4, 320.55, 3.6725, "12:00:00", live=True)
assert "2,750.12" in html and "320.55" in html and "LIVE" in html and "▲" in html
html2 = ui.ticker_html(2750.12, -3.4, 320.55, 3.6725, "12:00:00", live=False)
assert "PAUSED" in html2 and "▼" in html2
print("[PASS] ticker_html (up/down, live/paused)")

# ---- karat table ----
fx = gm.usd_to_aed_rate(None)
kt = gm.karat_price_table(2750.0, fx)
karat_html = ui.karat_table_html(kt)
assert "24K" in karat_html and "22K" in karat_html and "18K" in karat_html
assert karat_html.count("t-card") == 3
print("[PASS] karat_table_html ->", [int(k) for k in kt["karat"]])

# ---- stat card ----
sc = ui.stat_card_html("GEO RISK", "4.5", "war/conflict headlines")
assert "GEO RISK" in sc and "4.5" in sc
print("[PASS] stat_card_html")

# ---- factor table ----
factors = ["t", "vel", "DXY", "US10Y"]
values = [120.0, 1.2, 99.5, 4.31]
weights = [0.0021, -1.34, -2.1, 5.6]
ft = ui.factor_table_html(factors, values, weights)
assert "DXY" in ft and "+5.6000" in ft and "-2.1000" in ft
assert ui.RED in ft and ui.GREEN in ft  # both colors used (mixed signs)
print("[PASS] factor_table_html")

# ---- news log ----
nl_empty = ui.news_log_html([])
assert "no headlines" in nl_empty
nl = ui.news_log_html(["Gold surges on war fears", "Fed signals rate cut"])
assert nl.count("log-line") == 2
print("[PASS] news_log_html (empty + populated)")

# ---- history table ----
ht_empty = ui.history_table_html([])
assert "no trades" in ht_empty
ht = ui.history_table_html([("BUY", 2700.5, 1000.0), ("SELL", 2750.0, 500.0)])
assert "BUY" in ht and "SELL" in ht and ui.GREEN in ht and ui.RED in ht
print("[PASS] history_table_html (empty + populated)")

# ---- terrain figure (signature visual) ----
mc = gm.monte_carlo_gbm(float(gold[-1]), mu=0.0006, sigma=0.012, steps=7, n_sims=1500, seed=2)
grid, centers = gm.mc_density_terrain(mc["paths"], price_bins=30)
fig = ui.terrain_figure(grid, centers)
assert fig.data[0].type == "surface"
assert fig.data[0].z.shape == (8, 30)
assert fig.layout.scene.camera.eye.x == 1.7
print("[PASS] terrain_figure -> surface", fig.data[0].z.shape)

# ---- forecast figure ----
kal_series, _ = gm.kalman_filter(series)
future_idx = pd.date_range(idx[-1], periods=8, freq="D")
ff = ui.forecast_figure(idx, gold, kal_series, future_idx, mc["mean"], mc["p10"], mc["p90"])
trace_names = [t.name for t in ff.data]
assert "GOLD" in trace_names and "KALMAN" in trace_names and "MC MEAN" in trace_names
assert "P10–P90" in trace_names
print("[PASS] forecast_figure -> traces:", trace_names)

# ---- IMF figure ----
imfs = gm.ceemdan_decompose(series.values, max_imfs=5)
imf_fig = ui.imf_figure(imfs)
assert len(imf_fig.data) == imfs.shape[0]
assert imf_fig.data[-1].name == "TREND"
print("[PASS] imf_figure ->", len(imf_fig.data), "components")

# ---- CSS sanity ----
assert "@import" in ui.CSS and ".ticker" in ui.CSS and ".t-card" in ui.CSS
assert "{{" not in ui.CSS  # no accidental f-string double-brace leakage
print("[PASS] CSS string well-formed, %d chars" % len(ui.CSS))

print("=" * 55)
print("ALL UI COMPONENTS PASSED")
print("=" * 55)
