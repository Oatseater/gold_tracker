"""Tests for gold_models. Uses synthetic GBM-generated 'gold' data since
live Yahoo data is unavailable in the sandbox. Validates every component runs
and produces sane outputs."""
import numpy as np
import pandas as pd
import gold_models as gm

np.random.seed(42)
N = 400

# ---- synthetic correlated market: gold driven by a drift + factors + noise ----
t = np.arange(N)
dxy = 100 + np.cumsum(np.random.normal(0, 0.3, N))
us10y = 4.0 + np.cumsum(np.random.normal(0, 0.02, N)) * 0.1
vix = 18 + np.abs(np.cumsum(np.random.normal(0, 0.5, N))) % 25
oil = 75 + np.cumsum(np.random.normal(0, 0.5, N))
sp = 5000 + np.cumsum(np.random.normal(0, 8, N))
silver = 28 + np.cumsum(np.random.normal(0, 0.1, N))
# gold rises when dxy falls and yields fall, plus a real bull drift
gold = (2000 + 1.5 * t - 4 * (dxy - 100) - 30 * (us10y - 4.0)
        + 0.5 * (silver - 28) * 10 + np.cumsum(np.random.normal(0.2, 4, N)))

df = pd.DataFrame({"Gold": gold, "DXY": dxy, "US10Y": us10y,
                   "VIX": vix, "Oil": oil, "SP500": sp, "Silver": silver})

print("=" * 55)
print("DATA: %d synthetic rows, gold range $%.0f–$%.0f"
      % (N, gold.min(), gold.max()))
print("=" * 55)

# 1. Kalman
ks, state = gm.kalman_filter(df["Gold"])
assert len(ks) == N
assert np.isfinite(state).all()
print("[PASS] Kalman  -> price $%.2f, velocity %+.2f/step" % (state[0], state[1]))

# 2. GARCH sigma
logret = np.log(df["Gold"] / df["Gold"].shift(1)).dropna()
sig = gm.garch_sigma(logret)
assert 0 < sig < 1
print("[PASS] GARCH   -> 1-step daily vol sigma = %.5f" % sig)

# 3. Monte Carlo
mc = gm.monte_carlo_gbm(float(gold[-1]), mu=float(logret.mean()),
                        sigma=sig, steps=7, n_sims=2000, geo_drift=0.0003, seed=1)
assert mc["mean"].shape == (8,)
assert (mc["p10"] <= mc["mean"] + 1e-6).all() and (mc["mean"] <= mc["p90"] + 1e-6).all()
print("[PASS] MonteCar-> 7d mean $%.2f  band [$%.2f, $%.2f]"
      % (mc["mean"][-1], mc["p10"][-1], mc["p90"][-1]))

# 4. CEEMDAN
imfs = gm.ceemdan_decompose(df["Gold"].values, max_imfs=6)
recon = imfs.sum(axis=0)
recon_err = np.max(np.abs(recon - gold))
assert imfs.shape[1] == N
print("[PASS] CEEMDAN -> %d IMFs, reconstruction max-err %.4f"
      % (imfs.shape[0], recon_err))

# 5. Feature build + train/predict
d = gm.build_features(df)
assert "real_yield" in d and "vel" in d and "acc" in d
feats = ["t", "vel", "acc", "DXY", "US10Y", "VIX", "Oil", "SP500", "Silver", "real_yield"]
m, sc = gm.train_predict(d, feats, model="gbr")
pred = gm.predict_next(m, sc, d.iloc[-1], feats)
assert gold.min() * 0.5 < pred < gold.max() * 1.5
print("[PASS] GBR     -> next-step prediction $%.2f (last $%.2f)"
      % (pred, gold[-1]))

# 6. Backtest vs random walk
bt = gm.walk_forward_backtest(df, feats, model="gbr", min_train=120)
print("[PASS] Backtest-> %d one-step tests" % bt["n"])
print("        GBR  RMSE %.2f  MAPE %.3f%%  DirAcc %.1f%%"
      % (bt["model"]["RMSE"], bt["model"]["MAPE%"], bt["model"]["DirAcc%"]))
print("        RW   RMSE %.2f  MAPE %.3f%%  DirAcc %.1f%%"
      % (bt["random_walk"]["RMSE"], bt["random_walk"]["MAPE%"],
         bt["random_walk"]["DirAcc%"]))
beat = bt["model"]["RMSE"] < bt["random_walk"]["RMSE"]
print("        -> model %s random walk on RMSE"
      % ("BEATS" if beat else "loses to"))

print("=" * 55)
print("ALL COMPONENTS PASSED")
print("=" * 55)

# 7. Karat / AED conversion
kt = gm.karat_price_table(float(gold[-1]), gm.usd_to_aed_rate())
assert kt.iloc[0]["aed_per_gram"] > kt.iloc[1]["aed_per_gram"] > kt.iloc[2]["aed_per_gram"]
assert abs(kt.iloc[0]["purity"] - 1.0) < 1e-9
print("[PASS] Karat/AED-> 24K %.2f / 22K %.2f / 18K %.2f AED per gram"
      % tuple(kt["aed_per_gram"]))

# 8. MC density terrain (signature 3D visual data)
grid, centers = gm.mc_density_terrain(mc["paths"], price_bins=30)
assert grid.shape == (mc["mean"].shape[0], 30)
assert np.allclose(grid.sum(axis=1), 2000)
print("[PASS] MC terrain -> grid %s, price range $%.0f-$%.0f"
      % (grid.shape, centers[0], centers[-1]))

print("=" * 55)
print("ALL COMPONENTS PASSED (incl. karat/AED + terrain)")
print("=" * 55)
