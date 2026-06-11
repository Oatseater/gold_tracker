"""Pure model logic for gold forecasting. No Streamlit — unit-testable."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------ #
# 1. PHYSICS: Kalman constant-velocity filter (real-time smoothing)  #
# ------------------------------------------------------------------ #
def kalman_filter(series):
    s = np.asarray(series, dtype=float)
    x = np.array([s[0], 0.0])
    P = np.eye(2)
    F = np.array([[1, 1], [0, 1]])
    H = np.array([[1, 0]])
    Q = np.eye(2) * 0.01
    R = np.array([[max(s.var() * 0.05, 1.0)]])
    est = []
    for z in s:
        x = F @ x
        P = F @ P @ F.T + Q
        y = z - (H @ x)[0]
        S = H @ P @ H.T + R
        K = (P @ H.T) / S
        x = x + K.flatten() * y
        P = (np.eye(2) - K @ H) @ P
        est.append(x[0])
    return np.array(est), x  # smoothed series, final [pos, vel]


# ------------------------------------------------------------------ #
# 2. GARCH volatility  ->  feeds Monte Carlo (replaces constant sigma)#
# ------------------------------------------------------------------ #
def garch_sigma(log_returns):
    """Return 1-step-ahead daily vol forecast. Falls back to std if GARCH fails."""
    r = np.asarray(log_returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 30:
        return float(np.std(r)) if len(r) else 0.01
    try:
        from arch import arch_model
        am = arch_model(r * 100, vol="GARCH", p=1, q=1, dist="normal")
        res = am.fit(disp="off")
        fvar = res.forecast(horizon=1).variance.values[-1, 0]
        return float(np.sqrt(fvar) / 100.0)
    except Exception:
        return float(np.std(r))


# ------------------------------------------------------------------ #
# 3. GBM Monte Carlo with GARCH vol + geopolitical drift             #
# ------------------------------------------------------------------ #
def monte_carlo_gbm(price, mu, sigma, steps, n_sims=1000, geo_drift=0.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    paths = np.zeros((n_sims, steps + 1))
    paths[:, 0] = price
    for s in range(1, steps + 1):
        z = np.random.standard_normal(n_sims)
        paths[:, s] = paths[:, s - 1] * np.exp(
            (mu + geo_drift - 0.5 * sigma ** 2) + sigma * z
        )
    return {
        "mean": paths.mean(axis=0),
        "p10": np.percentile(paths, 10, axis=0),
        "p90": np.percentile(paths, 90, axis=0),
        "paths": paths,
    }


# ------------------------------------------------------------------ #
# 4. CEEMDAN signal decomposition (state-of-the-art preprocessing)   #
# ------------------------------------------------------------------ #
def ceemdan_decompose(series, max_imfs=8):
    """Decompose into IMFs. Returns array (n_imfs, len). Falls back to raw."""
    s = np.asarray(series, dtype=float)
    try:
        from PyEMD import CEEMDAN
        c = CEEMDAN(trials=20)
        imfs = c.ceemdan(s, max_imf=max_imfs)
        return imfs
    except Exception:
        return s.reshape(1, -1)


# ------------------------------------------------------------------ #
# 5. Multi-factor model (gradient boosting) + linear baseline        #
# ------------------------------------------------------------------ #
def build_features(df, price_col="Gold"):
    d = df.copy()
    d["t"] = np.arange(len(d))
    d["vel"] = d[price_col].diff().fillna(0)
    d["acc"] = d["vel"].diff().fillna(0)
    if "US10Y" in d and "VIX" in d:
        # crude real-yield proxy: nominal yield minus a vol-based inflation-fear term
        d["real_yield"] = d["US10Y"] - (d["VIX"] / 10.0)
    return d


def train_predict(df, feature_cols, target="Gold", model="gbr"):
    d = df.dropna(subset=feature_cols + [target])
    X = d[feature_cols].values
    y = d[target].values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    if model == "gbr":
        m = GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.05)
    else:
        m = LinearRegression()
    m.fit(Xs, y)
    return m, scaler


def predict_next(model, scaler, next_row, feature_cols):
    x = scaler.transform(np.asarray(next_row[feature_cols].values, dtype=float).reshape(1, -1))
    return float(model.predict(x)[0])


# ------------------------------------------------------------------ #
# 6. Benchmarks + walk-forward backtest                              #
# ------------------------------------------------------------------ #
def random_walk_forecast(series):
    """Tomorrow = today. The benchmark every model must beat."""
    return float(np.asarray(series)[-1])


def metrics(actual, pred):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    err = a - p
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err / a))) * 100
    # directional accuracy
    da = float(np.mean(np.sign(np.diff(a)) == np.sign(p[1:] - a[:-1]))) * 100 if len(a) > 1 else 0.0
    return {"RMSE": rmse, "MAE": mae, "MAPE%": mape, "DirAcc%": da}


def walk_forward_backtest(df, feature_cols, target="Gold", model="gbr", min_train=60):
    """One-step-ahead expanding-window backtest vs random walk."""
    d = build_features(df)
    d = d.dropna(subset=feature_cols + [target]).reset_index(drop=True)
    preds, rw, actual = [], [], []
    for i in range(min_train, len(d) - 1):
        train = d.iloc[:i]
        m, sc = train_predict(train, feature_cols, target, model)
        nxt = d.iloc[i]
        preds.append(predict_next(m, sc, nxt, feature_cols))
        rw.append(random_walk_forecast(train[target]))
        actual.append(d.iloc[i + 1][target])
    return {
        "model": metrics(actual, preds),
        "random_walk": metrics(actual, rw),
        "n": len(actual),
    }
