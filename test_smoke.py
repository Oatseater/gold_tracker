"""Full-app smoke test. Mocks streamlit, yfinance, feedparser with synthetic
data, then exec()s the REAL gold_app.py source, asserting every code path
runs and key computed variables are sane."""
import sys
import types
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# Fake yfinance
# ------------------------------------------------------------------ #
BASE = {"GC=F": 2700, "DX-Y.NYB": 100, "^TNX": 4.3, "^VIX": 18,
        "CL=F": 75, "^GSPC": 5000, "SI=F": 30, "AED=X": 3.6725}


def fake_download(tickers, period=None, interval=None, group_by=None):
    n = 150
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    if isinstance(tickers, list):
        cols = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Volume"]])
        df = pd.DataFrame(np.random.randn(n, len(cols)) * 0.01, index=idx, columns=cols)
        for tk in tickers:
            base = BASE.get(tk, 100)
            df[(tk, "Close")] = base + np.cumsum(np.random.normal(0, base * 0.002, n))
        return df
    base = BASE.get(tickers, 100)
    close = base + np.cumsum(np.random.normal(0, base * 0.002, n))
    return pd.DataFrame({"Open": close, "High": close, "Low": close,
                          "Close": close, "Volume": 1000}, index=idx)


fake_yf = types.ModuleType("yfinance")
fake_yf.download = fake_download
sys.modules["yfinance"] = fake_yf

# ------------------------------------------------------------------ #
# Fake feedparser
# ------------------------------------------------------------------ #
fake_fp = types.ModuleType("feedparser")


class _Entry:
    def __init__(self, title):
        self.title = title


class _Feed:
    def __init__(self, entries):
        self.entries = entries


def fake_parse(url):
    return _Feed([_Entry("Gold rallies as conflict tensions rise"),
                   _Entry("Fed signals possible rate cut")])


fake_fp.parse = fake_parse
sys.modules["feedparser"] = fake_fp

# ------------------------------------------------------------------ #
# Fake streamlit
# ------------------------------------------------------------------ #
class _Ctx:
    def __init__(self, parent=None): self._parent = parent
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name):
        if self._parent is not None:
            return getattr(self._parent, name)
        raise AttributeError(name)


class _SessionState(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e
    def __setattr__(self, k, v): self[k] = v
    def __contains__(self, k): return dict.__contains__(self, k)


class FakeSt:
    def __init__(self):
        self.session_state = _SessionState()
        self.sidebar = self
        self.calls = []

    def set_page_config(self, **kw): pass
    def markdown(self, *a, **kw): self.calls.append("markdown")
    def write(self, *a, **kw): pass
    def error(self, msg): raise RuntimeError("st.error: " + str(msg))
    def stop(self): raise SystemExit("st.stop")
    def success(self, msg): self.calls.append(("success", msg))
    def warning(self, msg): self.calls.append(("warning", msg))
    def rerun(self): pass

    def selectbox(self, label, options, index=0, **kw): return options[index]
    def radio(self, label, options, index=0, **kw): return options[index]
    def multiselect(self, label, options, default=None, **kw): return default or []
    def slider(self, label, *args, **kw):
        return kw.get("value", args[2] if len(args) > 2 else args[0])
    def checkbox(self, label, value=False, **kw):
        if "refresh" in label.lower() or "auto" in label.lower():
            return False  # avoid rerun loop in smoke test
        return value
    def number_input(self, label, **kw): return kw.get("value", 0.0)
    def button(self, label, **kw): return False

    def columns(self, spec, **kw):
        n = spec if isinstance(spec, int) else len(spec)
        return [_Ctx(self) for _ in range(n)]
    def expander(self, *a, **kw): return _Ctx(self)
    def spinner(self, *a, **kw): return _Ctx(self)

    def plotly_chart(self, fig, **kw):
        assert hasattr(fig, "data") and len(fig.data) > 0, "invalid figure passed to plotly_chart"
        self.calls.append(("plotly_chart", fig.data[0].type, len(fig.data)))

    def cache_data(self, *a, **kw):
        def deco(f): return f
        return deco


fake_st = FakeSt()
fake_st_module = types.ModuleType("streamlit")
for name in dir(fake_st):
    if not name.startswith("_"):
        setattr(fake_st_module, name, getattr(fake_st, name))
fake_st_module.session_state = fake_st.session_state
fake_st_module.sidebar = fake_st
sys.modules["streamlit"] = fake_st_module

# ------------------------------------------------------------------ #
# Exec the real app
# ------------------------------------------------------------------ #
src = open("/home/claude/gold_app.py").read()
g = {"__name__": "__main__", "__file__": "/home/claude/gold_app.py"}
print("=" * 55)
print("EXECUTING gold_app.py with mocked st/yfinance/feedparser")
print("=" * 55)
exec(compile(src, "gold_app.py", "exec"), g)

# ------------------------------------------------------------------ #
# Assertions on resulting state
# ------------------------------------------------------------------ #
assert g["price"] > 0
assert g["fx"] > 0
assert g["mc"]["mean"].shape[0] == g["horizon"] + 1
assert g["grid"].shape[1] == 36
assert set(g["kt"]["karat"]) == {24, 22, 18}
assert g["kt"].iloc[0]["aed_per_gram"] > g["kt"].iloc[-1]["aed_per_gram"]
assert g["pred_next"] != 0
assert len(g["headlines"]) == 4  # 2 feeds x 2 fake entries each

plotly_calls = [c for c in fake_st.calls if isinstance(c, tuple) and c[0] == "plotly_chart"]
print("plotly_chart calls:", plotly_calls)
assert any(t == "surface" for _, t, _ in plotly_calls), "terrain (surface) not rendered"
assert len(plotly_calls) >= 3  # terrain, forecast, IMF

print("-" * 55)
print(f"price=${g['price']:.2f}  fx={g['fx']:.4f}  pred_next=${g['pred_next']:.2f}")
print(f"24K={g['kt'].iloc[0]['aed_per_gram']:.2f} AED/g  "
      f"22K={g['kt'].iloc[1]['aed_per_gram']:.2f}  18K={g['kt'].iloc[2]['aed_per_gram']:.2f}")
print(f"{g['horizon']}d MC mean=${g['pred_horizon']:.2f}  geo_score={g['geo_score']}")
print("=" * 55)
print("SMOKE TEST PASSED — full app executed with no exceptions")
print("=" * 55)
