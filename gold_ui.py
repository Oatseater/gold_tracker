"""Terminal-aesthetic UI building blocks. Pure functions returning HTML
strings / Plotly figures — no Streamlit calls, so unit-testable."""
import plotly.graph_objects as go

# ------------------------------------------------------------------ #
# Design tokens
# ------------------------------------------------------------------ #
VOID = "#07080A"
PANEL = "#0D0F12"
LINE = "rgba(255,255,255,0.08)"
LINE_SOFT = "rgba(255,255,255,0.04)"
TEXT = "#E7E5DE"
DIM = "#6E7480"
GOLD = "#E8B84B"
CYAN = "#6FE7DD"
GREEN = "#57D38C"
RED = "#F2545B"

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family:'JetBrains Mono', monospace !important; }

.stApp{
  background-color:#07080A;
  background-image:
    linear-gradient(rgba(255,255,255,0.028) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.028) 1px, transparent 1px);
  background-size: 42px 42px;
  color:#E7E5DE;
}
#MainMenu, footer {visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:0.8rem; padding-bottom:3rem; max-width:1440px;}

[data-testid="stSidebar"]{ background-color:#0D0F12; border-right:1px solid rgba(255,255,255,0.08); }
[data-testid="stSidebar"] *{ font-family:'JetBrains Mono',monospace !important; }
[data-testid="stSidebar"] label{ color:#6E7480 !important; font-size:10px !important; letter-spacing:0.18em; text-transform:uppercase; }

.stSelectbox div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input{
  background-color:#111318 !important; border:1px solid rgba(255,255,255,0.08) !important;
  color:#E7E5DE !important; border-radius:2px !important; font-family:'JetBrains Mono',monospace !important;
}
.stButton>button{
  background-color:#111318; color:#E7E5DE; border:1px solid rgba(255,255,255,0.08);
  border-radius:2px; font-family:'JetBrains Mono',monospace; letter-spacing:0.1em;
  text-transform:uppercase; font-size:12px; font-weight:600; width:100%;
}
.stButton>button:hover{ border-color:#E8B84B; color:#E8B84B; }

h1,h2,h3{ font-family:'JetBrains Mono',monospace !important; letter-spacing:0.04em; font-weight:700 !important; }
[data-testid="stExpander"]{ background-color:#0D0F12; border:1px solid rgba(255,255,255,0.08); border-radius:2px; }
hr{ border-color: rgba(255,255,255,0.08) !important; }

.t-card{ background:#0D0F12; border:1px solid rgba(255,255,255,0.08); padding:10px 14px; border-radius:2px; margin-bottom:8px; }
.t-label{ color:#6E7480; font-size:10px; letter-spacing:0.18em; text-transform:uppercase; }
.t-value{ font-size:20px; font-weight:800; font-variant-numeric:tabular-nums; letter-spacing:-0.01em; margin-top:2px; color:#E7E5DE; }
.t-sub{ font-size:11px; color:#6E7480; margin-top:2px; font-variant-numeric:tabular-nums; }

.ticker{ display:flex; align-items:center; gap:32px; padding:10px 4px 14px;
  border-bottom:1px solid rgba(255,255,255,0.08); margin-bottom:14px; flex-wrap:wrap; }
.ticker .pair{ color:#6E7480; font-size:11px; letter-spacing:0.18em; }
.ticker .px{ font-size:30px; font-weight:800; color:#E8B84B; font-variant-numeric:tabular-nums; }
.ticker .px2{ font-size:24px; font-weight:800; color:#E7E5DE; font-variant-numeric:tabular-nums; }
.ticker .delta-up{ color:#57D38C; font-size:13px; font-variant-numeric:tabular-nums; }
.ticker .delta-down{ color:#F2545B; font-size:13px; font-variant-numeric:tabular-nums; }
.live{ display:flex; align-items:center; gap:6px; margin-left:auto; color:#6E7480; font-size:11px; letter-spacing:0.18em; }
.live .dot{ width:8px; height:8px; border-radius:50%; background:#57D38C; animation:pulse 1.6s infinite; }
.live .dot.paused{ background:#6E7480; animation:none; }
@keyframes pulse{ 0%,100%{opacity:1;} 50%{opacity:.25;} }

.term-table{ width:100%; border-collapse:collapse; font-size:12px; }
.term-table th{ text-align:left; color:#6E7480; font-size:10px; letter-spacing:0.15em;
  text-transform:uppercase; border-bottom:1px solid rgba(255,255,255,0.08); padding:6px 8px; font-weight:600; }
.term-table td{ padding:6px 8px; border-bottom:1px solid rgba(255,255,255,0.04); font-variant-numeric:tabular-nums; }
.term-table tr:last-child td{ border-bottom:none; }

.log-line{ font-size:12px; color:#E7E5DE; padding:5px 0; border-bottom:1px solid rgba(255,255,255,0.04); }
.log-line span{ color:#E8B84B; margin-right:8px; }

.section-label{ color:#6E7480; font-size:10px; letter-spacing:0.22em; text-transform:uppercase;
  border-bottom:1px solid rgba(255,255,255,0.08); padding-bottom:6px; margin-bottom:10px; }
"""

PLOTLY_FONT = dict(family="JetBrains Mono, monospace", size=10, color=DIM)


# ------------------------------------------------------------------ #
# Formatting
# ------------------------------------------------------------------ #
def fmt(x, d=2):
    return f"{x:,.{d}f}"


# ------------------------------------------------------------------ #
# HTML builders
# ------------------------------------------------------------------ #
def section_label(text):
    return f'<div class="section-label">{text}</div>'


def ticker_html(price_usd, delta_usd, price_aed_24k, fx_rate, sync_time, live):
    arrow = "▲" if delta_usd >= 0 else "▼"
    cls = "delta-up" if delta_usd >= 0 else "delta-down"
    dot_cls = "dot" if live else "dot paused"
    status = "LIVE" if live else "PAUSED"
    return f"""
<div class="ticker">
  <div><div class="pair">XAU / USD &middot; SPOT</div>
       <div class="px">${fmt(price_usd)}</div>
       <div class="{cls}">{arrow} {fmt(abs(delta_usd))}</div></div>
  <div><div class="pair">XAU / AED &middot; 24K /G</div>
       <div class="px2">{fmt(price_aed_24k)}</div>
       <div class="t-sub">FX 1 USD = {fmt(fx_rate, 4)} AED</div></div>
  <div class="live"><span class="{dot_cls}"></span>{status} &middot; SYNC {sync_time} UTC</div>
</div>
"""


def karat_card_html(karat, purity, aed_per_gram, aed_per_oz, usd_per_gram):
    return f"""
<div class="t-card">
  <div class="t-label">{int(karat)}K &middot; {purity*100:.1f}% PURE</div>
  <div class="t-value">{fmt(aed_per_gram)} <span style="font-size:11px;color:#6E7480;font-weight:500;">AED/G</span></div>
  <div class="t-sub">{fmt(aed_per_oz)} AED/OZ &middot; {fmt(usd_per_gram)} USD/G</div>
</div>
"""


def karat_table_html(karat_df):
    return "".join(
        karat_card_html(r.karat, r.purity, r.aed_per_gram, r.aed_per_oz, r.usd_per_gram)
        for r in karat_df.itertuples()
    )


def stat_card_html(label, value, sub=""):
    sub_html = f'<div class="t-sub">{sub}</div>' if sub else ""
    return f"""
<div class="t-card">
  <div class="t-label">{label}</div>
  <div class="t-value">{value}</div>
  {sub_html}
</div>
"""


def factor_table_html(factors, values, weights):
    rows = []
    for f, v, w in zip(factors, values, weights):
        color = GREEN if w >= 0 else RED
        rows.append(
            f"<tr><td>{f}</td><td>{fmt(v, 3)}</td>"
            f"<td style='color:{color}'>{w:+.4f}</td></tr>"
        )
    return (
        "<table class='term-table'><tr><th>Factor</th><th>Value</th><th>Weight</th></tr>"
        + "".join(rows) + "</table>"
    )


def news_log_html(headlines):
    if not headlines:
        return "<div class='log-line'><span>&rsaquo;</span>no headlines fetched</div>"
    return "".join(f"<div class='log-line'><span>&rsaquo;</span>{h}</div>" for h in headlines)


def history_table_html(history):
    if not history:
        return "<div class='log-line'><span>&rsaquo;</span>no trades yet</div>"
    rows = []
    for action, px, usd in history:
        color = GREEN if action == "BUY" else RED
        rows.append(
            f"<tr><td style='color:{color}'>{action}</td>"
            f"<td>${fmt(px)}</td><td>${fmt(usd)}</td></tr>"
        )
    return (
        "<table class='term-table'><tr><th>Action</th><th>Price</th><th>USD</th></tr>"
        + "".join(rows) + "</table>"
    )


# ------------------------------------------------------------------ #
# Figure builders
# ------------------------------------------------------------------ #
def terrain_figure(grid, centers, title="PRICE DENSITY TERRAIN — t+0..t+H (USD/OZ)"):
    """3D wireframe-style probability terrain from MC density grid."""
    time_idx = list(range(grid.shape[0]))
    fig = go.Figure(data=[go.Surface(
        x=centers, y=time_idx, z=grid,
        colorscale=[[0, VOID], [0.35, "#23262E"], [0.7, "#5b6270"], [1, "#F4F2EC"]],
        showscale=False,
        contours=dict(
            z=dict(show=True, color="rgba(255,255,255,0.18)", width=1),
            x=dict(show=True, color="rgba(255,255,255,0.05)", width=1),
            y=dict(show=True, color="rgba(255,255,255,0.05)", width=1),
        ),
        lighting=dict(ambient=0.65, diffuse=0.5, specular=0.05),
    )])
    axis = dict(backgroundcolor=PANEL, gridcolor="rgba(255,255,255,0.05)",
                color=DIM, tickfont=dict(size=9))
    fig.update_layout(
        height=440, margin=dict(l=0, r=0, t=28, b=0),
        paper_bgcolor=PANEL,
        title=dict(text=title, font=dict(family="JetBrains Mono, monospace", size=11, color=DIM), x=0.02),
        scene=dict(
            xaxis=dict(title="PRICE", **axis),
            yaxis=dict(title="T+DAYS", **axis),
            zaxis=dict(title="DENSITY", showticklabels=False, **axis),
            camera=dict(eye=dict(x=1.7, y=-1.9, z=0.55)),
            aspectratio=dict(x=1.4, y=1, z=0.6),
        ),
        font=PLOTLY_FONT,
    )
    return fig


def forecast_figure(gold_index, gold_values, kalman_series, future_idx, mc_mean, mc_p10, mc_p90):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gold_index, y=gold_values, name="GOLD",
                              line=dict(color=GOLD, width=1.6)))
    fig.add_trace(go.Scatter(x=gold_index, y=kalman_series, name="KALMAN",
                              line=dict(color=DIM, width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=future_idx, y=mc_p90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future_idx, y=mc_p10, fill="tonexty", name="P10–P90",
                              fillcolor="rgba(111,231,221,0.12)", line=dict(width=0), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future_idx, y=mc_mean, name="MC MEAN",
                              line=dict(color=CYAN, width=1.6)))
    grid_kw = dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)",
                    color=DIM, tickfont=dict(size=10))
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=PLOTLY_FONT,
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10, color=DIM)),
        xaxis=grid_kw, yaxis=grid_kw,
    )
    return fig


def imf_figure(imfs):
    fig = go.Figure()
    palette = [GOLD, CYAN, GREEN, RED, "#9D8CFF", "#FF9F6F", TEXT]
    for i, imf in enumerate(imfs):
        name = f"IMF {i+1}" if i < len(imfs) - 1 else "TREND"
        fig.add_trace(go.Scatter(y=imf, name=name,
                                  line=dict(color=palette[i % len(palette)], width=1)))
    grid_kw = dict(gridcolor="rgba(255,255,255,0.05)", color=DIM, tickfont=dict(size=10))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=PLOTLY_FONT,
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=9, color=DIM)),
        xaxis=grid_kw, yaxis=grid_kw,
    )
    return fig
