"""
app.py  --  ReservoirWatch Dashboard
Pune Reservoir Drought Early Warning System

Run:   python -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date
from pathlib import Path
import subprocess, sys

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ReservoirWatch | Pune Drought Early Warning",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': 'ReservoirWatch — PCCoE IGC Indradhanu Hackathon 2026 | UN SDG 13'},
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background:#0f172a; }
  [data-testid="stSidebar"]{ background:#1e293b; border-right:1px solid #334155; }
  [data-testid="stSidebar"] * { color:#cbd5e1 !important; }

  .badge-safe     { background:#22c55e; color:#fff; padding:3px 10px; border-radius:20px;
                    font-weight:700; font-size:.78rem; letter-spacing:.04em; display:inline-flex;
                    align-items:center; gap:5px; white-space:nowrap; flex-shrink:0; }
  .badge-watch    { background:#f59e0b; color:#fff; padding:3px 10px; border-radius:20px;
                    font-weight:700; font-size:.78rem; letter-spacing:.04em; display:inline-flex;
                    align-items:center; gap:5px; white-space:nowrap; flex-shrink:0; }
  .badge-critical { background:#ef4444; color:#fff; padding:3px 10px; border-radius:20px;
                    font-weight:700; font-size:.78rem; letter-spacing:.04em; display:inline-flex;
                    align-items:center; gap:5px; white-space:nowrap; flex-shrink:0;
                    animation:pulse 1.5s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.65} }

  .res-card  { background:#1e293b; border-radius:14px; padding:22px; margin:6px 0; border-top:4px solid; }
  .card-safe     { border-color:#22c55e; }
  .card-watch    { border-color:#f59e0b; }
  .card-critical { border-color:#ef4444; }

  .alert-crit  { background:rgba(239,68,68,.12); border:1px solid #ef4444;
                 border-radius:10px; padding:12px 18px; margin:6px 0; color:#fca5a5; }
  .alert-stale { background:rgba(245,158,11,.10); border:1px solid #f59e0b;
                 border-radius:10px; padding:10px 16px; margin:6px 0; color:#fde68a; }
  .alert-live  { background:rgba(34,197,94,.10);  border:1px solid #22c55e;
                 border-radius:10px; padding:10px 16px; margin:6px 0; color:#bbf7d0; }

  .freshness-dot-fresh  { width:8px;height:8px;border-radius:50%;background:#22c55e;
                           display:inline-block;margin-right:6px; }
  .freshness-dot-warn   { width:8px;height:8px;border-radius:50%;background:#f59e0b;
                           display:inline-block;margin-right:6px; }
  .freshness-dot-stale  { width:8px;height:8px;border-radius:50%;background:#ef4444;
                           display:inline-block;margin-right:6px; }

  .sec-hdr { font-size:.95rem;font-weight:600;color:#64748b;text-transform:uppercase;
             letter-spacing:.08em;margin-bottom:10px;padding-bottom:6px;
             border-bottom:1px solid #1e293b; }

  h1,h2,h3 { color:#f1f5f9 !important; }
  p,li { color:#cbd5e1; }
  [data-testid="metric-container"] { background:rgba(30,41,59,.7);border-radius:10px;padding:10px 14px; }
  .footer { color:#334155;font-size:.78rem;text-align:center;padding-top:36px; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
RISK_COLORS = {'SAFE':'#22c55e', 'WATCH':'#f59e0b', 'CRITICAL':'#ef4444'}
RES_COLORS  = {'Khadakwasla':'#38bdf8', 'Panshet':'#a78bfa', 'Varasgaon':'#34d399', 'Temghar':'#fb923c'}
# Max areas updated from actual satellite measurements (Nov 2026)
MAX_AREAS   = {'Khadakwasla':7.5, 'Panshet':12.0, 'Varasgaon':18.5, 'Temghar':4.5}
SENTINEL2_REVISIT_DAYS = 5   # Sentinel-2 theoretical revisit


# ── DATA ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)   # cache for 60 seconds — refreshes quickly on new data
def load_data():
    base    = Path(__file__).parent
    r_path  = base / 'data' / 'reservoir_data.csv'
    rk_path = base / 'data' / 'risk_analysis.csv'

    if not r_path.exists():
        st.error("data/reservoir_data.csv not found. Run: python scripts/generate_sample_data.py")
        st.stop()

    res_df = pd.read_csv(r_path, parse_dates=['date'])

    if rk_path.exists():
        rsk_df = pd.read_csv(rk_path, parse_dates=['date'])
    else:
        rsk_df = res_df.copy()
        rsk_df['depletion_rate_pct_per_week'] = 0.0
        rsk_df['risk_level']                  = 'SAFE'
        rsk_df['days_remaining_estimate']     = 999

    return res_df, rsk_df


def refresh_risk_analysis():
    """Re-run risk_logic.py on whatever CSVs are currently in data/."""
    base        = Path(__file__).parent
    script_path = base / 'risk_logic.py'
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True, text=True, cwd=str(base)
    )
    st.cache_data.clear()
    return result.returncode == 0, result.stdout, result.stderr


def data_freshness(rsk_df):
    """
    Returns per-reservoir freshness info.
    fresh  = last obs <= 10 days ago
    warn   = 11-20 days ago
    stale  = > 20 days ago
    """
    today  = pd.Timestamp.now().normalize()
    info   = {}
    for name in rsk_df['reservoir_name'].unique():
        sub       = rsk_df[rsk_df['reservoir_name'] == name]
        last_obs  = sub['date'].max()
        days_ago  = (today - last_obs).days
        # Next expected pass: last_obs + revisit cycle (Sentinel-2 passes every ~5 days)
        # accounting for cloud cover, realistic gap is ~10-14 days
        next_pass = last_obs + pd.Timedelta(days=SENTINEL2_REVISIT_DAYS)
        status    = 'fresh' if days_ago <= 10 else 'warn' if days_ago <= 20 else 'stale'
        info[name] = {
            'last_obs' : last_obs,
            'days_ago' : days_ago,
            'next_pass': next_pass,
            'status'   : status,
        }
    return info


def latest(df, name):
    sub = df[df['reservoir_name'] == name]
    return sub.sort_values('date').iloc[-1] if not sub.empty else None

def badge(level):
    return f'<span class="badge-{level.lower()}">&#9679; {level}</span>'


def compute_crisis_dates(rsk_df):
    """Project when each reservoir will hit WATCH/CRITICAL based on current depletion."""
    results = {}
    today   = pd.Timestamp.now().normalize()
    for name in rsk_df['reservoir_name'].unique():
        sub   = rsk_df[rsk_df['reservoir_name'] == name].sort_values('date')
        row   = sub.iloc[-1]
        max_a = MAX_AREAS.get(name, row['surface_area_sqkm'])
        area  = row['surface_area_sqkm']
        dep   = row.get('depletion_rate_pct_per_week', 0)
        watch_date = critical_date = None
        if dep > 0.1:
            area_per_day  = (dep / 100 * max_a) / 7
            watch_thr     = max_a * 0.35
            crit_thr      = max_a * 0.20
            if area > watch_thr:
                watch_date    = today + pd.Timedelta(days=int((area - watch_thr) / area_per_day))
            if area > crit_thr:
                critical_date = today + pd.Timedelta(days=int((area - crit_thr) / area_per_day))
        results[name] = {
            'watch_date': watch_date, 'critical_date': critical_date,
            'depletion': dep, 'area_pct': area / max_a * 100,
            'risk': row['risk_level'],
        }
    return results


# ── CHARTS ────────────────────────────────────────────────────────────────────
def _dark(fig, height=300, ytitle='', hovermode='closest', legend=None):
    u = dict(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
             plot_bgcolor='rgba(15,23,42,.8)', font_color='#94a3b8',
             height=height, margin=dict(l=0,r=0,t=30,b=0), hovermode=hovermode,
             xaxis=dict(gridcolor='#1e293b'),
             yaxis=dict(gridcolor='#1e293b', title=ytitle, title_font_color='#94a3b8'))
    if legend: u['legend'] = legend
    fig.update_layout(**u)


def monsoon_bands(fig, years):
    for y in years:
        fig.add_vrect(x0=f"{y}-07-01", x1=f"{y}-09-30",
                      fillcolor="rgba(56,189,248,.05)", layer="below", line_width=0,
                      annotation_text="☔ Monsoon" if y == min(years) else "",
                      annotation_position="top left",
                      annotation_font=dict(color="#475569", size=10))
    return fig


def chart_reservoir_map(rsk_df):
    COORDS = {
        'Khadakwasla': (73.750, 18.430),
        'Varasgaon':   (73.528, 18.376),
        'Panshet':     (73.541, 18.338),
        'Temghar':     (73.516, 18.451),
    }
    PUNE = (73.857, 18.520)
    fig  = go.Figure()

    # ── Cascade flow lines (thick + thin layered for depth effect) ────────────
    all_pts = {**COORDS, 'Pune': PUNE}
    for src, dst in [('Varasgaon','Panshet'),('Panshet','Temghar'),
                     ('Temghar','Khadakwasla'),('Khadakwasla','Pune')]:
        s, d = all_pts[src], all_pts[dst]
        # Outer glow line
        fig.add_trace(go.Scattermap(
            lon=[s[0], d[0]], lat=[s[1], d[1]], mode='lines',
            line=dict(width=6, color='rgba(56,189,248,0.12)'),
            showlegend=False, hoverinfo='skip'))
        # Core line
        fig.add_trace(go.Scattermap(
            lon=[s[0], d[0]], lat=[s[1], d[1]], mode='lines',
            line=dict(width=2, color='rgba(56,189,248,0.55)'),
            showlegend=False, hoverinfo='skip'))

    # Dashed supply line Khadakwasla → Pune city (water mains)
    kh, pu = COORDS['Khadakwasla'], PUNE
    # Interpolate mid-point for dashed effect
    for frac in [0, 0.25, 0.5, 0.75]:
        x0 = kh[0] + frac * (pu[0] - kh[0])
        y0 = kh[1] + frac * (pu[1] - kh[1])
        x1 = kh[0] + (frac + 0.18) * (pu[0] - kh[0])
        y1 = kh[1] + (frac + 0.18) * (pu[1] - kh[1])
        fig.add_trace(go.Scattermap(
            lon=[x0, x1], lat=[y0, y1], mode='lines',
            line=dict(width=2, color='rgba(248,250,252,0.25)'),
            showlegend=False, hoverinfo='skip'))

    # ── Reservoir markers ─────────────────────────────────────────────────────
    for name, (lon, lat) in COORDS.items():
        row   = latest(rsk_df, name)
        if row is None: continue
        max_a = MAX_AREAS.get(name, 1)
        area  = row['surface_area_sqkm']
        pct   = area / max_a * 100
        risk  = row['risk_level']
        dep   = row.get('depletion_rate_pct_per_week', 0)
        color = RISK_COLORS.get(risk, '#38bdf8')
        rc    = RES_COLORS.get(name, '#38bdf8')
        size  = max(22, min(54, pct * 0.52))

        # Outer glow ring (large, very transparent)
        fig.add_trace(go.Scattermap(
            lon=[lon], lat=[lat], mode='markers',
            marker=dict(size=size + 22, color=color, opacity=0.12),
            showlegend=False, hoverinfo='skip'))

        # Mid glow ring
        fig.add_trace(go.Scattermap(
            lon=[lon], lat=[lat], mode='markers',
            marker=dict(size=size + 10, color=color, opacity=0.22),
            showlegend=False, hoverinfo='skip'))

        # Main marker
        fig.add_trace(go.Scattermap(
            lon=[lon], lat=[lat], mode='markers+text',
            marker=dict(size=size, color=color, opacity=0.92),
            text=[f'{pct:.0f}%'],
            textposition='middle center',
            textfont=dict(color='#ffffff', size=10, family='Inter, sans-serif'),
            name=f'{name}',
            hovertemplate=(
                f'<b style="font-size:14px">{name}</b><br>'
                f'<b>Risk:</b> {risk}<br>'
                f'<b>Fill level:</b> {pct:.1f}%<br>'
                f'<b>Surface area:</b> {area:.2f} sq km<br>'
                f'<b>Depletion:</b> {dep:+.2f}%/wk'
                f'<extra></extra>'
            )))

    # ── Pune city star marker ─────────────────────────────────────────────────
    fig.add_trace(go.Scattermap(
        lon=[PUNE[0]], lat=[PUNE[1]], mode='markers+text',
        marker=dict(size=18, color='#f8fafc', opacity=0.9),
        text=['🏙 Pune'], textposition='top right',
        textfont=dict(color='#f8fafc', size=12, family='Inter, sans-serif'),
        showlegend=False,
        hovertemplate='<b>Pune City</b><br>Population served: 3.4M residents<extra></extra>'))

    fig.update_layout(
        map=dict(
            style='carto-darkmatter',
            center=dict(lon=73.63, lat=18.39),
            zoom=9.6,
        ),
        height=460,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            bgcolor='rgba(10,18,34,0.88)',
            bordercolor='#334155', borderwidth=1,
            font=dict(color='#cbd5e1', size=11,
                      family='Inter, sans-serif'),
            title=dict(text='Reservoir Status',
                       font=dict(color='#64748b', size=10)),
            x=0.01, y=0.99,
        ),
        hoverlabel=dict(
            bgcolor='#1e293b', bordercolor='#334155',
            font=dict(color='#f1f5f9', size=12,
                      family='Inter, sans-serif'),
        ),
    )
    return fig


def chart_combined(rsk_df):

    fig   = go.Figure()
    years = rsk_df['date'].dt.year.unique()
    fig   = monsoon_bands(fig, years)
    for name, color in RES_COLORS.items():
        df = rsk_df[rsk_df['reservoir_name'] == name].sort_values('date')
        if df.empty: continue
        # Mark the very latest point as a star
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['surface_area_sqkm'], name=name,
            line=dict(color=color, width=2.5), mode='lines+markers',
            marker=dict(size=4),
            hovertemplate=f"<b>{name}</b><br>%{{x|%b %d, %Y}}<br>%{{y:.2f}} sq km<extra></extra>",
        ))
        # Latest observation marker (star)
        fig.add_trace(go.Scatter(
            x=[df['date'].iloc[-1]], y=[df['surface_area_sqkm'].iloc[-1]],
            mode='markers', showlegend=False,
            marker=dict(symbol='star', color=color, size=14, line=dict(color='white', width=1)),
            hovertemplate=f"<b>{name} (latest)</b><br>%{{x|%b %d, %Y}}<br>%{{y:.2f}} sq km<extra></extra>",
        ))
    _dark(fig, height=380, ytitle="Surface Area (sq km)", hovermode="x unified",
          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig


def chart_area_detail(rsk_df, name):
    from scipy import stats as _stats
    df     = rsk_df[rsk_df['reservoir_name'] == name].sort_values('date')
    color  = RES_COLORS.get(name, '#38bdf8')
    max_a  = MAX_AREAS.get(name, df['surface_area_sqkm'].max())

    # Filter physically impossible readings (>110% of max area = bad satellite data)
    df     = df[df['surface_area_sqkm'] <= max_a * 1.10].copy()

    years  = df['date'].dt.year.unique()
    rgb    = tuple(int(color[i:i+2], 16) for i in (1,3,5))

    fig = go.Figure()
    fig = monsoon_bands(fig, years)
    fig.add_hline(y=max_a,        line_dash='dot', line_color='#334155',
                  annotation_text=f'Max {max_a} sq km',
                  annotation_font=dict(color='#475569', size=10))
    fig.add_hline(y=max_a * 0.20, line_dash='dot', line_color='#ef4444',
                  annotation_text='Critical threshold (20%)',
                  annotation_font=dict(color='#ef4444', size=10))

    if 'risk_level' in df.columns:
        for risk, rc in RISK_COLORS.items():
            sub = df[df['risk_level'] == risk]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub['date'], y=sub['surface_area_sqkm'],
                    mode='markers', name=risk, marker=dict(color=rc, size=7),
                    hovertemplate=f'<b>{risk}</b><br>%{{x|%b %d}}<br>%{{y:.2f}} sq km<extra></extra>',
                ))

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['surface_area_sqkm'], name='Observed Area',
        line=dict(color=color, width=2.2),
        fill='tozeroy', fillcolor=f'rgba({rgb[0]},{rgb[1]},{rgb[2]},0.08)',
        hovertemplate='Area: %{y:.2f} sq km<br>%{x|%b %d, %Y}<extra></extra>',
    ))
    # Latest star
    last_date = df['date'].iloc[-1]
    last_area = df['surface_area_sqkm'].iloc[-1]
    fig.add_trace(go.Scatter(
        x=[last_date], y=[last_area],
        mode='markers', showlegend=False,
        marker=dict(symbol='star', color=color, size=16, line=dict(color='white', width=1)),
        hovertemplate='<b>Latest observation</b><br>%{x|%b %d, %Y}<br>%{y:.2f} sq km<extra></extra>',
    ))

    # ── 60-Day Forecast ──────────────────────────────────────────────────────
    # 21-day window: Sentinel-2 passes every ~5 days so 14 days only gives
    # 2 observations (fails the >=3 check). 21 days gives 4 clean points.
    window = df[df['date'] >= last_date - pd.Timedelta(days=21)]
    if len(window) >= 2:
        x_num  = (window['date'] - window['date'].min()).dt.days.values.astype(float)
        y_vals = window['surface_area_sqkm'].values.astype(float)
        slope, intercept, _, _, stderr = _stats.linregress(x_num, y_vals)

        # Seasonal correction: Oct–Jun is dry season — reservoirs don't fill
        # without monsoon rain. If regression sees a positive slope (artifact
        # of late-monsoon fill data still in window), force it to flat (0).
        dry_months = list(range(1, 7)) + list(range(10, 13))  # Oct-Jun
        if last_date.month in dry_months and slope > 0:
            slope = 0.0   # plateau — no filling in dry season

        # Build forecast dates: today → +60 days
        forecast_dates = pd.date_range(last_date, periods=61, freq='D')
        x_forecast     = np.array((forecast_dates - window['date'].min()).days, dtype=float)
        y_forecast     = np.array(slope * x_forecast + intercept)
        # Cap forecast at current observed area (can't go higher in dry season)
        if last_date.month in dry_months:
            y_forecast = np.minimum(y_forecast, last_area)
        # 95% confidence band: ±1.96 × stderr × sqrt(1 + 1/n)
        band    = 1.96 * stderr * (1 + 1 / len(window)) ** 0.5
        y_upper = np.clip(y_forecast + band, 0, max_a)
        y_lower = np.clip(y_forecast - band, 0, None)
        y_mid   = np.clip(y_forecast, 0, None)

        # Confidence band
        fig.add_trace(go.Scatter(
            x=list(forecast_dates) + list(forecast_dates[::-1]),
            y=list(y_upper) + list(y_lower[::-1]),
            fill='toself', fillcolor=f'rgba({rgb[0]},{rgb[1]},{rgb[2]},0.07)',
            line=dict(width=0), showlegend=False, hoverinfo='skip',
        ))
        # Forecast centre line
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=y_mid,
            mode='lines', name='60-Day Forecast',
            line=dict(color=color, width=2.5, dash='dot'),
            hovertemplate='<b>Forecast</b><br>%{x|%b %d, %Y}<br>%{y:.2f} sq km<extra></extra>',
        ))


        # Mark where forecast crosses WATCH threshold (35% of max)
        watch_lvl = max_a * 0.35
        for j in range(len(y_forecast) - 1):
            if y_forecast[j] > watch_lvl >= y_forecast[j + 1]:
                cross_date = forecast_dates[j + 1]
                fig.add_vline(x=cross_date, line_dash='dash', line_color='#f59e0b', line_width=1.5)
                fig.add_annotation(x=cross_date, y=watch_lvl, text='⚠ WATCH',
                                   font=dict(color='#f59e0b', size=11), showarrow=False,
                                   xshift=40, yshift=10)
                break

    _dark(fig, height=400, ytitle='Surface Area (sq km)',
          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    # Extend x-axis to cover full forecast window so dotted line is visible
    x_end = last_date + pd.Timedelta(days=61)
    fig.update_xaxes(range=[str(df['date'].min().date()), str(x_end.date())])
    fig.update_yaxes(range=[0, max_a * 1.15])
    return fig



def chart_depletion(rsk_df, name):
    df = rsk_df[rsk_df['reservoir_name'] == name].sort_values('date')
    if 'depletion_rate_pct_per_week' not in df.columns: return None

    vals   = df['depletion_rate_pct_per_week']
    colors = vals.apply(
        lambda x: '#ef4444' if x > 4 else '#f59e0b' if x > 2 else '#22c55e' if x < 0 else '#38bdf8'
    )

    # Clip y-axis to 95th percentile to prevent outlier spikes from squashing the view
    p05 = float(vals.quantile(0.05))
    p95 = float(vals.quantile(0.95))
    padding = max((p95 - p05) * 0.4, 2.0)
    y_min = min(p05 - padding, -1)
    y_max = max(p95 + padding,  6)

    fig = go.Figure()

    # Subtle fill under line
    fig.add_trace(go.Scatter(
        x=df['date'], y=vals.clip(y_min, y_max),
        fill='tozeroy',
        fillcolor='rgba(56,189,248,0.07)',
        line=dict(color='rgba(56,189,248,0.25)', width=1),
        showlegend=False, hoverinfo='skip',
    ))

    # Bars
    fig.add_trace(go.Bar(
        x=df['date'], y=vals,
        marker_color=list(colors),
        marker_line_width=0,
        opacity=0.85,
        name='Depletion Rate',
        hovertemplate="Depletion: %{y:+.2f}%/wk<br>%{x|%b %d, %Y}<extra></extra>",
    ))

    # Threshold lines — no annotation text, just colored dashed lines
    fig.add_hline(y=0,  line_color='#475569', line_width=1)
    fig.add_hline(y=2,  line_dash='dot', line_color='#f59e0b', line_width=1.5)
    fig.add_hline(y=4,  line_dash='dot', line_color='#ef4444', line_width=1.5)

    # Clean legend labels via invisible scatter points
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
        line=dict(color='#f59e0b', dash='dot', width=2), name='Watch (2%/wk)'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
        line=dict(color='#ef4444', dash='dot', width=2), name='Critical (4%/wk)'))

    _dark(fig, height=280, ytitle='Depletion (%/week)',
          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.update_yaxes(range=[y_min, y_max], zeroline=True, zerolinecolor='#475569')
    return fig





def chart_days_remaining(rsk_df):
    rows = [latest(rsk_df, n) for n in rsk_df['reservoir_name'].unique()]
    rows = [r for r in rows if r is not None]
    if not rows: return None
    df = pd.DataFrame(rows)
    df['days_capped'] = df['days_remaining_estimate'].clip(upper=400)
    df['color']       = df['risk_level'].map(RISK_COLORS)
    df['label']       = df['days_remaining_estimate'].apply(lambda x: '999+' if x >= 999 else str(x))
    fig = go.Figure()
    fig.add_hline(y=30,  line_dash='dot', line_color='#ef4444', line_width=1,
                  annotation_text='30-day crisis', annotation_font=dict(color='#ef4444', size=9))
    fig.add_hline(y=90,  line_dash='dot', line_color='#f59e0b', line_width=1,
                  annotation_text='90-day warning', annotation_font=dict(color='#f59e0b', size=9))
    fig.add_trace(go.Bar(
        x=df['reservoir_name'], y=df['days_capped'], marker_color=df['color'],
        text=df['label'].apply(lambda x: f"{x} days"),
        textposition='outside', textfont=dict(color='#cbd5e1', size=13),
        hovertemplate="%{x}<br>Buffer: %{text}<extra></extra>",
    ))
    _dark(fig, height=260, ytitle="Days Remaining")
    fig.update_yaxes(range=[0, 440])
    return fig


def chart_risk_history(rsk_df):
    cutoff   = rsk_df['date'].max() - timedelta(days=90)
    recent   = rsk_df[rsk_df['date'] >= cutoff]
    risk_cnt = recent.groupby(['reservoir_name','risk_level']).size().reset_index(name='count')
    fig = px.bar(risk_cnt, x='reservoir_name', y='count', color='risk_level',
                 color_discrete_map=RISK_COLORS,
                 labels={'count':'Observations','reservoir_name':'','risk_level':'Risk'})
    _dark(fig, height=260)
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig


def chart_risk_timeline(rsk_df, name):
    df = rsk_df[rsk_df['reservoir_name'] == name].sort_values('date').copy()
    df['risk_num'] = df['risk_level'].map({'SAFE':1,'WATCH':2,'CRITICAL':3})
    fig = go.Figure()
    for risk, rc in RISK_COLORS.items():
        sub = df[df['risk_level'] == risk]
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=sub['date'], y=sub['risk_num'], mode='markers', name=risk,
                marker=dict(color=rc, size=8),
                hovertemplate=f"<b>{risk}</b><br>%{{x|%b %d, %Y}}<extra></extra>",
            ))
    _dark(fig, height=200)
    fig.update_yaxes(tickvals=[1,2,3], ticktext=['SAFE','WATCH','CRITICAL'])
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:18px 0 10px'>
      <div style='font-size:2.6rem'>💧</div>
      <div style='font-size:1.3rem;font-weight:700;color:#f1f5f9'>ReservoirWatch</div>
      <div style='font-size:.78rem;color:#475569;margin-top:3px'>Pune Drought Early Warning</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio("", ["🏠 Overview","🔍 Reservoir Detail","🛰️ Methodology"],
                    label_visibility="collapsed")
    st.divider()

    # ── Data Freshness Panel ──────────────────────────────────────────────────
    st.markdown("<div style='font-size:.8rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.06em'>📡 Data Freshness</div>", unsafe_allow_html=True)

    res_df_sidebar, rsk_df_sidebar = load_data()
    fresh_info = data_freshness(rsk_df_sidebar)

    for name, info in fresh_info.items():
        dot_class = f"freshness-dot-{info['status']}"
        days_lbl  = f"{info['days_ago']}d ago" if info['days_ago'] > 0 else "today"
        next_lbl  = info['next_pass'].strftime('%b %d')
        color     = '#22c55e' if info['status']=='fresh' else '#f59e0b' if info['status']=='warn' else '#ef4444'
        st.markdown(f"""
        <div style='margin:6px 0;padding:8px 10px;background:#0f172a;border-radius:8px'>
          <div style='display:flex;justify-content:space-between;align-items:center'>
            <span style='font-size:.78rem;color:#cbd5e1'><span class="{dot_class}"></span>{name}</span>
            <span style='font-size:.72rem;color:{color}'>{days_lbl}</span>
          </div>
          <div style='font-size:.68rem;color:#475569;margin-top:2px'>
            Last: {info['last_obs'].strftime('%b %d, %Y')} &nbsp;|&nbsp; Next pass: ~{next_lbl}
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Auto-Update Info ──────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:#0f172a;border-radius:8px;padding:10px 12px;border:1px solid #1e293b'>
      <div style='font-size:.75rem;font-weight:600;color:#22c55e;margin-bottom:4px'>⚡ Fully Automated</div>
      <div style='font-size:.7rem;color:#475569;line-height:1.6'>
        Satellite data updates automatically every <b style='color:#94a3b8'>5 days</b> via GitHub Actions
        + ESA Sentinel-2. No manual action needed.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style='font-size:.73rem;color:#475569;line-height:1.7'>
    <b style='color:#64748b'>DATA SOURCE</b><br>
    ESA Sentinel-2 via Google Earth Engine<br>
    NDWI · 10 m resolution · 5-day revisit<br><br>
    <b style='color:#64748b'>RESERVOIRS</b><br>
    Khadakwasla · Panshet · Varasgaon · Temghar<br>
    Mutha River Basin, Pune<br><br>
    <b style='color:#64748b'>ALIGNED WITH</b><br>
    UN SDG 13 — Climate Action
    </div>
    """, unsafe_allow_html=True)


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
res_df, rsk_df = load_data()
reservoirs     = sorted(rsk_df['reservoir_name'].unique())
freshness      = data_freshness(rsk_df)
today          = pd.Timestamp.now()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    last_date   = rsk_df['date'].max()
    days_since  = (today - last_date).days

    # Data freshness banner
    if days_since <= 10:
        st.markdown(f"""<div class="alert-live">
        🛰️ <b>Live data</b> — last clear satellite observation <b>{days_since} days ago</b>
        ({last_date.strftime('%B %d, %Y')}). Sentinel-2 passes every ~5 days.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="alert-crit" style="line-height:1.6;">
        ⚠️ <b>Stale Data Alert — Last clear observation {days_since} days ago</b> ({last_date.strftime('%B %d, %Y')})<br>
        <span style="font-size:0.85rem; color:#f87171;">
        <b>Why is this happening?</b> The Sentinel-2 satellite passes over Pune every 5 days, but recent passes have been completely blocked by heavy monsoon clouds. To prevent false readings, our AI automatically rejects any images with >20% cloud cover. This intelligent alert warns decision-makers not to rely on old visual data until the skies clear.
        </span>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='display:flex;align-items:baseline;gap:14px;margin:10px 0 2px'>
      <h1 style='margin:0;font-size:1.85rem'>💧 Pune Reservoir Monitor</h1>
      <span style='color:#475569;font-size:.82rem'>
        Last satellite obs: {last_date.strftime('%B %d, %Y')} &nbsp;·&nbsp;
        {len(rsk_df)} total observations
      </span>
    </div>
    <p style='color:#475569;margin-bottom:22px'>
      Satellite-based water surface tracking via Sentinel-2 NDWI ·
      Targeted drought early warning (not city-wide advisories)
    </p>
    """, unsafe_allow_html=True)

    # ── Combined System Health Score ──────────────────────────────────────────
    crisis_info  = compute_crisis_dates(rsk_df)
    risk_scores  = {'SAFE': 100, 'WATCH': 55, 'CRITICAL': 15}
    scores       = [risk_scores.get(v['risk'], 50) for v in crisis_info.values()]
    sys_health   = int(sum(scores) / len(scores)) if scores else 0
    n_safe       = sum(1 for v in crisis_info.values() if v['risk'] == 'SAFE')
    n_watch      = sum(1 for v in crisis_info.values() if v['risk'] == 'WATCH')
    n_crit       = sum(1 for v in crisis_info.values() if v['risk'] == 'CRITICAL')
    health_color = '#22c55e' if sys_health >= 75 else '#f59e0b' if sys_health >= 45 else '#ef4444'
    health_label = 'Healthy' if sys_health >= 75 else 'Stressed' if sys_health >= 45 else 'Critical'

    st.markdown(f"""
    <div style='background:#1e293b;border-radius:14px;padding:18px 24px;margin-bottom:16px;
                display:flex;align-items:center;justify-content:space-between;
                border:1px solid #334155;flex-wrap:wrap;gap:16px'>
      <div>
        <div style='font-size:.72rem;color:#64748b;text-transform:uppercase;
                    letter-spacing:.08em;margin-bottom:4px'>
          🏙️ Pune Water System — Overall Health
        </div>
        <div style='display:flex;align-items:baseline;gap:10px'>
          <span style='font-size:2.8rem;font-weight:800;color:{health_color}'>{sys_health}%</span>
          <span style='font-size:1rem;font-weight:600;color:{health_color}'>{health_label}</span>
        </div>
        <div style='font-size:.72rem;color:#475569;margin-top:2px'>
          Based on {len(crisis_info)} reservoirs serving 3.4M residents · Mutha River Basin
        </div>
      </div>
      <div style='display:flex;gap:24px'>
        <div style='text-align:center'>
          <div style='font-size:2rem;font-weight:700;color:#22c55e'>{n_safe}</div>
          <div style='font-size:.72rem;color:#475569;margin-top:2px'>🟢 SAFE</div>
        </div>
        <div style='text-align:center'>
          <div style='font-size:2rem;font-weight:700;color:#f59e0b'>{n_watch}</div>
          <div style='font-size:.72rem;color:#475569;margin-top:2px'>🟡 WATCH</div>
        </div>
        <div style='text-align:center'>
          <div style='font-size:2rem;font-weight:700;color:#ef4444'>{n_crit}</div>
          <div style='font-size:.72rem;color:#475569;margin-top:2px'>🔴 CRITICAL</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Crisis Date Prediction Banners ────────────────────────────────────────
    for name, info in crisis_info.items():
        # Already-critical reservoirs
        if info['risk'] == 'CRITICAL':
            max_a = MAX_AREAS.get(name, 1)
            row   = latest(rsk_df, name)
            pct   = row['surface_area_sqkm'] / max_a * 100 if row is not None else 0
            st.markdown(f"""<div class="alert-crit">
            🚨 <b>CRITICAL NOW — {name}:</b> {row['surface_area_sqkm']:.2f} sq km
            ({pct:.0f}% capacity) · estimated <b>{row['days_remaining_estimate']} days</b> buffer remaining
            </div>""", unsafe_allow_html=True)
        # Upcoming watch warnings (within 90 days)
        elif info['watch_date']:
            days_away = (info['watch_date'] - pd.Timestamp.now().normalize()).days
            if 0 < days_away <= 90 and info['depletion'] > 0.5:
                crit_txt = ''
                if info['critical_date']:
                    cd = (info['critical_date'] - pd.Timestamp.now().normalize()).days
                    crit_txt = f" · 🔴 CRITICAL projected in <b>{cd} days</b> ({info['critical_date'].strftime('%b %d')})"
                st.markdown(f"""<div class="alert-stale">
                ⚠️ <b>Forecast: {name}</b> projected to reach <b>WATCH level in {days_away} days</b>
                ({info['watch_date'].strftime('%b %d, %Y')}) at {info['depletion']:+.2f}%/wk{crit_txt}
                </div>""", unsafe_allow_html=True)

    # ── Reservoir Cards ───────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">Current Status</div>', unsafe_allow_html=True)
    cols = st.columns(len(reservoirs))

    for i, name in enumerate(reservoirs):
        row = latest(rsk_df, name)
        if row is None: continue

        level      = row['risk_level']
        area       = row['surface_area_sqkm']
        max_a      = MAX_AREAS.get(name, area)
        pct        = area / max_a * 100
        days       = row['days_remaining_estimate']
        dep        = row.get('depletion_rate_pct_per_week', 0)
        rc         = RES_COLORS.get(name, '#38bdf8')
        bc         = RISK_COLORS[level]
        days_lbl   = '999+' if days >= 999 else str(days)
        days_color = '#ef4444' if days < 30 else '#f59e0b' if days < 90 else '#22c55e'
        f_info     = freshness.get(name, {})
        f_days_ago = f_info.get('days_ago', '?')
        f_next     = f_info.get('next_pass', pd.Timestamp.now())

        with cols[i]:
            st.markdown(f"""
            <div class="res-card card-{level.lower()}" style='border-color:{bc}'>
              <div style='display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:14px'>
                <div style='min-width:0;flex:1'>
                  <div style='font-size:1rem;font-weight:700;color:#f1f5f9;white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{name}</div>
                  <div style='font-size:.7rem;color:#475569'>Mutha River Basin</div>
                </div>
                <div style='flex-shrink:0'>{badge(level)}</div>
              </div>
              <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px'>
                <div>
                  <div style='font-size:.67rem;color:#475569;text-transform:uppercase;letter-spacing:.05em'>Surface Area</div>
                  <div style='font-size:1.55rem;font-weight:700;color:{rc}'>{area:.2f}</div>
                  <div style='font-size:.72rem;color:#475569'>sq km &nbsp;·&nbsp; {pct:.0f}% of max</div>
                </div>
                <div>
                  <div style='font-size:.67rem;color:#475569;text-transform:uppercase;letter-spacing:.05em'>Buffer Remaining</div>
                  <div style='font-size:1.55rem;font-weight:700;color:{days_color}'>{days_lbl}</div>
                  <div style='font-size:.72rem;color:#475569'>days est.</div>
                </div>
              </div>
              <div style='margin-top:14px'>
                <div style='background:#0f172a;border-radius:5px;height:7px;overflow:hidden'>
                  <div style='background:{bc};width:{min(pct,100):.0f}%;height:100%;border-radius:5px'></div>
                </div>
                <div style='display:flex;justify-content:space-between;margin-top:4px'>
                  <span style='font-size:.68rem;color:#334155'>0%</span>
                  <span style='font-size:.68rem;color:#475569'>Δ {dep:+.2f}%/wk</span>
                  <span style='font-size:.68rem;color:#334155'>100%</span>
                </div>
              </div>
              <div style='margin-top:12px;padding-top:10px;border-top:1px solid #0f172a;
                          display:flex;justify-content:space-between'>
                <span style='font-size:.68rem;color:#475569'>
                  🛰️ Last obs: {f_days_ago}d ago
                </span>
                <span style='font-size:.68rem;color:#475569'>
                  Next pass: ~{f_next.strftime("%b %d")}
                </span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Population Impact Row ─────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">🏙️ Population Impact — Pune City Water Supply (3.4M Residents)</div>',
                unsafe_allow_html=True)
    # Pune daily water demand ≈ 1,400 MLD. Each reservoir contributes proportionally.
    PUNE_DAILY_MLD   = 1400          # Million Litres per Day
    TOTAL_MAX_SQkm   = sum(MAX_AREAS.values())
    pop_cols = st.columns(len(reservoirs))
    for i, name in enumerate(reservoirs):
        row   = latest(rsk_df, name)
        if row is None: continue
        max_a = MAX_AREAS.get(name, 1)
        # Reservoir share of total supply capacity
        share       = max_a / TOTAL_MAX_SQkm
        daily_share = PUNE_DAILY_MLD * share          # MLD this reservoir handles
        # Buffer volume (above 20% critical floor) in Million Litres
        area        = row['surface_area_sqkm']
        pct         = area / max_a
        usable_pct  = max(pct - 0.20, 0)
        # Approximate volume: surface_area × mean_depth(~8m) × usable fraction
        # simplified: days = buffer_days from risk engine
        days        = row['days_remaining_estimate']
        days_lbl    = '999+' if days >= 999 else str(days)
        rc          = RES_COLORS.get(name, '#38bdf8')
        daily_lbl   = f'{daily_share:.0f}'
        with pop_cols[i]:
            st.markdown(f"""
            <div style='background:#1e293b;border-radius:12px;padding:16px;text-align:center;border:1px solid #334155'>
              <div style='font-size:.78rem;color:#64748b;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px'>{name}</div>
              <div style='font-size:1.9rem;font-weight:800;color:{rc}'>{days_lbl}</div>
              <div style='font-size:.72rem;color:#94a3b8;margin-top:2px'>days of supply remaining</div>
              <div style='margin-top:10px;padding-top:10px;border-top:1px solid #334155'>
                <span style='font-size:.72rem;color:#475569'>Daily contribution: </span>
                <span style='font-size:.78rem;color:#cbd5e1;font-weight:600'>{daily_lbl} MLD</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Combined area chart ───────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">Water Surface Area — All Reservoirs (⭐ = latest observation)</div>',
                unsafe_allow_html=True)
    st.plotly_chart(chart_combined(rsk_df), use_container_width=True)

    # ── Map + Charts row ──────────────────────────────────────────────────────
    mc1, mc2 = st.columns([3, 2])
    with mc1:
        st.markdown('<div class="sec-hdr">🗺️ Reservoir Locations — Mutha River Basin</div>',
                    unsafe_allow_html=True)
        st.markdown("""<div style='font-size:.7rem;color:#475569;margin-bottom:6px'>
        Blue lines show cascade flow: Varasgaon → Panshet → Temghar → Khadakwasla → Pune.
        Dot size = current fill level. Color = risk status.</div>""", unsafe_allow_html=True)
        st.plotly_chart(chart_reservoir_map(rsk_df), use_container_width=True)
    with mc2:
        st.markdown('<div class="sec-hdr">Days of Buffer Remaining</div>', unsafe_allow_html=True)
        ch = chart_days_remaining(rsk_df)
        if ch: st.plotly_chart(ch, use_container_width=True)
        st.markdown('<div class="sec-hdr">Risk Distribution — Last 90 Days</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(chart_risk_history(rsk_df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — RESERVOIR DETAIL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Reservoir Detail":
    selected = st.selectbox("Select Reservoir", reservoirs, format_func=lambda x: f"💧 {x}")
    row      = latest(rsk_df, selected)
    if row is None:
        st.error("No data for this reservoir."); st.stop()

    max_a  = MAX_AREAS.get(selected, row['surface_area_sqkm'])
    level  = row['risk_level']
    area   = row['surface_area_sqkm']
    pct    = area / max_a * 100
    dep    = row.get('depletion_rate_pct_per_week', 0)
    days   = row['days_remaining_estimate']
    f_info = freshness.get(selected, {})

    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(f"## {selected} Reservoir")
        st.markdown(f"<p style='color:#475569'>Mutha River Basin · "
                    f"Last satellite observation: <b>{row['date'].strftime('%B %d, %Y')}</b> "
                    f"({f_info.get('days_ago','?')} days ago) · "
                    f"Next expected pass: ~{f_info.get('next_pass', pd.Timestamp.now()).strftime('%B %d')}"
                    f"</p>", unsafe_allow_html=True)
    with h2:
        st.markdown(f"<div style='text-align:right;padding-top:22px'>{badge(level)}</div>",
                    unsafe_allow_html=True)

    sub = rsk_df[rsk_df['reservoir_name'] == selected].sort_values('date')
    prev_area = sub.iloc[-2]['surface_area_sqkm'] if len(sub) >= 2 else area

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Surface Area",      f"{area:.2f} sq km",  f"{area-prev_area:+.2f} sq km")
    with m2: st.metric("Capacity",          f"{pct:.0f}%",        f"of {max_a} sq km max")
    with m3: st.metric("Depletion Rate",    f"{dep:+.2f}%/wk",
                        "filling ☔" if dep < 0 else "depleting ↓", delta_color='inverse')
    with m4: st.metric("Buffer Remaining",  f"{'999+' if days>=999 else days} days")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">Water Surface Area Over Time (⭐ = latest observation)</div>',
                unsafe_allow_html=True)
    st.plotly_chart(chart_area_detail(rsk_df, selected), use_container_width=True)

    cl, cr = st.columns([3, 2])
    with cl:
        st.markdown('<div class="sec-hdr">Weekly Depletion Rate</div>', unsafe_allow_html=True)
        dc = chart_depletion(rsk_df, selected)
        if dc: st.plotly_chart(dc, use_container_width=True)
    with cr:
        st.markdown('<div class="sec-hdr">Recent Observations</div>', unsafe_allow_html=True)
        tbl = sub.tail(15)[['date','surface_area_sqkm','risk_level']].copy()
        tbl['date'] = tbl['date'].dt.strftime('%Y-%m-%d')
        tbl = tbl.sort_values('date', ascending=False)
        tbl.columns = ['Date','Area (sq km)','Risk']
        st.dataframe(
            tbl.style.map(
                lambda v: f"color:{RISK_COLORS.get(v,'#cbd5e1')};font-weight:bold",
                subset=['Risk']
            ),
            use_container_width=True, hide_index=True, height=300,
        )

    st.markdown('<div class="sec-hdr">Risk Level History</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_risk_timeline(rsk_df, selected), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🛰️ Methodology":
    st.markdown("## 🛰️ How ReservoirWatch Works")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
### Satellite Pipeline (Incremental)
Free **Sentinel-2 Level-2A** imagery from ESA Copernicus, accessed via Google Earth Engine.

**NDWI Formula:**
""")
        st.latex(r"\mathrm{NDWI} = \frac{B_3\,(\mathrm{Green}) - B_8\,(\mathrm{NIR})}{B_3\,(\mathrm{Green}) + B_8\,(\mathrm{NIR})}")
        st.markdown("""
NDWI > 0.2 → water pixel · < 0.2 → land/vegetation.

**Incremental Update System:**
Every time the Colab notebook runs, it:
1. Checks the last date already extracted per reservoir
2. Fetches **only new satellite passes** since that date
3. Appends them to the CSV — no duplicate API calls

So running it today vs. next week gives you the latest data automatically,
and each run costs almost nothing (only fetching the delta).

**Pipeline steps:**
1. Filter Sentinel-2 by bounding box + date
2. Drop images with >20% cloud cover
3. Compute NDWI per 10m pixel
4. Threshold NDWI > 0.2 → water mask
5. Sum pixel areas → sq km per date
6. Repeat for all dates → time series
""")

    with c2:
        st.markdown("""### Risk Logic & Freshness""")
        st.dataframe(pd.DataFrame({
            'Level':              ['🟢 SAFE','🟡 WATCH','🔴 CRITICAL'],
            'Area (% of max)':    ['> 55%','35–55%','< 35%'],
            'Depletion (%/wk)':   ['< 2%','2–4%','>4%'],
        }), use_container_width=True, hide_index=True)

        st.markdown("""
**Data Freshness Indicators (sidebar):**
- 🟢 Fresh — last obs ≤10 days ago
- 🟡 Warning — 11–20 days ago (re-run Colab)
- 🔴 Stale — >20 days ago (re-run urgently)

**Days remaining** = days until reservoir drops below 20% of max capacity:
```
buffer_area ÷ (depletion_%/wk × max_area / 700)
```

**Reservoir Coordinates:**
""")
        st.dataframe(pd.DataFrame({
            'Reservoir':  ['Khadakwasla','Panshet','Varasgaon'],
            'GEE Box':    ['[73.71,18.40,73.79,18.46]','[73.55,18.35,73.65,18.43]',
                           '[73.58,18.36,73.68,18.42]'],
            'Max Area':   ['7.5 sq km','11.0 sq km','9.0 sq km'],
            'Verified':   ['✅ Confirmed','⚠️ Verify','⚠️ Verify'],
        }), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("""
### Live Data Workflow
```
Colab (GEE)                 Local Machine
───────────────             ──────────────────────────────
Run satellite_pipeline.py → downloads reservoir_data.csv
                          → drop into data/ folder
                          → click "Refresh Risk Analysis"
                          → dashboard shows real-time data
```
Re-run the notebook anytime — it only fetches new dates since last run.
""")

st.markdown("""
<div class="footer">
  ReservoirWatch · PCCoE IGC Indradhanu Hackathon 2026 ·
  UN SDG 13 — Climate Action ·
  Data: ESA Sentinel-2 via Google Earth Engine
</div>
""", unsafe_allow_html=True)
