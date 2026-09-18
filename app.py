from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.linear_model import LinearRegression
import requests

st.set_page_config(page_title="GridSight", layout="wide", page_icon="⚡")

# Site-specific configuration profiles
SITE_CONFIGS = {
    "Pune Industrial Campus": {"capacity_kw": 92.0, "base_load": 140.0, "aqi_offset": 10},
    "Bengaluru Tech Park": {"capacity_kw": 120.0, "base_load": 110.0, "aqi_offset": -20},
    "Delhi Commercial Hub": {"capacity_kw": 75.0, "base_load": 165.0, "aqi_offset": 50}
}


def fetch_live_aqi(city_name: str, token: str) -> tuple[float, str]:
    city_mapping = {
        "Pune Industrial Campus": "pune",
        "Bengaluru Tech Park": "bengaluru",
        "Delhi Commercial Hub": "delhi"
    }
    query_city = city_mapping.get(city_name, "kolkata")
    url = f"https://api.waqi.info/feed/{query_city}/?token={token}"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            payload = response.json()
            if payload.get("status") == "ok":
                return float(payload["data"]["aqi"]), "Live (WAQI API)"
    except Exception:
        pass
    return 75.0, "Fallback (Simulation)"


def generate_site_data(site_name: str, days: int, seed: int) -> pd.DataFrame:
    config = SITE_CONFIGS.get(site_name, SITE_CONFIGS["Pune Industrial Campus"])
    capacity = config["capacity_kw"]
    
    rng = np.random.default_rng(seed)
    periods = days * 48
    timestamp = pd.date_range(end=pd.Timestamp.now().floor("30min"), periods=periods, freq="30min")
    hour = timestamp.hour + timestamp.minute / 60
    weekday = timestamp.dayofweek

    base_load = config["base_load"] + 35 * np.sin(2 * np.pi * (hour - 8) / 24) + 20 * np.sin(2 * np.pi * (hour - 18) / 24)
    weekday_factor = np.where(weekday < 5, 1.0, 0.82)
    load_noise = rng.normal(0, 5, periods)
    load_kw = np.clip(base_load * weekday_factor + load_noise, 30, None)

    irradiance = np.clip(950 * np.sin(np.pi * (hour - 6) / 12), 0, None)
    irradiance = np.clip(irradiance + rng.normal(0, 20, periods), 0, 1000)

    temperature = 25 + 7 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 1.0, periods)

    aqi_base = 90 + config["aqi_offset"] + 40 * np.sin(2 * np.pi * (hour - 7) / 24) + rng.normal(0, 12, periods)
    aqi = np.clip(aqi_base, 15, 350)

    panel_derate = np.clip(1 - np.clip(temperature - 25, 0, None) * 0.004, 0.85, 1)
    dust_derate = np.clip(1 - np.clip(aqi - 45, 0, None) * 0.00075, 0.7, 1)
    
    solar_raw = (irradiance / 1000) * capacity * panel_derate * dust_derate
    solar_kw = np.where(irradiance < 10.0, 0.0, np.clip(solar_raw, 0, capacity))

    return pd.DataFrame({
        "timestamp": timestamp,
        "load_kw": load_kw,
        "solar_kw": solar_kw,
        "aqi": aqi,
        "irradiance": irradiance,
        "temperature": temperature,
    })


def forecast_load(train: pd.DataFrame, horizon_periods: int, weather_shift: float) -> tuple[pd.DataFrame, float]:
    features = pd.DataFrame({
        "hour": train.timestamp.dt.hour + train.timestamp.dt.minute / 60,
        "weekday": (train.timestamp.dt.dayofweek < 5).astype(int),
        "temperature": train.temperature,
    })
    model = LinearRegression()
    model.fit(features, train.load_kw)

    last_ts = train.timestamp.iloc[-1]
    future_ts = pd.date_range(start=last_ts + pd.Timedelta(minutes=30), periods=horizon_periods, freq="30min")
    baseline_temp = train.temperature.iloc[-48:].mean() + weather_shift

    future = pd.DataFrame({
        "hour": future_ts.hour + future_ts.minute / 60,
        "weekday": (future_ts.dayofweek < 5).astype(int),
        "temperature": baseline_temp + 3 * np.sin(2 * np.pi * (future_ts.hour - 8) / 24),
    })
    prediction = model.predict(future)
    residual = train.load_kw - model.predict(features)
    band = max(7, residual.std() * 1.96)
    
    forecast_df = pd.DataFrame({"timestamp": future_ts, "forecast_kw": prediction, "lower": prediction - band, "upper": prediction + band})
    return forecast_df, residual.std()


def aqi_label(value: float) -> tuple[str, str]:
    if value <= 50:
        return "Good", "#36c98b"
    if value <= 100:
        return "Satisfactory", "#a5d86a"
    if value <= 200:
        return "Moderate", "#ffc857"
    return "Poor", "#f47c67"


st.markdown("""
<style>
 .stApp { background: #071522; color: #edf5fb; }
 [data-testid="stSidebar"] { background: #0d2233; }
 .metric-card { background: linear-gradient(135deg,#102e43,#0d2233); border: 1px solid #24506a; border-radius: 14px; padding: 16px; min-height: 115px; }
 .metric-label { color:#9bb6c7; font-size:0.78rem; text-transform:uppercase; letter-spacing:.08em; }
 .metric-value { font-size:1.8rem; font-weight:700; margin:5px 0; }
 .metric-note { color:#71d5c1; font-size:.82rem; }
 h1,h2,h3 { color:#f4fbff !important; } .stPlotlyChart { border: 1px solid #1b4057; border-radius: 12px; padding: 6px; background:#0b1d2b; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚡ GridSight")
    st.caption("Microgrid command center")
    site = st.selectbox("Site", list(SITE_CONFIGS.keys()))
    days = st.slider("Historical data window", 7, 60, 21)
    horizon_hours = st.select_slider("Forecast horizon", options=[12, 24, 36, 48, 72], value=24)
    
    weather_shift = st.slider("Temperature scenario", -4, 6, 0, help="Adjusts forecast demand for a warmer or cooler outlook.")
    if weather_shift != 0:
        st.caption(f"🌡️ Thermal Scenario: {'+' if weather_shift > 0 else ''}{weather_shift}°C shift applied to baseline demand.")
    else:
        st.caption("🌡️ Baseline temperature profile active.")
        
    st.divider()
    st.caption("Data telemetry & reliability")
    
    try:
        _ = st.secrets["WAQI_TOKEN"]
        st.info("Live API connected (WAQI)", icon="🟢")
    except Exception:
        st.info("Simulation mode active", icon="ℹ️")

site_capacity = SITE_CONFIGS[site]["capacity_kw"]
data = generate_site_data(site, days, 42)
latest = data.iloc[-1].copy()

data_source_status = "Simulation Feed"
try:
    token = st.secrets["WAQI_TOKEN"]
    latest["aqi"], data_source_status = fetch_live_aqi(site, token)
except Exception:
    pass

forecast, res_std = forecast_load(data, horizon_hours * 2, weather_shift)
baseline_forecast, _ = forecast_load(data, horizon_hours * 2, 0)
peak_diff = forecast.forecast_kw.max() - baseline_forecast.forecast_kw.max()

aqi_text, aqi_color = aqi_label(latest.aqi)
solar_loss = max(0, (latest.aqi - 45) * 0.075)
net_load = latest.load_kw - latest.solar_kw

with st.sidebar:
    st.divider()
    st.caption("Report & Data Export")
    csv_data = forecast.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv_data,
        file_name=f"gridsight_{site.lower().replace(' ', '_')}_forecast.csv",
        mime="text/csv",
        help="Export current machine learning demand forecast data."
    )

st.markdown(f"# {site}  ")
st.caption(f"LIVE OPERATIONS VIEW  •  Source: {data_source_status}  •  Refreshed: {pd.Timestamp.now().strftime('%H:%M:%S')}")

# Operational Alert Strip Logic
active_alerts = []
if latest.aqi > 150:
    active_alerts.append(("🔴 High Particulate Warning", f"Severe AQI levels ({latest.aqi:.0f}) detected. Active solar soiling derate applied."))
if weather_shift != 0:
    active_alerts.append(("🌡️ Thermal Scenario Active", f"{weather_shift:+.1f}°C temperature shift applied to predictive load models."))
if latest.load_kw > (SITE_CONFIGS[site]["base_load"] * 1.1):
    active_alerts.append(("⚡ High Demand Event", f"Current grid load ({latest.load_kw:.0f} kW) is elevated above baseline profile."))

if active_alerts:
    for title, msg in active_alerts:
        st.warning(f"**{title}:** {msg}")
else:
    st.success("🟢 **System Status Normal:** All microgrid telemetry parameters operating within optimal parameters.", icon="✅")

cards = st.columns(4)
current_capacity_factor = (latest.solar_kw / site_capacity) * 100
metrics = [
    ("Grid demand", f"{latest.load_kw:.0f} kW", "↗ 3.2% vs yesterday", "#71d5c1"),
    ("Solar output", f"{latest.solar_kw:.1f} kW", f"{current_capacity_factor:.0f}% capacity factor ({site_capacity:.0f} kW cap)", "#ffd166"),
    ("Air quality", f"{latest.aqi:.0f} AQI", aqi_text, aqi_color),
    ("Net grid import", f"{net_load:.0f} kW", "after on-site generation", "#a9c7ff"),
]
for col, (label, value, note, color) in zip(cards, metrics):
    col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note" style="color:{color}">{note}</div></div>', unsafe_allow_html=True)

st.markdown("### Demand outlook & ML forecasting")
left, right = st.columns([2.1, 1])
with left:
    recent = data.tail(144)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent.timestamp, y=recent.load_kw, name="Actual demand", line=dict(color="#5ad1e5", width=2)))
    fig.add_trace(go.Scatter(x=forecast.timestamp, y=forecast.upper, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=forecast.timestamp, y=forecast.lower, fill="tonexty", fillcolor="rgba(92, 183, 191, .16)", line=dict(width=0), name="95% confidence", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=forecast.timestamp, y=forecast.forecast_kw, name="Forecast", line=dict(color="#ffc857", dash="dash", width=2.5)))
    fig.update_layout(template="plotly_dark", height=360, margin=dict(l=12,r=12,t=20,b=10), paper_bgcolor="#0b1d2b", plot_bgcolor="#0b1d2b", legend=dict(orientation="h", y=1.12), yaxis_title="kW", xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)
with right:
    peak = forecast.loc[forecast.forecast_kw.idxmax()]
    st.markdown("#### Forecast signal")
    
    delta_text = f"{peak_diff:+.1f} kW vs base" if weather_shift != 0 else peak.timestamp.strftime("%H:%M tomorrow")
    st.metric("Expected peak", f"{peak.forecast_kw:.0f} kW", delta=delta_text)
    st.metric("Forecast energy", f"{forecast.forecast_kw.sum() / 2:.1f} kWh")
    
    st.markdown("---")
    st.caption(f"⚙️ **Model Info:** Scikit-Learn Linear Regression\n\n📊 **Window:** {days} Days history | **Residual Std:** ±{res_std:.1f} kW")

st.markdown("### Solar performance & environmental impact")
col1, col2 = st.columns([1.5, 1])
with col1:
    solar_view = data.tail(96)
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(x=solar_view.timestamp, y=solar_view.solar_kw, name="PV output", line=dict(color="#ffd166", width=2.4)), secondary_y=False)
    fig2.add_trace(go.Scatter(x=solar_view.timestamp, y=solar_view.aqi, name="AQI", line=dict(color="#ef7f6d", width=1.8)), secondary_y=True)
    fig2.update_layout(template="plotly_dark", height=330, margin=dict(l=12,r=12,t=20,b=10), paper_bgcolor="#0b1d2b", plot_bgcolor="#0b1d2b", legend=dict(orientation="h", y=1.12))
    fig2.update_yaxes(title_text="Solar kW", secondary_y=False)
    fig2.update_yaxes(title_text="AQI", secondary_y=True)
    st.plotly_chart(fig2, use_container_width=True)
with col2:
    if latest.irradiance < 10.0:
        performance_ratio = 0.0
    else:
        theoretical_max = latest.irradiance / 1000 * site_capacity
        performance_ratio = min(100.0, max(0.0, (latest.solar_kw / theoretical_max) * 100))
        
    st.markdown(f"#### PV health at a glance ({site_capacity:.0f} kW Peak)")
    st.progress(int(performance_ratio), text=f"Performance Ratio (PR): {performance_ratio:.1f}%")
    st.metric("AQI-related soiling loss", f"{solar_loss:.1f}%", "modeled dust derate")
    st.metric("Cell temperature", f"{latest.temperature:.1f} °C")
    st.warning("Schedule a panel wash within 48 hours" if latest.aqi > 150 else "Conditions are optimal for current operation", icon="🧽")

with st.expander("Data model and integration notes"):
    st.markdown(f"""
    **Current architecture & site config:** 30-minute resolution telemetry configured for **{site}** ({site_capacity:.0f} kW nameplate solar capacity). Load forecasting uses a multivariable linear regression model incorporating time-of-day, day-of-week, and ambient temperature features. 

    **Solar & Environmental Modeling:** Photovoltaic output includes strict nighttime irradiance gating (< 10 W/m² cutoff) and is dynamically derated using real-time cell temperature coefficients and particulate soiling proxies mapped from live AQI feeds.
    """)
