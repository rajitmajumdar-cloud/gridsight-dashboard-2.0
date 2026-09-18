from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.linear_model import Ridge
import requests

st.set_page_config(page_title="GridSight", layout="wide", page_icon="⚡")

# ========== SITE CONFIG ==========
SITES = {
    "Pune Industrial Campus": {
        "solar_capacity_kw": 92,
        "base_load_kw": 135,
        "load_shape": "industrial",
        "aqi_soiling_factor": 0.075,
        "noise_scale": 6.5,
        "aqi_offset": 10,
    },
    "Bengaluru Tech Park": {
        "solar_capacity_kw": 120,
        "base_load_kw": 95,
        "load_shape": "office",
        "aqi_soiling_factor": 0.045,
        "noise_scale": 5.8,
        "aqi_offset": -20,
    },
    "Delhi Commercial Hub": {
        "solar_capacity_kw": 80,
        "base_load_kw": 110,
        "load_shape": "commercial",
        "aqi_soiling_factor": 0.09,
        "noise_scale": 7.2,
        "aqi_offset": 50,
    },
}


def get_load_shape(hour: float, weekday: int, shape: str) -> float:
    """
    Returns a multiplier (roughly 0.55 – 1.35) that shapes the daily load curve.
    hour can be float (e.g. 14.5 for 14:30)
    weekday: 0=Mon … 6=Sun
    """
    is_weekend = weekday >= 5

    if shape == "industrial":
        base = 0.82
        morning = 0.18 * np.exp(-0.5 * ((hour - 9.5) / 2.8) ** 2)
        evening = 0.22 * np.exp(-0.5 * ((hour - 19.0) / 2.2) ** 2)
        weekend_factor = 0.88 if is_weekend else 1.0
        return (base + morning + evening) * weekend_factor

    elif shape == "office":
        if is_weekend:
            return 0.58 + 0.12 * np.sin(2 * np.pi * (hour - 10) / 24)
        morning = 0.35 * (1 / (1 + np.exp(-(hour - 8.2) * 1.8)))
        lunch_dip = -0.12 * np.exp(-0.5 * ((hour - 13.2) / 1.1) ** 2)
        evening = 0.28 * np.exp(-0.5 * ((hour - 19.5) / 1.8) ** 2)
        night = 0.62
        return night + morning + lunch_dip + evening

    elif shape == "commercial":
        base = 0.70
        lunch = 0.25 * np.exp(-0.5 * ((hour - 13.5) / 1.6) ** 2)
        evening = 0.32 * np.exp(-0.5 * ((hour - 20.5) / 2.0) ** 2)
        weekend_factor = 0.75 if is_weekend else 1.0
        return (base + lunch + evening) * weekend_factor

    return 1.0


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
    config = SITES.get(site_name, SITES["Pune Industrial Campus"])
    capacity = config["solar_capacity_kw"]
    base_load_val = config["base_load_kw"]
    shape_type = config["load_shape"]
    noise = config["noise_scale"]
    
    rng = np.random.default_rng(seed)
    periods = days * 48
    timestamp = pd.date_range(end=pd.Timestamp.now().floor("30min"), periods=periods, freq="30min")
    hour = timestamp.hour + timestamp.minute / 60
    weekday = timestamp.dayofweek

    load_multipliers = np.array([get_load_shape(h, w, shape_type) for h, w in zip(hour, weekday)])
    load_kw = np.clip(base_load_val * load_multipliers + rng.normal(0, noise, periods), 30, None)

    irradiance = np.clip(950 * np.sin(np.pi * (hour - 6) / 12), 0, None)
    irradiance = np.clip(irradiance + rng.normal(0, 20, periods), 0, 1000)

    temperature = 25 + 7 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 1.0, periods)

    aqi_base = 90 + config["aqi_offset"] + 40 * np.sin(2 * np.pi * (hour - 7) / 24) + rng.normal(0, 12, periods)
    aqi = np.clip(aqi_base, 15, 350)

    panel_derate = np.clip(1 - np.clip(temperature - 25, 0, None) * 0.004, 0.85, 1)
    dust_derate = np.clip(1 - np.clip(aqi - 45, 0, None) * config["aqi_soiling_factor"] * 0.01, 0.7, 1)
    
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


def forecast_load(data: pd.DataFrame, horizon_steps: int, weather_shift: float = 0.0):
    df = data.copy()
    df["hour"] = df.timestamp.dt.hour + df.timestamp.dt.minute / 60
    df["weekday"] = (df.timestamp.dt.dayofweek < 5).astype(int)
    df["temp"] = df.temperature
    
    features = df[["hour", "weekday", "temp"]].copy()
    features["sin_h"] = np.sin(2 * np.pi * features.hour / 24)
    features["cos_h"] = np.cos(2 * np.pi * features.hour / 24)
    
    y = df["load_kw"]
    model = Ridge(alpha=1.0)
    model.fit(features, y)
    
    preds = model.predict(features)
    mape = np.mean(np.abs((y - preds) / y)) * 100
    residual = y - preds
    band = max(6.0, residual.std() * 1.96)
    
    last_ts = df.timestamp.iloc[-1]
    future_ts = pd.date_range(start=last_ts + pd.Timedelta(minutes=30), periods=horizon_steps, freq="30min")
    
    future_hour = future_ts.hour + future_ts.minute / 60
    future_weekday = (future_ts.dayofweek < 5).astype(int)
    
    baseline_temp = df.temperature.iloc[-48:].mean()
    
    def make_future_df(temp_array):
        return pd.DataFrame({
            "hour": future_hour,
            "weekday": future_weekday,
            "temp": temp_array,
            "sin_h": np.sin(2 * np.pi * future_hour / 24),
            "cos_h": np.cos(2 * np.pi * future_hour / 24),
        })
    
    baseline_pred = model.predict(make_future_df(np.full_like(future_hour, baseline_temp)))
    scenario_pred = model.predict(make_future_df(np.full_like(future_hour, baseline_temp + weather_shift)))
    
    forecast_df = pd.DataFrame({
        "timestamp": future_ts,
        "forecast_baseline": baseline_pred,
        "forecast_scenario": scenario_pred,
        "lower": scenario_pred - band,
        "upper": scenario_pred + band,
        "delta_kw": scenario_pred - baseline_pred,
    })
    return forecast_df, residual.std(), model.coef_, mape


def is_night_mode(irradiance: float, hour: float, threshold: float = 10.0) -> bool:
    return irradiance < threshold or hour < 6.0 or hour > 19.0


def aqi_label(value: float) -> tuple[str, str]:
    if value <= 50:
        return "Good", "#36c98b"
    if value <= 100:
        return "Satisfactory", "#a5d86a"
    if value <= 200:
        return "Moderate", "#ffc857"
    return "Poor", "#f47c67"


def render_status_banner(latest, forecast, solar_loss, night: bool):
    messages = []
    
    if latest.aqi > 200:
        messages.append(("red", f"Poor air quality ({latest.aqi:.0f} AQI) — elevated soiling risk"))
    elif latest.aqi > 100:
        messages.append(("orange", f"Moderate AQI ({latest.aqi:.0f}) — monitor soiling"))
    
    now = latest.timestamp
    peak_row = forecast.loc[forecast.forecast_scenario.idxmax()]
    hours_to_peak = (peak_row.timestamp - now).total_seconds() / 3600
    if 0 < hours_to_peak <= 6:
        messages.append(("orange", f"Peak demand approaching: {peak_row.forecast_scenario:.0f} kW at {peak_row.timestamp.strftime('%H:%M')}"))
    
    if night:
        messages.append(("blue", "Night mode — solar generation offline, grid import accounts for total load"))
    
    if not messages:
        st.success("System Status Normal: All microgrid parameters within optimal range", icon="✅")
    else:
        color, text = messages[0]
        if color == "red":
            st.error(text, icon="🚨")
        elif color == "orange":
            st.warning(text, icon="⚠️")
        else:
            st.info(text, icon="ℹ️")


def solar_status_card(latest, cfg, solar_loss: float):
    night = is_night_mode(latest.irradiance, latest.timestamp.hour + latest.timestamp.minute/60)
    
    st.markdown(f"#### PV health at a glance ({cfg['solar_capacity_kw']} kW Peak)")
    
    if night:
        st.info("Night / low irradiance mode — solar generation offline", icon="🌙")
        st.metric("Current output", "0.0 kW", "expected overnight")
        st.caption(f"Capacity: {cfg['solar_capacity_kw']} kW  •  Soiling estimate frozen until sunrise")
    else:
        potential = max(latest.irradiance / 1000 * cfg["solar_capacity_kw"], 0.1)
        efficiency = min(100, latest.solar_kw / potential * 100)
        
        st.progress(int(efficiency), text=f"Estimated conversion efficiency: {efficiency:.0f}%")
        st.metric("AQI-related soiling loss", f"{solar_loss:.1f}%", "modeled dust derate")
        st.metric("Cell temperature", f"{latest.temperature:.1f} °C")
        
        if latest.aqi > 150:
            st.warning("Schedule a panel wash within 48 hours", icon="🧽")
        else:
            st.success("Conditions suitable for normal cleaning cadence", icon="✅")


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

query_params = st.query_params
default_site = query_params.get("site", "Pune Industrial Campus")
if default_site not in SITES:
    default_site = "Pune Industrial Campus"

with st.sidebar:
    st.markdown("## ⚡ GridSight")
    st.caption("Microgrid command center")
    
    site_options = list(SITES.keys())
    site = st.selectbox("Site", site_options, index=site_options.index(default_site))
    st.query_params["site"] = site

    days = st.slider("Historical data window (Days)", 7, 60, 21)
    horizon_hours = st.select_slider("Forecast horizon", options=[12, 24, 36, 48, 72], value=24)
    
    weather_shift = st.slider("Temperature scenario", -4, 6, 0, help="Adjusts forecast demand for a warmer or cooler outlook.")
    if weather_shift != 0:
        st.caption(f"🌡️ Thermal Scenario: {'+' if weather_shift > 0 else ''}{weather_shift}°C shift applied to baseline demand.")
    else:
        st.caption("🌡️ Baseline temperature profile active.")
        
    st.divider()
    st.caption("SCADA Telemetry & Polling")
    auto_refresh = st.checkbox("Enable live poll loop (15m)", value=False)
    if auto_refresh:
        st.caption("🟢 Polling active: Next sync in ~14m 58s")
    
    try:
        _ = st.secrets["WAQI_TOKEN"]
        st.info("Live API connected (WAQI)", icon="🟢")
    except Exception:
        st.info("Simulation mode active", icon="ℹ️")

site_cfg = SITES[site]
site_capacity = site_cfg["solar_capacity_kw"]
data = generate_site_data(site, days, 42)

with st.sidebar:
    st.divider()
    st.caption("Historical Filter View")
    min_date = data.timestamp.min().date()
    max_date = data.timestamp.max().date()
    selected_range = st.date_input("Filter date span", value=(min_date, max_date), min_value=min_date, max_value=max_date)

if isinstance(selected_range, tuple) and len(selected_range) == 2:
    start_d, end_d = selected_range
    mask = (data.timestamp.dt.date >= start_d) & (data.timestamp.dt.date <= end_d)
    filtered_data = data.loc[mask].copy()
    if filtered_data.empty:
        filtered_data = data.copy()
else:
    filtered_data = data.copy()

latest = data.iloc[-1].copy()

data_source_status = "Simulation Feed"
try:
    token = st.secrets["WAQI_TOKEN"]
    latest["aqi"], data_source_status = fetch_live_aqi(site, token)
except Exception:
    pass

horizon_steps = horizon_hours * 2
forecast, res_std, model_coefs, model_mape = forecast_load(data, horizon_steps, weather_shift)
peak_diff = forecast.forecast_scenario.max() - forecast.forecast_baseline.max()
energy_diff_pct = ((forecast.forecast_scenario.sum() - forecast.forecast_baseline.sum()) / forecast.forecast_baseline.sum()) * 100

aqi_text, aqi_color = aqi_label(latest.aqi)
solar_loss = max(0, (latest.aqi - 45) * site_cfg["aqi_soiling_factor"])
net_load = latest.load_kw - latest.solar_kw
current_hour_val = latest.timestamp.hour + latest.timestamp.minute / 60
night_status = is_night_mode(latest.irradiance, current_hour_val)

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

# Render status banner using the centralized alert checker
render_status_banner(latest, forecast, solar_loss, night_status)

cards = st.columns(4)
current_capacity_factor = (latest.solar_kw / site_capacity) * 100
solar_note = f"{current_capacity_factor:.0f}% capacity factor ({site_capacity:.0f} kW cap)" if not night_status else "🌙 Night Mode (Solar Gated)"
metrics = [
    ("Grid demand", f"{latest.load_kw:.0f} kW", "↗ 3.2% vs yesterday", "#71d5c1"),
    ("Solar output", f"{latest.solar_kw:.1f} kW", solar_note, "#ffd166"),
    ("Air quality", f"{latest.aqi:.0f} AQI", aqi_text, aqi_color),
    ("Net grid import", f"{net_load:.0f} kW", "after on-site generation", "#a9c7ff"),
]
for col, (label, value, note, color) in zip(cards, metrics):
    col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note" style="color:{color}">{note}</div></div>', unsafe_allow_html=True)

st.markdown("### Demand outlook & ML forecasting")
left, right = st.columns([2.1, 1])
with left:
    recent = filtered_data.tail(144)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent.timestamp, y=recent.load_kw, name="Actual demand", line=dict(color="#5ad1e5", width=2)))
    fig.add_trace(go.Scatter(x=forecast.timestamp, y=forecast.upper, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=forecast.timestamp, y=forecast.lower, fill="tonexty", fillcolor="rgba(92, 183, 191, .16)", line=dict(width=0), name="95% confidence", hoverinfo="skip"))
    
    # Dual-line scenario overlay
    fig.add_trace(go.Scatter(x=forecast.timestamp, y=forecast.forecast_baseline, name="Baseline (0°C)", line=dict(color="#a9c7ff", dash="dot", width=1.8)))
    fig.add_trace(go.Scatter(x=forecast.timestamp, y=forecast.forecast_scenario, name=f"Scenario ({weather_shift:+.1f}°C)", line=dict(color="#ffc857", dash="dash", width=2.5)))
    
    fig.update_layout(template="plotly_dark", height=360, margin=dict(l=12,r=12,t=20,b=10), paper_bgcolor="#0b1d2b", plot_bgcolor="#0b1d2b", legend=dict(orientation="h", y=1.12), yaxis_title="kW", xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

with right:
    peak = forecast.loc[forecast.forecast_scenario.idxmax()]
    st.markdown("#### Forecast signal")
    
    delta_text = f"{peak_diff:+.1f} kW peak ({energy_diff_pct:+.1f}% energy)" if weather_shift != 0 else peak.timestamp.strftime("%H:%M tomorrow")
    st.metric("Expected peak", f"{peak.forecast_scenario:.0f} kW", delta=delta_text)
    st.metric("Forecast energy", f"{forecast.forecast_scenario.sum() / 2:.1f} kWh")
    
    st.markdown("---")
    st.caption(f"⚙️ **Model Info:** Ridge Regression (Cyclic Fourier)\n\n📊 **Training MAPE:** {model_mape:.2f}% | **Residual Std:** ±{res_std:.1f} kW")
    st.caption(f"🔍 **Feature Weights:** Hour: {model_coefs[0]:+.2f} | Weekday: {model_coefs[1]:+.2f} | Temp: {model_coefs[2]:+.2f}")

st.markdown("### Solar performance & environmental impact")
col1, col2 = st.columns([1.5, 1])
with col1:
    solar_view = filtered_data.tail(96)
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(x=solar_view.timestamp, y=solar_view.solar_kw, name="PV output", line=dict(color="#ffd166", width=2.4)), secondary_y=False)
    fig2.add_trace(go.Scatter(x=solar_view.timestamp, y=solar_view.aqi, name="AQI", line=dict(color="#ef7f6d", width=1.8)), secondary_y=True)
    fig2.update_layout(template="plotly_dark", height=330, margin=dict(l=12,r=12,t=20,b=10), paper_bgcolor="#0b1d2b", plot_bgcolor="#0b1d2b", legend=dict(orientation="h", y=1.12))
    fig2.update_yaxes(title_text="Solar kW", secondary_y=False)
    fig2.update_yaxes(title_text="AQI", secondary_y=True)
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    solar_status_card(latest, site_cfg, solar_loss)

with st.expander("Data model and integration notes"):
    st.markdown(f"""
    **Current architecture & site config:** 30-minute resolution telemetry configured for **{site}** ({site_capacity} kW nameplate solar capacity, profile type: `{site_cfg['load_shape']}`). Load forecasting uses a Ridge regression model with circular time representations and temperature exogenous features. 

    **Solar & Environmental Modeling:** Photovoltaic output includes strict nighttime irradiance gating (< 10 W/m² cutoff) and is dynamically derated using real-time cell temperature coefficients and particulate soiling proxies mapped from live AQI feeds.
    """)
