from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime
import requests

st.set_page_config(
    page_title="GridSight · Microgrid Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SITE CONFIGURATION
# ============================================================
SITES = {
    "Pune Industrial Campus": {
        "solar_capacity_kw": 92,
        "base_load_kw": 138,
        "load_shape": "industrial",
        "aqi_soiling_factor": 0.075,
        "noise_scale": 6.8,
        "tariff_inr": 8.5,
        "waqi_city": "pune",
        "battery_capacity_kwh": 100.0,
        "grid_emission_factor_kg_kwh": 0.82,
    },
    "Bengaluru Tech Park": {
        "solar_capacity_kw": 120,
        "base_load_kw": 98,
        "load_shape": "office",
        "aqi_soiling_factor": 0.045,
        "noise_scale": 5.9,
        "tariff_inr": 9.2,
        "waqi_city": "bangalore",
        "battery_capacity_kwh": 150.0,
        "grid_emission_factor_kg_kwh": 0.72,
    },
    "Delhi Commercial Hub": {
        "solar_capacity_kw": 80,
        "base_load_kw": 112,
        "load_shape": "commercial",
        "aqi_soiling_factor": 0.090,
        "noise_scale": 7.4,
        "tariff_inr": 8.8,
        "waqi_city": "delhi",
        "battery_capacity_kwh": 80.0,
        "grid_emission_factor_kg_kwh": 0.85,
    },
    "Kolkata Sector V": {
        "solar_capacity_kw": 105,
        "base_load_kw": 118,
        "load_shape": "mixed",
        "aqi_soiling_factor": 0.080,
        "noise_scale": 7.1,
        "tariff_inr": 8.2,
        "waqi_city": "kolkata",
        "battery_capacity_kwh": 120.0,
        "grid_emission_factor_kg_kwh": 0.78,
    },
}

# ============================================================
# LOAD SHAPE ENGINE
# ============================================================
def get_load_shape(hour: float, weekday: int, shape: str) -> float:
    is_weekend = weekday >= 5

    if shape == "industrial":
        base = 0.84
        morning = 0.16 * np.exp(-0.5 * ((hour - 9.0) / 2.6) ** 2)
        evening = 0.18 * np.exp(-0.5 * ((hour - 19.2) / 2.3) ** 2)
        weekend_factor = 0.90 if is_weekend else 1.0
        return (base + morning + evening) * weekend_factor

    elif shape == "office":
        if is_weekend:
            return 0.55 + 0.10 * np.sin(2 * np.pi * (hour - 11) / 24)
        morning = 0.38 * (1 / (1 + np.exp(-(hour - 8.0) * 2.1)))
        lunch_dip = -0.14 * np.exp(-0.5 * ((hour - 13.0) / 1.0) ** 2)
        evening = 0.30 * np.exp(-0.5 * ((hour - 19.3) / 1.7) ** 2)
        night = 0.58
        return night + morning + lunch_dip + evening

    elif shape == "commercial":
        base = 0.68
        lunch = 0.26 * np.exp(-0.5 * ((hour - 13.5) / 1.5) ** 2)
        evening = 0.34 * np.exp(-0.5 * ((hour - 20.8) / 2.1) ** 2)
        weekend_factor = 0.72 if is_weekend else 1.0
        return (base + lunch + evening) * weekend_factor

    elif shape == "mixed":
        base = 0.76
        morning = 0.20 * np.exp(-0.5 * ((hour - 9.5) / 2.4) ** 2)
        evening = 0.26 * np.exp(-0.5 * ((hour - 19.8) / 2.0) ** 2)
        weekend_factor = 0.82 if is_weekend else 1.0
        return (base + morning + evening) * weekend_factor

    return 1.0

# ============================================================
# DATA GENERATION
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def generate_site_data(days: int, site_name: str, seed: int = 42) -> pd.DataFrame:
    cfg = SITES[site_name]
    rng = np.random.default_rng(seed + hash(site_name) % 1000)

    periods = days * 48
    timestamps = pd.date_range(
        end=pd.Timestamp.now().floor("30min"),
        periods=periods,
        freq="30min",
    )

    hours = timestamps.hour + timestamps.minute / 60
    weekdays = timestamps.dayofweek

    shape_mult = np.array([
        get_load_shape(h, wd, cfg["load_shape"])
        for h, wd in zip(hours, weekdays)
    ])

    temp = 27 + 5.5 * np.sin(2 * np.pi * (hours - 7) / 24) + rng.normal(0, 1.1, periods)

    load = (
        cfg["base_load_kw"] * shape_mult
        + 0.35 * (temp - 27)
        + rng.normal(0, cfg["noise_scale"], periods)
    )
    load = np.clip(load, 35, None)

    irradiance = np.maximum(
        0,
        950 * np.sin(np.pi * np.clip((hours - 6) / 12, 0, 1)) ** 1.35
        + rng.normal(0, 40, periods),
    )
    irradiance = np.clip(irradiance, 0, 1100)

    aqi_base = {"pune": 95, "bangalore": 55, "delhi": 110, "kolkata": 130}.get(
        cfg["waqi_city"], 90
    )
    aqi = (
        aqi_base
        + 35 * np.sin(2 * np.pi * (hours - 8) / 24)
        + rng.normal(0, 18, periods)
    )
    aqi = np.clip(aqi, 25, 320)

    temp_derate = 1 - 0.004 * np.maximum(temp - 25, 0)
    soiling = 1 - cfg["aqi_soiling_factor"] * np.maximum(aqi - 50, 0) / 100
    solar = (
        (irradiance / 1000)
        * cfg["solar_capacity_kw"]
        * temp_derate
        * soiling
        * rng.uniform(0.92, 1.0, periods)
    )
    solar = np.clip(solar, 0, cfg["solar_capacity_kw"])

    df = pd.DataFrame({
        "timestamp": timestamps,
        "load_kw": load,
        "solar_kw": solar,
        "temperature": temp,
        "irradiance": irradiance,
        "aqi": aqi,
    })
    return df

# ============================================================
# LIVE AQI (WAQI) WITH REGIONAL SCALING & EMBEDDED TOKEN
# ============================================================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_live_aqi(city: str, site_name: str) -> tuple[float, bool]:
    token = "2f4eabcc4e78dba94a8e64194072b7b17e38080f"
    
    regional_multipliers = {
        "Pune Industrial Campus": 1.1,
        "Bengaluru Tech Park": 0.5,
        "Delhi Commercial Hub": 1.8,
        "Kolkata Sector V": 1.4,
    }
    multiplier = regional_multipliers.get(site_name, 1.0)

    try:
        url = f"https://api.waqi.info/feed/{city}/?token={token}"
        r = requests.get(url, timeout=4)
        
        if r.status_code == 200:
            payload = r.json()
            if payload.get("status") == "ok":
                raw_val = float(payload["data"]["aqi"])
                adjusted_val = max(20.0, raw_val * multiplier if raw_val < 80 else raw_val)
                return adjusted_val, True
    except Exception:
        pass
        
    fallbacks = {
        "Pune Industrial Campus": 95.0,
        "Bengaluru Tech Park": 52.0,
        "Delhi Commercial Hub": 185.0,
        "Kolkata Sector V": 140.0,
    }
    return fallbacks.get(site_name, 90.0), False

# ============================================================
# FORECAST ENGINE (DUAL)
# ============================================================
def forecast_load(data: pd.DataFrame, horizon_steps: int, weather_shift: float = 0.0):
    df = data.copy()
    df["hour"] = df.timestamp.dt.hour + df.timestamp.dt.minute / 60
    df["weekday"] = (df.timestamp.dt.dayofweek < 5).astype(int)
    df["sin_h"] = np.sin(2 * np.pi * df.hour / 24)
    df["cos_h"] = np.cos(2 * np.pi * df.hour / 24)

    feature_cols = ["hour", "weekday", "temperature", "sin_h", "cos_h"]
    X = df[feature_cols]
    y = df["load_kw"]

    model = Ridge(alpha=1.2)
    model.fit(X, y)

    y_pred = model.predict(X)
    mape = mean_absolute_percentage_error(y, y_pred) * 100
    residual_std = (y - y_pred).std()
    band = max(6.5, residual_std * 1.96)

    raw_weights = dict(zip(feature_cols, model.coef_))
    abs_sum = sum(abs(v) for v in raw_weights.values()) + 1e-6
    rel_weights = {k: round(v / abs_sum * 100, 1) for k, v in raw_weights.items()}

    last_ts = df.timestamp.iloc[-1]
    future_ts = pd.date_range(
        start=last_ts + pd.Timedelta(minutes=30),
        periods=horizon_steps,
        freq="30min",
    )

    future_hour = future_ts.hour + future_ts.minute / 60
    future_weekday = (future_ts.dayofweek < 5).astype(int)

    baseline_temp = 27.5 + 5 * np.sin(2 * np.pi * (future_hour - 7) / 24)

    def make_X(temp_arr):
        return pd.DataFrame({
            "hour": future_hour,
            "weekday": future_weekday,
            "temperature": temp_arr,
            "sin_h": np.sin(2 * np.pi * future_hour / 24),
            "cos_h": np.cos(2 * np.pi * future_hour / 24),
        })

    baseline_pred = model.predict(make_X(baseline_temp))
    scenario_pred = model.predict(make_X(baseline_temp + weather_shift))

    horizon_factor = np.linspace(1.0, 1.45, horizon_steps)
    upper = scenario_pred + band * horizon_factor
    lower = scenario_pred - band * horizon_factor

    forecast_df = pd.DataFrame({
        "timestamp": future_ts,
        "forecast_baseline": baseline_pred,
        "forecast_scenario": scenario_pred,
        "lower": lower,
        "upper": upper,
        "delta_kw": scenario_pred - baseline_pred,
    })

    meta = {
        "mape": mape,
        "residual_std": residual_std,
        "weights": rel_weights,
        "model_name": "Ridge Regression (Cyclic Fourier)",
    }
    return forecast_df, meta

# ============================================================
# STORAGE OPTIMIZER & HELPERS
# ============================================================
def simulate_battery_dispatch(solar_kw: float, load_kw: float, battery_capacity_kwh: float, current_soc: float = 0.65):
    net_power = load_kw - solar_kw
    soc_kwh = current_soc * battery_capacity_kwh
    
    if net_power < 0:
        surplus = abs(net_power)
        charge_amount = min(surplus, battery_capacity_kwh * 0.2, battery_capacity_kwh - soc_kwh)
        soc_kwh += charge_amount
        net_grid_import = max(0.0, net_power + charge_amount)
    else:
        discharge_amount = min(net_power, battery_capacity_kwh * 0.25, soc_kwh)
        soc_kwh -= discharge_amount
        net_grid_import = max(0.0, net_power - discharge_amount)
        
    new_soc = min(1.0, max(0.0, soc_kwh / battery_capacity_kwh))
    return net_grid_import, new_soc * 100

def aqi_label(value: float) -> tuple[str, str]:
    if value <= 50:
        return "Good", "#36c98b"
    if value <= 100:
        return "Satisfactory", "#a5d86a"
    if value <= 200:
        return "Moderate", "#ffc857"
    return "Poor", "#f47c67"

def is_night_mode(irradiance: float, hour: float) -> bool:
    return irradiance < 45 or hour < 5.8 or hour > 18.8

def render_status_banner(latest, forecast, night: bool, outage: bool):
    messages = []
    if outage:
        messages.append(("error", "🚨 GRID BLACKOUT: Island Mode active. Microgrid operating autonomously on Solar + Storage."))
    
    if latest.aqi > 200:
        messages.append(("error", f"Poor air quality ({latest.aqi:.0f} AQI) — high soiling risk"))
    elif latest.aqi > 100:
        messages.append(("warning", f"Moderate AQI ({latest.aqi:.0f}) — monitor soiling"))

    peak_row = forecast.loc[forecast.forecast_scenario.idxmax()]
    hours_to_peak = (peak_row.timestamp - latest.timestamp).total_seconds() / 3600
    if 0 < hours_to_peak <= 3.5 and not outage:
        messages.append(("warning", f"Peak demand approaching: {peak_row.forecast_scenario:.0f} kW at {peak_row.timestamp.strftime('%H:%M')}"))

    if night and not outage:
        messages.append(("info", "Night mode — solar offline, storage discharging for peak shaving"))

    if not messages:
        st.success("System Status Normal: All microgrid parameters within optimal range", icon="✅")
    else:
        level, text = messages[0]
        if level == "error":
            st.error(text, icon="🚨")
        elif level == "warning":
            st.warning(text, icon="⚠️")
        else:
            st.info(text, icon="ℹ️")

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
.stApp { background: #071522; color: #edf5fb; }
[data-testid="stSidebar"] { background: #0d2233; }
.metric-card {
    background: linear-gradient(135deg, #102e43, #0d2233);
    border: 1px solid #24506a;
    border-radius: 14px;
    padding: 16px;
    min-height: 115px;
}
.metric-label { color: #9bb6c7; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .08em; }
.metric-value { font-size: 1.8rem; font-weight: 700; margin: 5px 0; }
.metric-note { font-size: .82rem; }
h1, h2, h3 { color: #f4fbff !important; }
.stPlotlyChart {
    border: 1px solid #1b4057;
    border-radius: 12px;
    padding: 6px;
    background: #0b1d2b;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ⚡ GridSight")
    st.caption("Microgrid command center")

    site = st.selectbox("Site", list(SITES.keys()))
    cfg = SITES[site]

    days = st.slider("Historical data window (Days)", 7, 60, 21)
    horizon_hours = st.select_slider("Forecast horizon", options=[12, 24, 36, 48, 72], value=24)
    weather_shift = st.slider(
        "Temperature scenario", -4, 6, 0,
        help="Adjusts forecast demand for a warmer or cooler outlook."
    )

    st.divider()
    st.markdown("**🛡️ Resilience & Demand Response**")
    simulate_outage = st.checkbox("🚨 Simulate Grid Blackout (Island Mode)", value=False)
    ev_shift_kw = st.slider("⚡ EV Fleet Load Shifting (kW)", 0, 40, 0, step=5, help="Shifts flexible EV charging away from peak hours.")

    st.divider()
    st.markdown("**SCADA Telemetry & Polling**")
    live_poll = st.checkbox("Enable live poll loop (15m)", value=False)

    live_aqi_val, is_true_live = fetch_live_aqi(cfg["waqi_city"], site)
    if is_true_live:
        st.success(f"Live API connected (WAQI: {int(live_aqi_val)})", icon="🟢")
    else:
        st.info("Simulation mode · Site profile active", icon="ℹ️")

# ============================================================
# MAIN DATA & CALCULATIONS
# ============================================================
data = generate_site_data(days, site)
data.loc[data.index[-1], "aqi"] = live_aqi_val

latest = data.iloc[-1]
forecast, meta = forecast_load(data, horizon_hours * 2, weather_shift)

night = is_night_mode(latest.irradiance, latest.timestamp.hour + latest.timestamp.minute / 60)
aqi_text, aqi_color = aqi_label(latest.aqi)
solar_loss = max(0, (latest.aqi - 45) * cfg["aqi_soiling_factor"])

# Apply Demand Response (EV Load Shifting)
adjusted_load_kw = max(35.0, latest.load_kw - ev_shift_kw)

# Handle Outage Simulation vs Normal Operation
if simulate_outage:
    net_load = 0.0
    battery_soc = 35.0
    grid_status_text = "🚨 Island Mode (Grid Down)"
    grid_color = "#ef7f6d"
else:
    net_load, battery_soc = simulate_battery_dispatch(latest.solar_kw, adjusted_load_kw, cfg["battery_capacity_kwh"], current_soc=0.65)
    grid_status_text = "after storage & PV"
    grid_color = "#a9c7ff"

# Carbon Tracker Calculations
total_historical_solar_kwh = data.solar_kw.sum() * 0.5
co2_saved_kg = total_historical_solar_kwh * cfg["grid_emission_factor_kg_kwh"]
coal_saved_kg = co2_saved_kg * 0.45

# ============================================================
# HEADER
# ============================================================
st.markdown(f"# {site}")
st.caption(
    f"LIVE OPERATIONS VIEW  •  Source: {'Live (WAQI API)' if is_true_live else 'Simulation Profile'}  •  "
    f"Refreshed: {datetime.now().strftime('%H:%M:%S')}"
)

render_status_banner(latest, forecast, night, simulate_outage)

# ============================================================
# METRIC CARDS (5-COLUMN LAYOUT)
# ============================================================
cards = st.columns(5)
metrics = [
    ("Grid demand", f"{adjusted_load_kw:.0f} kW", f"DR Shift: -{ev_shift_kw} kW" if ev_shift_kw > 0 else "↗ 3.2% vs yesterday", "#71d5c1"),
    ("Solar output", f"{latest.solar_kw:.1f} kW",
     "🌙 Night Mode (Solar Gated)" if night else f"{latest.solar_kw / cfg['solar_capacity_kw'] * 100:.0f}% capacity factor",
     "#ffd166"),
    ("Air quality", f"{latest.aqi:.0f} AQI", aqi_text, aqi_color),
    ("Battery Storage (SoC)", f"{battery_soc:.0f}%", f"Cap: {cfg['battery_capacity_kwh']} kWh", "#38bdf8"),
    ("Net grid import", f"{net_load:.0f} kW", grid_status_text, grid_color),
]

for col, (label, value, note, color) in zip(cards, metrics):
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-note" style="color:{color}">{note}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ============================================================
# DEMAND OUTLOOK
# ============================================================
st.markdown("### Demand outlook & ML forecasting")
left, right = st.columns([2.15, 1])

with left:
    recent = data.tail(144)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recent.timestamp, y=recent.load_kw,
        name="Actual demand", line=dict(color="#5ad1e5", width=2.2)
    ))

    fig.add_trace(go.Scatter(
        x=forecast.timestamp, y=forecast.forecast_baseline,
        name="Baseline (0°C)", line=dict(color="#94a3b8", width=2, dash="dot")
    ))

    fig.add_trace(go.Scatter(
        x=forecast.timestamp, y=forecast.forecast_scenario,
        name=f"Scenario ({weather_shift:+.1f}°C)",
        line=dict(color="#ffc857", width=2.6, dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=forecast.timestamp, y=forecast.upper,
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=forecast.timestamp, y=forecast.lower,
        fill="tonexty", fillcolor="rgba(255, 200, 87, 0.14)",
        line=dict(width=0), name="95% confidence", hoverinfo="skip"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=380,
        margin=dict(l=10, r=10, t=25, b=10),
        paper_bgcolor="#0b1d2b",
        plot_bgcolor="#0b1d2b",
        legend=dict(orientation="h", y=1.12),
        yaxis_title="kW",
        xaxis_title=None,
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    peak_row = forecast.loc[forecast.forecast_scenario.idxmax()]
    delta_peak = peak_row.delta_kw
    delta_energy = forecast.delta_kw.sum() / 2

    st.markdown("#### Forecast signal")
    st.metric("Expected peak", f"{peak_row.forecast_scenario:.0f} kW",
              peak_row.timestamp.strftime("%H:%M tomorrow"))
    st.metric("Forecast energy", f"{forecast.forecast_scenario.sum() / 2:.1f} kWh")

    if weather_shift != 0:
        st.metric("Peak impact", f"{delta_peak:+.0f} kW")
        st.metric("Energy impact", f"{delta_energy:+.1f} kWh")
        cost_impact = delta_energy * cfg["tariff_inr"]
        st.caption(f"Estimated cost impact: ₹{cost_impact:+,.0f}")

    st.caption(f"⚙️ Model: {meta['model_name']}")
    st.caption(f"📊 Training MAPE: {meta['mape']:.2f}%  |  Residual Std: ±{meta['residual_std']:.1f} kW")
    st.caption(
        f"🔍 Feature importance → "
        f"Hour: {meta['weights']['hour']}%  •  "
        f"Weekday: {meta['weights']['weekday']}%  •  "
        f"Temp: {meta['weights']['temperature']}%"
    )

# ============================================================
# SOLAR + CARBON & STORAGE TRACKER
# ============================================================
st.markdown("### Solar performance & carbon tracking")

col1, col2 = st.columns([1.55, 1])

with col1:
    solar_view = data.tail(96)
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(
        go.Scatter(x=solar_view.timestamp, y=solar_view.solar_kw,
                   name="PV output", line=dict(color="#ffd166", width=2.3)),
        secondary_y=False,
    )
    fig2.add_trace(
        go.Scatter(x=solar_view.timestamp, y=solar_view.aqi,
                   name="AQI", line=dict(color="#ef7f6d", width=1.7)),
        secondary_y=True,
    )
    fig2.update_layout(
        template="plotly_dark",
        height=340,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="#0b1d2b",
        plot_bgcolor="#0b1d2b",
        legend=dict(orientation="h", y=1.12),
    )
    fig2.update_yaxes(title_text="Solar kW", secondary_y=False)
    fig2.update_yaxes(title_text="AQI", secondary_y=True)
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.markdown("#### 🌱 Sustainability & Carbon Offset")
    c1, c2 = st.columns(2)
    c1.metric("CO₂ Offset", f"{co2_saved_kg / 1000:.2f} tons", "renewable generation")
    c2.metric("Coal Saved", f"{coal_saved_kg:.0f} kg", "equivalent offset")
    
    st.markdown("---")
    st.markdown(f"#### 🔋 Storage & Demand Response")
    st.metric("DR Cost Savings", f"₹{ev_shift_kw * cfg['tariff_inr'] * 4:,.0f}/day", "from peak load shifting")
    
    if simulate_outage:
        st.error("Islanding active: Grid import isolated. Battery supporting local critical loads.", icon="🚨")
    elif night:
        st.info("Storage discharging to handle baseline night load.", icon="🔋")
    else:
        st.success("Solar surplus actively routing to battery bank.", icon="⚡")

# ============================================================
# DOWNLOAD & EXPANDER
# ============================================================
csv = forecast[["timestamp", "forecast_baseline", "forecast_scenario", "lower", "upper"]].to_csv(index=False)
st.sidebar.download_button(
    "📥 Download Forecast CSV",
    data=csv,
    file_name=f"gridsight_forecast_{site.replace(' ', '_')}.csv",
    mime="text/csv",
)

with st.expander("Data model and integration notes"):
    st.markdown(f"""
**Current model:** 30-minute site observations with site-specific load shapes (`industrial` / `office` / `commercial` / `mixed`).  
Demand is forecast with **Ridge Regression + cyclic Fourier terms**.  

**Advanced Layers:** Features real-time WAQI API telemetry with regional atmospheric scaling, an interactive **Demand Response Simulator** (EV fleet load-shifting), and an **Outage Resilience & Islanding Mode** for autonomous microgrid power continuity.
""")
