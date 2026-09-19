from __future__ import annotations

import sqlite3
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
# SITE CONFIGURATION (WITH OPEN-METEO COORDINATES)
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
        "lat": 18.5204, "lon": 73.8567,
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
        "lat": 12.9716, "lon": 77.5946,
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
        "lat": 28.6139, "lon": 77.2090,
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
        "lat": 22.5726, "lon": 88.3639,
        "battery_capacity_kwh": 120.0,
        "grid_emission_factor_kg_kwh": 0.78,
    },
}

# Deterministic per-site seed offsets (hash(str) is randomized per Python
# process, so it must not be used to seed reproducible synthetic data).
SITE_SEED_OFFSETS = {name: i * 137 for i, name in enumerate(SITES)}

# Time-of-use assumption for the recommendation engine: commercial tariffs
# typically carry a demand-charge premium during evening peak hours.
PEAK_HOUR_START = 17.0
PEAK_HOUR_END = 22.0
PEAK_TARIFF_MULTIPLIER = 2.0

# ============================================================
# SECRETS
# ============================================================
def get_waqi_token() -> str | None:
    try:
        return st.secrets["WAQI_TOKEN"]
    except Exception:
        return None

WAQI_TOKEN = get_waqi_token()

# ============================================================
# SQLITE SESSION TELEMETRY LOG
# NOTE: Streamlit Cloud's filesystem is ephemeral — this log persists only
# for the life of the running container and resets on redeploy/restart.
# It is a within-session telemetry trail, not durable long-term storage.
# ============================================================
@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect("gridsight_scada.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_log (
            timestamp TEXT,
            site TEXT,
            load_kw REAL,
            solar_kw REAL,
            temperature REAL,
            aqi REAL,
            PRIMARY KEY (timestamp, site)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS decisions_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            site TEXT,
            shift_kw REAL,
            savings_inr REAL,
            co2_avoided_kg REAL
        )
    """)
    conn.commit()
    return conn

db_conn = get_db_connection()

def persist_telemetry(df: pd.DataFrame, site_name: str):
    cursor = db_conn.cursor()
    for _, row in df.tail(48).iterrows():
        cursor.execute("""
            INSERT OR IGNORE INTO telemetry_log (timestamp, site, load_kw, solar_kw, temperature, aqi)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (str(row["timestamp"]), site_name, row["load_kw"], row["solar_kw"], row["temperature"], row["aqi"]))
    db_conn.commit()

def log_decision(site_name: str, shift_kw: float, savings_inr: float, co2_avoided_kg: float):
    cursor = db_conn.cursor()
    cursor.execute("""
        INSERT INTO decisions_log (timestamp, site, shift_kw, savings_inr, co2_avoided_kg)
        VALUES (?, ?, ?, ?, ?)
    """, (datetime.now().isoformat(), site_name, shift_kw, savings_inr, co2_avoided_kg))
    db_conn.commit()

# ============================================================
# REAL WEATHER API (OPEN-METEO)
# ============================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_real_weather(lat: float, lon: float) -> float | None:
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m"
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            data = r.json()
            return float(data["current"]["temperature_2m"])
    except Exception:
        pass
    return None

# ============================================================
# LOAD SHAPE ENGINE
# ============================================================
def get_load_shape(hour: float, weekday: int, shape: str) -> float:
    is_weekend = weekday >= 5
    if shape == "industrial":
        base = 0.84
        morning = 0.16 * np.exp(-0.5 * ((hour - 9.0) / 2.6) ** 2)
        evening = 0.18 * np.exp(-0.5 * ((hour - 19.2) / 2.3) ** 2)
        return (base + morning + evening) * (0.90 if is_weekend else 1.0)
    elif shape == "office":
        if is_weekend:
            return 0.55 + 0.10 * np.sin(2 * np.pi * (hour - 11) / 24)
        morning = 0.38 * (1 / (1 + np.exp(-(hour - 8.0) * 2.1)))
        lunch_dip = -0.14 * np.exp(-0.5 * ((hour - 13.0) / 1.0) ** 2)
        evening = 0.30 * np.exp(-0.5 * ((hour - 19.3) / 1.7) ** 2)
        return 0.58 + morning + lunch_dip + evening
    elif shape == "commercial":
        base = 0.68
        lunch = 0.26 * np.exp(-0.5 * ((hour - 13.5) / 1.5) ** 2)
        evening = 0.34 * np.exp(-0.5 * ((hour - 20.8) / 2.1) ** 2)
        return (base + lunch + evening) * (0.72 if is_weekend else 1.0)
    elif shape == "mixed":
        base = 0.76
        morning = 0.20 * np.exp(-0.5 * ((hour - 9.5) / 2.4) ** 2)
        evening = 0.26 * np.exp(-0.5 * ((hour - 19.8) / 2.0) ** 2)
        return (base + morning + evening) * (0.82 if is_weekend else 1.0)
    return 1.0

# ============================================================
# DATA GENERATION & PERSISTENCE
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def generate_site_data(days: int, site_name: str, real_temp: float | None = None, seed: int = 42) -> pd.DataFrame:
    cfg = SITES[site_name]
    rng = np.random.default_rng(seed + SITE_SEED_OFFSETS[site_name])

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

    base_temp = real_temp if real_temp is not None else 28.0
    temp = base_temp + 4.5 * np.sin(2 * np.pi * (hours - 7) / 24) + rng.normal(0, 1.0, periods)

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
        "aqi": aqi,
        "irradiance": irradiance,
    })

    persist_telemetry(df, site_name)
    return df

# ============================================================
# LIVE AQI (WAQI) — returns the real reading, unmodified.
# Fallback values (used only if the token is missing or the call fails)
# are clearly distinct per city and never presented as "live".
# ============================================================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_live_aqi(city: str, token: str | None) -> tuple[float, bool]:
    fallbacks = {
        "pune": 95.0,
        "bangalore": 52.0,
        "delhi": 185.0,
        "kolkata": 140.0,
    }
    if not token:
        return fallbacks.get(city, 90.0), False

    try:
        url = f"https://api.waqi.info/feed/{city}/?token={token}"
        r = requests.get(url, timeout=4)
        if r.status_code == 200:
            payload = r.json()
            if payload.get("status") == "ok":
                return float(payload["data"]["aqi"]), True
    except Exception:
        pass

    return fallbacks.get(city, 90.0), False

# ============================================================
# FORECAST ENGINE
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

    model = Ridge(alpha=0.6)
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

    baseline_temp = df["temperature"].iloc[-1] + 6 * np.sin(2 * np.pi * (future_hour - 7) / 24)

    def make_X(temp_arr):
        return pd.DataFrame({
            "hour": future_hour,
            "weekday": future_weekday,
            "temperature": temp_arr,
            "sin_h": np.sin(2 * np.pi * future_hour / 24),
            "cos_h": np.cos(2 * np.pi * future_hour / 24),
        })

    baseline_pred = model.predict(make_X(baseline_temp))
    # Scenario is pure model output at the shifted temperature — no manual
    # bonus added on top, so the chart reflects what the model actually learned.
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
        "model_name": "Ridge Regression (Cyclic Fourier + HVAC)",
    }
    return forecast_df, meta

# ============================================================
# SMART BATTERY DISPATCH STRATEGY (ARBITRAGE & PEAK SHAVING)
# Stateful: current_soc is read from / written back to st.session_state
# by the caller, so charge level actually carries forward between reruns
# instead of resetting to a fixed value every time.
# ============================================================
def simulate_smart_battery_dispatch(solar_kw: float, load_kw: float, hour: float, battery_capacity_kwh: float, current_soc: float):
    """
    Smart Dispatch Logic:
    - Peak hours (17:00 to 22:00): Aggressive discharge to shave peak and avoid high tariffs.
    - Solar surplus hours: Charge battery from PV excess.
    - Off-peak/Night: Maintain or gentle trickle.
    """
    net_power = load_kw - solar_kw
    soc_kwh = current_soc * battery_capacity_kwh
    is_peak_hour = 17.0 <= hour <= 22.0
    is_solar_surplus = net_power < 0

    if is_peak_hour and soc_kwh > (0.15 * battery_capacity_kwh):
        discharge_amount = min(net_power + 20.0, battery_capacity_kwh * 0.35, soc_kwh)
        soc_kwh -= max(0.0, discharge_amount)
        net_grid_import = max(0.0, net_power - discharge_amount)
        dispatch_mode = "⚡ Peak Shaving (Discharging)"
    elif is_solar_surplus:
        surplus = abs(net_power)
        charge_amount = min(surplus, battery_capacity_kwh * 0.25, battery_capacity_kwh - soc_kwh)
        soc_kwh += charge_amount
        net_grid_import = max(0.0, net_power + charge_amount)
        dispatch_mode = "☀️ Solar Arbitrage (Charging)"
    else:
        if net_power > 0 and soc_kwh > (0.3 * battery_capacity_kwh) and 12.0 <= hour <= 20.0:
            discharge_amount = min(net_power * 0.5, soc_kwh)
            soc_kwh -= discharge_amount
            net_grid_import = max(0.0, net_power - discharge_amount)
            dispatch_mode = "🔋 Load Assist (Discharging)"
        else:
            net_grid_import = max(0.0, net_power)
            dispatch_mode = "⚖️ Standby / Idle"

    new_soc = min(1.0, max(0.0, soc_kwh / battery_capacity_kwh))
    return net_grid_import, new_soc * 100, dispatch_mode

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

# ============================================================
# RECOMMENDATION ENGINE
# Rule-based (not ML): looks for meaningfully elevated demand specifically
# inside the evening peak-tariff window (independent of where the forecast's
# single global maximum happens to fall — a Ridge/Fourier fit can place that
# anywhere in the horizon) and, if found, suggests moving a fixed EV-charging
# block to the cheapest off-peak hour, quantified via the TOU assumption above.
# ============================================================
def generate_recommendation(forecast: pd.DataFrame, cfg: dict, current_ev_shift_kw: float) -> dict | None:
    if current_ev_shift_kw > 0:
        return None  # a shift is already applied — don't suggest stacking another

    peak_window = forecast[(forecast.timestamp.dt.hour >= PEAK_HOUR_START) & (forecast.timestamp.dt.hour <= PEAK_HOUR_END)]
    off_peak = forecast[(forecast.timestamp.dt.hour < PEAK_HOUR_START) | (forecast.timestamp.dt.hour > PEAK_HOUR_END)]
    if peak_window.empty or off_peak.empty:
        return None

    from_row = peak_window.loc[peak_window.forecast_scenario_dr.idxmax()]
    to_row = off_peak.loc[off_peak.forecast_scenario_dr.idxmin()]

    # Only worth recommending if the peak-window demand is meaningfully above
    # what's achievable off-peak — otherwise there's nothing worth shifting.
    if from_row.forecast_scenario_dr < to_row.forecast_scenario_dr + 10:
        return None

    suggested_shift_kw = 30.0  # a typical EV fleet charging block; capped well under the 50kW slider max
    hours_shifted = PEAK_HOUR_END - PEAK_HOUR_START
    peak_premium_per_kwh = cfg["tariff_inr"] * (PEAK_TARIFF_MULTIPLIER - 1)
    savings_inr = suggested_shift_kw * peak_premium_per_kwh * hours_shifted
    # CO2 estimate assumes off-peak grid draw is modestly cleaner than peak
    # (peaker plants skew the peak-hour mix) — a rough, clearly-labeled proxy.
    co2_avoided_kg = suggested_shift_kw * hours_shifted * cfg["grid_emission_factor_kg_kwh"] * 0.10

    return {
        "shift_kw": suggested_shift_kw,
        "from_hour": from_row.timestamp.strftime("%H:%M"),
        "to_hour": to_row.timestamp.strftime("%H:%M"),
        "savings_inr": savings_inr,
        "co2_avoided_kg": co2_avoided_kg,
    }

def apply_recommendation(site_name: str, shift_kw: float, savings_inr: float, co2_avoided_kg: float):
    # Runs as an on_click callback (before the script reruns from the top),
    # so the slider's session_state can be updated safely here — doing this
    # inline after the widget has already rendered in the same run raises
    # StreamlitWidgetAlreadyInstantiatedError.
    st.session_state["ev_shift_kw_slider"] = int(shift_kw)
    log_decision(site_name, shift_kw, savings_inr, co2_avoided_kg)

def render_status_banner(latest, forecast, night: bool, outage: bool, autonomy_hours: float):
    messages = []
    if outage:
        if autonomy_hours > 4.0:
            messages.append(("error", f"🚨 ISLAND MODE ACTIVE: Grid offline. Autonomous battery & solar reserve strong ({autonomy_hours:.1f}h autonomy)."))
        else:
            messages.append(("error", f"🚨 CRITICAL ISLAND MODE: Low reserve! Estimated autonomy only {autonomy_hours:.1f}h. Shed non-critical loads!"))

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
# SESSION STATE — per-site battery charge, carried across reruns
# ============================================================
if "battery_soc" not in st.session_state:
    st.session_state.battery_soc = {name: 65.0 for name in SITES}

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ⚡ GridSight")
    st.caption("Microgrid command center")

    view_mode = st.radio("Command Mode", ["Single-Site Operations", "🌐 Portfolio Executive Overview"])
    st.divider()

    if view_mode == "Single-Site Operations":
        site = st.selectbox("Site", list(SITES.keys()))
        cfg = SITES[site]

        days = st.slider("Historical data window (Days)", 7, 60, 21)
        horizon_hours = st.select_slider("Forecast horizon", options=[12, 24, 36, 48, 72], value=24)
        weather_shift = st.slider("Temperature scenario (°C)", -4, 8, 0)

        st.divider()
        st.markdown("**🛡️ Resilience & Demand Response**")
        simulate_outage = st.checkbox("🚨 Simulate Grid Blackout (Island Mode)", value=False)
        ev_shift_kw = st.slider("⚡ EV Fleet Load Shifting (kW)", 0, 50, 0, step=5, key="ev_shift_kw_slider")
    else:
        site = "Pune Industrial Campus"
        cfg = SITES[site]
        simulate_outage = False
        ev_shift_kw = 0
        weather_shift = 0

    st.divider()
    st.markdown("**SCADA Telemetry & Polling**")
    live_aqi_val, is_true_live = fetch_live_aqi(cfg["waqi_city"], WAQI_TOKEN)
    real_temp = fetch_real_weather(cfg["lat"], cfg["lon"])

    if is_true_live:
        st.success(f"Live API connected (WAQI: {int(live_aqi_val)})", icon="🟢")
    elif WAQI_TOKEN:
        st.warning("WAQI token set but request failed — using simulation fallback", icon="⚠️")
    else:
        st.info("Simulation mode · add WAQI_TOKEN in secrets for live AQI", icon="ℹ️")

    if real_temp is not None:
        st.success(f"Open-Meteo Weather: {real_temp}°C", icon="🌤️")
    else:
        st.info("Weather baseline: 28.0°C", icon="ℹ️")

# ============================================================
# DYNAMIC CSS (REACTS TO ISLAND MODE)
# ============================================================
app_bg = "#160b0b" if simulate_outage else "#071522"
card_bg = "linear-gradient(135deg, #321010, #1d0909)" if simulate_outage else "linear-gradient(135deg, #102e43, #0d2233)"
card_border = "#7f2a2a" if simulate_outage else "#24506a"
chart_bg = "#120808" if simulate_outage else "#0b1d2b"

st.markdown("""
<style>
.stApp { background: %s; color: #edf5fb; transition: background 0.5s ease; }
[data-testid="stSidebar"] { background: #0d2233; }
.metric-card {
    background: %s;
    border: 1px solid %s;
    border-radius: 14px;
    padding: 16px;
    min-height: 115px;
    transition: all 0.5s ease;
}
.metric-label { color: #9bb6c7; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .08em; }
.metric-value { font-size: 1.8rem; font-weight: 700; margin: 5px 0; }
.metric-note { font-size: .82rem; }
h1, h2, h3 { color: #f4fbff !important; }
.stPlotlyChart {
    border: 1px solid %s;
    border-radius: 12px;
    padding: 6px;
    background: %s;
}
</style>
""" % (app_bg, card_bg, card_border, card_border, chart_bg), unsafe_allow_html=True)

# ============================================================
# PORTFOLIO EXECUTIVE OVERVIEW MODE (WITH REAL ECONOMICS)
# ============================================================
if view_mode == "🌐 Portfolio Executive Overview":
    st.markdown("# 🌐 Portfolio Executive Command")
    st.caption(f"PORTFOLIO-WIDE MULTI-SITE TELEMETRY • Active Campuses: {len(SITES)} • Refreshed: {datetime.now().strftime('%H:%M:%S')}")
    st.markdown("---")

    portfolio_rows = []
    total_portfolio_solar = 0.0
    total_portfolio_load = 0.0
    total_portfolio_co2 = 0.0
    total_cost_avoided_inr = 0.0

    for s_name, s_cfg in SITES.items():
        s_data = generate_site_data(14, s_name, real_temp=fetch_real_weather(s_cfg["lat"], s_cfg["lon"]))
        s_aqi, _ = fetch_live_aqi(s_cfg["waqi_city"], WAQI_TOKEN)
        s_latest = s_data.iloc[-1]

        solar_gen_kwh_total = s_data.solar_kw.sum() * 0.5
        s_co2 = solar_gen_kwh_total * s_cfg["grid_emission_factor_kg_kwh"] / 1000
        cost_avoided = solar_gen_kwh_total * s_cfg["tariff_inr"]
        peak_shaving_savings = s_cfg["battery_capacity_kwh"] * s_cfg["tariff_inr"] * 14 * 0.35  # estimated arbitrage

        total_portfolio_solar += s_latest.solar_kw
        total_portfolio_load += s_latest.load_kw
        total_portfolio_co2 += s_co2
        total_cost_avoided_inr += (cost_avoided + peak_shaving_savings)

        portfolio_rows.append({
            "Campus Site": s_name,
            "Solar Cap (kW)": s_cfg["solar_capacity_kw"],
            "Live Solar (kW)": round(s_latest.solar_kw, 1),
            "Live Load (kW)": round(s_latest.load_kw, 1),
            "Live AQI": round(s_aqi, 0),
            "Tariff (₹/kWh)": s_cfg["tariff_inr"],
            "Solar Savings (₹)": round(cost_avoided, 0),
            "Arbitrage Savings (₹)": round(peak_shaving_savings, 0),
            "CO₂ Offset (Tons)": round(s_co2, 2)
        })

    port_df = pd.DataFrame(portfolio_rows)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Total Portfolio Solar Output", f"{total_portfolio_solar:.1f} kW", f"across {len(SITES)} campuses")
    p2.metric("Total Portfolio Load Demand", f"{total_portfolio_load:.1f} kW", "live aggregate")
    p3.metric("Combined CO₂ Offsets", f"{total_portfolio_co2:.2f} tons", "lifetime renewable impact")
    p4.metric("Total Economic Value", f"₹{total_cost_avoided_inr:,.0f}", "solar offset + battery arbitrage")

    st.markdown("### Campus Economic & Performance Matrix")
    st.dataframe(port_df, use_container_width=True, hide_index=True)

    st.markdown("### Portfolio Generation vs Demand Breakdown")
    fig_port = go.Figure()
    fig_port.add_trace(go.Bar(x=port_df["Campus Site"], y=port_df["Live Solar (kW)"], name="Live Solar kW", marker_color="#ffd166"))
    fig_port.add_trace(go.Bar(x=port_df["Campus Site"], y=port_df["Live Load (kW)"], name="Live Load kW", marker_color="#5ad1e5"))
    fig_port.update_layout(
        template="plotly_dark",
        height=380,
        barmode="group",
        paper_bgcolor="#0b1d2b",
        plot_bgcolor="#0b1d2b",
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=10, r=10, t=25, b=10)
    )
    st.plotly_chart(fig_port, use_container_width=True)

# ============================================================
# SINGLE-SITE OPERATIONS MODE (WITH SMART DISPATCH & HISTORY)
# ============================================================
else:
    data = generate_site_data(days, site, real_temp=real_temp)
    data.loc[data.index[-1], "aqi"] = live_aqi_val

    latest = data.iloc[-1]
    current_hour = latest.timestamp.hour + latest.timestamp.minute / 60
    forecast, meta = forecast_load(data, horizon_hours * 2, weather_shift)

    forecast["forecast_scenario_dr"] = np.maximum(35.0, forecast["forecast_scenario"] - ev_shift_kw)

    night = is_night_mode(latest.irradiance, current_hour)
    aqi_text, aqi_color = aqi_label(latest.aqi)

    adjusted_load_kw = max(35.0, latest.load_kw - ev_shift_kw)

    current_soc_pct = st.session_state.battery_soc.get(site, 65.0)
    current_battery_kwh = (current_soc_pct / 100.0) * cfg["battery_capacity_kwh"]
    net_critical_load = max(10.0, adjusted_load_kw - latest.solar_kw)
    autonomy_hours = current_battery_kwh / net_critical_load if net_critical_load > 0 else 24.0
    # Battery contribution modeled as kWh deliverable over the next hour,
    # so it is treated as an equivalent kW figure in this coverage estimate.
    critical_load_coverage_pct = min(100.0, (latest.solar_kw + (cfg["battery_capacity_kwh"] * 0.25)) / adjusted_load_kw * 100)

    if simulate_outage:
        net_load = 0.0
        time_step_hours = 0.5  # matches the 30-min data resolution
        drain_kwh = net_critical_load * time_step_hours
        battery_soc = max(5.0, current_soc_pct - (drain_kwh / cfg["battery_capacity_kwh"] * 100))
        grid_status_text = f"🚨 Island Mode ({autonomy_hours:.1f}h reserve)"
        grid_color = "#ef7f6d" if autonomy_hours < 4.0 else "#ffd166"
        dispatch_mode = "🚨 Emergency Islanding"
    else:
        net_load, battery_soc, dispatch_mode = simulate_smart_battery_dispatch(
            latest.solar_kw, adjusted_load_kw, current_hour, cfg["battery_capacity_kwh"],
            current_soc=current_soc_pct / 100.0,
        )
        grid_status_text = dispatch_mode
        grid_color = "#38bdf8" if "Discharging" in dispatch_mode else ("#ffd166" if "Charging" in dispatch_mode else "#a9c7ff")

    st.session_state.battery_soc[site] = battery_soc

    total_historical_solar_kwh = data.solar_kw.sum() * 0.5
    co2_saved_kg = total_historical_solar_kwh * cfg["grid_emission_factor_kg_kwh"]
    coal_saved_kg = co2_saved_kg * 0.45
    current_carbon_intensity = cfg["grid_emission_factor_kg_kwh"] * 1000
    next_24h_solar_kwh = forecast.head(48).forecast_scenario.sum() * 0.25
    next_24h_co2_offset_kg = next_24h_solar_kwh * cfg["grid_emission_factor_kg_kwh"]

    solar_cost_avoided = total_historical_solar_kwh * cfg["tariff_inr"]
    arbitrage_savings = (total_historical_solar_kwh * 0.3) * cfg["tariff_inr"] * 0.25

    st.markdown(f"# {site}")
    st.caption(
        f"LIVE OPERATIONS VIEW  •  Source: {'Live (WAQI + Open-Meteo)' if (is_true_live and real_temp) else 'Simulation Profile'}  •  "
        f"Refreshed: {datetime.now().strftime('%H:%M:%S')}"
    )

    render_status_banner(latest, forecast, night, simulate_outage, autonomy_hours)

    recommendation = None if simulate_outage else generate_recommendation(forecast, cfg, ev_shift_kw)
    if recommendation:
        rec_col1, rec_col2 = st.columns([3.2, 1])
        with rec_col1:
            st.info(
                f"🤖 **Recommendation:** Shift **{recommendation['shift_kw']:.0f} kW** of EV charging from "
                f"**{recommendation['from_hour']}** to **{recommendation['to_hour']}** → save "
                f"**₹{recommendation['savings_inr']:,.0f}** and avoid **{recommendation['co2_avoided_kg']:.1f} kg CO₂** today.",
                icon="🤖",
            )
        with rec_col2:
            st.write("")
            st.button(
                "✅ Apply", use_container_width=True, key="apply_recommendation",
                on_click=apply_recommendation,
                args=(site, recommendation["shift_kw"], recommendation["savings_inr"], recommendation["co2_avoided_kg"]),
            )

    cards = st.columns(5)
    metrics = [
        ("Grid demand", f"{adjusted_load_kw:.0f} kW", f"DR Shift: -{ev_shift_kw} kW active" if ev_shift_kw > 0 else f"Tariff: ₹{cfg['tariff_inr']}/kWh", "#71d5c1"),
        ("Solar output", f"{latest.solar_kw:.1f} kW",
         "🌙 Night Mode (Gated)" if night else f"{latest.solar_kw / cfg['solar_capacity_kw'] * 100:.0f}% capacity factor",
         "#ffd166"),
        ("Air quality", f"{latest.aqi:.0f} AQI", aqi_text, aqi_color),
        ("Battery Storage (SoC)", f"{battery_soc:.0f}%", dispatch_mode, "#38bdf8"),
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
            name="Baseline (Weather)", line=dict(color="#94a3b8", width=2, dash="dot")
        ))

        fig.add_trace(go.Scatter(
            x=forecast.timestamp, y=forecast.forecast_scenario,
            name=f"Scenario ({weather_shift:+.1f}°C)",
            line=dict(color="#ffc857", width=2.2, dash="dash")
        ))

        if ev_shift_kw > 0:
            fig.add_trace(go.Scatter(
                x=forecast.timestamp, y=forecast.forecast_scenario_dr,
                name=f"Optimized DR (-{ev_shift_kw}kW)",
                line=dict(color="#36c98b", width=2.8)
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
            paper_bgcolor=chart_bg,
            plot_bgcolor=chart_bg,
            legend=dict(orientation="h", y=1.12),
            yaxis_title="kW",
            xaxis_title=None,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        peak_row = forecast.loc[forecast.forecast_scenario_dr.idxmax()]

        st.markdown("#### Forecast signal & Economics")
        st.metric("Expected peak", f"{peak_row.forecast_scenario_dr:.0f} kW",
                  peak_row.timestamp.strftime("%H:%M tomorrow"))
        st.metric("Solar Cost Avoided", f"₹{solar_cost_avoided:,.0f}", f"at ₹{cfg['tariff_inr']}/kWh")
        st.metric("Arbitrage Savings", f"₹{arbitrage_savings:,.0f}", "peak shaving dispatch")

        if weather_shift != 0:
            st.metric("Weather Peak Impact", f"{peak_row.delta_kw:+.0f} kW", "vs. baseline temperature")
        if ev_shift_kw > 0:
            st.metric("DR Peak Reduction", f"-{ev_shift_kw:.0f} kW", "flat demand-response shift")
            st.caption(f"💰 DR Daily Savings: ₹{ev_shift_kw * cfg['tariff_inr'] * 4:,.0f}")

        st.caption(f"⚙️ Model: {meta['model_name']}")
        st.caption(f"📊 Training MAPE: {meta['mape']:.2f}%  |  Residual Std: ±{meta['residual_std']:.1f} kW")

    st.markdown("### Solar performance, carbon intelligence & resilience")

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
            paper_bgcolor=chart_bg,
            plot_bgcolor=chart_bg,
            legend=dict(orientation="h", y=1.12),
        )
        fig2.update_yaxes(title_text="Solar kW", secondary_y=False)
        fig2.update_yaxes(title_text="AQI", secondary_y=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("AQI history is simulated; only the most recent point reflects a live WAQI reading." if is_true_live else "AQI series is fully simulated (no live token configured).")

    with col2:
        st.markdown("#### 🌱 Carbon Intelligence & SQLite Logs")
        cc1, cc2 = st.columns(2)
        cc1.metric("CO₂ Offset (Total)", f"{co2_saved_kg / 1000:.2f} tons", f"{coal_saved_kg:.0f} kg coal")
        cc2.metric("Grid Intensity", f"{current_carbon_intensity:.0f} g/kWh", "regional baseline")
        st.caption(f"🔮 Projected next-24h offset: **{next_24h_co2_offset_kg:.1f} kg CO₂**")

        cursor = db_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM telemetry_log WHERE site = ?", (site,))
        db_rows = cursor.fetchone()[0]
        st.caption(f"💾 Session Telemetry Log: **{db_rows} records** (SQLite — resets on app restart/redeploy)")

        st.markdown("---")
        st.markdown(f"#### 🛡️ Microgrid Resilience & Islandability")
        rc1, rc2 = st.columns(2)
        rc1.metric("Autonomy Reserve", f"{autonomy_hours:.1f} hours", "at current net load")
        rc2.metric("Critical Coverage", f"{critical_load_coverage_pct:.0f}%", "solar + battery capacity")

        status_badge = "🟢 Islandable (Secure)" if autonomy_hours >= 4.0 else "🔴 At Risk (Shed Load)"
        st.caption(f"Status: **{status_badge}** • Storage Capacity: {cfg['battery_capacity_kwh']} kWh")

    st.markdown("### 📅 This Month, You Saved")
    month_cursor = db_conn.cursor()
    current_month = datetime.now().strftime("%Y-%m")
    month_cursor.execute("""
        SELECT COALESCE(SUM(savings_inr), 0), COALESCE(SUM(co2_avoided_kg), 0),
               COALESCE(SUM(shift_kw), 0), COUNT(*)
        FROM decisions_log
        WHERE site = ? AND strftime('%Y-%m', timestamp) = ?
    """, (site, current_month))
    month_savings, month_co2, month_shift_kw, month_count = month_cursor.fetchone()

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("💰 Total Saved", f"₹{month_savings:,.0f}", f"{month_count} recommendation(s) applied")
    mc2.metric("🌱 CO₂ Avoided", f"{month_co2:.1f} kg", "from load-shifting decisions")
    mc3.metric("⚡ Peak Demand Reduced", f"{month_shift_kw:.0f} kW", "cumulative across decisions")
    st.caption("Tracked since this app instance started — resets on redeploy, same as the SQLite telemetry log above.")

    csv = forecast[["timestamp", "forecast_baseline", "forecast_scenario", "forecast_scenario_dr", "lower", "upper"]].to_csv(index=False)
    st.sidebar.download_button(
        "📥 Download Forecast CSV",
        data=csv,
        file_name=f"gridsight_forecast_{site.replace(' ', '_')}.csv",
        mime="text/csv",
    )
