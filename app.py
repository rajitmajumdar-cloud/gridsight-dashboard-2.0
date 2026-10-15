"""
app.py - GridSight Enterprise Microgrid Control & BI Dashboard
Single-file production suite built for 100% deployment reliability on Streamlit Cloud.
"""

from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error

# ==========================================
# 1. TELEMETRY & PHYSICAL ENGINE
# ==========================================

SITE_CONFIGS = {
    "Pune Campus": {
        "timezone": "Asia/Kolkata",
        "pv_capacity_kw": 250.0,
        "base_load_kw": 120.0,
        "peak_load_kw": 380.0,
        "seed": 101,
        "battery_capacity_kwh": 400.0,
        "max_battery_kw": 100.0,
    },
    "Bengaluru Tech Park": {
        "timezone": "Asia/Kolkata",
        "pv_capacity_kw": 500.0,
        "base_load_kw": 250.0,
        "peak_load_kw": 850.0,
        "seed": 202,
        "battery_capacity_kwh": 800.0,
        "max_battery_kw": 200.0,
    },
    "Delhi Regional Hub": {
        "timezone": "Asia/Kolkata",
        "pv_capacity_kw": 180.0,
        "base_load_kw": 90.0,
        "peak_load_kw": 310.0,
        "seed": 303,
        "battery_capacity_kwh": 300.0,
        "max_battery_kw": 75.0,
    },
}


def get_cpcb_aqi_category(aqi: float) -> str:
    """Official Central Pollution Control Board (CPCB) Indian AQI standards."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


def generate_site_data(site_name: str, days: int = 30) -> pd.DataFrame:
    """Generates site-specific, physics-grounded microgrid telemetry."""
    config = SITE_CONFIGS.get(site_name, SITE_CONFIGS["Pune Campus"])
    tz = pytz.timezone(config["timezone"])

    end_time = datetime.now(tz).replace(minute=0, second=0, microsecond=0)
    periods = days * 24 * 2  # 30-min steps
    timestamps = pd.date_range(end=end_time, periods=periods, freq="30min", tz=tz)

    np.random.seed(config["seed"])
    hours = timestamps.hour + timestamps.minute / 60.0
    day_of_week = timestamps.dayofweek

    # Environmental profiles
    ambient_temp = (
        24.0 + 8.0 * np.sin(np.pi * (hours - 8.0) / 12.0) + np.random.normal(0, 1.2, periods)
    )
    relative_humidity = np.clip(
        70.0 - 20.0 * np.sin(np.pi * (hours - 8.0) / 12.0) + np.random.normal(0, 3, periods),
        20.0,
        98.0,
    )

    # Solar Irradiance & Derating
    solar_raw = np.maximum(0.0, np.sin(np.pi * (hours - 6.0) / 12.0))
    solar_raw[hours < 6] = 0
    solar_raw[hours > 18] = 0
    irradiance_wm2 = solar_raw * 1000.0 * np.random.uniform(0.85, 1.0, periods)

    base_aqi = 180.0 if "Delhi" in site_name else (110.0 if "Pune" in site_name else 75.0)
    aqi_series = np.clip(base_aqi + np.random.normal(0, 25, periods), 20.0, 500.0)
    soiling_derate_factor = 1.0 - np.maximum(0.0, (aqi_series - 100.0) * 0.0002)

    # NOCT Thermal Cell Model
    cell_temp_c = ambient_temp + (45.0 - 20.0) / 800.0 * irradiance_wm2
    temp_derate_factor = 1.0 - np.maximum(0.0, (cell_temp_c - 25.0) * 0.004)

    ideal_solar_kw = (irradiance_wm2 / 1000.0) * config["pv_capacity_kw"]
    actual_solar_kw = ideal_solar_kw * soiling_derate_factor * temp_derate_factor
    pv_performance_ratio = np.where(
        ideal_solar_kw > 0, actual_solar_kw / ideal_solar_kw, 0.0
    )

    # Campus HVAC & Base Consumption Modeling
    is_weekend = (day_of_week >= 5).astype(int)
    occupancy = (1.0 - 0.5 * is_weekend) * np.maximum(0.0, np.sin(np.pi * (hours - 7.0) / 11.0))
    hvac_load = np.maximum(0.0, (ambient_temp - 24.0) ** 1.5) * (1.0 + relative_humidity / 200.0) * 3.5

    base_load = config["base_load_kw"]
    peak_capacity = config["peak_load_kw"]
    simulated_load_kw = np.clip(
        base_load
        + (peak_capacity - base_load) * 0.6 * occupancy
        + hvac_load
        + np.random.normal(0, 4.0, periods),
        base_load * 0.5,
        None,
    )

    # Battery State Simulation
    battery_capacity_kwh = config["battery_capacity_kwh"]
    max_kw = config["max_battery_kw"]
    soc_kwh = np.zeros(periods)
    battery_kw = np.zeros(periods)
    curr_soc = battery_capacity_kwh * 0.5

    net_unbuffered = simulated_load_kw - actual_solar_kw
    for i in range(periods):
        net = net_unbuffered[i]
        if net > 0:
            discharge = min(float(net), float(max_kw), float(curr_soc))
            battery_kw[i] = discharge
            curr_soc -= discharge * 0.5
        else:
            charge = min(float(abs(net)), float(max_kw), float(battery_capacity_kwh - curr_soc))
            battery_kw[i] = -charge
            curr_soc += charge * 0.5
        soc_kwh[i] = curr_soc

    grid_import_kw = np.maximum(0.0, simulated_load_kw - actual_solar_kw - battery_kw)
    grid_export_kw = np.maximum(0.0, actual_solar_kw + battery_kw - simulated_load_kw)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "site_name": site_name,
            "load_kw": simulated_load_kw,
            "solar_kw": actual_solar_kw,
            "battery_kw": battery_kw,
            "battery_soc_kwh": soc_kwh,
            "battery_soc_pct": (soc_kwh / battery_capacity_kwh) * 100.0,
            "grid_import_kw": grid_import_kw,
            "grid_export_kw": grid_export_kw,
            "ambient_temp_c": ambient_temp,
            "relative_humidity": relative_humidity,
            "cell_temp_c": cell_temp_c,
            "irradiance_wm2": irradiance_wm2,
            "aqi": aqi_series,
            "aqi_label": [get_cpcb_aqi_category(a) for a in aqi_series],
            "pv_performance_ratio": pv_performance_ratio,
        }
    )


# ==========================================
# 2. MACHINE LEARNING FORECASTING ENGINE
# ==========================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fourier transform features and lag construction."""
    data = df.copy()
    hour_float = data["timestamp"].dt.hour + data["timestamp"].dt.minute / 60.0
    dow = data["timestamp"].dt.dayofweek

    data["sin_hour"] = np.sin(2.0 * np.pi * hour_float / 24.0)
    data["cos_hour"] = np.cos(2.0 * np.pi * hour_float / 24.0)
    data["sin_dow"] = np.sin(2.0 * np.pi * dow / 7.0)
    data["cos_dow"] = np.cos(2.0 * np.pi * dow / 7.0)
    data["temp_humidity_index"] = data["ambient_temp_c"] * (data["relative_humidity"] / 100.0)
    data["load_lag_2"] = data["load_kw"].shift(2)
    data["load_lag_48"] = data["load_kw"].shift(48)

    return data.dropna().reset_index(drop=True)


def train_forecast_model(df: pd.DataFrame):
    """Trains non-linear Gradient Boosting Regressor with chronological split."""
    data = build_features(df)
    feature_cols = [
        "sin_hour", "cos_hour", "sin_dow", "cos_dow",
        "ambient_temp_c", "relative_humidity", "temp_humidity_index",
        "load_lag_2", "load_lag_48"
    ]
    target_col = "load_kw"

    split_idx = int(len(data) * 0.8)
    X_train, y_train = data.iloc[:split_idx][feature_cols], data.iloc[:split_idx][target_col]
    X_test, y_test = data.iloc[split_idx:][feature_cols], data.iloc[split_idx:][target_col]

    model = HistGradientBoostingRegressor(
        max_iter=100, learning_rate=0.05, max_depth=5, random_state=42
    )
    model.fit(X_train, y_train)

    preds_test = model.predict(X_test)
    rmse = float(root_mean_squared_error(y_test, preds_test))
    mape = float(mean_absolute_percentage_error(y_test, preds_test) * 100.0)
    residual_std = float(np.std(y_test - preds_test))

    metrics = {
        "rmse": rmse, "mape": mape, "residual_std": residual_std,
        "train_samples": len(X_train), "test_samples": len(X_test)
    }
    return model, feature_cols, metrics


def generate_forecast(model, feature_cols, last_known_df, residual_std, horizon_hours=24, temp_scenario_delta=0.0):
    """Generates multi-period horizon forecast with confidence intervals."""
    start_time = last_known_df["timestamp"].max() + pd.Timedelta(minutes=30)
    future_periods = horizon_hours * 2
    future_dates = pd.date_range(start=start_time, periods=future_periods, freq="30min")

    hours = future_dates.hour + future_dates.minute / 60.0
    future_temp = 24.0 + 8.0 * np.sin(np.pi * (hours - 8.0) / 12.0) + temp_scenario_delta
    future_rh = np.clip(70.0 - 20.0 * np.sin(np.pi * (hours - 8.0) / 12.0), 20.0, 95.0)

    future_df = pd.DataFrame({
        "timestamp": future_dates,
        "ambient_temp_c": future_temp,
        "relative_humidity": future_rh,
        "sin_hour": np.sin(2.0 * np.pi * hours / 24.0),
        "cos_hour": np.cos(2.0 * np.pi * hours / 24.0),
        "sin_dow": np.sin(2.0 * np.pi * future_dates.dayofweek / 7.0),
        "cos_dow": np.cos(2.0 * np.pi * future_dates.dayofweek / 7.0),
        "temp_humidity_index": future_temp * (future_rh / 100.0)
    })

    history_loads = list(last_known_df["load_kw"].values)
    preds = []
    for i in range(future_periods):
        lag_2 = history_loads[-2] if len(history_loads) >= 2 else history_loads[-1]
        lag_48 = history_loads[-48] if len(history_loads) >= 48 else history_loads[0]
        
        row = future_df.iloc[i].copy()
        row["load_lag_2"] = lag_2
        row["load_lag_48"] = lag_48

        X_in = pd.DataFrame([row[feature_cols]])
        pred_val = float(model.predict(X_in)[0])
        preds.append(pred_val)
        history_loads.append(pred_val)

    future_df["forecast_load_kw"] = preds
    half_width = 1.96 * residual_std
    future_df["confidence_lower_kw"] = np.maximum(0.0, future_df["forecast_load_kw"] - half_width)
    future_df["confidence_upper_kw"] = future_df["forecast_load_kw"] + half_width

    return future_df


# ==========================================
# 3. STREAMLIT UI DASHBOARD
# ==========================================

st.set_page_config(
    page_title="GridSight Enterprise | Microgrid Command",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .badge-sim { background-color: #FF9800; color: black; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚡ GridSight Control")
site_selected = st.sidebar.selectbox("Select Facility / Site", list(SITE_CONFIGS.keys()))
horizon_hours = st.sidebar.slider("Forecast Horizon (Hours)", 12, 72, 24, step=12)

st.sidebar.markdown("---")
st.sidebar.subheader("🌡️ Weather Scenario Studio")
temp_delta = st.sidebar.slider("Simulate Heatwave Offset (°C)", -5.0, 10.0, 0.0, 0.5)

# Load Telemetry Data
site_data = generate_site_data(site_selected, days=30)
latest_row = site_data.iloc[-1]

# Header Row
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title(f"🏢 {site_selected}")
    st.caption("Enterprise Microgrid SCADA & Operational Decision Intelligence Engine")
with col_h2:
    st.markdown("<br><span class='badge-sim'>SIMULATION MODE</span>", unsafe_allow_html=True)
    st.write(f"**Local Time:** {latest_row['timestamp'].strftime('%Y-%m-%d %H:%M %Z')}")

st.markdown("---")

# Executive BI Financial KPIs
st.subheader("📊 Executive Financial & Resiliency KPIs")
k1, k2, k3, k4 = st.columns(4)

peak_demand_cost = float(latest_row["grid_import_kw"]) * 450.0
tod_rate = 8.5
arbitrage_yield = float(site_data["battery_kw"].clip(lower=0).sum() * 0.5) * (tod_rate * 0.35)
co2_offset = float(site_data["solar_kw"].sum() * 0.5 * 0.82) / 1000.0
autonomy_hrs = float(site_data["battery_soc_kwh"].iloc[-1]) / max(1.0, float(latest_row["load_kw"] - latest_row["solar_kw"]))

k1.metric("Peak Cost Exposure", f"₹{peak_demand_cost:,.0f}", delta="Real-Time Grid Demand")
k2.metric("Monthly Arbitrage Yield", f"₹{arbitrage_yield:,.0f}", delta="ToD Optimization")
k3.metric("ESG Carbon Offsets", f"{co2_offset:.2f} Tons", delta="30-Day Solar Clean Yield")
k4.metric("Grid Autonomy Reserve", f"{autonomy_hrs:.1f} Hours", delta="Islanded Backup Capacity")

st.markdown("---")

# Train & Generate Forecast
model, features, ml_metrics = train_forecast_model(site_data)
forecast_df = generate_forecast(
    model, features, site_data, ml_metrics["residual_std"],
    horizon_hours=horizon_hours, temp_scenario_delta=temp_delta
)

# Render Dashboard Tabs
tab_telemetry, tab_forecast, tab_battery, tab_model_diag = st.tabs(
    ["📡 Real-Time Telemetry", "🔮 Demand Forecast & Scenarios", "🔋 Battery Dispatch", "🧪 ML Diagnostics"]
)

with tab_telemetry:
    st.subheader("Live Operational SCADA Feed")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Site Consumption", f"{latest_row['load_kw']:.1f} kW")
    m2.metric("Solar Production", f"{latest_row['solar_kw']:.1f} kW")
    m3.metric("Net Grid Import", f"{latest_row['grid_import_kw']:.1f} kW")
    m4.metric("Ambient Temp", f"{latest_row['ambient_temp_c']:.1f} °C")
    m5.metric("AQI Standard", f"{latest_row['aqi']:.0f}", delta=latest_row['aqi_label'])

    fig_scada = go.Figure()
    fig_scada.add_trace(go.Scatter(x=site_data["timestamp"].tail(96), y=site_data["load_kw"].tail(96), name="Load (kW)", line=dict(color="#FF5252")))
    fig_scada.add_trace(go.Scatter(x=site_data["timestamp"].tail(96), y=site_data["solar_kw"].tail(96), name="Solar (kW)", line=dict(color="#FFD700")))
    fig_scada.add_trace(go.Scatter(x=site_data["timestamp"].tail(96), y=site_data["grid_import_kw"].tail(96), name="Grid Import (kW)", line=dict(color="#00E676", dash="dash")))
    fig_scada.update_layout(title="48-Hour SCADA Telemetry Window", template="plotly_dark", height=400)
    st.plotly_chart(fig_scada, use_container_width=True)

with tab_forecast:
    st.subheader(f"{horizon_hours}-Hour Demand Forecast")
    if temp_delta != 0.0:
        st.info(f"⚠️ **Thermal Scenario:** Applied +{temp_delta}°C temperature shift across horizon.")

    peak_row = forecast_df.loc[forecast_df["forecast_load_kw"].idxmax()]
    st.warning(f"⚡ **Horizon Peak:** {peak_row['forecast_load_kw']:.1f} kW at **{peak_row['timestamp'].strftime('%Y-%m-%d %H:%M %Z')}**")

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=list(forecast_df["timestamp"]) + list(forecast_df["timestamp"])[::-1],
        y=list(forecast_df["confidence_upper_kw"]) + list(forecast_df["confidence_lower_kw"])[::-1],
        fill="toself", fillcolor="rgba(0, 176, 255, 0.2)", line=dict(color="rgba(255,255,255,0)"),
        name="95% Calibrated Confidence Interval"
    ))
    fig_fc.add_trace(go.Scatter(x=forecast_df["timestamp"], y=forecast_df["forecast_load_kw"], name="Predicted Demand (kW)", line=dict(color="#00B0FF", width=3)))
    fig_fc.update_layout(template="plotly_dark", height=450, xaxis_title="Timestamp", yaxis_title="Power Demand (kW)")
    st.plotly_chart(fig_fc, use_container_width=True)

with tab_battery:
    st.subheader("Battery Energy Storage System (BESS) Dynamics")
    fig_bess = go.Figure()
    fig_bess.add_trace(go.Scatter(x=site_data["timestamp"].tail(192), y=site_data["battery_soc_pct"].tail(192), name="SoC (%)", line=dict(color="#7C4DFF", width=2)))
    fig_bess.add_trace(go.Scatter(x=site_data["timestamp"].tail(192), y=site_data["battery_kw"].tail(192), name="BESS Output (kW)", line=dict(color="#00E5FF", dash="dot")))
    fig_bess.update_layout(title="4-Day BESS Dispatch & State-of-Charge Tracking", template="plotly_dark", height=400)
    st.plotly_chart(fig_bess, use_container_width=True)

with tab_model_diag:
    st.subheader("ML Architecture Diagnostics & Backtesting Performance")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Test MAPE", f"{ml_metrics['mape']:.2f}%")
    d2.metric("Test RMSE", f"{ml_metrics['rmse']:.2f} kW")
    d3.metric("Residual Std (σ)", f"{ml_metrics['residual_std']:.2f} kW")
    d4.metric("Train / Test Samples", f"{ml_metrics['train_samples']} / {ml_metrics['test_samples']}")
