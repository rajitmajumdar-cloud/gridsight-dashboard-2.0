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

    # Battery State Simulation (Greedy Dispatch)
    battery_capacity_kwh = config["battery_capacity_kwh"]
    max_kw = config["max_battery_kw"]
    soc_kwh = np.zeros(periods)
    battery_kw = np.zeros(periods)
    curr_soc = battery_capacity_kwh * 0.5

    net_unbuffered = simulated_load_kw - actual_solar_kw
    for i in range(periods):
        net = net_unbuffered[i]
        if net > 0:  # Discharge
            discharge = min(float(net), float(max_kw), float(curr_soc))
            battery_kw[i] = discharge
            curr_soc -= discharge * 0.5
        else:  # Charge
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
