"""
app.py - GridSight Enterprise Microgrid Control & BI Dashboard
Single-file enterprise suite containing:
1. Physics & SCADA Telemetry Simulation (CPCB AQI standards, NOCT thermal cell models)
2. Machine Learning Pipeline (Fourier cyclic encodings, XGBoost forecasting, calibrated confidence bands)
3. Streamlit 1.66+ Executive SCADA UI & Financial KPI Engine
"""

from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
import xgboost as xgb

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
    cell_temp_c = ambient_temp + (45.0 -
