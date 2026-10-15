"""
engine.py - Physics-grounded Telemetry and Microgrid Simulation Engine
Handles multi-site parameters, timezone safety, Indian CPCB AQI standards,
and physical solar/thermal derating models.
"""

from datetime import datetime
import numpy as np
import pandas as pd
import pytz

# Site Configurations & Physical Parameters
SITE_CONFIGS = {
    "Pune Campus": {
        "timezone": "Asia/Kolkata",
        "pv_capacity_kw": 250.0,
        "base_load_kw": 120.0,
        "peak_load_kw": 380.0,
        "seed": 101,
        "coords": (18.5204, 73.8567),
        "battery_capacity_kwh": 400.0,
        "max_battery_kw": 100.0,
    },
    "Bengaluru Tech Park": {
        "timezone": "Asia/Kolkata",
        "pv_capacity_kw": 500.0,
        "base_load_kw": 250.0,
        "peak_load_kw": 850.0,
        "seed": 202,
        "coords": (12.9716, 77.5946),
        "battery_capacity_kwh": 800.0,
        "max_battery_kw": 200.0,
    },
    "Delhi Regional Hub": {
        "timezone": "Asia/Kolkata",
        "pv_capacity_kw": 180.0,
        "base_load_kw": 90.0,
        "peak_load_kw": 310.0,
        "seed": 303,
        "coords": (28.6139, 77.2090),
        "battery_capacity_kwh": 300.0,
        "max_battery_kw": 75.0,
    },
}


def get_cpcb_aqi_category(aqi: float) -> str:
    """Official CPCB (Indian Central Pollution Control Board) AQI categories."""
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
    """
    Generates realistic, site-specific 30-minute telemetry data.
    Incorporates relative humidity, thermal cell temp models, and soiling heuristics.
    """
    config = SITE_CONFIGS.get(site_name, SITE_CONFIGS["Pune Campus"])
    tz = pytz.timezone(config["timezone"])

    end_time = datetime.now(tz).replace(minute=0, second=0, microsecond=0)
    periods = days * 24 * 2  # 30-min resolution
    timestamps = pd.date_range(end=end_time, periods=periods, freq="30min", tz=tz)

    np.random.seed(config["seed"])

    hours = timestamps.hour + timestamps.minute / 60.0
    day_of_week = timestamps.dayofweek

    # Ambient Temperature Profile (Diurnal cycle)
    ambient_temp = (
        24
        + 8 * np.sin(np.pi * (hours - 8) / 12)
        + np.random.normal(0, 1.2, periods)
    )

    # Relative Humidity Profile (%)
    relative_humidity = np.clip(
        70 - 20 * np.sin(np.pi * (hours - 8) / 12) + np.random.normal(0, 3, periods),
        20,
        98,
    )

    # Solar Irradiance (W/m2)
    solar_raw = np.maximum(0, np.sin(np.pi * (hours - 6) / 12))
    solar_raw[hours < 6] = 0
    solar_raw[hours > 18] = 0
    irradiance_wm2 = solar_raw * 1000.0 * np.random.uniform(0.85, 1.0, periods)

    # AQI & Soiling Loss Heuristic
    base_aqi = 180 if "Delhi" in site_name else (110 if "Pune" in site_name else 75)
    aqi_series = np.clip(base_aqi + np.random.normal(0, 25, periods), 20, 500)
    soiling_derate_factor = 1.0 - np.maximum(0, (aqi_series - 100) * 0.0002)

    # NOCT Thermal Cell Model: T_cell = T_amb + (NOCT-20)/800 * Irradiance
    cell_temp_c = ambient_temp + (45 - 20) / 800.0 * irradiance_wm2
    temp_derate_factor = 1.0 - np.maximum(0, (cell_temp_c - 25) * 0.004)

    # Ideal vs Actual PV Generation
    ideal_solar_kw = (irradiance_wm2 / 1000.0) * config["pv_capacity_kw"]
    actual_solar_kw = ideal_solar_kw * soiling_derate_factor * temp_derate_factor

    pv_performance_ratio = np.where(
        ideal_solar_kw > 0, actual_solar_kw / ideal_solar_kw, 0.0
    )

    # Campus HVAC & Base Consumption Modeling
    is_weekend = (day_of_week >= 5).astype(int)
    occupancy = (1 - 0.5 * is_weekend) * np.maximum(0, np.sin(np.pi * (hours - 7) / 11))
    
    # Non-linear thermal load (HVAC cooling spike with humidity interaction)
    hvac_load = np.maximum(0, (ambient_temp - 24) ** 1.5) * (1 + relative_humidity / 200.0) * 3.5

    base_load = config["base_load_kw"]
    peak_capacity = config["peak_load_kw"]
    simulated_load_kw = (
        base_load
        + (peak_capacity - base_load) * 0.6 * occupancy
        + hvac_load
        + np.random.normal(0, 4.0, periods)
    )
    simulated_load_kw = np.clip(simulated_load_kw, base_load * 0.5, None)

    # Battery State Simulation (Greedy Self-Consumption Dispatch)
    battery_capacity_kwh = config["battery_capacity_kwh"]
    max_kw = config["max_battery_kw"]
    
    soc_kwh = np.zeros(periods)
    battery_kw = np.zeros(periods) # + = discharging to load, - = charging
    curr_soc = battery_capacity_kwh * 0.5

    net_unbuffered = simulated_load_kw - actual_solar_kw

    for i in range(periods):
        net = net_unbuffered[i]
        if net > 0:  # Deficit -> Discharge battery
            discharge = min(net, max_kw, curr_soc)
            battery_kw[i] = discharge
            curr_soc -= discharge * 0.5
        else:  # Surplus -> Charge battery
            charge = min(abs(net), max_kw, battery_capacity_kwh - curr_soc)
            battery_kw[i] = -charge
            curr_soc += charge * 0.5
        
        soc_kwh[i] = curr_soc

    # True Grid Import / Export Math
    grid_import_kw = np.maximum(0, simulated_load_kw - actual_solar_kw - battery_kw)
    grid_export_kw = np.maximum(0, actual_solar_kw + battery_kw - simulated_load_kw)

    df = pd.DataFrame(
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

    return df
