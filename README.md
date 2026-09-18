# GridSight — Smart Microgrid & Environmental Impact Dashboard

A Streamlit prototype that combines smart-grid load forecasting with solar generation, weather and AQI monitoring.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Included

- Half-hourly simulated smart-meter, weather, irradiance, AQI and PV telemetry
- Temperature-aware load forecast with a confidence interval
- PV/AQI overlay and AQI-derived dust-loss estimate
- Scenario control for forecast temperature, historical period and forecast horizon

`generate_site_data()` is the explicit integration point for real smart-meter, weather, and AQI provider data.
