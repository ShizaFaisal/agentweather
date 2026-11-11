"""Central configuration. Everything can be overridden with environment variables."""
import os

CITY = os.getenv("CITY", "Karachi")
LAT = float(os.getenv("LAT", "24.8607"))
LON = float(os.getenv("LON", "67.0011"))

HORIZON = 24  # hours ahead

# The nine modelled signals. Six user-facing metrics map onto them:
#   AQI -> aqi | Temperature -> temperature | Precipitation (mm) -> precipitation
#   Rain probability -> rain_flag (0/1 target, model output is a probability)
#   Humidity -> humidity | UV -> uv_index
#   Wind speed -> wind_speed | Wind direction -> wind_sin / wind_cos (circular)
TARGETS = ["aqi", "temperature", "precipitation", "rain_flag", "humidity",
           "uv_index", "wind_speed", "wind_sin", "wind_cos"]
RAW_COLS = ["aqi", "temperature", "precipitation", "humidity", "uv_index",
            "wind_speed", "wind_dir", "pm2_5", "pm10"]
DRIFT_COLS = ["aqi", "temperature", "precipitation", "humidity", "uv_index", "wind_speed"]

RAIN_THRESHOLD_MM = 0.1
REPORT_HOUR_UTC = int(os.getenv("REPORT_HOUR_UTC", "3"))   # daily report time

# Drift thresholds
DATA_DRIFT_KS = 0.30        # KS statistic live-vs-training
MODEL_DRIFT_RATIO = 1.5     # recent MAE / validation MAE
DRIFT_COOLDOWN_H = 12       # min hours between drift-triggered trainings

# Fixed alert rules: (metric, "max"/"min", threshold, level, label)
ALERT_RULES = [
    ("aqi", "max", 100, "watch", "Unhealthy air for sensitive groups"),
    ("aqi", "max", 150, "warning", "Unhealthy air quality"),
    ("aqi", "max", 200, "severe", "Very unhealthy air quality"),
    ("aqi", "max", 300, "extreme", "Hazardous air quality"),
    ("temperature", "max", 38, "warning", "Heat warning"),
    ("temperature", "max", 42, "severe", "Extreme heat"),
    ("temperature", "min", 2, "warning", "Near-freezing temperatures"),
    ("temperature", "min", -5, "severe", "Hard freeze"),
    ("precipitation", "max", 5, "watch", "Heavy rain possible"),
    ("precipitation", "max", 15, "warning", "Very heavy rain"),
    ("rain_flag", "max", 0.7, "watch", "High chance of rain"),
    ("humidity", "max", 90, "watch", "Very high humidity"),
    ("humidity", "min", 20, "watch", "Very low humidity / dry air"),
    ("uv_index", "max", 6, "watch", "High UV"),
    ("uv_index", "max", 8, "warning", "Very high UV"),
    ("uv_index", "max", 11, "severe", "Extreme UV"),
    ("wind_speed", "max", 40, "warning", "Strong winds"),
    ("wind_speed", "max", 60, "severe", "Damaging winds"),
]
LEVEL_ORDER = {"watch": 1, "warning": 2, "severe": 3, "extreme": 4}

GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]  # fallback chain
