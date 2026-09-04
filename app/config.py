"""
FloodGuard AI - Shared configuration
Central place for feature names, risk thresholds, and file paths so every
module (data generator, trainer, API, dashboard) stays in sync.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

TRAINING_DATA_PATH = DATA_DIR / "flood_training_data.csv"
LOCATIONS_PATH = DATA_DIR / "demo_locations.csv"

MODEL_PATH = MODEL_DIR / "flood_rf_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

DB_PATH = BASE_DIR / "floodguard.db"

# ---------------------------------------------------------------------------
# Model features
# ---------------------------------------------------------------------------
# Order matters: this exact order is used for training AND inference.
FEATURE_NAMES = [
    "rainfall_1h_mm",
    "rainfall_6h_mm",
    "rainfall_24h_mm",
    "river_level_m",
    "river_level_change_m",
    "soil_moisture_pct",
    "temperature_c",
    "humidity_pct",
    "elevation_m",
    "slope_deg",
]

# Human-readable labels for dashboard / explanations
FEATURE_LABELS = {
    "rainfall_1h_mm": "Rainfall (last 1h, mm)",
    "rainfall_6h_mm": "Rainfall (last 6h, mm)",
    "rainfall_24h_mm": "Rainfall (last 24h, mm)",
    "river_level_m": "River / water level (m)",
    "river_level_change_m": "River level change rate (m/h)",
    "soil_moisture_pct": "Soil moisture (%)",
    "temperature_c": "Temperature (C)",
    "humidity_pct": "Humidity (%)",
    "elevation_m": "Elevation (m)",
    "slope_deg": "Terrain slope (degrees)",
}

# Reasonable physical bounds used for input validation (hackathon-grade, not
# meteorological gospel -- generous enough to allow extreme events).
FEATURE_BOUNDS = {
    "rainfall_1h_mm": (0.0, 250.0),
    "rainfall_6h_mm": (0.0, 600.0),
    "rainfall_24h_mm": (0.0, 1200.0),
    "river_level_m": (0.0, 30.0),
    "river_level_change_m": (-5.0, 5.0),
    "soil_moisture_pct": (0.0, 100.0),
    "temperature_c": (-10.0, 55.0),
    "humidity_pct": (0.0, 100.0),
    "elevation_m": (0.0, 6000.0),
    "slope_deg": (0.0, 60.0),
}

# ---------------------------------------------------------------------------
# Risk classification thresholds (on 0-100 flood probability scale)
# ---------------------------------------------------------------------------
RISK_THRESHOLDS = {
    "Low": (0, 25),
    "Moderate": (25, 50),
    "High": (50, 75),
    "Extreme": (75, 100),
}

RISK_COLORS = {
    "Low": "#2ecc71",       # green
    "Moderate": "#f1c40f",  # yellow
    "High": "#e67e22",      # orange
    "Extreme": "#e74c3c",   # red
}


def classify_risk(probability_pct: float) -> str:
    """Map a 0-100 flood probability to a risk category label."""
    if probability_pct < 25:
        return "Low"
    elif probability_pct < 50:
        return "Moderate"
    elif probability_pct < 75:
        return "High"
    else:
        return "Extreme"


RANDOM_SEED = 42
