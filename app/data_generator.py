"""
FloodGuard AI - Synthetic training data generator
====================================================
THIS DATA IS SYNTHETIC / DEMO DATA. It is generated procedurally to mimic
plausible flash-flood-triggering conditions in hilly terrain. It is NOT
sourced from any real sensor, gauge, or government agency, and must never be
presented as real observational data.

Design notes
------------
We do NOT hand the model a final label directly copied from a simple linear
formula the demo could be accused of "faking". Instead we:
  1. Sample each of the 10 raw features from realistic distributions.
  2. Compute a *continuous* underlying "flood stress" score from a nonlinear
     combination of features (interactions + thresholds), inspired loosely
     on hydrological intuition (e.g. rainfall matters much more when soil is
     already saturated and slope is steep; river level change rate matters
     more than absolute level).
  3. Add Gaussian noise to the stress score (real-world unpredictability).
  4. Convert stress -> probability with a logistic squashing function.
  5. Sample a binary "flood_occurred" outcome using that probability
     (Bernoulli draw) -- so the label has genuine randomness, not a
     deterministic mirror of the inputs.
  6. Train the RandomForestClassifier on the BINARY outcome. At inference
     time we use predict_proba() to get a genuine model-estimated
     probability -- the model must learn the feature relationships itself
     from the noisy binary outcomes, it is never given the stress formula.

This means the Random Forest cannot simply "look up" the answer -- it has to
learn patterns from noisy examples, exactly like a real ML pipeline would
from historical flood records.
"""

import numpy as np
import pandas as pd

from app.config import FEATURE_NAMES, TRAINING_DATA_PATH, LOCATIONS_PATH, RANDOM_SEED


def _sample_raw_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample raw environmental features from realistic-ish distributions."""

    # Rainfall: mostly light, occasionally heavy (Gamma distribution gives a
    # realistic long right tail for storm events).
    rainfall_1h = rng.gamma(shape=1.3, scale=6.0, size=n)
    rainfall_1h = np.clip(rainfall_1h, 0, 200)

    # 6h and 24h rainfall correlated with 1h but with their own accumulation
    rainfall_6h = rainfall_1h * rng.uniform(2.5, 5.5, size=n) + rng.gamma(1.0, 5.0, size=n)
    rainfall_6h = np.clip(rainfall_6h, rainfall_1h, 550)

    rainfall_24h = rainfall_6h * rng.uniform(1.6, 3.2, size=n) + rng.gamma(1.0, 8.0, size=n)
    rainfall_24h = np.clip(rainfall_24h, rainfall_6h, 1100)

    # River level: baseline + contribution from recent rainfall accumulation
    base_river_level = rng.normal(3.0, 1.2, size=n)
    river_level = base_river_level + rainfall_24h * 0.015 + rng.normal(0, 0.4, size=n)
    river_level = np.clip(river_level, 0.1, 25)

    # River level change rate - correlated with short-term rainfall intensity
    river_level_change = (rainfall_1h * 0.02 + rainfall_6h * 0.005
                           + rng.normal(0, 0.3, size=n))
    river_level_change = np.clip(river_level_change, -3, 4.5)

    # Soil moisture - saturates with recent rainfall, has natural baseline variability
    soil_moisture = 30 + rainfall_24h * 0.09 + rng.normal(0, 8, size=n)
    soil_moisture = np.clip(soil_moisture, 5, 100)

    # Temperature & humidity - weakly linked to rainfall (storms cool things, raise humidity)
    temperature = rng.normal(24, 6, size=n) - rainfall_1h * 0.03
    temperature = np.clip(temperature, -5, 48)

    humidity = 55 + rainfall_6h * 0.06 + rng.normal(0, 10, size=n)
    humidity = np.clip(humidity, 10, 100)

    # Terrain: elevation and slope - independent geographic features
    elevation = rng.uniform(50, 3500, size=n)
    slope = rng.gamma(shape=2.0, scale=6.0, size=n)
    slope = np.clip(slope, 0, 55)

    return pd.DataFrame({
        "rainfall_1h_mm": rainfall_1h,
        "rainfall_6h_mm": rainfall_6h,
        "rainfall_24h_mm": rainfall_24h,
        "river_level_m": river_level,
        "river_level_change_m": river_level_change,
        "soil_moisture_pct": soil_moisture,
        "temperature_c": temperature,
        "humidity_pct": humidity,
        "elevation_m": elevation,
        "slope_deg": slope,
    })


def _compute_flood_stress(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """
    Nonlinear "ground truth" hazard score used only to GENERATE noisy labels.
    The trained model never sees this formula -- it only sees the resulting
    noisy binary outcomes and must learn the relationships itself.
    """
    r1, r6, r24 = df["rainfall_1h_mm"], df["rainfall_6h_mm"], df["rainfall_24h_mm"]
    river = df["river_level_m"]
    river_chg = df["river_level_change_m"]
    soil = df["soil_moisture_pct"]
    slope = df["slope_deg"]
    elevation = df["elevation_m"]

    # Normalize a few terms to comparable scales
    rain_term = (0.35 * r1 / 50 + 0.30 * r6 / 150 + 0.35 * r24 / 300)
    river_term = 0.6 * (river / 10) + 0.9 * np.clip(river_chg, 0, None) / 2
    soil_term = (soil / 100) ** 1.5
    slope_term = slope / 45  # steeper slope -> faster runoff -> higher flash flood risk
    elevation_term = -0.15 * (elevation / 3500)  # higher elevation -> slightly lower risk (less pooling)

    # Interaction: saturated soil + heavy rain compounds risk (soil can't absorb more)
    interaction = 1.4 * (soil / 100) * (r6 / 150)
    # Interaction: steep slope + heavy short rain -> fast flash-flood response
    interaction2 = 1.2 * (slope / 45) * (r1 / 50)

    stress = (
        2.2 * rain_term
        + 1.6 * river_term
        + 1.1 * soil_term
        + 0.8 * slope_term
        + elevation_term
        + interaction
        + interaction2
        - 1.6  # bias so baseline conditions sit at low risk
    )

    noise = rng.normal(0, 0.55, size=len(df))
    return stress + noise


def _stress_to_probability(stress: np.ndarray) -> np.ndarray:
    """Logistic squashing of the stress score into a 0-1 probability."""
    return 1 / (1 + np.exp(-1.15 * stress))


def generate_training_dataset(n_samples: int = 6000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate the full synthetic training dataset with binary flood_occurred labels."""
    rng = np.random.default_rng(seed)

    df = _sample_raw_features(n_samples, rng)
    stress = _compute_flood_stress(df, rng)
    true_probability = _stress_to_probability(stress)

    # Bernoulli draw -> genuine noisy binary outcome (not a deterministic threshold)
    flood_occurred = rng.binomial(1, true_probability)

    df["flood_occurred"] = flood_occurred
    # Keep the underlying probability only for reference/analysis, not used in training features
    df["_reference_probability"] = true_probability

    # Reorder columns: features first, then labels
    df = df[FEATURE_NAMES + ["flood_occurred", "_reference_probability"]]
    return df


def generate_demo_locations(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a small set of DEMO hilly-region locations with plausible
    lat/lon/elevation/slope for the Folium risk map. Coordinates are loosely
    inspired by real hilly regions in India for demo realism but are NOT
    guaranteed to be precise -- this is illustrative demo data only.
    """
    rng = np.random.default_rng(seed + 1)

    locations = [
        {"name": "Rudraprayag Valley", "lat": 30.2849, "lon": 78.9814, "elevation_m": 895, "slope_deg": 28},
        {"name": "Munnar Hills", "lat": 10.0889, "lon": 77.0595, "elevation_m": 1600, "slope_deg": 22},
        {"name": "Darjeeling Ridge", "lat": 27.0410, "lon": 88.2663, "elevation_m": 2050, "slope_deg": 33},
        {"name": "Shimla Slopes", "lat": 31.1048, "lon": 77.1734, "elevation_m": 2200, "slope_deg": 25},
        {"name": "Wayanad Basin", "lat": 11.6854, "lon": 76.1320, "elevation_m": 780, "slope_deg": 19},
        {"name": "Nainital Catchment", "lat": 29.3803, "lon": 79.4636, "elevation_m": 1938, "slope_deg": 30},
        {"name": "Kalimpong Ridge", "lat": 27.0714, "lon": 88.4700, "elevation_m": 1250, "slope_deg": 27},
        {"name": "Manali Gorge", "lat": 32.2432, "lon": 77.1892, "elevation_m": 2050, "slope_deg": 35},
    ]

    df = pd.DataFrame(locations)
    # Give each demo location a live, randomized snapshot of current sensor-like readings
    n = len(df)
    live = _sample_raw_features(n, rng)
    # Override elevation/slope with the fixed terrain values for the named locations
    live["elevation_m"] = df["elevation_m"].values
    live["slope_deg"] = df["slope_deg"].values

    combined = pd.concat([df[["name", "lat", "lon"]].reset_index(drop=True),
                           live.reset_index(drop=True)], axis=1)
    return combined


def main():
    print("Generating synthetic FloodGuard AI training dataset (DEMO DATA)...")
    df = generate_training_dataset()
    df.to_csv(TRAINING_DATA_PATH, index=False)
    print(f"Saved {len(df)} rows to {TRAINING_DATA_PATH}")
    print(f"Flood-occurred positive rate: {df['flood_occurred'].mean():.3f}")

    locs = generate_demo_locations()
    locs.to_csv(LOCATIONS_PATH, index=False)
    print(f"Saved {len(locs)} demo locations to {LOCATIONS_PATH}")


if __name__ == "__main__":
    main()
