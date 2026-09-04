"""
FloodGuard AI - Automated test suite
=======================================
Run:
    pytest -v

Covers:
  - Dataset generation shape/columns/label sanity
  - Model artifacts exist and load
  - Predictor input validation (missing fields, bad types, out-of-range)
  - Predictor produces varying, non-hardcoded probabilities for different inputs
  - Risk classification thresholds
  - Explanation output structure
  - Database read/write roundtrip
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import FEATURE_NAMES, classify_risk, MODEL_PATH, SCALER_PATH
from app.data_generator import generate_training_dataset, generate_demo_locations
from app.predictor import get_predictor, ValidationError, FloodPredictor


# ---------------------------------------------------------------------------
# Data generation tests
# ---------------------------------------------------------------------------

def test_generate_training_dataset_shape():
    df = generate_training_dataset(n_samples=500, seed=1)
    assert len(df) == 500
    for col in FEATURE_NAMES:
        assert col in df.columns
    assert "flood_occurred" in df.columns
    assert set(df["flood_occurred"].unique()).issubset({0, 1})


def test_generate_training_dataset_reasonable_positive_rate():
    df = generate_training_dataset(n_samples=2000, seed=2)
    rate = df["flood_occurred"].mean()
    # Should not be degenerate (all 0s or all 1s) -- a real learnable dataset
    assert 0.05 < rate < 0.95


def test_generate_training_dataset_feature_bounds_reasonable():
    df = generate_training_dataset(n_samples=1000, seed=3)
    assert (df["rainfall_1h_mm"] >= 0).all()
    assert (df["soil_moisture_pct"] >= 0).all()
    assert (df["soil_moisture_pct"] <= 100).all()
    assert (df["humidity_pct"] <= 100).all()


def test_generate_demo_locations():
    locs = generate_demo_locations()
    assert len(locs) >= 5
    for col in ["name", "lat", "lon"] + FEATURE_NAMES:
        assert col in locs.columns


def test_dataset_generation_is_reproducible_with_seed():
    df1 = generate_training_dataset(n_samples=300, seed=99)
    df2 = generate_training_dataset(n_samples=300, seed=99)
    assert np.allclose(df1["rainfall_1h_mm"].values, df2["rainfall_1h_mm"].values)
    assert (df1["flood_occurred"].values == df2["flood_occurred"].values).all()


# ---------------------------------------------------------------------------
# Model artifact tests
# ---------------------------------------------------------------------------

def test_model_artifacts_exist():
    assert MODEL_PATH.exists(), "Run `python -m app.train_model` before testing."
    assert SCALER_PATH.exists(), "Run `python -m app.train_model` before testing."


def test_predictor_loads_without_error():
    predictor = get_predictor()
    predictor._ensure_loaded()
    assert predictor._model is not None
    assert predictor._scaler is not None


# ---------------------------------------------------------------------------
# Risk classification tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prob,expected", [
    (0, "Low"), (10, "Low"), (24.9, "Low"),
    (25, "Moderate"), (40, "Moderate"), (49.9, "Moderate"),
    (50, "High"), (60, "High"), (74.9, "High"),
    (75, "Extreme"), (90, "Extreme"), (100, "Extreme"),
])
def test_classify_risk_thresholds(prob, expected):
    assert classify_risk(prob) == expected


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

def valid_features(**overrides):
    base = {
        "rainfall_1h_mm": 10.0, "rainfall_6h_mm": 30.0, "rainfall_24h_mm": 60.0,
        "river_level_m": 3.0, "river_level_change_m": 0.2, "soil_moisture_pct": 40.0,
        "temperature_c": 25.0, "humidity_pct": 60.0, "elevation_m": 1000.0, "slope_deg": 12.0,
    }
    base.update(overrides)
    return base


def test_validation_accepts_valid_input():
    predictor = get_predictor()
    cleaned = predictor.validate_input(valid_features())
    assert cleaned["rainfall_1h_mm"] == 10.0


def test_validation_rejects_missing_field():
    predictor = get_predictor()
    feats = valid_features()
    del feats["soil_moisture_pct"]
    with pytest.raises(ValidationError):
        predictor.validate_input(feats)


def test_validation_rejects_non_numeric():
    predictor = get_predictor()
    feats = valid_features(rainfall_1h_mm="heavy")
    with pytest.raises(ValidationError):
        predictor.validate_input(feats)


def test_validation_rejects_out_of_range():
    predictor = get_predictor()
    feats = valid_features(humidity_pct=250.0)
    with pytest.raises(ValidationError):
        predictor.validate_input(feats)


def test_validation_rejects_nan():
    predictor = get_predictor()
    feats = valid_features(rainfall_1h_mm=float("nan"))
    with pytest.raises(ValidationError):
        predictor.validate_input(feats)


def test_validation_rejects_negative_rainfall():
    predictor = get_predictor()
    feats = valid_features(rainfall_1h_mm=-5.0)
    with pytest.raises(ValidationError):
        predictor.validate_input(feats)


# ---------------------------------------------------------------------------
# Prediction behavior tests
# ---------------------------------------------------------------------------

def test_prediction_returns_valid_probability_range():
    predictor = get_predictor()
    result = predictor.predict(valid_features())
    assert 0.0 <= result.flood_probability_pct <= 100.0


def test_prediction_risk_level_matches_probability():
    predictor = get_predictor()
    result = predictor.predict(valid_features())
    assert result.risk_level == classify_risk(result.flood_probability_pct)


def test_prediction_is_not_hardcoded_low_vs_high_differ():
    """The model must actually respond to inputs -- low and extreme
    scenarios should produce meaningfully different probabilities."""
    predictor = get_predictor()

    low = valid_features(
        rainfall_1h_mm=0.5, rainfall_6h_mm=1.0, rainfall_24h_mm=3.0,
        river_level_m=1.0, river_level_change_m=-0.2, soil_moisture_pct=15.0,
    )
    high = valid_features(
        rainfall_1h_mm=80.0, rainfall_6h_mm=250.0, rainfall_24h_mm=500.0,
        river_level_m=12.0, river_level_change_m=3.5, soil_moisture_pct=95.0,
        slope_deg=40.0,
    )

    low_result = predictor.predict(low)
    high_result = predictor.predict(high)

    assert high_result.flood_probability_pct > low_result.flood_probability_pct
    assert (high_result.flood_probability_pct - low_result.flood_probability_pct) > 20


def test_prediction_changes_with_single_feature_change():
    """Changing just rainfall should change the predicted probability --
    proves the model is actually using the feature, not ignoring it."""
    predictor = get_predictor()
    base = valid_features()
    wetter = valid_features(rainfall_1h_mm=90.0, rainfall_6h_mm=300.0, rainfall_24h_mm=600.0)

    base_result = predictor.predict(base)
    wetter_result = predictor.predict(wetter)

    assert wetter_result.flood_probability_pct != base_result.flood_probability_pct


def test_prediction_explanation_structure():
    predictor = get_predictor()
    result = predictor.predict(valid_features())
    assert len(result.top_factors) > 0
    for factor in result.top_factors:
        assert "feature" in factor
        assert "label" in factor
        assert "direction" in factor
        assert factor["direction"] in ("increasing", "reducing")


def test_predictor_raises_before_model_trained(tmp_path, monkeypatch):
    """A fresh predictor pointed at a non-existent model path should fail
    clearly rather than silently returning a fake prediction."""
    fresh_predictor = FloodPredictor()
    monkeypatch.setattr("app.predictor.MODEL_PATH", tmp_path / "nonexistent_model.joblib")
    with pytest.raises(FileNotFoundError):
        fresh_predictor._ensure_loaded()


# ---------------------------------------------------------------------------
# Database tests
# ---------------------------------------------------------------------------

def test_database_roundtrip(tmp_path, monkeypatch):
    test_db_path = tmp_path / "test_floodguard.db"
    monkeypatch.setattr("app.database.DB_PATH", test_db_path)

    from app.database import init_db, save_prediction, get_recent_predictions, clear_history

    init_db()
    row_id = save_prediction(
        features=valid_features(),
        probability=42.5,
        risk_level="Moderate",
        top_factors=[{"feature": "rainfall_1h_mm", "direction": "increasing"}],
        location_name="Test Site",
    )
    assert row_id is not None

    rows = get_recent_predictions(limit=10)
    assert len(rows) == 1
    assert rows[0]["location_name"] == "Test Site"
    assert rows[0]["risk_level"] == "Moderate"
    assert rows[0]["flood_probability"] == 42.5
    assert isinstance(rows[0]["top_factors"], list)

    clear_history()
    rows_after_clear = get_recent_predictions(limit=10)
    assert len(rows_after_clear) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
