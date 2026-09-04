"""
FloodGuard AI - Prediction engine
====================================
Loads the trained RandomForest model + scaler and exposes a single
`predict_flood_risk()` function used by BOTH the FastAPI backend and the
Streamlit dashboard, so behavior is always consistent.

Includes:
  - Input validation (types, bounds, required fields)
  - Flood probability (0-100%) from the real trained model
  - Risk classification (Low / Moderate / High / Extreme)
  - A model-grounded explanation of which features pushed the prediction up
    or down, computed by ablating each feature to a calm baseline and
    re-querying the actual trained model (see `_explain` for details) --
    not a hand-coded rule.
"""

import json
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np

from app.config import (FEATURE_NAMES, FEATURE_LABELS, FEATURE_BOUNDS,
                         MODEL_PATH, SCALER_PATH, FEATURE_IMPORTANCE_PATH,
                         RISK_COLORS, classify_risk)

# "Safe baseline" values used only for the human-readable explanation (NOT
# used in the model's actual prediction). Represents fairly calm conditions.
SAFE_BASELINE = {
    "rainfall_1h_mm": 2.0,
    "rainfall_6h_mm": 8.0,
    "rainfall_24h_mm": 20.0,
    "river_level_m": 2.0,
    "river_level_change_m": 0.0,
    "soil_moisture_pct": 30.0,
    "temperature_c": 24.0,
    "humidity_pct": 55.0,
    "elevation_m": 1000.0,
    "slope_deg": 10.0,
}


class ValidationError(ValueError):
    """Raised when input features fail validation."""
    pass


@dataclass
class PredictionResult:
    flood_probability_pct: float
    risk_level: str
    risk_color: str
    top_factors: list
    feature_importances: dict
    inputs_used: dict


class FloodPredictor:
    """Wraps the trained model + scaler and provides a stable prediction API."""

    def __init__(self):
        self._model = None
        self._scaler = None
        self._importances = None

    def _ensure_loaded(self):
        if self._model is None:
            if not MODEL_PATH.exists() or not SCALER_PATH.exists():
                raise FileNotFoundError(
                    "Model artifacts not found. Run `python -m app.train_model` "
                    "first to train and save the model."
                )
            self._model = joblib.load(MODEL_PATH)
            self._scaler = joblib.load(SCALER_PATH)
            if FEATURE_IMPORTANCE_PATH.exists():
                with open(FEATURE_IMPORTANCE_PATH) as f:
                    self._importances = json.load(f)
            else:
                self._importances = dict(zip(
                    FEATURE_NAMES, self._model.feature_importances_.tolist()
                ))

    def validate_input(self, features: dict) -> dict:
        """
        Validate that all required features are present, numeric, and within
        plausible physical bounds. Returns a cleaned dict of floats.
        Raises ValidationError with a clear message on failure.
        """
        cleaned = {}
        missing = [f for f in FEATURE_NAMES if f not in features or features[f] is None]
        if missing:
            raise ValidationError(f"Missing required feature(s): {', '.join(missing)}")

        for name in FEATURE_NAMES:
            raw_value = features[name]
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                raise ValidationError(f"Feature '{name}' must be numeric, got: {raw_value!r}")

            if not np.isfinite(value):
                raise ValidationError(f"Feature '{name}' must be a finite number, got: {value}")

            low, high = FEATURE_BOUNDS[name]
            if value < low or value > high:
                raise ValidationError(
                    f"Feature '{name}'={value} is outside the plausible range "
                    f"[{low}, {high}]. Please check the input."
                )
            cleaned[name] = value

        return cleaned

    def _model_proba(self, feat_dict: dict) -> float:
        X = np.array([[feat_dict[f] for f in FEATURE_NAMES]])
        X_scaled = self._scaler.transform(X)
        return float(self._model.predict_proba(X_scaled)[0, 1])

    def _explain(self, cleaned: dict, base_probability: float) -> list:
        """
        Single-feature ablation: for each feature, ask the ACTUAL trained
        model "if this one feature were reset to a calm baseline (all
        others unchanged), how would the predicted probability change?"

        contribution_i = P(actual inputs) - P(inputs with feature_i reset to baseline)

        This is evaluated as a real (non-infinitesimal) difference in model
        output, so unlike a plain derivative it stays informative even when
        the model's prediction is saturated near 0% or 100% -- it directly
        measures each feature's real effect on this prediction rather than
        the local slope at that single point.
        """
        base_proba = base_probability / 100.0
        contributions = []
        for name in FEATURE_NAMES:
            counterfactual = dict(cleaned)
            counterfactual[name] = SAFE_BASELINE[name]
            cf_proba = self._model_proba(counterfactual)
            contribution = base_proba - cf_proba  # >0 means this feature is pushing risk UP
            contributions.append((name, contribution, cleaned[name]))

        # Sort by absolute contribution, descending
        contributions.sort(key=lambda x: -abs(x[1]))
        top = contributions[:4]

        explanations = []
        for name, contribution, value in top:
            label = FEATURE_LABELS.get(name, name)
            direction = "increasing" if contribution > 0 else "reducing"
            explanations.append({
                "feature": name,
                "label": label,
                "value": round(value, 2),
                "direction": direction,
                "influence_score": round(float(contribution) * 100, 2),  # in probability-% points
            })
        return explanations

    def predict(self, features: dict) -> PredictionResult:
        self._ensure_loaded()
        cleaned = self.validate_input(features)

        X = np.array([[cleaned[f] for f in FEATURE_NAMES]])
        X_scaled = self._scaler.transform(X)

        # predict_proba returns [P(no flood), P(flood)] -- genuine model output,
        # not a hard-coded value. This is what makes the % change per input.
        probability = float(self._model.predict_proba(X_scaled)[0, 1]) * 100.0
        probability = round(probability, 2)

        risk_level = classify_risk(probability)
        top_factors = self._explain(cleaned, probability)

        return PredictionResult(
            flood_probability_pct=probability,
            risk_level=risk_level,
            risk_color=RISK_COLORS[risk_level],
            top_factors=top_factors,
            feature_importances=self._importances,
            inputs_used=cleaned,
        )


# Singleton instance shared across API and dashboard
_predictor_instance: Optional[FloodPredictor] = None


def get_predictor() -> FloodPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = FloodPredictor()
    return _predictor_instance
