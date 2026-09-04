"""
FloodGuard AI - FastAPI backend
==================================
Run:
    uvicorn app.api:app --reload --port 8000

Docs:
    http://127.0.0.1:8000/docs
"""

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import FEATURE_NAMES, FEATURE_BOUNDS
from app.predictor import get_predictor, ValidationError
from app.database import init_db, save_prediction, get_recent_predictions, clear_history

app = FastAPI(
    title="FloodGuard AI API",
    description=(
        "Prototype flash-flood risk prediction API for hilly regions. "
        "Built for a student hackathon demo. Uses a RandomForest model "
        "trained on SYNTHETIC demo data. "
        "This system is a PROTOTYPE and does NOT constitute an official "
        "emergency warning service."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class FloodPredictionRequest(BaseModel):
    rainfall_1h_mm: float = Field(..., description="Rainfall in the last 1 hour (mm)")
    rainfall_6h_mm: float = Field(..., description="Rainfall in the last 6 hours (mm)")
    rainfall_24h_mm: float = Field(..., description="Rainfall in the last 24 hours (mm)")
    river_level_m: float = Field(..., description="Current river/water level (m)")
    river_level_change_m: float = Field(..., description="River level change rate (m/h)")
    soil_moisture_pct: float = Field(..., description="Soil moisture (%)")
    temperature_c: float = Field(..., description="Temperature (C)")
    humidity_pct: float = Field(..., description="Relative humidity (%)")
    elevation_m: float = Field(..., description="Elevation (m)")
    slope_deg: float = Field(..., description="Terrain slope (degrees)")
    location_name: Optional[str] = Field(None, description="Optional location label for history logging")

    class Config:
        json_schema_extra = {
            "example": {
                "rainfall_1h_mm": 42.0,
                "rainfall_6h_mm": 150.0,
                "rainfall_24h_mm": 260.0,
                "river_level_m": 7.2,
                "river_level_change_m": 1.4,
                "soil_moisture_pct": 78.0,
                "temperature_c": 23.0,
                "humidity_pct": 88.0,
                "elevation_m": 900.0,
                "slope_deg": 30.0,
                "location_name": "Demo Valley"
            }
        }


class TopFactor(BaseModel):
    feature: str
    label: str
    value: float
    direction: str
    influence_score: float


class FloodPredictionResponse(BaseModel):
    flood_probability_pct: float
    risk_level: str
    risk_color: str
    top_factors: list[TopFactor]
    feature_importances: dict
    disclaimer: str = (
        "This is a student hackathon PROTOTYPE using synthetic demo data. "
        "It is NOT an official emergency warning system."
    )


class HistoryItem(BaseModel):
    id: int
    timestamp: str
    location_name: Optional[str] = None
    flood_probability: float
    risk_level: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "FloodGuard AI API",
        "status": "running",
        "note": "Hackathon prototype. Not an official warning system.",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    predictor = get_predictor()
    try:
        predictor._ensure_loaded()
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False
    return {"status": "ok", "model_loaded": model_loaded}


@app.get("/features")
def list_features():
    """Return the feature schema (names + valid bounds) the model expects."""
    return {
        "features": FEATURE_NAMES,
        "bounds": FEATURE_BOUNDS,
    }


@app.post("/predict", response_model=FloodPredictionResponse)
def predict(payload: FloodPredictionRequest):
    predictor = get_predictor()
    features = payload.dict(exclude={"location_name"})

    try:
        result = predictor.predict(features)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        save_prediction(
            features=result.inputs_used,
            probability=result.flood_probability_pct,
            risk_level=result.risk_level,
            top_factors=result.top_factors,
            location_name=payload.location_name,
        )
    except Exception:
        # History logging must never break the prediction response itself.
        pass

    return FloodPredictionResponse(
        flood_probability_pct=result.flood_probability_pct,
        risk_level=result.risk_level,
        risk_color=result.risk_color,
        top_factors=result.top_factors,
        feature_importances=result.feature_importances,
    )


@app.get("/history", response_model=list[HistoryItem])
def history(limit: int = 50):
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500")
    rows = get_recent_predictions(limit=limit)
    return [
        HistoryItem(
            id=r["id"],
            timestamp=r["timestamp"],
            location_name=r.get("location_name"),
            flood_probability=r["flood_probability"],
            risk_level=r["risk_level"],
        )
        for r in rows
    ]


@app.delete("/history")
def delete_history():
    clear_history()
    return {"status": "cleared"}
