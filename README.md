# 🌊 FloodGuard AI

**AI-based Flash Flood Prediction and Early Warning System for Hilly Regions**
*Hackathon prototype — Python / Scikit-learn / FastAPI / Streamlit*

> ⚠️ **Disclaimer:** This is a **student hackathon prototype**. It runs entirely
> on **synthetic (demo) data** and a demonstration ML model. It is **not**
> validated for real-world use and **must not** be treated as, or represent
> itself as, an official emergency warning system. Real deployment would
> require real sensor/satellite data feeds, historical flood records, and
> review by qualified hydrologists and disaster-management authorities.

---

## What it does

FloodGuard AI takes 10 environmental readings (rainfall at 1h/6h/24h windows,
river/water level and its rate of change, soil moisture, temperature,
humidity, elevation, and slope) and:

1. Predicts a **flood probability (0–100%)** using a trained RandomForest model.
2. Classifies the result into **Low / Moderate / High / Extreme** risk.
3. Explains **which factors** are driving the prediction (model-grounded, not hard-coded).
4. Visualizes risk on an interactive **Folium map** across demo hilly-region locations.
5. Lets you play with a **What-If simulator** (sliders for rainfall, river level, soil moisture) to see risk update live.
6. Serves predictions via a **FastAPI** backend and logs history to **SQLite**.
7. Provides an interactive **Streamlit** dashboard that works standalone (no need to run the API separately).

---

## Project structure

```
floodguard-ai/
├── app/
│   ├── config.py          # feature list, bounds, risk thresholds, paths
│   ├── data_generator.py  # synthetic (DEMO) dataset + demo locations generator
│   ├── train_model.py     # trains + evaluates + saves the RandomForest model
│   ├── predictor.py       # validation, prediction, model-grounded explanation
│   ├── database.py        # SQLite prediction history
│   ├── api.py              # FastAPI backend
│   └── dashboard.py        # Streamlit dashboard (map + what-if + history)
├── data/                    # generated: flood_training_data.csv, demo_locations.csv
├── models/                  # generated: flood_rf_model.joblib, scaler.joblib, metrics.json
├── tests/
│   └── test_pipeline.py    # automated test suite (pytest)
├── setup_project.py         # one-command: generate data + train model
├── requirements.txt
└── README.md
```

---

## 1. Installation

Requires **Python 3.10+**.

```bash
# from the project root
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## 2. Set up the data and model (one command)

This generates the synthetic training dataset + demo locations, trains the
RandomForest model, evaluates it, and saves all artifacts to `models/`.

```bash
python setup_project.py
```

You should see output ending with:
```
Setup complete. You can now run the API and/or dashboard.
```

*(You can re-run this anytime to regenerate data / retrain from scratch.
Individual steps are also available: `python -m app.data_generator` and
`python -m app.train_model`.)*

---

## 3. Run the FastAPI backend

```bash
uvicorn app.api:app --reload --port 8000
```

- API root: http://127.0.0.1:8000/
- Interactive docs (Swagger UI): **http://127.0.0.1:8000/docs**
- Health check: http://127.0.0.1:8000/health

### Example request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "rainfall_1h_mm": 42,
    "rainfall_6h_mm": 150,
    "rainfall_24h_mm": 260,
    "river_level_m": 7.2,
    "river_level_change_m": 1.4,
    "soil_moisture_pct": 78,
    "temperature_c": 23,
    "humidity_pct": 88,
    "elevation_m": 900,
    "slope_deg": 30,
    "location_name": "Demo Valley"
  }'
```

Other endpoints:
- `GET /features` — feature schema + valid input bounds
- `GET /history?limit=50` — recent prediction history
- `DELETE /history` — clear prediction history

---

## 4. Run the Streamlit dashboard

In a **separate terminal** (with the same venv activated):

```bash
streamlit run app/dashboard.py
```

This opens a browser tab (usually http://localhost:8501) with:
- **Live Prediction** — manual input form + gauge + explanation
- **Regional Risk Map** — Folium map of demo hilly-region locations colored by risk
- **What-If Simulator** — sliders for rainfall / river level / soil moisture with live-updating predictions
- **Prediction History** — table + chart of everything logged to SQLite
- **About** — methodology and limitations

> The dashboard talks directly to the same `predictor` module the API uses —
> it works fully standalone, you do **not** need the FastAPI server running
> to use the dashboard. Both share the same SQLite history file
> (`floodguard.db`) if run from the same project folder.

---

## 5. Run the automated tests

```bash
pytest -v tests/test_pipeline.py
```

The suite checks:
- Synthetic dataset generation shape, columns, and label sanity
- Model artifacts exist and load correctly
- Input validation (missing fields, non-numeric values, out-of-range values, NaN)
- The model produces **different** probabilities for different inputs (i.e. it
  is not hard-coded) and low-risk vs. high-risk scenarios are meaningfully separated
- Risk-level classification thresholds
- Explanation output structure
- SQLite history read/write roundtrip

---

## How the model avoids being "hard-coded"

This was a specific design requirement, addressed as follows:

- The synthetic training labels (`flood_occurred`) are generated from a
  **nonlinear hazard score with injected Gaussian noise**, then converted to
  a genuinely random **Bernoulli draw** — not a deterministic threshold. The
  RandomForest never sees the underlying formula, only noisy historical-style
  examples, exactly like a real ML pipeline trained on messy real-world data.
- At inference time, `model.predict_proba()` is called on the **actual scaled
  input vector** — the probability changes continuously as you change any
  input (verified in `tests/test_pipeline.py` and in `train_model.py`'s
  built-in sensitivity check, which refuses to save a degenerate model).
- The "why" explanation is computed by **ablating each feature to a calm
  baseline one at a time and re-querying the real trained model**, so the
  explanation reflects what the model actually learned — not a hand-coded
  rule (e.g. it correctly discovers that higher elevation tends to *reduce*
  risk, without being told that).

---

## Notes on data

- `data/flood_training_data.csv` and `data/demo_locations.csv` are
  **procedurally generated synthetic demo data**, clearly labeled as such in
  code comments and in the dashboard's "About" page and sidebar disclaimer.
- External live data APIs (e.g. real rainfall/river gauges) are **optional
  future extensions** — the core demo is fully self-contained and works
  offline.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: Model artifacts not found` | Run `python setup_project.py` first |
| `ModuleNotFoundError` | Make sure your venv is activated and `pip install -r requirements.txt` completed |
| Streamlit map not rendering | Ensure `streamlit-folium` installed correctly (`pip show streamlit-folium`) |
| Port already in use | Change port: `uvicorn app.api:app --reload --port 8001` |
