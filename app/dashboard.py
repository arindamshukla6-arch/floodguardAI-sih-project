"""
FloodGuard AI - Streamlit Dashboard
=====================================
Run:
    streamlit run app/dashboard.py

This dashboard talks DIRECTLY to the predictor module (no need for the
FastAPI server to be running) so it works standalone during a demo. It also
optionally logs every prediction to the same SQLite history the API uses.
"""

import sys
from pathlib import Path

# Allow running via `streamlit run app/dashboard.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from app.config import (FEATURE_NAMES, FEATURE_LABELS, FEATURE_BOUNDS,
                         RISK_COLORS, LOCATIONS_PATH)
from app.database import get_recent_predictions, save_prediction, init_db
from app.predictor import get_predictor, ValidationError

st.set_page_config(
    page_title="FloodGuard AI",
    page_icon="🌊",
    layout="wide",
)

init_db()
predictor = get_predictor()

# ---------------------------------------------------------------------------
# Sidebar - navigation + disclaimer
# ---------------------------------------------------------------------------
st.sidebar.title("🌊 FloodGuard AI")
st.sidebar.caption("Flash Flood Prediction Prototype - Hilly Regions")
page = st.sidebar.radio(
    "Navigate",
    ["Live Prediction", "Regional Risk Map", "What-If Simulator", "Prediction History", "About"],
)
st.sidebar.markdown("---")
st.sidebar.warning(
    "⚠️ **Hackathon prototype.** Uses synthetic demo data. "
    "This is **not** an official emergency warning system."
)


def default_features():
    return {
        "rainfall_1h_mm": 10.0,
        "rainfall_6h_mm": 35.0,
        "rainfall_24h_mm": 70.0,
        "river_level_m": 3.5,
        "river_level_change_m": 0.3,
        "soil_moisture_pct": 45.0,
        "temperature_c": 24.0,
        "humidity_pct": 65.0,
        "elevation_m": 1200.0,
        "slope_deg": 15.0,
    }


def feature_input_form(prefix: str, values: dict) -> dict:
    """Render number inputs for all 10 features, split across columns."""
    result = {}
    col1, col2 = st.columns(2)
    cols = [col1, col2]
    for i, name in enumerate(FEATURE_NAMES):
        low, high = FEATURE_BOUNDS[name]
        col = cols[i % 2]
        with col:
            result[name] = st.number_input(
                FEATURE_LABELS[name],
                min_value=float(low),
                max_value=float(high),
                value=float(values[name]),
                key=f"{prefix}_{name}",
            )
    return result


def risk_gauge(probability: float, risk_level: str):
    color = RISK_COLORS[risk_level]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        number={"suffix": "%"},
        title={"text": f"Flood Probability - {risk_level} Risk"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 25], "color": "#eafaf1"},
                {"range": [25, 50], "color": "#fef9e7"},
                {"range": [50, 75], "color": "#fdf0e2"},
                {"range": [75, 100], "color": "#fdedec"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": probability,
            },
        },
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def factors_chart(top_factors: list):
    labels = [f["label"] for f in top_factors]
    scores = [f["influence_score"] for f in top_factors]
    colors = ["#e74c3c" if s > 0 else "#2ecc71" for s in scores]
    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{s:+.2f} pp" for s in scores],
        textposition="auto",
    ))
    fig.update_layout(
        title="Top Contributing Factors (probability points, vs. calm baseline)",
        xaxis_title="Effect on flood probability (pp)",
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def run_prediction_and_render(features: dict, location_name=None, log_history=True):
    try:
        result = predictor.predict(features)
    except ValidationError as e:
        st.error(f"Input validation error: {e}")
        return None

    if log_history:
        try:
            save_prediction(
                features=result.inputs_used,
                probability=result.flood_probability_pct,
                risk_level=result.risk_level,
                top_factors=result.top_factors,
                location_name=location_name,
            )
        except Exception as e:
            st.warning(f"Could not log prediction to history: {e}")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(risk_gauge(result.flood_probability_pct, result.risk_level), use_container_width=True)
    with c2:
        st.plotly_chart(factors_chart(result.top_factors), use_container_width=True)

    badge_color = RISK_COLORS[result.risk_level]
    st.markdown(
        f"<div style='padding:12px;border-radius:8px;background:{badge_color}22;"
        f"border:2px solid {badge_color};font-size:18px;'>"
        f"<b>Risk Level: {result.risk_level}</b> — "
        f"{result.flood_probability_pct}% estimated flood probability</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Why this prediction? (model explanation)"):
        for f in result.top_factors:
            arrow = "🔺" if f["direction"] == "increasing" else "🔻"
            st.write(f"{arrow} **{f['label']}** = {f['value']} → {f['direction']} risk "
                     f"by ~{abs(f['influence_score'])} percentage points vs. a calm baseline")

    return result


# ---------------------------------------------------------------------------
# Page: Live Prediction
# ---------------------------------------------------------------------------
if page == "Live Prediction":
    st.title("🌊 Live Flash Flood Risk Prediction")
    st.caption("Enter current environmental readings (or use the demo defaults) to get a live risk estimate.")

    with st.form("live_prediction_form"):
        location_name = st.text_input("Location label (optional)", value="Demo Site")
        feats = feature_input_form("live", default_features())
        submitted = st.form_submit_button("Predict Flood Risk", type="primary")

    if submitted:
        run_prediction_and_render(feats, location_name=location_name)

# ---------------------------------------------------------------------------
# Page: Regional Risk Map
# ---------------------------------------------------------------------------
elif page == "Regional Risk Map":
    st.title("🗺️ Regional Flood Risk Map")
    st.caption("Demo hilly-region locations with simulated current readings, colored by predicted risk.")

    if not LOCATIONS_PATH.exists():
        st.error("Demo locations file not found. Run `python -m app.data_generator` first.")
    else:
        locs = pd.read_csv(LOCATIONS_PATH)

        rows = []
        for _, row in locs.iterrows():
            feats = {name: row[name] for name in FEATURE_NAMES}
            try:
                result = predictor.predict(feats)
            except ValidationError:
                continue
            rows.append({
                "name": row["name"],
                "lat": row["lat"],
                "lon": row["lon"],
                "probability": result.flood_probability_pct,
                "risk": result.risk_level,
            })

        risk_df = pd.DataFrame(rows)

        m = folium.Map(location=[22.0, 82.0], zoom_start=5, tiles="CartoDB positron")
        for _, r in risk_df.iterrows():
            color = RISK_COLORS[r["risk"]]
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=14,
                popup=folium.Popup(
                    f"<b>{r['name']}</b><br>Probability: {r['probability']}%<br>Risk: {r['risk']}",
                    max_width=250,
                ),
                tooltip=f"{r['name']}: {r['risk']} ({r['probability']}%)",
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                weight=2,
            ).add_to(m)

        st_folium(m, width=None, height=520)

        st.subheader("Location Risk Summary")
        st.dataframe(
            risk_df[["name", "probability", "risk"]].rename(
                columns={"name": "Location", "probability": "Flood Probability (%)", "risk": "Risk Level"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Note: readings for these demo locations are simulated (synthetic), regenerated each app run "
                   "unless the demo_locations.csv file is fixed. Not real sensor data.")

# ---------------------------------------------------------------------------
# Page: What-If Simulator
# ---------------------------------------------------------------------------
elif page == "What-If Simulator":
    st.title("🎛️ What-If Simulator")
    st.caption("Adjust rainfall, river level, and soil moisture with sliders to see how predicted risk changes in real time. Other factors stay fixed.")

    base = default_features()

    st.subheader("Fixed context (terrain & weather)")
    ctx_col1, ctx_col2, ctx_col3, ctx_col4 = st.columns(4)
    with ctx_col1:
        base["temperature_c"] = st.number_input("Temperature (C)", 
            float(FEATURE_BOUNDS["temperature_c"][0]), float(FEATURE_BOUNDS["temperature_c"][1]), base["temperature_c"])
    with ctx_col2:
        base["humidity_pct"] = st.number_input("Humidity (%)", 
            float(FEATURE_BOUNDS["humidity_pct"][0]), float(FEATURE_BOUNDS["humidity_pct"][1]), base["humidity_pct"])
    with ctx_col3:
        base["elevation_m"] = st.number_input("Elevation (m)", 
            float(FEATURE_BOUNDS["elevation_m"][0]), float(FEATURE_BOUNDS["elevation_m"][1]), base["elevation_m"])
    with ctx_col4:
        base["slope_deg"] = st.number_input("Slope (deg)", 
            float(FEATURE_BOUNDS["slope_deg"][0]), float(FEATURE_BOUNDS["slope_deg"][1]), base["slope_deg"])

    st.subheader("Adjustable simulation variables")
    sc1, sc2 = st.columns(2)
    with sc1:
        base["rainfall_1h_mm"] = st.slider("Rainfall - last 1h (mm)", 0.0, 200.0, base["rainfall_1h_mm"])
        base["rainfall_6h_mm"] = st.slider("Rainfall - last 6h (mm)", 0.0, 550.0, base["rainfall_6h_mm"])
        base["rainfall_24h_mm"] = st.slider("Rainfall - last 24h (mm)", 0.0, 1100.0, base["rainfall_24h_mm"])
    with sc2:
        base["river_level_m"] = st.slider("River / water level (m)", 0.0, 25.0, base["river_level_m"])
        base["river_level_change_m"] = st.slider("River level change rate (m/h)", -3.0, 4.5, base["river_level_change_m"])
        base["soil_moisture_pct"] = st.slider("Soil moisture (%)", 0.0, 100.0, base["soil_moisture_pct"])

    st.markdown("---")
    run_prediction_and_render(base, location_name="What-If Simulation", log_history=False)

    st.info("Predictions here are computed live by the trained model as you move the sliders — "
            "they are not looked up from a table or fixed formula.")

# ---------------------------------------------------------------------------
# Page: Prediction History
# ---------------------------------------------------------------------------
elif page == "Prediction History":
    st.title("📜 Prediction History")
    st.caption("Most recent predictions logged to the local SQLite database (floodguard.db).")

    rows = get_recent_predictions(limit=100)
    if not rows:
        st.info("No predictions logged yet. Make a prediction on the 'Live Prediction' page or the Regional Map first.")
    else:
        df = pd.DataFrame(rows)
        display_cols = ["timestamp", "location_name", "flood_probability", "risk_level"] + FEATURE_NAMES
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        st.subheader("Risk level distribution")
        counts = df["risk_level"].value_counts()
        fig = go.Figure(go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=[RISK_COLORS.get(k, "#999") for k in counts.index],
        ))
        fig.update_layout(height=300, xaxis_title="Risk Level", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
elif page == "About":
    st.title("ℹ️ About FloodGuard AI")
    st.markdown("""
**FloodGuard AI** is a student hackathon prototype demonstrating how machine
learning could support flash-flood early warning in hilly regions using
multi-source environmental data.

**How it works**
1. A RandomForest classifier is trained on synthetic (demo) data that mimics
   plausible flood-triggering conditions, including realistic noise so the
   model must genuinely learn patterns rather than memorize a formula.
2. Given 10 live input features (rainfall at multiple windows, river level and
   its rate of change, soil moisture, temperature, humidity, elevation, and
   slope), the model outputs a flood probability (0-100%).
3. The probability is mapped to a Low / Moderate / High / Extreme risk category.
4. An explanation module shows which inputs are pushing the prediction up or
   down, computed directly from the trained model's own behavior.

**Important limitations**
- All training and demo location data is **synthetic**, generated
  procedurally for this prototype. It is **not** real sensor, satellite, or
  government hydrological data.
- This prototype is **not** validated for operational use and **must not**
  be treated as an official emergency warning system.
- Real deployment would require calibration against historical flood
  records, real-time data feeds (IMD, CWC gauges, satellite rainfall
  estimates), and review by domain hydrologists and disaster-management
  authorities.

**Tech stack:** Python, Pandas, NumPy, Scikit-learn, FastAPI, Streamlit,
Plotly, Folium, SQLite, Joblib.
""")
