"""
FloodGuard AI - Model training pipeline
==========================================
Trains a RandomForestClassifier on the synthetic (DEMO) dataset to predict
flood probability from the 10 environmental features, then saves:
  - the fitted model (joblib)
  - the fitted StandardScaler (joblib)
  - feature importances (json)
  - evaluation metrics (json)

Run:
    python -m app.train_model
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.config import (FEATURE_NAMES, TRAINING_DATA_PATH, MODEL_PATH,
                         SCALER_PATH, FEATURE_IMPORTANCE_PATH, METRICS_PATH,
                         RANDOM_SEED)
from app.data_generator import generate_training_dataset


def load_or_generate_dataset() -> pd.DataFrame:
    if TRAINING_DATA_PATH.exists():
        df = pd.read_csv(TRAINING_DATA_PATH)
        # sanity check the file actually has the columns we expect
        if all(col in df.columns for col in FEATURE_NAMES + ["flood_occurred"]):
            return df
    df = generate_training_dataset()
    df.to_csv(TRAINING_DATA_PATH, index=False)
    return df


def train():
    print("=" * 60)
    print("FloodGuard AI - Model Training (using SYNTHETIC demo data)")
    print("=" * 60)

    df = load_or_generate_dataset()
    print(f"Loaded dataset: {len(df)} rows")

    X = df[FEATURE_NAMES].values
    y = df["flood_occurred"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Scale features (helps some downstream uses; RF itself is scale-invariant
    # but we keep a scaler so the API/dashboard have a consistent, reusable
    # preprocessing step if the model type changes later).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=9,
        min_samples_split=8,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    start = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start
    print(f"Model trained in {train_time:.2f}s")

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "trained_on": "synthetic_demo_data",
    }

    print("\nEvaluation on held-out test set:")
    for k, v in metrics.items():
        if k not in ("confusion_matrix",):
            print(f"  {k}: {v}")
    print(f"  confusion_matrix: {metrics['confusion_matrix']}")

    # Feature importances
    importances = model.feature_importances_
    importance_dict = {
        name: round(float(imp), 4)
        for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
    }
    print("\nFeature importances:")
    for name, imp in importance_dict.items():
        print(f"  {name}: {imp}")

    # Sanity check: verify predictions genuinely respond to feature changes
    # (guards against an accidental constant/degenerate model).
    sample = X_test_scaled[0:1].copy()
    base_pred = model.predict_proba(sample)[0, 1]
    perturbed = scaler.transform(
        (scaler.inverse_transform(sample) + np.array([[40, 80, 100, 3, 1, 20, 0, 0, 0, 0]]))
    )
    perturbed_pred = model.predict_proba(perturbed)[0, 1]
    sensitivity_ok = abs(perturbed_pred - base_pred) > 1e-4
    print(f"\nSensitivity check (prediction must change with inputs): "
          f"base={base_pred:.4f} perturbed={perturbed_pred:.4f} -> "
          f"{'PASS' if sensitivity_ok else 'FAIL'}")
    if not sensitivity_ok:
        raise RuntimeError(
            "Model failed sensitivity check: prediction did not change when "
            "inputs were perturbed. Refusing to save a degenerate model."
        )

    # Save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(importance_dict, f, indent=2)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")
    print(f"Saved feature importances to {FEATURE_IMPORTANCE_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    train()
