"""
FloodGuard AI - One-command setup
====================================
Generates the synthetic demo dataset (if missing) and trains the model.

Run:
    python setup_project.py
"""

from app.data_generator import main as generate_data
from app.train_model import train as train_model


def main():
    print("STEP 1/2: Generating synthetic demo dataset + demo locations...")
    generate_data()
    print("\nSTEP 2/2: Training RandomForest flood-risk model...")
    train_model()
    print("\nSetup complete. You can now run the API and/or dashboard.")
    print("  API:       uvicorn app.api:app --reload --port 8000")
    print("  Dashboard: streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
