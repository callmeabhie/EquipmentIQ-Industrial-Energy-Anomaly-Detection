
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "cleaned_telemetry.csv"
)

ARTIFACT_DIR = ROOT_DIR / "artifacts"

MODEL_FILE = (
    ARTIFACT_DIR
    / "equipmentiq_three_asset_model.joblib"
)


# Features used for anomaly detection
FEATURE_COLUMNS = [
    "power_kw",
    "flow_rate",
    "supply_temp_c",
    "return_temp_c",
    "pressure_bar",
    "delta_t",
    "specific_energy",
    "rolling_power_mean_1h",
    "load_ratio",
]


def main():

    print("\nEQUIPMENTIQ MODEL TRAINING STARTED\n")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    print("Dataset shape:", df.shape)

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    models_by_asset = {}
    score_ranges = {}
    training_counts = {}

    assets = sorted(df["asset_id"].unique())

    for asset_id in assets:

        print(f"\nTraining model for {asset_id}...")

        # Use RUNNING records only
        asset_data = df[
            (df["asset_id"] == asset_id)
            & (df["status"] == "RUNNING")
        ].copy()

        X = asset_data[FEATURE_COLUMNS]

        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        X = X.fillna(X.median())

        if len(X) < 100:
            raise ValueError(
                f"Not enough RUNNING records for {asset_id}"
            )

        # StandardScaler + Isolation Forest
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                IsolationForest(
                    n_estimators=200,
                    contamination=0.02,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ])

        pipeline.fit(X)

        # Lower decision scores indicate more anomalous records
        raw_scores = pipeline.decision_function(X)

        score_min = float(np.min(raw_scores))
        score_max = float(np.max(raw_scores))

        if score_max <= score_min:
            raise ValueError(
                f"Invalid score range for {asset_id}"
            )

        models_by_asset[asset_id] = pipeline

        score_ranges[asset_id] = {
            "min": score_min,
            "max": score_max
        }

        training_counts[asset_id] = len(X)

        print("Training records:", len(X))
        print("Score minimum:", round(score_min, 5))
        print("Score maximum:", round(score_max, 5))

    # Save all three models in one separate artifact
    artifact = {
        "models_by_asset": models_by_asset,
        "feature_columns": FEATURE_COLUMNS,
        "score_ranges": score_ranges,
        "training_counts": training_counts,
        "score_thresholds": {
            "warning": 0.50,
            "critical": 0.75
        }
    }

    joblib.dump(
        artifact,
        MODEL_FILE
    )

    print("\nMODEL TRAINING COMPLETED")

    print("Saved model:", MODEL_FILE)

    print("\nTraining records by asset:")

    for asset_id, count in training_counts.items():
        print(f"{asset_id}: {count}")

    print("\nAll three machine-learning models saved successfully.")


if __name__ == "__main__":
    main()
