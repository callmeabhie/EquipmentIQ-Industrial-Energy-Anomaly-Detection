
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT_DIR / "artifacts" / "equipmentiq_three_asset_model.joblib"
)

DATA_PATH = (
    ROOT_DIR / "data" / "processed" / "cleaned_telemetry.csv"
)

# Load the three trained machine models
artifact = joblib.load(MODEL_PATH)

models_by_asset = artifact["models_by_asset"]
feature_columns = artifact["feature_columns"]
score_ranges = artifact["score_ranges"]
score_thresholds = artifact["score_thresholds"]

WARNING_THRESHOLD = score_thresholds["warning"]
CRITICAL_THRESHOLD = score_thresholds["critical"]

ASSETS = [
    "Chiller-01",
    "Compressor-02",
    "Pump-03"
]


# Load training-data reference values
reference_df = pd.read_csv(DATA_PATH)

baseline_power_by_asset = {}
specific_energy_median_by_asset = {}

for asset_id in ASSETS:

    asset_df = reference_df[
        reference_df["asset_id"] == asset_id
    ]

    running_df = asset_df[
        asset_df["status"] == "RUNNING"
    ]

    baseline_power_by_asset[asset_id] = float(
        running_df["power_kw"].median()
    )

    specific_energy_median_by_asset[asset_id] = float(
        asset_df["specific_energy"].median()
    )


def diagnose_readings(readings: list[dict]) -> list[dict]:
    """
    Diagnose a sequence of readings using the
    corresponding machine-specific Isolation Forest model.
    """

    if not readings:
        return []

    df = pd.DataFrame(readings).copy()

    df["_original_order"] = range(len(df))

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    if df["timestamp"].isna().any():
        raise ValueError("Invalid timestamp in readings.")

    if not df["asset_id"].isin(ASSETS).all():
        raise ValueError("Unknown asset_id supplied.")

    df = df.sort_values(
        ["asset_id", "timestamp"]
    ).copy()

    # Feature 1: Temperature difference
    df["delta_t"] = (
        df["return_temp_c"] - df["supply_temp_c"]
    )

    # Feature 2: Specific energy
    denominator = df["flow_rate"] * df["delta_t"]

    df["specific_energy"] = np.where(
        denominator > 0,
        df["power_kw"] / denominator,
        np.nan
    )

    # Feature 3: Rolling average power (60 readings)
    df["rolling_power_mean_1h"] = (
        df.groupby("asset_id")["power_kw"]
        .transform(
            lambda values: values.rolling(
                window=60,
                min_periods=1
            ).mean()
        )
    )

    # Feature 4: Load ratio using training-data baseline
    df["load_ratio"] = 0.0

    for asset_id in ASSETS:

        mask = df["asset_id"] == asset_id

        baseline = baseline_power_by_asset[asset_id]

        if baseline > 0:
            df.loc[mask, "load_ratio"] = (
                df.loc[mask, "power_kw"] / baseline
            )

        # Match training-data missing-value handling
        df.loc[mask, "specific_energy"] = (
            df.loc[mask, "specific_energy"].fillna(
                specific_energy_median_by_asset[asset_id]
            )
        )

    # Prepare the exact nine model features
    features = df[feature_columns].replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    df["anomaly_score"] = 0.0

    # Score each machine with its own model
    for asset_id, model in models_by_asset.items():

        mask = df["asset_id"] == asset_id

        if not mask.any():
            continue

        asset_features = features.loc[mask]

        raw_scores = model.decision_function(
            asset_features
        )

        score_min = score_ranges[asset_id]["min"]
        score_max = score_ranges[asset_id]["max"]

        if score_max == score_min:
            raise ValueError(
                f"Invalid score range for {asset_id}"
            )

        normalized_scores = (
            (score_max - raw_scores)
            / (score_max - score_min)
        )

        df.loc[mask, "anomaly_score"] = np.clip(
            normalized_scores,
            0,
            1
        )

    results = []

    for _, row in df.iterrows():

        score = float(row["anomaly_score"])

        if row["status"] == "OFF":

            severity = "NORMAL"
            diagnosis = "Equipment is switched off."

        elif score >= CRITICAL_THRESHOLD:

            severity = "CRITICAL"
            diagnosis = (
                "Critical abnormal equipment behaviour detected."
            )

        elif score >= WARNING_THRESHOLD:

            severity = "WARNING"
            diagnosis = (
                "Potential equipment anomaly detected."
            )

        else:

            severity = "NORMAL"
            diagnosis = (
                "Equipment is operating within expected limits."
            )

        results.append({
            "_original_order": int(row["_original_order"]),
            "timestamp": row["timestamp"].to_pydatetime(),
            "asset_id": row["asset_id"],
            "anomaly_score": score,
            "severity": severity,
            "diagnosis": diagnosis
        })

    # Restore the original input order
    results.sort(
        key=lambda result: result["_original_order"]
    )

    for result in results:
        result.pop("_original_order")

    return results


def diagnose_reading(reading: dict) -> dict:
    """
    Diagnose one sensor reading.
    """

    return diagnose_readings([reading])[0]
