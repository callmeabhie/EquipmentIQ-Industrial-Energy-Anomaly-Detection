
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT_DIR / "data" / "processed" / "cleaned_telemetry.csv"
MODEL_FILE = ROOT_DIR / "artifacts" / "equipmentiq_three_asset_model.joblib"
OUTPUT_DIR = ROOT_DIR / "artifacts" / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("\nEQUIPMENTIQ VISUALIZATION STARTED")

    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    artifact = joblib.load(MODEL_FILE)

    models_by_asset = artifact["models_by_asset"]
    feature_columns = artifact["feature_columns"]
    score_ranges = artifact["score_ranges"]

    df["anomaly_score"] = 0.0

    for asset_id, model in models_by_asset.items():
        mask = df["asset_id"] == asset_id
        asset_df = df.loc[mask]

        features = asset_df[feature_columns].fillna(0)
        raw_scores = model.decision_function(features)

        score_min = score_ranges[asset_id]["min"]
        score_max = score_ranges[asset_id]["max"]

        scores = (score_max - raw_scores) / (score_max - score_min)
        df.loc[mask, "anomaly_score"] = scores.clip(0, 1)

    df["severity"] = "NORMAL"
    df.loc[df["anomaly_score"] >= 0.50, "severity"] = "WARNING"
    df.loc[df["anomaly_score"] >= 0.75, "severity"] = "CRITICAL"

    # Chart 1
    plt.figure(figsize=(14, 6))

    for asset in df["asset_id"].unique():
        asset_df = df[df["asset_id"] == asset]
        plt.plot(
            asset_df["timestamp"],
            asset_df["power_kw"].rolling(60).mean(),
            label=asset
        )

    plt.title("Industrial Equipment Power Consumption Trends")
    plt.xlabel("Timestamp")
    plt.ylabel("Power Consumption (kW)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trends.png", dpi=150)
    plt.close()

    # Chart 2
    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=df[df["status"] == "RUNNING"],
        x="flow_rate",
        y="specific_energy",
        hue="asset_id",
        alpha=0.5
    )

    plt.title("Flow Rate vs Specific Energy")
    plt.xlabel("Flow Rate")
    plt.ylabel("Specific Energy")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "efficiency_correlation.png", dpi=150)
    plt.close()

    # Chart 3
    plt.figure(figsize=(10, 6))

    sns.countplot(
        data=df,
        x="asset_id",
        hue="severity",
        order=sorted(df["asset_id"].unique())
    )

    plt.title("Anomalies Detected by Equipment")
    plt.xlabel("Equipment")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "anomalies_detected.png", dpi=150)
    plt.close()

    print("\nVISUALIZATION COMPLETED")
    print("Saved charts in:", OUTPUT_DIR)

    for file in OUTPUT_DIR.glob("*.png"):
        print("Created:", file.name)


if __name__ == "__main__":
    main()
