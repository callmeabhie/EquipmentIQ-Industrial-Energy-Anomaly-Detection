
from pathlib import Path
import pandas as pd
import numpy as np
import json

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT_DIR / "data" / "processed" / "cleaned_telemetry.csv"
OUTPUT_DIR = ROOT_DIR / "artifacts" / "insights"

ENERGY_COST_PER_KWH = 0.12

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("\nEQUIPMENTIQ INSIGHTS STARTED")

    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(["asset_id", "timestamp"]).copy()

    # Use the first 24 hours of RUNNING data as the baseline period
    first_day = df["timestamp"].min() + pd.Timedelta(hours=24)

    baseline_data = df[
        (df["timestamp"] < first_day) &
        (df["status"] == "RUNNING")
    ].copy()

    baseline_data["flow_bin"] = (
        baseline_data["flow_rate"].round(-1)
    )

    baseline_table = (
        baseline_data.groupby(["asset_id", "flow_bin"])["power_kw"]
        .median()
    )

    asset_baselines = (
        baseline_data.groupby("asset_id")["power_kw"].median()
    )

    # Match each reading to its asset and flow-rate baseline
    df["flow_bin"] = df["flow_rate"].round(-1)

    baseline_map = baseline_table.to_dict()
    asset_baseline_map = asset_baselines.to_dict()

    df["baseline_power_kw"] = [
        baseline_map.get(
            (asset, flow),
            asset_baseline_map.get(asset, 0)
        )
        for asset, flow in zip(df["asset_id"], df["flow_bin"])
    ]

    # Calculate excess energy and cost
    df["excess_power_kw"] = (
        df["power_kw"] - df["baseline_power_kw"]
    ).clip(lower=0)

    df.loc[df["status"] != "RUNNING", "excess_power_kw"] = 0

    df["excess_energy_kwh"] = df["excess_power_kw"] / 60

    df["estimated_cost_loss"] = (
        df["excess_energy_kwh"] * ENERGY_COST_PER_KWH
    )

    # Diagnostics
    df["diagnosis"] = "NORMAL"

    running = df["status"] == "RUNNING"

    df.loc[
        running & (df["power_kw"] > df["baseline_power_kw"] * 1.20),
        "diagnosis"
    ] = "POWER_SURGE"

    if "sensor_frozen" in df.columns:
        frozen = df["sensor_frozen"].astype(str).str.lower().isin(
            ["true", "1"]
        )
        df.loc[running & frozen, "diagnosis"] = "SENSOR_FREEZE"

    # Detect sustained efficiency degradation
    for asset in df["asset_id"].unique():
        mask = df["asset_id"] == asset

        rolling_power = (
            df.loc[mask]
            .set_index("timestamp")["power_kw"]
            .rolling("6h", min_periods=60)
            .median()
        )

        baseline = asset_baseline_map.get(asset, 0)

        degraded_times = rolling_power[
            rolling_power >= baseline * 1.15
        ].index

        df.loc[
            mask & df["timestamp"].isin(degraded_times) & running,
            "diagnosis"
        ] = "GRADUAL_EFFICIENCY_DEGRADATION"

    # Asset-wise summary
    summary = []

    for asset in df["asset_id"].unique():
        asset_df = df[df["asset_id"] == asset]

        summary.append({
            "asset_id": asset,
            "baseline_power_kw": round(
                asset_baseline_map.get(asset, 0), 2
            ),
            "total_excess_energy_kwh": round(
                asset_df["excess_energy_kwh"].sum(), 3
            ),
            "estimated_cost_loss": round(
                asset_df["estimated_cost_loss"].sum(), 2
            ),
            "power_surge_records": int(
                (asset_df["diagnosis"] == "POWER_SURGE").sum()
            ),
            "sensor_freeze_records": int(
                (asset_df["diagnosis"] == "SENSOR_FREEZE").sum()
            ),
            "degradation_records": int(
                (asset_df["diagnosis"] ==
                 "GRADUAL_EFFICIENCY_DEGRADATION").sum()
            )
        })

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        OUTPUT_DIR / "asset_insights.csv",
        index=False
    )

    df.to_csv(
        OUTPUT_DIR / "detailed_diagnostics.csv",
        index=False
    )

    report = {
        "total_excess_energy_kwh": round(
            df["excess_energy_kwh"].sum(), 3
        ),
        "total_estimated_cost_loss": round(
            df["estimated_cost_loss"].sum(), 2
        ),
        "energy_cost_per_kwh": ENERGY_COST_PER_KWH,
        "assets": summary
    }

    with open(OUTPUT_DIR / "insights_report.json", "w") as file:
        json.dump(report, file, indent=4)

    print("\nINSIGHTS COMPLETED")
    print("\nASSET SUMMARY:")
    print(summary_df.to_string(index=False))
    print("\nTotal excess energy:",
          report["total_excess_energy_kwh"], "kWh")
    print("Estimated cost loss: $",
          report["total_estimated_cost_loss"])
    print("\nOutput folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
