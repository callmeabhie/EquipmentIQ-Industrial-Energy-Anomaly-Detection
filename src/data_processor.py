
from pathlib import Path

import numpy as np
import pandas as pd


# Project folders
ROOT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT_DIR / "data" / "raw" / "three_machine_telemetry.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "cleaned_telemetry.csv"


def load_data():
    """Load the raw three-machine telemetry dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    df = df.dropna(subset=["timestamp"])

    df = df.sort_values(
        ["asset_id", "timestamp"]
    ).reset_index(drop=True)

    print("Raw dataset loaded:", df.shape)

    return df


def clean_physical_values(df):
    """Replace physically invalid sensor values with NaN."""

    sensor_bounds = {
        "power_kw": (0, 1000),
        "flow_rate": (0, 5000),
        "supply_temp_c": (-20, 100),
        "return_temp_c": (-20, 150),
        "pressure_bar": (0, 30),
    }

    for column, (minimum, maximum) in sensor_bounds.items():
        invalid = (
            (df[column] < minimum)
            | (df[column] > maximum)
        )

        count = int(invalid.sum())

        df.loc[invalid, column] = np.nan

        print(f"{column}: {count} invalid values corrected")

    return df


def detect_frozen_sensors(df):
    """Detect a supply temperature frozen for at least 45 minutes."""

    df["sensor_frozen"] = False

    for asset_id in df["asset_id"].unique():

        asset_mask = df["asset_id"] == asset_id

        running_mask = (
            asset_mask
            & (df["status"] == "RUNNING")
        )

        running_data = df.loc[running_mask].copy()

        rolling_std = (
            running_data["supply_temp_c"]
            .rolling(window=45, min_periods=45)
            .std()
        )

        frozen_indices = rolling_std[
            rolling_std < 1e-9
        ].index

        df.loc[frozen_indices, "sensor_frozen"] = True

        # Treat frozen sensor measurements as invalid
        df.loc[frozen_indices, "supply_temp_c"] = np.nan

    print(
        "Frozen sensor readings detected:",
        int(df["sensor_frozen"].sum())
    )

    return df


def handle_missing_values(df):
    """Interpolate short gaps and fill remaining missing values."""

    numeric_columns = [
        "power_kw",
        "flow_rate",
        "supply_temp_c",
        "return_temp_c",
        "pressure_bar",
    ]

    for asset_id in df["asset_id"].unique():

        asset_mask = df["asset_id"] == asset_id

        asset_data = df.loc[asset_mask].copy()

        asset_data = asset_data.set_index("timestamp")

        for column in numeric_columns:

            asset_data[column] = (
                asset_data[column]
                .interpolate(
                    method="time",
                    limit=5,
                    limit_direction="both"
                )
            )

            median_value = asset_data[column].median()

            asset_data[column] = (
                asset_data[column].fillna(median_value)
            )

        df.loc[asset_mask, numeric_columns] = (
            asset_data[numeric_columns].to_numpy()
        )

    print("Missing sensor values handled.")

    return df


def create_features(df):
    """Create energy and equipment performance features."""

    df = df.sort_values(
        ["asset_id", "timestamp"]
    ).copy()

    # Temperature difference
    df["delta_t"] = (
        df["return_temp_c"] - df["supply_temp_c"]
    )

    # Specific energy proxy
    denominator = df["flow_rate"] * df["delta_t"]

    df["specific_energy"] = np.where(
        denominator > 0,
        df["power_kw"] / denominator,
        np.nan
    )

    df["specific_energy"] = (
        df["specific_energy"]
        .replace([np.inf, -np.inf], np.nan)
    )

    # One-hour rolling average power
    df["rolling_power_mean_1h"] = (
        df.groupby("asset_id")["power_kw"]
        .transform(
            lambda values: values.rolling(
                window=60,
                min_periods=1
            ).mean()
        )
    )

    # Load ratio based on each asset's RUNNING baseline
    df["load_ratio"] = np.nan

    for asset_id in df["asset_id"].unique():

        asset_mask = df["asset_id"] == asset_id

        running_mask = (
            asset_mask
            & (df["status"] == "RUNNING")
        )

        baseline_power = df.loc[
            running_mask, "power_kw"
        ].median()

        if pd.notna(baseline_power) and baseline_power > 0:

            df.loc[asset_mask, "load_ratio"] = (
                df.loc[asset_mask, "power_kw"]
                / baseline_power
            )

    df["specific_energy"] = (
        df.groupby("asset_id")["specific_energy"]
        .transform(lambda values: values.fillna(values.median()))
    )

    df["load_ratio"] = df["load_ratio"].fillna(0)

    print("Feature engineering completed.")

    return df


def main():

    print("\nEQUIPMENTIQ DATA PROCESSING STARTED\n")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_data()

    df = clean_physical_values(df)

    df = detect_frozen_sensors(df)

    df = handle_missing_values(df)

    df = create_features(df)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nDATA PROCESSING COMPLETED")

    print("Cleaned file:", OUTPUT_FILE)

    print("Final shape:", df.shape)

    print("\nFinal columns:")
    print(list(df.columns))

    print("\nRemaining missing values:")
    print(df.isnull().sum())

    print("\nAsset record counts:")
    print(df["asset_id"].value_counts())


if __name__ == "__main__":
    main()
