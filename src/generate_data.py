
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "three_machine_telemetry.csv"

RANDOM_SEED = 42
RECORDS_PER_MACHINE = 10080

ASSETS = ["Chiller-01", "Compressor-02", "Pump-03"]


def generate_asset_data(asset_id, rng):
    timestamps = pd.date_range(
        start="2026-01-01",
        periods=RECORDS_PER_MACHINE,
        freq="min",
        tz="UTC"
    )

    minute = np.arange(RECORDS_PER_MACHINE)
    hour = minute / 60

    # Daily operating schedule:
    # 16 hours RUNNING, 2 hours STANDBY, 6 hours OFF
    running = (minute % 1440 < 960)
    standby = (minute % 1440 >= 960) & (minute % 1440 < 1080)

    status = np.where(
        running,
        "RUNNING",
        np.where(standby, "STANDBY", "OFF")
    )

    ambient = (
        27
        + 4 * np.sin(2 * np.pi * hour / 24)
        + rng.normal(0, 0.5, RECORDS_PER_MACHINE)
    )

    if asset_id == "Chiller-01":
        flow = np.where(running, 450, np.where(standby, 100, 0)).astype(float)
        supply = np.where(running, 7, 22).astype(float)
        delta = np.where(running, 5.5, 1.0)
        power = np.where(running, 150, np.where(standby, 12, 2)).astype(float)
        pressure = np.where(running, 4.2, 0.5).astype(float)

    elif asset_id == "Compressor-02":
        flow = np.where(running, 850, np.where(standby, 150, 0)).astype(float)
        supply = np.where(running, 32, 25).astype(float)
        delta = np.where(running, 12, 1.0)
        power = np.where(running, 110, np.where(standby, 10, 2)).astype(float)
        pressure = np.where(running, 7.5, 0.5).astype(float)

    else:
        flow = np.where(running, 600, np.where(standby, 100, 0)).astype(float)
        supply = np.where(running, 18, 24).astype(float)
        delta = np.where(running, 4, 1.0)
        power = np.where(running, 65, np.where(standby, 7, 1)).astype(float)
        pressure = np.where(running, 3.0, 0.4).astype(float)

    # Add realistic sensor noise
    flow = np.where(
    running | standby,
    np.maximum(flow + rng.normal(0, 3, RECORDS_PER_MACHINE), 0.1),
    0.0
)
    power = np.maximum(power + rng.normal(0, 2, RECORDS_PER_MACHINE), 0.1)
    supply += rng.normal(0, 0.15, RECORDS_PER_MACHINE)
    pressure += rng.normal(0, 0.05, RECORDS_PER_MACHINE)

    # Gradual 25% power increase over 48 hours for the chiller
    if asset_id == "Chiller-01":
        degradation_start = 5000
        degradation_end = 7880

        degradation_mask = (
            running
            & (minute >= degradation_start)
            & (minute < degradation_end)
        )

        ramp = np.linspace(
            1.0,
            1.25,
            degradation_end - degradation_start
        )

        power[degradation_mask] *= ramp[
            minute[degradation_mask] - degradation_start
        ]

        # Keep flow constant during this degradation period
        flow[degradation_mask] = 450.0

    # Calculate return temperature
    return_temp = supply + delta

    # Inject a power spike during RUNNING
    power[2200:2215] *= 1.8

    # Freeze supply temperature for 60 consecutive RUNNING minutes
    supply[3000:3060] = supply[3000]

    # Physical outliers
    power[9000] = -10
    supply[9001] = 999

    # 20-minute sensor dropout (retain rows with missing values)
    power[7000:7020] = np.nan
    flow[7000:7020] = np.nan
    supply[7000:7020] = np.nan
    return_temp[7000:7020] = np.nan
    pressure[7000:7020] = np.nan

    df = pd.DataFrame({
        "timestamp": timestamps,
        "asset_id": asset_id,
        "power_kw": power,
        "flow_rate": flow,
        "supply_temp_c": supply,
        "return_temp_c": return_temp,
        "pressure_bar": pressure,
        "status": status
    })

    return df


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_assets = []

    for asset_id in ASSETS:
        print(f"Generating data for {asset_id}...")
        asset_df = generate_asset_data(asset_id, rng)
        all_assets.append(asset_df)

    final_df = pd.concat(all_assets, ignore_index=True)

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\nDATA GENERATION COMPLETED")
    print("Output file:", OUTPUT_FILE)
    print("Total records:", len(final_df))
    print("\nRecords per asset:")
    print(final_df["asset_id"].value_counts())
    print("\nStatus counts:")
    print(final_df["status"].value_counts())
    print("\nColumns:")
    print(list(final_df.columns))
    print("\nMissing values:")
    print(final_df.isnull().sum())


if __name__ == "__main__":
    main()
