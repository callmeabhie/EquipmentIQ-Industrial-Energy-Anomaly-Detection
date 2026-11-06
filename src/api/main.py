
from fastapi import FastAPI, HTTPException
import pandas as pd

from src.models.schemas import (
    SensorReading,
    DiagnosisResponse,
    BatchAuditRequest,
    BatchAuditResponse
)

from src.services.diagnosis import (
    diagnose_reading,
    diagnose_readings,
    baseline_power_by_asset
)


app = FastAPI(
    title="EquipmentIQ API",
    description="Industrial Energy Anomaly Detection and Diagnosis Service",
    version="2.0.0"
)

ELECTRICITY_COST_PER_KWH = 0.12


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "EquipmentIQ",
        "models_loaded": list(baseline_power_by_asset.keys())
    }


@app.post(
    "/api/v1/diagnose",
    response_model=DiagnosisResponse
)
def diagnose(data: SensorReading):

    try:
        reading = data.model_dump()
        result = diagnose_reading(reading)
        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Diagnosis failed: {str(error)}"
        )


@app.post(
    "/api/v1/batch-audit",
    response_model=BatchAuditResponse
)
def batch_audit(data: BatchAuditRequest):

    try:
        readings = [
            reading.model_dump()
            for reading in data.readings
        ]

        results = diagnose_readings(readings)

        batch_df = pd.DataFrame(readings)

        batch_df["timestamp"] = pd.to_datetime(
            batch_df["timestamp"],
            utc=True
        )

        batch_df["original_index"] = range(len(batch_df))

        # ------------------------------------------------
        # SENSOR FREEZE DETECTION
        # ------------------------------------------------

        for asset_id in batch_df["asset_id"].unique():

            asset_df = batch_df[
                (batch_df["asset_id"] == asset_id)
                & (batch_df["status"] == "RUNNING")
            ].sort_values("timestamp").copy()

            if len(asset_df) < 45:
                continue

            for column in [
                "supply_temp_c",
                "return_temp_c"
            ]:

                rolling_std = (
                    asset_df[column]
                    .rolling(window=45, min_periods=45)
                    .std()
                )

                frozen_indices = asset_df.index[
                    rolling_std < 1e-9
                ].tolist()

                for idx in frozen_indices:

                    original_idx = int(
                        batch_df.loc[idx, "original_index"]
                    )

                    results[original_idx]["severity"] = "CRITICAL"

                    results[original_idx]["anomaly_score"] = max(
                        results[original_idx]["anomaly_score"],
                        0.75
                    )

                    results[original_idx]["diagnosis"] = (
                        f"CRITICAL: {column} sensor was frozen "
                        "for at least 45 consecutive readings."
                    )

        # ------------------------------------------------
        # EXCESS ENERGY AND COST ESTIMATION
        # ------------------------------------------------

        total_excess_energy = 0.0

        for asset_id in batch_df["asset_id"].unique():

            asset_df = batch_df[
                batch_df["asset_id"] == asset_id
            ].sort_values("timestamp").copy()

            asset_df = asset_df[
                asset_df["status"] == "RUNNING"
            ].copy()

            if asset_df.empty:
                continue

            baseline = baseline_power_by_asset[asset_id]

            intervals = (
                asset_df["timestamp"].shift(-1)
                - asset_df["timestamp"]
            ).dt.total_seconds() / 60

            valid_intervals = intervals[intervals > 0]

            if not valid_intervals.empty:
                last_interval = float(valid_intervals.median())
            else:
                last_interval = 1.0

            intervals = intervals.fillna(last_interval)
            intervals = intervals.clip(lower=0)

            excess_power = (
                asset_df["power_kw"] - baseline
            ).clip(lower=0)

            excess_energy = (
                excess_power * intervals / 60
            ).sum()

            total_excess_energy += float(excess_energy)

        estimated_cost_loss = (
            total_excess_energy * ELECTRICITY_COST_PER_KWH
        )

        # ------------------------------------------------
        # ANOMALY SUMMARY
        # ------------------------------------------------

        anomaly_count = sum(
            result["severity"] != "NORMAL"
            for result in results
        )

        return {
            "total_records": len(results),
            "anomaly_count": anomaly_count,
            "excess_energy_kwh": round(total_excess_energy, 3),
            "estimated_cost_loss": round(estimated_cost_loss, 4),
            "results": results
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Batch audit failed: {str(error)}"
        )


@app.get("/metrics")
def get_metrics():

    return {
        "service": "EquipmentIQ",
        "status": "operational",
        "available_endpoints": 4,
        "supported_assets": list(
            baseline_power_by_asset.keys()
        ),
        "electricity_cost_per_kwh": ELECTRICITY_COST_PER_KWH
    }
