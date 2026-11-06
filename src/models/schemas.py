
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    timestamp: datetime

    asset_id: Literal[
        "Chiller-01",
        "Compressor-02",
        "Pump-03"
    ]

    power_kw: float = Field(ge=0)
    flow_rate: float = Field(ge=0)
    supply_temp_c: float
    return_temp_c: float
    pressure_bar: float = Field(ge=0)

    status: Literal[
        "RUNNING",
        "STANDBY",
        "OFF"
    ]


class DiagnosisResponse(BaseModel):
    timestamp: datetime
    asset_id: str

    anomaly_score: float = Field(ge=0, le=1)

    severity: Literal[
        "NORMAL",
        "WARNING",
        "CRITICAL"
    ]

    diagnosis: str


class BatchAuditRequest(BaseModel):
    readings: list[SensorReading] = Field(min_length=1)


class BatchAuditResponse(BaseModel):
    total_records: int
    anomaly_count: int

    excess_energy_kwh: float = 0.0
    estimated_cost_loss: float = 0.0

    results: list[DiagnosisResponse]
