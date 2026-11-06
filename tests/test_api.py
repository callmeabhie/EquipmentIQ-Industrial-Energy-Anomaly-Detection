
from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def sample_reading(timestamp="2026-01-01T07:40:00"):
    return {
        "timestamp": timestamp,
        "asset_id": "Chiller-01",
        "power_kw": 149.76,
        "flow_rate": 445.34,
        "supply_temp_c": 6.76,
        "return_temp_c": 12.26,
        "pressure_bar": 4.19,
        "status": "RUNNING"
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_diagnose():
    response = client.post(
        "/api/v1/diagnose",
        json=sample_reading()
    )

    assert response.status_code == 200

    result = response.json()

    assert 0 <= result["anomaly_score"] <= 1
    assert result["asset_id"] == "Chiller-01"
    assert result["severity"] in ["NORMAL", "WARNING", "CRITICAL"]


def test_batch_audit():
    readings = [
        sample_reading("2026-01-01T07:40:00"),
        sample_reading("2026-01-01T07:41:00"),
    ]

    response = client.post(
        "/api/v1/batch-audit",
        json={"readings": readings}
    )

    assert response.status_code == 200

    result = response.json()

    assert result["total_records"] == 2
    assert "anomaly_count" in result
    assert "excess_energy_kwh" in result
    assert "estimated_cost_loss" in result


def test_empty_batch_rejected():
    response = client.post(
        "/api/v1/batch-audit",
        json={"readings": []}
    )

    assert response.status_code == 422


def test_sensor_freeze_detection():
    start = datetime(2026, 1, 3, 10, 0)

    readings = []

    for i in range(50):
        reading = sample_reading(
            (start + timedelta(minutes=i)).isoformat()
        )

        reading["supply_temp_c"] = 6.6
        reading["return_temp_c"] = 12.1 + (i * 0.01)

        readings.append(reading)

    response = client.post(
        "/api/v1/batch-audit",
        json={"readings": readings}
    )

    assert response.status_code == 200

    results = response.json()["results"]

    assert any(
        result["severity"] == "CRITICAL"
        and "frozen" in result["diagnosis"].lower()
        for result in results
    )
