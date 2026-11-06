
# EquipmentIQ
## Industrial Energy Anomaly Detection & Diagnostic Service

EquipmentIQ is a machine-learning-powered industrial energy monitoring and diagnostic service. It analyzes equipment telemetry to identify abnormal energy consumption, sensor faults, and potential equipment efficiency degradation.

The project combines unsupervised machine learning, rule-based diagnostics, energy analysis, and REST API integration.

## 1. Project Objectives

- Detect abnormal industrial equipment power consumption.
- Identify frozen temperature sensors.
- Detect potential equipment efficiency degradation.
- Generate normalized anomaly scores.
- Classify equipment conditions into NORMAL, WARNING, and CRITICAL.
- Estimate excess energy consumption and associated cost.
- Provide real-time-style diagnosis through REST API endpoints.

## 2. Dataset

The project uses synthetically generated industrial telemetry for three assets.

| Asset | Equipment |
|---|---|
| Chiller-01 | Industrial chiller |
| Compressor-02 | Industrial compressor |
| Pump-03 | Industrial pump |

**Dataset characteristics:**

- Duration: 7 days
- Sampling frequency: 1 minute
- Records per asset: 10,080
- Total records: 30,240

### Sensor Features

| Feature | Description |
|---|---|
| timestamp | Telemetry timestamp |
| asset_id | Equipment identifier |
| power_kw | Power consumption in kW |
| flow_rate | Equipment flow rate |
| supply_temp_c | Supply temperature |
| return_temp_c | Return temperature |
| pressure_bar | Operating pressure |
| status | RUNNING, STANDBY, or OFF |

### Injected Anomalies

1. Missing telemetry and sensor dropout periods.
2. Physically invalid readings, including negative power and extreme temperature values.
3. Frozen temperature sensor readings.
4. Sudden power consumption spikes.
5. Gradual power consumption increase over 48 hours.

## 3. Data Processing and Feature Engineering

The data processing pipeline performs:

- Missing-value handling.
- Physical sensor-bound validation.
- Frozen sensor detection.
- Timestamp processing.
- Feature engineering.

### Engineered Features

| Feature | Description |
|---|---|
| delta_t | Difference between return and supply temperature |
| specific_energy | Equipment-specific energy efficiency indicator |
| rolling_power_mean_1h | One-hour rolling average power |
| load_ratio | Power consumption relative to asset baseline |

Processed telemetry is saved to:

`data/processed/cleaned_telemetry.csv`

## 4. Machine Learning

EquipmentIQ uses **Isolation Forest**, an unsupervised anomaly detection algorithm.

Separate models are trained for the three equipment assets using their RUNNING telemetry records.

### Anomaly Scoring

Anomaly scores are normalized between 0 and 1.

| Score | Severity |
|---|---|
| Below 0.50 | NORMAL |
| 0.50 to below 0.75 | WARNING |
| 0.75 and above | CRITICAL |

These scores represent relative anomaly scores, not probabilities.

A hybrid diagnostic approach combines machine learning predictions with rule-based sensor-freeze detection.

### Model Artifacts

Trained models are saved in:

`artifacts/equipmentiq_three_asset_model.joblib`

## 5. Energy Insights and Diagnostics

The insights module estimates:

- Baseline power consumption for each asset.
- Excess energy consumption in kWh.
- Estimated electricity cost loss.
- Power surge records.
- Sensor freeze records.
- Potential efficiency degradation records.

The electricity cost assumption is **$0.12 per kWh**.

Generated reports:

- `artifacts/insights/asset_insights.csv`
- `artifacts/insights/detailed_diagnostics.csv`
- `artifacts/insights/insights_report.json`

These are preliminary estimates based on synthetic telemetry and configured baselines, not verified operational savings.

## 6. Visualizations

The project generates the following diagnostic visualizations:

- `trends.png` – Equipment telemetry trends.
- `efficiency_correlation.png` – Relationships between operating variables and efficiency.
- `anomalies_detected.png` – Detected anomaly patterns.

Location:

`artifacts/plots/`

## 7. REST API

The application is developed using FastAPI and Pydantic.

### Available Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check and loaded models |
| POST | `/api/v1/diagnose` | Diagnose a single sensor reading |
| POST | `/api/v1/batch-audit` | Audit multiple sensor readings |
| GET | `/metrics` | Service information and supported assets |

### API Documentation

After starting the application, access:

`http://127.0.0.1:8000/docs`

### Batch Audit Response

The batch audit provides:

- Total records processed.
- Number of detected anomalies.
- Estimated excess energy consumption.
- Estimated electricity cost loss.
- Individual diagnosis results.

The batch endpoint currently accepts JSON sensor readings.

## 8. Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- FastAPI
- Pydantic
- Joblib
- Uvicorn
- Pytest
- Docker

## 9. Project Structure

```text
EquipmentIQ/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── artifacts/
│   ├── insights/
│   ├── plots/
│   └── equipmentiq_three_asset_model.joblib
│
├── src/
│   ├── api/
│   │   └── main.py
│   ├── models/
│   │   └── schemas.py
│   ├── services/
│   │   └── diagnosis.py
│   ├── generate_data.py
│   ├── data_processor.py
│   ├── train.py
│   ├── visualize.py
│   └── insights.py
│
├── tests/
│   └── test_api.py
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 10. Installation and Execution

### Step 1: Clone the repository

```bash
git clone <your-github-repository-url>
cd EquipmentIQ
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Generate telemetry data

```bash
python src/generate_data.py
```

### Step 4: Process telemetry

```bash
python src/data_processor.py
```

### Step 5: Train machine learning models

```bash
python src/train.py
```

### Step 6: Generate visualizations

```bash
python src/visualize.py
```

### Step 7: Generate equipment insights

```bash
python src/insights.py
```

### Step 8: Start the API

```bash
uvicorn src.api.main:app --reload
```

Open the interactive API documentation:

`http://127.0.0.1:8000/docs`

## 11. Running Tests

Run the test suite using:

```bash
python -m pytest -v
```

The API test suite covers health checks, individual diagnosis, batch auditing, empty-batch validation, and sensor-freeze detection.

## 12. Docker Deployment

Build the Docker image:

```bash
docker compose build
```

Start the service:

```bash
docker compose up
```

Access the API documentation:

`http://localhost:8000/docs`

## 13. Limitations and Future Improvements

- Validate anomaly thresholds using real industrial operating data.
- Improve detection of gradual degradation using time-series analysis.
- Incorporate timestamp-continuity checks for sensor freeze detection.
- Improve historical feature calculation for individual API readings.
- Integrate real-time telemetry ingestion.
- Validate energy and cost estimates against actual operational measurements.
- Expand monitoring and diagnostic capabilities.

## 14. Author

**Abhijith K.**

B.Tech – Computer Science and Engineering

GitHub: https://github.com/callmeabhie

LinkedIn: https://linkedin.com/in/abhijith-k-abhie7
