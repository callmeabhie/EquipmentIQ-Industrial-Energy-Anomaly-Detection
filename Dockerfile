
# Stage 1: Build environment
FROM python:3.11-slim AS builder

WORKDIR /app

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


# Stage 2: Runtime environment
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app

# Copy application source code
COPY src/ ./src/

# Copy trained machine learning models
COPY artifacts/ ./artifacts/

# Copy processed telemetry required by diagnosis service
COPY data/processed/ ./data/processed/

# Create non-root application user
RUN useradd --create-home appuser

USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

