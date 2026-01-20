FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir evidently

# Copy source code
COPY src/ ./src/
COPY data/processed/ ./data/processed/

# Expose port
EXPOSE 8001

# Run drift detection API
CMD ["uvicorn", "src.drift_api:app", "--host", "0.0.0.0", "--port", "8001"]
