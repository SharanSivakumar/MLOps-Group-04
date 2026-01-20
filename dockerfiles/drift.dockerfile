FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir evidently

# Copy source code
COPY src/ ./src/

# Expose port
EXPOSE 8080

# Run drift detection API
CMD ["uvicorn", "src.drift_api:app", "--host", "0.0.0.0", "--port", "8080"]
