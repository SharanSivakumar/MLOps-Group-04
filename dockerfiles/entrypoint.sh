#!/bin/bash
set -e

echo "Pulling data with DVC..."
# Configure DVC to work without git (for Docker containers)
uv run dvc config core.no_scm true || true
if ! uv run dvc pull; then
    echo "ERROR: Failed to pull data from DVC remote. Check your credentials and remote configuration."
    exit 1
fi

echo "Verifying data was pulled..."
if [ ! -d "data/time_series/AF" ] || [ ! -d "data/time_series/Noise" ] || [ ! -d "data/time_series/NSR" ]; then
    echo "ERROR: Data directories not found after DVC pull. Check your DVC configuration."
    exit 1
fi

echo "Data pull successful. Starting training..."
exec uv run src/train.py "$@"

