#!/bin/sh
set -e

PORT=${PORT:-8080}
exec uvicorn drift_api:app --host 0.0.0.0 --port "$PORT"
