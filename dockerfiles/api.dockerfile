FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

WORKDIR /app

COPY uv.lock pyproject.toml README.md ./

RUN uv sync --frozen --no-install-project

COPY src/ ./src/

RUN uv sync --frozen

ENV PORT=8000

ENTRYPOINT ["uv", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Checkpoints are downloaded at container startup from GCS; do not copy at build time.
