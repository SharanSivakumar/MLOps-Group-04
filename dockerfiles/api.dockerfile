FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

WORKDIR /app

COPY uv.lock pyproject.toml README.md ./

RUN uv sync --frozen --no-install-project

COPY src/ ./src/
COPY drift/ ./drift/
COPY dockerfiles/api_entrypoint.sh /app/api_entrypoint.sh
RUN uv sync --frozen && chmod +x /app/api_entrypoint.sh

ENV PORT=8080

ENTRYPOINT ["/app/api_entrypoint.sh"]

# Checkpoints are downloaded at container startup from GCS; do not copy at build time.
