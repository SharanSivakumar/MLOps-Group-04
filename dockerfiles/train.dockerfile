FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md

RUN uv sync --frozen --no-install-project

COPY src src/
COPY .dvc .dvc/
COPY .dvcignore .dvcignore
COPY data/time_series.dvc data/time_series.dvc
COPY dockerfiles/entrypoint.sh /entrypoint.sh

RUN uv sync --frozen

RUN chmod +x /entrypoint.sh
RUN mkdir -p data/time_series

ENTRYPOINT ["/entrypoint.sh"]
