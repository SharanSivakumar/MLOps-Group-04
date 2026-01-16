FROM ghcr.io/astral-sh/uv:python3.12-alpine AS base

# Install build dependencies for packages like scikit-learn, scipy, etc.
RUN apk add --no-cache gcc g++ musl-dev linux-headers gfortran openblas-dev

# Copy dependency files
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md

# Install dependencies without installing the project itself
RUN uv sync --frozen --no-install-project

# Copy source code
COPY src/ src/

# Install the project
RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "python", "-u", "tasks.py"]
