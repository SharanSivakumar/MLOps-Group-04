FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /drift

WORKDIR /drift

COPY requirements.txt /drift/requirements.txt
COPY drift_api.py /drift/drift_api.py
COPY drift_detection.py /drift/drift_detection.py
COPY __init__.py /drift/__init__.py
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

EXPOSE 8080
CMD ["sh", "-c", "python -m uvicorn drift_api:app --host 0.0.0.0 --port ${PORT:-8080}"]
