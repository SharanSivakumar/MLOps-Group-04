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
COPY entrypoint.sh /drift/entrypoint.sh
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt && \
    chmod +x /drift/entrypoint.sh

EXPOSE 8080
ENV PORT=8080

CMD ["/drift/entrypoint.sh"]
