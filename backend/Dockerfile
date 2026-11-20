# Multi-stage: the build stage carries compilers for the wheels that need
# them (PyMuPDF, onnxruntime); the runtime stage does not, which keeps the
# final image smaller and removes a class of container CVEs that only ever
# come from build tooling.

FROM python:3.12-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /wheels
COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt


FROM python:3.12-slim AS runtime

# libgl/libglib: OpenCV, which RapidOCR pulls in for image preprocessing.
# Without them the OCR path fails at import with a bare libGL.so error that
# is genuinely hard to diagnose from inside a container.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Never run as root. The graph writes uploads, a SQLite database and
# checkpoints, so the app owns exactly those directories and nothing else.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY --from=build /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY --chown=appuser:appuser . .

RUN mkdir -p /app/graph_storage/uploads /app/storage \
    && chown -R appuser:appuser /app/graph_storage /app/storage

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRAPH_SQLITE_PATH=/app/graph_storage/graph.db \
    CHECKPOINT_DB_PATH=/app/graph_storage/checkpoints.db

EXPOSE 8000

# Checks the graph's own health route, not just "is the port open" — a process
# that is listening but cannot reach its database is not healthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; \
r=urllib.request.urlopen('http://127.0.0.1:8000/graph/health',timeout=8); \
sys.exit(0 if r.status==200 else 1)"

# One worker by default. The extraction node already runs a thread pool per
# document, so multiple uvicorn workers multiply concurrent LLM calls and blow
# through the provider rate limit. Scale with replicas and a shared Postgres
# checkpointer instead — see docs/GRAPH_DEPLOYMENT.md.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
