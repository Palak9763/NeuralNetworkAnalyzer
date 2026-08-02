# -- Stage 1: Build (install packages into /root/.local) ----------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools needed for some ML packages (psycopg2, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into /root/.local so we can copy the whole dir to the final stage.
# gunicorn is added here (not in requirements.txt per se, but we pin it in
# requirements.txt too for consistency; the --user flag is key for the copy).
RUN pip install --no-cache-dir --user -r requirements.txt

# -- Stage 2: Runtime (lean final image) --------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Only the runtime system deps (libpq for psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application source (only what .dockerignore does not exclude)
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Suppress TensorFlow C++ startup noise; keep container logs readable
ENV TF_CPP_MIN_LOG_LEVEL=2

# Ensure the uploads directory exists inside the container
RUN mkdir -p /app/storage/uploads

EXPOSE 8000

# Healthcheck using stdlib urllib (no extra deps needed).
# start-period=120s gives gunicorn time to boot before Docker marks it unhealthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" \
    || exit 1

# Production command:
#   --workers 1              one process: avoids duplicating ML library memory
#   --max-requests 50        recycle worker after ~50 requests (releases accumulated
#                            memory from mixed torch/tf usage across requests)
#   --max-requests-jitter 10 stagger recycling to avoid all workers dying at once
#   --timeout 120            allow up to 120s per request (HF config download +
#                            heavy parses; tune up if you see WORKER TIMEOUT in logs)
#   --access-logfile -       log to stdout (Docker json-file driver handles rotation)
#   --error-logfile  -       same
CMD ["gunicorn", "app.main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--workers", "1", \
     "--max-requests", "50", \
     "--max-requests-jitter", "10", \
     "--timeout", "120", \
     "--bind", "0.0.0.0:8000", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
