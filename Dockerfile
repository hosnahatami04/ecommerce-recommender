# --- Builder stage: install dependencies into a venv ---
FROM python:3.13-slim AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Runtime stage: copy the venv, app code, and trained artifacts ---
FROM python:3.13-slim

# libgomp1 (GNU OpenMP runtime) is required by implicit's compiled ALS
# extension at import time -- not present on the slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV OPENBLAS_NUM_THREADS=1

COPY src/ src/
COPY config/ config/
COPY artifacts/ artifacts/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
