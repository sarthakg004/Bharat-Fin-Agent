# syntax=docker/dockerfile:1.6
# =============================================================================
# FinAgent — single image for Google Cloud Run.
# Serves the React SPA + FastAPI API on one origin. The prebuilt Chroma vector
# store is baked into the image, so full hybrid retrieval (BM25 + dense + rerank)
# works without any external vector DB.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 — build the Vite SPA
# -----------------------------------------------------------------------------
FROM node:20-alpine AS spa
WORKDIR /spa
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
ENV VITE_API_URL=""
RUN npm run build


# -----------------------------------------------------------------------------
# Stage 2 — Python runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    SENTENCE_TRANSFORMERS_HOME=/app/.hf \
    STATIC_DIR=/app/static \
    CHROMA_DIR=/app/data/chroma \
    PYTHONPATH=/app \
    # Per-session memory: the client carries history, nothing persisted server-side.
    STATELESS=1 \
    PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first (Cloud Run has no GPU) so sentence-transformers reuses it
# instead of pulling the multi-GB CUDA build — smaller image, faster cold start.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Bake the bge models into the image so a cold start loads from local disk.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('BAAI/bge-reranker-base')"

# App code + built SPA + the prebuilt Chroma store (read at $CHROMA_DIR).
COPY finagent/      ./finagent/
COPY pyproject.toml ./
COPY --from=spa /spa/dist ./static
COPY data/chroma/   ./data/chroma/

EXPOSE 8080

# Cloud Run sets $PORT (default 8080). Single worker keeps the models + Chroma
# in one process.
CMD ["sh", "-c", "uvicorn finagent.api.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
