# syntax=docker/dockerfile:1.6
# FinAgent: one image for Google Cloud Run, serving the API and the built frontend.
# The filings live in Qdrant, so the image holds code and one model only.

# Stage 1: build the frontend
FROM node:20-alpine AS spa
WORKDIR /spa
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
ENV VITE_API_URL=""
RUN npm run build


# Stage 2: install Python packages and download models (has compilers)
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    SENTENCE_TRANSFORMERS_HOME=/app/.hf \
    FASTEMBED_CACHE_PATH=/app/.fastembed \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv

# CPU-only torch first, so sentence-transformers does not pull the CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# `unstructured` downloads this spaCy model on first use; bake it instead.
RUN python -m spacy download en_core_web_sm

# The local reranker, used only when Cohere is unavailable. Baked because the
# runtime is offline for Hugging Face (HF_HUB_OFFLINE=1).
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"

# The BM25 encoder for the lexical half of hybrid search.
RUN python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25')"


# Stage 3: slim runtime (no compilers)
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf \
    SENTENCE_TRANSFORMERS_HOME=/app/.hf \
    FASTEMBED_CACHE_PATH=/app/.fastembed \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    STATIC_DIR=/app/static \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8080

# libgomp1: torch. libmagic1: `unstructured` sniffs a fetched filing's type.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /opt/venv       /opt/venv
COPY --from=builder /app/.hf        /app/.hf
COPY --from=builder /app/.fastembed /app/.fastembed
COPY finagent/      ./finagent/
COPY --from=spa /spa/dist ./static

# The runtime is offline; fail the build if the baked BM25 model will not load
# (fastembed once refused its own cache and every filing search failed).
RUN python -c "from finagent.vectorstore import get_sparse_embeddings; get_sparse_embeddings()"

EXPOSE 8080
CMD ["sh", "-c", "uvicorn finagent.api.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
