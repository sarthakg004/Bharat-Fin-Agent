# syntax=docker/dockerfile:1.6
# =============================================================================
# FinAgent — single image for Google Cloud Run.
# Serves the React SPA + FastAPI API on one origin. The corpus lives in a Qdrant
# cluster (hybrid dense + BM25-sparse search with server-side RRF), so the image
# ships code and models only — no vector data.
#
# Three stages: (1) build the SPA, (2) a builder that compiles/installs the
# Python deps + bakes the models with build tools present, (3) a slim runtime
# that copies only the finished venv + model cache — no compilers in the final
# image. This trims the runtime image (smaller pull = faster cold start) and
# guarantees deps that need a C/C++ toolchain still
# build, because the toolchain lives in the throwaway builder stage.
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
# Stage 2 — Python builder (has compilers; produces /opt/venv + /app/.hf)
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    SENTENCE_TRANSFORMERS_HOME=/app/.hf \
    FASTEMBED_CACHE_PATH=/app/.fastembed \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv so we can copy exactly the installed packages into the runtime.
RUN python -m venv /opt/venv

# CPU-only torch first (Cloud Run has no GPU) so sentence-transformers reuses it
# instead of pulling the multi-GB CUDA build. torchvision comes from the same
# index so the pair matches — docling pulls it transitively, and a PyPI
# torchvision against a CPU-index torch breaks (torchvision::nms missing).
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Bake the bge models into the venv-side HF cache so cold start loads from disk.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('BAAI/bge-reranker-v2-m3')"

# Bake the BM25 sparse encoder (~0.1 MB) that produces the lexical half of
# hybrid retrieval. Without it the runtime tries to fetch from HF on first
# query and dies under HF_HUB_OFFLINE=1: "Could not load model Qdrant/bm25".
RUN python -c "from fastembed import SparseTextEmbedding; \
SparseTextEmbedding('Qdrant/bm25')"

# Bake the Docling models (layout + TableFormer only, ~500 MB — no OCR, no
# code/picture classifiers) so the upload parser loads offline at runtime.
RUN python -c "from pathlib import Path; \
from docling.utils.model_downloader import download_models; \
download_models(output_dir=Path('/app/.docling'), with_code_formula=False, \
with_picture_classifier=False, with_rapidocr=False)"


# -----------------------------------------------------------------------------
# Stage 3 — slim runtime (no compilers, no pip caches)
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf \
    SENTENCE_TRANSFORMERS_HOME=/app/.hf \
    FASTEMBED_CACHE_PATH=/app/.fastembed \
    STATIC_DIR=/app/static \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8080 \
    # The corpus lives in Qdrant, so a dynamically fetched filing can be
    # upserted into the shared index and stay there — the server handles
    # concurrent read/write, and deterministic point ids make a duplicate write
    # harmless. QDRANT_URL / QDRANT_API_KEY are set on the service.
    PERSIST_DYNAMIC_FETCH=true \
    # Models are baked above — force offline loading so cold start makes no HF
    # Hub network call (faster + robust if HF is down/rate-limits).
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    # Docling loads the baked layout/table models from here (no network).
    DOCLING_ARTIFACTS_PATH=/app/.docling

# Runtime-only system libs (no build-essential): libgomp1 for torch/onnx,
# ca-certificates for outbound HTTPS (Groq/Tavily), curl for debugging.
# libgl1 + libglib2.0-0: opencv (cv2) links libGL/libglib/libX11 — docling's
# TableFormer imports cv2 mid-parse, so uploads 422 without them (works
# locally only because desktops ship these).
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The finished Python environment + model caches from the builder.
COPY --from=builder /opt/venv     /opt/venv
COPY --from=builder /app/.hf      /app/.hf
COPY --from=builder /app/.docling /app/.docling
COPY --from=builder /app/.fastembed /app/.fastembed

# App code + built SPA. The corpus is NOT in the image any more — it lives in
# Qdrant, which is why this build no longer needs the 2.2 GB data directory.
COPY finagent/      ./finagent/
COPY pyproject.toml ./
COPY --from=spa /spa/dist ./static

EXPOSE 8080

# Cloud Run sets $PORT (default 8080). Single worker keeps the models
# in one process.
CMD ["sh", "-c", "uvicorn finagent.api.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
