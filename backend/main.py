"""
FastAPI wrapper around the existing RAG code.

Endpoints:
    GET  /api/health
    GET  /api/configs
    GET  /api/history
    POST /api/query     (Server-Sent Events stream)

Run with:
    uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator

# Make `src` importable when uvicorn launches from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend import history, rag_service
from backend.models import (
    ConfigInfo,
    ConfigsResponse,
    HealthResponse,
    HistoryItem,
    HistoryResponse,
    QueryRequest,
)


app = FastAPI(title="FinAgent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://*.hf.space",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# A dedicated pool for the sync RAG calls so SSE stays responsive.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag")


# --------------------------------------------------------------------------- #
# Health / configs / history
# --------------------------------------------------------------------------- #

@app.get("/api/health", response_model=HealthResponse)
def healthcheck():
    return HealthResponse(**rag_service.health())


@app.get("/api/configs", response_model=ConfigsResponse)
def list_configs():
    return ConfigsResponse(configs=[
        ConfigInfo(
            id="naive",
            label="Naive RAG",
            model="llama-3.1-8b-instant",
            description="Single-shot embed → retrieve → stuff-into-prompt → generate.",
        ),
        ConfigInfo(
            id="agentic",
            label="Agentic RAG",
            model="llama-3.3-70b-versatile",
            description=(
                "Planner → hybrid retrieve (BM25 + dense + reranker) → grader → "
                "rewrite/synthesize → critic. Loops on low-grade retrieval and "
                "unsupported claims."
            ),
        ),
    ])


@app.get("/api/history", response_model=HistoryResponse)
def list_history(limit: int = 50):
    rows = history.list_recent(limit=limit)
    items = [HistoryItem(
        id=r["id"], question=r["question"], config=r["config"],
        market=r["market"], answer=r["answer"] or "",
        latency=r["latency"] or 0.0, created_at=r["created_at"],
    ) for r in rows]
    return HistoryResponse(items=items)


# --------------------------------------------------------------------------- #
# Query — Server-Sent Events
# --------------------------------------------------------------------------- #

def _sse(event: dict) -> str:
    """Encode one event for an SSE stream."""
    return f"data: {json.dumps(event, default=str, ensure_ascii=False)}\n\n"


async def _run_rag(request: QueryRequest) -> dict:
    """Run the chosen RAG in a thread and return the normalised result."""
    loop = asyncio.get_event_loop()
    if request.config == "naive":
        return await loop.run_in_executor(
            _executor, rag_service.run_naive,
            request.market, request.question, request.top_k, request.company_filter,
        )
    return await loop.run_in_executor(
        _executor, rag_service.run_agentic,
        request.market, request.question, request.top_k, request.company_filter,
    )


async def _stream_answer(request: QueryRequest) -> AsyncGenerator[str, None]:
    """
    The streaming protocol the frontend expects:
        chunk    incremental answer text
        sources  retrieved chunks + run metadata
        metrics  final latency / token info
        done     close the stream

    We get the full answer from the synchronous RAG call (existing code path) and
    then feed it back in word-sized pieces so the UI behaves like it's streaming.
    Retrieval status is emitted as `chunk` events with role=status — see the
    React side for how those are rendered as the progress row.
    """
    t0 = time.time()

    yield _sse({"type": "status", "stage": "retrieving",
                "label": f"Retrieving from {request.market.upper()} filings..."})

    try:
        result = await _run_rag(request)
    except Exception as e:
        yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})
        yield _sse({"type": "done"})
        return

    answer = result.get("answer") or ""
    chunks = result.get("chunks") or []
    meta = result.get("metadata") or {}

    yield _sse({
        "type": "sources",
        "chunks": chunks,
        "metadata": meta,
    })

    yield _sse({"type": "status", "stage": "generating",
                "label": f"Generating answer ({meta.get('model','')})..."})

    # Pseudo-stream the answer in word-sized pieces. ~25ms per word is fast
    # enough that the UI feels live but slow enough that the eye can follow.
    for piece in _piecewise(answer, words_per_chunk=3):
        yield _sse({"type": "chunk", "content": piece})
        await asyncio.sleep(0.025)

    latency = time.time() - t0
    meta["latency"] = round(latency, 3)

    yield _sse({
        "type": "metrics",
        "latency": meta["latency"],
        "model": meta.get("model"),
        "input_tokens": meta.get("input_tokens"),
        "output_tokens": meta.get("output_tokens"),
        "agentic": meta if request.config == "agentic" else None,
    })

    try:
        history.log_query(
            question=request.question, config=request.config,
            market=request.market, answer=answer, latency=latency,
        )
    except Exception:
        pass

    yield _sse({"type": "done"})


def _piecewise(text: str, words_per_chunk: int = 3) -> list[str]:
    """Split text into space-preserving pieces."""
    if not text:
        return []
    parts: list[str] = []
    words = text.split(" ")
    for i in range(0, len(words), words_per_chunk):
        seg = " ".join(words[i:i + words_per_chunk])
        if i + words_per_chunk < len(words):
            seg += " "
        parts.append(seg)
    return parts


@app.post("/api/query")
async def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Empty question.")
    return StreamingResponse(
        _stream_answer(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering
        },
    )
