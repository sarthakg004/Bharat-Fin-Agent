"""
FastAPI app — agentic RAG.

The single agentic pipeline (`AgenticRAGv4`) drives every answer. The server is
stateless: it stores nothing between requests. The client owns the conversation
thread and replays recent turns via `QueryRequest.chat_history`, which the agent
uses as memory.

Endpoints
---------
    GET    /api/health
    POST   /api/upload               # parse a PDF/DOCX into ephemeral chunks
    POST   /api/query                # SSE stream — one answer
    POST   /api/research             # SSE stream — Deep Research Mode

Streaming events on /api/query: status, sources, chart, chunk, metrics, done.
/api/research streams research_plan, agent_start, agent_done, then the same
sources/chunk/metrics/done frames.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator

# Repo root on sys.path so `import finagent...` resolves when this module is
# imported directly (e.g. `uvicorn finagent.api.main:app`). api/ is two levels
# below the root: finagent/api/main.py -> parents[2] == repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os

# Cloud Run egress is IPv4-only. Some API hosts (e.g. api.groq.com behind
# Cloudflare) resolve to an IPv6 address first, and httpx then fails to connect
# → groq.APIConnectionError. When FORCE_IPV4 is set we make DNS return only
# IPv4 addresses, so every outbound call (Groq, Tavily, yfinance) uses IPv4.
if os.getenv("FORCE_IPV4", "").strip().lower() in ("1", "true", "yes"):
    import socket as _socket

    _orig_getaddrinfo = _socket.getaddrinfo

    def _getaddrinfo_ipv4(host, port, family=0, *args, **kwargs):
        return _orig_getaddrinfo(host, port, _socket.AF_INET, *args, **kwargs)

    _socket.getaddrinfo = _getaddrinfo_ipv4

# Native-thread safety. The graph runs in a ThreadPoolExecutor, so embedding /
# tokenizer / embedding work happens off the main thread. The HuggingFace
# `tokenizers` Rust parallelism is not fork/thread-safe and can SIGSEGV; disable
# it before transformers is imported (it reads this at import time). Overridable.
# On a GPU box you may also want FINAGENT_DEVICE=cpu locally, since CUDA from a
# worker thread is fragile — production (Cloud Run) is CPU anyway.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from finagent.api import rag_service
from finagent.api.models import (
    HealthResponse,
    QueryRequest,
    ResearchRequest,
    UploadResponse,
)


app = FastAPI(title="FinAgent API", version="2.0.0")

# CORS — the SPA is hosted on Firebase (a different origin) and calls this API
# directly. Set ALLOWED_ORIGINS on Cloud Run to your Firebase URL(s),
# comma-separated. Local dev origins are always allowed.
_ALLOWED_ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    *[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()],
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["*"], allow_headers=["*"],
)

# Graph runs are serialized on a SINGLE worker. The storage layer no longer
# requires it — Qdrant is a server and handles concurrent read/write itself, and
# request state lives in a per-request RuntimeContext rather than on the shared
# agent. What remains is local: the cross-encoder and HF tokenizers are the
# memory-heavy, not-fully-thread-safe part, and one CPU serving several graph
# runs just time-slices the same core. Raising RAG_MAX_WORKERS is now a capacity
# decision, not a correctness one.
_executor = ThreadPoolExecutor(
    max_workers=int(os.getenv("RAG_MAX_WORKERS", "1")), thread_name_prefix="rag"
)

# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #

@app.get("/api/health", response_model=HealthResponse)
def healthcheck():
    """Liveness only — the SPA polls this to tell a cold start from a dead
    backend, and reads nothing but the status code. Deliberately does no work:
    it must not probe the LLM (burns quota) or the index (slows every poll)."""
    return HealthResponse(status="ok")


# --------------------------------------------------------------------------- #
# Query — Server-Sent Events
# --------------------------------------------------------------------------- #

def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str, ensure_ascii=False)}\n\n"


def _agent_history(request: QueryRequest) -> list[dict]:
    """The agent's `chat_history` — the last 6 client-supplied turns, truncated.

    The server stores nothing: the SPA owns the thread (sessionStorage) and
    replays recent turns with each request, so memory is exactly what the
    client sent and no more.
    """
    return [
        {"role": t.role, "content": (t.content or "")[:1200]}
        for t in (request.chat_history or [])
    ][-6:]


# Friendly, market-neutral labels for each graph node, surfaced live as the
# agent "thinks". Repeated nodes (retrieve/grade across rewrite loops) reuse the
# same label; the UI de-dupes consecutive repeats.
_STEP_LABELS = {
    "planner":        "Planning the approach…",
    "router":         "Routing the sub-questions…",
    "fetch_filing":   "Fetching latest filing…",
    "retrieve":       "Searching the filings…",
    "grader":         "Weighing the evidence…",
    "rewrite":        "Refining the search…",
    "xbrl":           "Looking up exact figures…",
    "calculator":     "Computing the metrics…",
    "table_agent":    "Crunching the numbers…",
    "market_data":    "Pulling market data…",
    "web_search":     "Searching the web…",
    "edgar_search":   "Searching EDGAR across companies…",
    "evidence_builder": "Organising the evidence…",
    "synthesize":     "Writing the answer…",
    "critic":         "Fact-checking the draft…",
    "verify_numbers": "Verifying every figure…",
    "confidence":     "Scoring confidence…",
}

# Canonical pipeline order (for the UI progress bar). The agent skips most of
# these per query (the dispatcher routes around them), but the position of the
# furthest stage reached still maps cleanly to "how far along" the run is.
PIPELINE_ORDER = list(_STEP_LABELS.keys())


# --------------------------------------------------------------------------- #
# Document upload — ephemeral, per-session
# --------------------------------------------------------------------------- #
# Uploaded documents are parsed (Docling) into in-memory chunks and held in a
# TTL dict keyed by upload_id; a query referencing the id rides the agent's
# existing `fetched_chunks` lane (ranked against the question, never written
# to the persistent index). Survives neither restarts nor scale-to-zero — the
# client re-uploads after an idle gap, exactly like the dynamic-fetch path.
# ponytail: in-memory, single-instance store; deploy pins --max-instances 1.
# Swap for a GCS-backed store if multi-instance is ever needed.
_UPLOADS: dict[str, tuple[float, dict]] = {}
_UPLOAD_TTL_S = 3600
_UPLOAD_MAX_ENTRIES = 20
_UPLOAD_MAX_BYTES = 15 * 1024 * 1024
_UPLOAD_SUFFIXES = {".pdf", ".docx"}
# Docling parse time scales with pages (~seconds/page on Cloud Run CPU); a big
# filing both bills the instance for minutes and blocks the single-worker
# executor behind it (observed: a 197-page 10-Q blew the 600s request timeout).
# Reject early with a cheap pypdfium2 page count — cost control for a
# portfolio deployment. ponytail: raise when a separate parse worker exists.
_UPLOAD_MAX_PAGES = 40


def _pdf_page_count(data: bytes):
    """Page count via pypdfium2 (already in the image as a docling dep).
    None when unreadable — Docling then gets its own try at the file."""
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(data)
        try:
            return len(doc)
        finally:
            doc.close()
    except Exception:
        return None


def _uploads_gc() -> None:
    now = time.time()
    expired = [k for k, (ts, _) in _UPLOADS.items() if now - ts > _UPLOAD_TTL_S]
    for k in expired:
        _UPLOADS.pop(k, None)
    while len(_UPLOADS) >= _UPLOAD_MAX_ENTRIES:     # evict oldest
        _UPLOADS.pop(min(_UPLOADS, key=lambda k: _UPLOADS[k][0]), None)


@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _UPLOAD_SUFFIXES:
        raise HTTPException(status_code=415,
                            detail="Only PDF and DOCX files are supported.")
    data = await file.read()
    if len(data) > _UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 15 MB limit.")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if suffix == ".pdf":
        pages = _pdf_page_count(data)
        if pages and pages > _UPLOAD_MAX_PAGES:
            raise HTTPException(
                status_code=413,
                detail=f"PDF has {pages} pages; the limit is {_UPLOAD_MAX_PAGES}. "
                       "Please upload the relevant section (e.g. the financial "
                       "statements and MD&A pages).")

    from finagent.ingestion.upload import parse_upload

    # Same single-worker executor as the agent: Docling parsing is serialized
    # behind RAG runs instead of racing them for CPU/memory.
    loop = asyncio.get_event_loop()
    try:
        parsed = await loop.run_in_executor(
            _executor, lambda: parse_upload(data, file.filename or "upload.pdf"))
    except Exception as e:
        # Full type + message in the server log (the 422 detail stays generic);
        # a bare class name made the cloud cv2/libGL failure a blind hunt.
        print(f"[upload error] {type(e).__name__}: {e}", flush=True)
        raise HTTPException(status_code=422,
                            detail="Could not parse the document.") from e
    if not parsed["ok"]:
        raise HTTPException(status_code=422,
                            detail="No extractable text found in the document "
                                   "(scanned/image-only files are not supported).")

    import uuid
    _uploads_gc()
    upload_id = uuid.uuid4().hex
    _UPLOADS[upload_id] = (time.time(), parsed)
    return UploadResponse(upload_id=upload_id, filename=parsed["filename"],
                          pages=parsed["pages"], tables=parsed["tables"],
                          chunks=len(parsed["chunks"]))


def _resolve_uploads(upload_ids: list[str]) -> tuple[list[dict], list[str]]:
    """(chunks, missing_ids) for the requested uploads; touches TTLs."""
    chunks: list[dict] = []
    missing: list[str] = []
    for uid in upload_ids:
        entry = _UPLOADS.get(uid)
        if entry is None or time.time() - entry[0] > _UPLOAD_TTL_S:
            _UPLOADS.pop(uid, None)
            missing.append(uid)
            continue
        _UPLOADS[uid] = (time.time(), entry[1])     # keep alive while in use
        chunks.extend(entry[1]["chunks"])
    return chunks, missing


def _classify_provider_error(e: Exception, pc) -> dict:
    """Classify a provider failure into a user-facing SSE error event
    (shared by /api/query and /api/research)."""
    from finagent.llm import is_daily_quota_error, is_rate_limit_error

    # Log the error type for debugging, but NOT the full exception value —
    # provider auth errors can echo the API key into logs.
    print(f"[query error] {type(e).__name__}", flush=True)

    rate_limited = is_rate_limit_error(e)
    provider = (pc.provider if pc else "groq")
    user_key = bool(pc and pc.api_key)
    prov_label = {"groq": "Groq", "gemini": "Gemini",
                  "openai": "OpenAI", "anthropic": "Anthropic"}.get(provider, provider)
    if rate_limited:
        code = "rate_limit"
        if user_key:
            # The user supplied THEIR OWN key — don't blame the shared keys.
            # Free tiers are tiny (Gemini = 5 req/min) and this agent makes
            # many model calls per question, so a single query can exhaust
            # them. Tell them what actually happened and how to recover.
            daily = is_daily_quota_error(e)
            window = "daily quota" if daily else "per-minute rate limit"
            message = (
                f"Your {prov_label} API key hit its {window}. This agent makes "
                f"several model calls per question, and free tiers are very low "
                f"(Gemini allows just 5 requests/min). "
                + ("Try again tomorrow, " if daily else "Wait a minute and retry, ")
                + f"or use a higher-tier {prov_label} key."
            )
        elif is_daily_quota_error(e):
            message = ("We've hit today's usage limit on the shared API keys. "
                       "Please try again tomorrow — or add your own API key "
                       "from the model picker to keep going now.")
        else:
            message = ("Limit exhausted: all shared API keys have hit their "
                       "rate limit. Please wait a minute and try again — or "
                       "add your own API key from the model picker to keep "
                       "going now.")
    else:
        code = "error"
        # Surface a clean provider error (the value may include a key, so the
        # llm layer already avoids logging it; here we keep the type + a short
        # hint without echoing the full provider payload).
        msg = str(e)
        message = f"{prov_label} error: {msg[:240]}" if user_key else f"{type(e).__name__}: {msg[:240]}"
    return {"type": "error", "code": code, "message": message}


async def _run_rag(request: QueryRequest, hist: list[dict],
                   extra_chunks: list[dict] | None = None,
                   on_step=None, on_step_done=None) -> dict:
    loop = asyncio.get_event_loop()
    pc = request.provider_config
    provider = (pc.provider if pc else "groq")
    synth_model = (pc.synth_model if pc else None)
    api_key = (pc.api_key if pc else None)
    # `run_in_executor` doesn't take kwargs — use a small lambda wrapper instead.
    return await loop.run_in_executor(
        _executor,
        lambda: rag_service.run_agentic(
            request.question, chat_history=hist,
            provider=provider, synth_model=synth_model, api_key=api_key,
            session_id=request.session_id or None,
            extra_chunks=extra_chunks,
            on_step=on_step, on_step_done=on_step_done,
        ),
    )


async def _stream_answer(request: QueryRequest) -> AsyncGenerator[str, None]:
    t0 = time.time()

    agent_history = _agent_history(request)

    # Resolve uploaded-document chunks up front so an expired upload fails the
    # request cleanly instead of silently answering without the document.
    upload_chunks: list[dict] = []
    if request.upload_ids:
        upload_chunks, missing = _resolve_uploads(request.upload_ids)
        if missing:
            yield _sse({"type": "error", "code": "upload_expired",
                        "message": "The uploaded document has expired "
                                   "(uploads are kept for 1 hour). Please "
                                   "re-attach the file and ask again."})
            yield _sse({"type": "done"})
            return

    # Emit the first pipeline step UP FRONT so the progress bar appears the
    # instant the user submits — instead of only after the first node finishes.
    # This matters most for slow/rate-limited first calls (e.g. a free-tier
    # Gemini key), where the planner LLM call can take seconds or fail: without
    # this the user sees a lone "…" with no sign the run started.
    yield _sse({"type": "status", "stage": "planner",
                "label": _STEP_LABELS["planner"],
                "index": 0, "total": len(PIPELINE_ORDER)})

    # Bridge the (synchronous, thread-pool) graph run to this async generator:
    # the graph pushes node names onto a thread-safe queue as it runs, and we
    # drain them here into live "status" events so the UI shows real progress
    # instead of a single static spinner.
    loop = asyncio.get_event_loop()
    step_queue: asyncio.Queue = asyncio.Queue()

    def _on_step(node: str) -> None:
        # Fired when a node STARTS — drives the spinner's current-activity label.
        label = _STEP_LABELS.get(node)
        if label:
            loop.call_soon_threadsafe(
                step_queue.put_nowait,
                {"type": "status", "stage": node, "label": label,
                 # Position in the canonical pipeline + total, so the UI can show
                 # a real progress bar / ETA instead of a static spinner.
                 "index": PIPELINE_ORDER.index(node), "total": len(PIPELINE_ORDER)},
            )

    def _on_step_done(node: str, detail) -> None:
        # Fired when a node FINISHES, with a short outcome ("12 passages",
        # "2 exact figures") — the UI checks the step off and shows the detail.
        if node in _STEP_LABELS:
            loop.call_soon_threadsafe(
                step_queue.put_nowait,
                {"type": "step_done", "stage": node, "detail": detail},
            )

    rag_task = asyncio.ensure_future(_run_rag(
        request, agent_history, extra_chunks=upload_chunks or None,
        on_step=_on_step, on_step_done=_on_step_done))

    try:
        while not (rag_task.done() and step_queue.empty()):
            try:
                evt = await asyncio.wait_for(step_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            yield _sse(evt)
        result = rag_task.result()
    except Exception as e:
        err = _classify_provider_error(e, request.provider_config)
        yield _sse(err)
        yield _sse({"type": "done"})
        return

    answer = result.get("answer") or ""
    chunks = result.get("chunks") or []
    charts = result.get("charts") or []
    meta = result.get("metadata") or {}

    yield _sse({"type": "sources", "chunks": chunks, "metadata": meta})

    # Charts go on their own channel so the UI attaches them to the message.
    for chart in charts:
        yield _sse({"type": "chart", "chart": chart})

    # Word-piece pseudo-stream — gives the UI the live feel without a deep
    # token-streaming refactor on the LLM side. Pacing is capped so the
    # artificial tail never adds more than ~1.5s on top of the real latency
    # (3 words / 25ms used to add ~5s to a long answer).
    for piece in _piecewise(answer, words_per_chunk=6):
        yield _sse({"type": "chunk", "content": piece})
        await asyncio.sleep(0.012)

    latency = round(time.time() - t0, 3)
    meta["latency"] = latency

    yield _sse({
        "type": "metrics", "latency": latency,
        "model": meta.get("model"),
        "input_tokens": meta.get("input_tokens"),
        "output_tokens": meta.get("output_tokens"),
        "agentic": meta,
    })

    yield _sse({"type": "done"})


def _piecewise(text: str, words_per_chunk: int = 3) -> list[str]:
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


def _validate_provider(request) -> None:
    """Fail a missing/unknown provider key at the boundary.

    The agent used to validate this in its constructor; now that it is
    provider-agnostic the check belongs here — and this is the better place
    anyway, since a bad key returns a clean 400 instead of dying mid-graph.
    """
    from finagent.llm import resolve_api_key

    pc = request.provider_config
    try:
        resolve_api_key(pc.provider if pc else "groq", pc.api_key if pc else None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/api/query")
async def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Empty question.")
    _validate_provider(request)
    return StreamingResponse(
        _stream_answer(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------- #
# Deep Research — Server-Sent Events
#
# An independent execution path next to /api/query: the orchestrator scopes
# the request, runs specialist research tasks through the SAME production
# agent (rag_service.run_agentic, injected), and writes a cited investment
# report. Streams research_plan / agent_start / agent_done progress events,
# then the standard sources / chunk / metrics / done frames — so the
# citations panel and answer streaming reuse the chat pipeline unchanged.
# --------------------------------------------------------------------------- #

async def _stream_research(request: ResearchRequest) -> AsyncGenerator[str, None]:
    t0 = time.time()

    # ResearchRequest carries the same chat_history field, so memory behaves
    # exactly like chat.
    agent_history = _agent_history(request)

    loop = asyncio.get_event_loop()
    events: asyncio.Queue = asyncio.Queue()

    def _on_event(evt: dict) -> None:
        loop.call_soon_threadsafe(events.put_nowait, evt)

    pc = request.provider_config
    provider = (pc.provider if pc else "groq")
    synth_model = (pc.synth_model if pc else None)
    api_key = (pc.api_key if pc else None)

    from finagent.research import DeepResearch

    def _run() -> dict:
        research = DeepResearch(
            # Every specialist task runs through the production agent; the
            # orchestrator itself never talks to retrieval or tools directly.
            run_fn=lambda q: rag_service.run_agentic(
                q, provider=provider, synth_model=synth_model, api_key=api_key,
                session_id=request.session_id or None),
            provider=provider, model=synth_model, api_key=api_key,
            max_agents=request.max_agents,
            session_id=request.session_id or None,
        )
        return research.run(request.question, chat_history=agent_history,
                            on_event=_on_event)

    task = asyncio.ensure_future(loop.run_in_executor(_executor, _run))
    try:
        while not (task.done() and events.empty()):
            try:
                evt = await asyncio.wait_for(events.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            yield _sse(evt)
        result = task.result()
    except Exception as e:
        err = _classify_provider_error(e, request.provider_config)
        yield _sse(err)
        yield _sse({"type": "done"})
        return

    report = result.get("report") or ""
    chunks = result.get("chunks") or []
    meta = result.get("metadata") or {}

    yield _sse({"type": "sources", "chunks": chunks, "metadata": meta})

    for piece in _piecewise(report, words_per_chunk=8):
        yield _sse({"type": "chunk", "content": piece})
        await asyncio.sleep(0.008)

    latency = round(time.time() - t0, 3)
    meta["latency"] = latency
    yield _sse({"type": "metrics", "latency": latency,
                "model": meta.get("model"),
                "input_tokens": meta.get("input_tokens"),
                "output_tokens": meta.get("output_tokens"),
                "agentic": meta})

    yield _sse({"type": "done"})


@app.post("/api/research")
async def research(request: ResearchRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Empty question.")
    _validate_provider(request)
    return StreamingResponse(
        _stream_research(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------- #
# Static SPA hosting
#
# When `STATIC_DIR` exists, FastAPI serves the built React app from it. The
# Dockerfile builds the Vite SPA in stage 1 and copies its dist/ into this
# location, so a single Hugging Face Space serves both the API and the UI
# at the same origin (no CORS).
# --------------------------------------------------------------------------- #

_STATIC_DIR = Path(os.getenv("STATIC_DIR", "static")).resolve()

if _STATIC_DIR.exists() and (_STATIC_DIR / "index.html").exists():
    # Serve hashed JS/CSS chunks under their real paths.
    app.mount(
        "/assets",
        StaticFiles(directory=str(_STATIC_DIR / "assets")),
        name="spa-assets",
    )

    @app.get("/", include_in_schema=False)
    def _spa_root():
        return FileResponse(_STATIC_DIR / "index.html")

    # SPA fallback — any non-/api path resolves to index.html so client-side
    # routing works if we ever add it (current app uses zustand, no router).
    @app.get("/{full_path:path}", include_in_schema=False)
    def _spa_fallback(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404)
        # Confine the read to STATIC_DIR. `full_path` is percent-DECODED after
        # routing, so "..%2f" and "%2e%2e" arrive here as "../" — Starlette's
        # raw-path normalisation never saw them. Before this guard,
        # GET /..%2f..%2fproc/self/environ served the container's environment,
        # i.e. every Secret Manager value on the service.
        candidate = (_STATIC_DIR / full_path).resolve()
        if candidate.is_relative_to(_STATIC_DIR) and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_STATIC_DIR / "index.html")
