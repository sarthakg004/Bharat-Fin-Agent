"""
FastAPI app — multi-chat agentic RAG.

The single agentic pipeline (`AgenticRAGv4`) drives every answer. Conversations
are organised into chat threads; the agent gets the last few turns of the
active thread as memory.

Endpoints
---------
    GET    /api/health
    GET    /api/configs              # legacy; returns just the agentic config
    GET    /api/chats                # list all chat threads
    POST   /api/chats                # create new chat
    GET    /api/chats/{id}           # chat metadata + messages
    PATCH  /api/chats/{id}           # rename
    DELETE /api/chats/{id}           # delete chat + its messages
    DELETE /api/chats                # delete all chats
    POST   /api/upload               # parse a PDF/DOCX into ephemeral chunks
    POST   /api/query                # SSE stream — appends to a chat
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
# tokenizer / Chroma work happens off the main thread. The HuggingFace
# `tokenizers` Rust parallelism is not fork/thread-safe and can SIGSEGV; disable
# it before transformers is imported (it reads this at import time). Overridable.
# On a GPU box you may also want FINAGENT_DEVICE=cpu locally, since CUDA from a
# worker thread is fragile — production (Cloud Run) is CPU anyway.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from finagent.api import history, rag_service
from finagent.api.models import (
    ChatListResponse,
    ChatMessage,
    ChatMessagesResponse,
    ChatSummary,
    ConfigInfo,
    ConfigsResponse,
    CreateChatRequest,
    DeleteResponse,
    HealthResponse,
    QueryRequest,
    RenameChatRequest,
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

# The agent graph touches non-thread-safe native stacks: Chroma/hnswlib (a
# concurrent read while the dynamic-fetch ingest WRITES the same collection
# segfaults) and CUDA from worker threads. So run the graph on a SINGLE worker —
# requests queue rather than racing the native libs. The API stays async for I/O
# (DB, SSE); only the heavy RAG call is serialized here. Raise RAG_MAX_WORKERS
# once the ingest write-path is moved off the live collection (Phase 12).
_executor = ThreadPoolExecutor(
    max_workers=int(os.getenv("RAG_MAX_WORKERS", "1")), thread_name_prefix="rag"
)

# Stateless mode (set on scale-to-zero hosts like Cloud Run): no server-side
# chat store — the client supplies conversation memory via QueryRequest.chat_history
# and nothing is persisted to disk. Local dev leaves this off and keeps the
# SQLite multi-chat store + /api/chats endpoints.
STATELESS = os.getenv("STATELESS", "").strip().lower() in ("1", "true", "yes")


# --------------------------------------------------------------------------- #
# Health + configs
# --------------------------------------------------------------------------- #

@app.get("/api/health", response_model=HealthResponse)
def healthcheck():
    return HealthResponse(**{k: v for k, v in rag_service.health().items() if k != "configs"})


@app.get("/api/configs", response_model=ConfigsResponse)
def list_configs():
    return ConfigsResponse(configs=[
        ConfigInfo(
            id="agentic",
            label="Agentic RAG",
            model="openai/gpt-oss-120b",
            description=(
                "Planner → router → hybrid retrieve → grader → rewrite/synthesize → "
                "table agent → market tools → web search → critic → verifier. "
                "Loops on poor retrieval and unsupported claims; refuses if it can't ground."
            ),
        ),
    ])


# --------------------------------------------------------------------------- #
# Chats — list / create / read / rename / delete
# --------------------------------------------------------------------------- #

def _to_summary(d: dict) -> ChatSummary:
    return ChatSummary(
        id=d["id"], title=d["title"],
        created_at=str(d["created_at"]), updated_at=str(d["updated_at"]),
        message_count=int(d.get("message_count", 0)),
        preview=(d.get("preview") or None),
    )


@app.get("/api/chats", response_model=ChatListResponse)
def list_chats():
    return ChatListResponse(chats=[_to_summary(c) for c in history.list_chats(200)])


@app.post("/api/chats", response_model=ChatSummary)
def create_chat(req: CreateChatRequest):
    chat = history.create_chat(title=req.title)
    if not chat:
        raise HTTPException(status_code=500, detail="Could not create chat.")
    chat["message_count"] = 0
    chat["preview"] = None
    return _to_summary(chat)


@app.get("/api/chats/{chat_id}", response_model=ChatMessagesResponse)
def get_chat(chat_id: int):
    chat = history.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found.")
    msgs = history.list_messages(chat_id)
    # The summary endpoint computes preview + count; do the same inline.
    chat["message_count"] = len(msgs)
    chat["preview"] = next((m["content"] for m in msgs if m["role"] == "user"), None)
    return ChatMessagesResponse(
        chat=_to_summary(chat),
        messages=[
            ChatMessage(
                id=m["id"], chat_id=m["chat_id"], role=m["role"],
                content=m["content"], chunks=m["chunks"], charts=m["charts"],
                metadata=m["metadata"], latency=m["latency"],
                created_at=str(m["created_at"]),
            )
            for m in msgs
        ],
    )


@app.patch("/api/chats/{chat_id}", response_model=ChatSummary)
def rename_chat(chat_id: int, req: RenameChatRequest):
    ok = history.rename_chat(chat_id, req.title)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found.")
    return _to_summary(history.get_chat(chat_id) or {})


@app.delete("/api/chats/{chat_id}", response_model=DeleteResponse)
def delete_chat(chat_id: int):
    ok = history.delete_chat(chat_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found.")
    return DeleteResponse(deleted=1)


@app.delete("/api/chats", response_model=DeleteResponse)
def delete_all_chats():
    return DeleteResponse(deleted=history.delete_all_chats())


# --------------------------------------------------------------------------- #
# Query — Server-Sent Events
# --------------------------------------------------------------------------- #

def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str, ensure_ascii=False)}\n\n"


def _build_history_for_agent(chat_id: int) -> list[dict]:
    """The agent's `chat_history` parameter — last 6 turns, truncated."""
    msgs = history.recent_turns(chat_id, k=6)
    return [
        {"role": m["role"], "content": (m["content"] or "")[:1200]}
        for m in msgs
    ]


def _resolve_memory(request: QueryRequest) -> tuple[int, list[dict]]:
    """(chat_id, agent_history) for this query.

    The client's `chat_history` always wins when supplied — the SPA owns the
    thread (sessionStorage) and never sends `chat_id`, so deriving memory from
    the SQLite store built it from a brand-new chat every time (i.e. none).
    The SQLite store remains the persistence log + fallback for callers that
    do pass a `chat_id`.
    """
    client = [
        {"role": t.role, "content": (t.content or "")[:1200]}
        for t in (request.chat_history or [])
    ][-6:]
    if STATELESS:
        return request.chat_id or 0, client
    chat_id = request.chat_id
    if chat_id is None or not history.get_chat(chat_id):
        chat_id = history.create_chat(title="New chat")["id"]
    # Persist the user message first so any failure mid-stream still keeps it.
    history.add_message(chat_id, role="user", content=request.question)
    history.auto_title_if_default(chat_id, request.question)
    return chat_id, client or _build_history_for_agent(chat_id)


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

    from finagent.ingestion.upload import parse_upload

    # Same single-worker executor as the agent: Docling parsing is serialized
    # behind RAG runs instead of racing them for CPU/memory.
    loop = asyncio.get_event_loop()
    try:
        parsed = await loop.run_in_executor(
            _executor, lambda: parse_upload(data, file.filename or "upload.pdf"))
    except Exception as e:
        print(f"[upload error] {type(e).__name__}", flush=True)
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
            request.question, request.top_k, None, hist,
            provider, synth_model, api_key,
            session_id=str(request.chat_id) if request.chat_id else None,
            extra_chunks=extra_chunks,
            on_step=on_step, on_step_done=on_step_done,
        ),
    )


async def _stream_answer(request: QueryRequest) -> AsyncGenerator[str, None]:
    t0 = time.time()

    # Resolve the conversation + memory (client history wins in both modes).
    chat_id, agent_history = _resolve_memory(request)
    yield _sse({"type": "chat", "chat_id": chat_id})

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
        code, message = err["code"], err["message"]
        if not STATELESS:
            history.add_message(
                chat_id, role="assistant", content="",
                metadata={"error": message},
            )
        yield _sse({"type": "error", "code": code, "message": message})
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

    if not STATELESS:
        try:
            history.add_message(
                chat_id, role="assistant", content=answer,
                chunks=chunks, charts=charts, metadata=meta, latency=latency,
            )
        except Exception:
            pass

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


@app.post("/api/query")
async def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Empty question.")
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

    # ResearchRequest carries the same question/chat_id/chat_history fields
    # _resolve_memory reads, so memory + persistence behave exactly like chat.
    chat_id, agent_history = _resolve_memory(request)
    yield _sse({"type": "chat", "chat_id": chat_id})

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
                session_id=str(chat_id) if chat_id else None),
            provider=provider, model=synth_model, api_key=api_key,
            max_agents=request.max_agents,
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
        if not STATELESS:
            history.add_message(chat_id, role="assistant", content="",
                                metadata={"error": err["message"]})
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

    if not STATELESS:
        try:
            history.add_message(chat_id, role="assistant", content=report,
                                chunks=chunks, metadata=meta, latency=latency)
        except Exception:
            pass

    yield _sse({"type": "done"})


@app.post("/api/research")
async def research(request: ResearchRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Empty question.")
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

_STATIC_DIR = Path(os.getenv("STATIC_DIR", "static"))

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
        candidate = _STATIC_DIR / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_STATIC_DIR / "index.html")
