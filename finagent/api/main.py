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
    POST   /api/query                # SSE stream — appends to a chat

Streaming events on /api/query: status, sources, chart, chunk, metrics, done.
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

from fastapi import FastAPI, HTTPException
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
        id=d["id"], title=d["title"], market=d["market"],
        created_at=str(d["created_at"]), updated_at=str(d["updated_at"]),
        message_count=int(d.get("message_count", 0)),
        preview=(d.get("preview") or None),
    )


@app.get("/api/chats", response_model=ChatListResponse)
def list_chats():
    return ChatListResponse(chats=[_to_summary(c) for c in history.list_chats(200)])


@app.post("/api/chats", response_model=ChatSummary)
def create_chat(req: CreateChatRequest):
    chat = history.create_chat(title=req.title, market=req.market)
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


async def _run_rag(request: QueryRequest, hist: list[dict],
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
            request.market, request.question, request.top_k, None, hist,
            provider, synth_model, api_key,
            on_step=on_step, on_step_done=on_step_done,
        ),
    )


async def _stream_answer(request: QueryRequest) -> AsyncGenerator[str, None]:
    t0 = time.time()

    # Resolve the conversation + memory. In stateless mode the client owns the
    # history (per-session, cleared on exit) and nothing is persisted; otherwise
    # we use the SQLite chat store.
    if STATELESS:
        chat_id = request.chat_id or 0
        agent_history = [t.model_dump() for t in (request.chat_history or [])][-6:]
    else:
        chat_id = request.chat_id
        if chat_id is None or not history.get_chat(chat_id):
            chat_id = history.create_chat(title="New chat", market=request.market)["id"]
        # Persist the user message first so any failure mid-stream still keeps it.
        history.add_message(chat_id, role="user", content=request.question)
        history.auto_title_if_default(chat_id, request.question)
        agent_history = _build_history_for_agent(chat_id)
    yield _sse({"type": "chat", "chat_id": chat_id})

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
        request, agent_history, on_step=_on_step, on_step_done=_on_step_done))

    try:
        while not (rag_task.done() and step_queue.empty()):
            try:
                evt = await asyncio.wait_for(step_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            yield _sse(evt)
        result = rag_task.result()
    except Exception as e:
        from finagent.llm import is_daily_quota_error, is_rate_limit_error

        # Log the error type for debugging, but NOT the full exception value —
        # provider auth errors can echo the API key into logs.
        print(f"[query error] {type(e).__name__}", flush=True)

        rate_limited = is_rate_limit_error(e)
        pc = request.provider_config
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
