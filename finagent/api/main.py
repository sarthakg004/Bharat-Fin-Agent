"""The FastAPI app. Stateless: the client owns the chat and replays recent turns.

    GET  /api/health    liveness
    GET  /api/config    which model does which job, and what the picker may offer
    POST /api/query     the answer, streamed as Server-Sent Events:
                        status, step_done, sources, chart, chunk, metrics, done, error
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator

# Cloud Run has no IPv6 egress, and some API hosts resolve to IPv6 first.
if os.getenv("FORCE_IPV4", "").strip().lower() in ("1", "true", "yes"):
    import socket as _socket

    _getaddrinfo = _socket.getaddrinfo
    _socket.getaddrinfo = lambda host, port, family=0, *a, **kw: _getaddrinfo(
        host, port, _socket.AF_INET, *a, **kw)

# The local fallback reranker's tokenizer is not thread-safe when parallel.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from finagent.api import service
from finagent.api.models import HealthResponse, QueryRequest
from finagent.llm import (CHAT_PROVIDERS, PROVIDER_LABELS, ProviderError, _chain,
                          classify_error, collect_provider_keys, format_wait)
from finagent.runtime import ROLES, WRITER_MODELS, RuntimeContext

app = FastAPI(title="FinAgent API", version="3.0.0")

# The frontend is hosted on Firebase, a different origin. ALLOWED_ORIGINS lists it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   *[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]],
    allow_methods=["*"], allow_headers=["*"],
)

# One question at a time: the graph is synchronous and runs in this worker.
# ponytail: one worker, one instance; raise RAG_MAX_WORKERS if traffic ever needs it.
_executor = ThreadPoolExecutor(max_workers=int(os.getenv("RAG_MAX_WORKERS", "1")),
                               thread_name_prefix="rag")

# Label shown in the UI while each step runs, in pipeline order.
STEP_LABELS = {
    "planner":      "Planning the approach…",
    "fetch_filing": "Checking the filing index…",
    "retrieve":     "Searching the filings…",
    "xbrl":         "Looking up exact figures…",
    "calculator":   "Computing the metrics…",
    "market_data":  "Pulling market data…",
    "web_search":   "Searching the web…",
    "edgar_search": "Searching EDGAR across companies…",
    "synthesize":   "Writing the answer…",
    "critic":       "Fact-checking the draft…",
}
_STEPS = list(STEP_LABELS)


@app.get("/api/health", response_model=HealthResponse)
def health():
    """Liveness only. Does no work, so the frontend can poll it cheaply."""
    return HealthResponse(status="ok")


@app.get("/api/config")
def config():
    """The model table the frontend renders. Keeps model names in one place."""
    return {
        "roles": {role: {"provider": p, "model": m} for role, (p, m) in ROLES.items()},
        "writer_models": WRITER_MODELS,
        # Providers the server has keys for; the others need the user's own key.
        "server_keys": [p for p in CHAT_PROVIDERS if collect_provider_keys(p)],
    }


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

def error_event(e: Exception, own_key: bool = False) -> dict:
    """An exception as the SSE error the UI shows.

    `code` says what happened; `retryable` and `retry_after` tell the UI whether
    to retry by itself and when.
    """
    from finagent.vectorstore import EmbeddingQuotaExhausted

    # Log the type only: a provider's auth error can echo the API key.
    print(f"[query error] {type(e).__name__}", flush=True)

    if any(isinstance(c, EmbeddingQuotaExhausted) for c in _chain(e)):
        return {"type": "error", "code": "quota", "retryable": False,
                "message": "Today's free embedding quota is used up, so filings cannot "
                           "be searched or indexed right now. It resets at midnight "
                           "Pacific time."}

    info = classify_error(e)
    provider = next((c.provider for c in _chain(e) if isinstance(c, ProviderError)), None)
    label = PROVIDER_LABELS.get(provider, "The model provider")
    whose = "Your" if own_key else "The shared"
    wait = info.retry_after
    messages = {
        "rate_limit": f"{label} is rate limiting requests. "
                      + (f"Retrying in {format_wait(wait)}." if wait else "Retrying shortly."),
        "busy": f"{label} is busy right now. Retrying shortly.",
        "quota": f"{whose} {label} key has used up today's free quota. It resets at "
                 f"midnight Pacific time. You can also switch the writer model or add "
                 f"your own API key in the model picker.",
        "too_large": "This question needs more context than the model accepts in one "
                     "request. Start a new chat to drop the history, or pick another "
                     "writer model.",
        "auth": f"{whose} {label} API key was rejected. Check the key in the model picker.",
        "not_found": f"{label} no longer serves the selected model. Pick another "
                     f"writer model.",
    }
    if info.kind not in messages:
        return {"type": "error", "code": "error", "retryable": False,
                "message": f"{type(e).__name__}: {str(e)[:240]}"}
    event = {"type": "error", "code": info.kind, "retryable": info.retryable,
             "message": messages[info.kind]}
    if info.retryable:
        event["retry_after"] = round(wait if wait is not None else 5.0, 1)
    return event


def _check_writer(request: QueryRequest) -> RuntimeContext:
    """Reject a writer choice that cannot work, before the run starts."""
    w = request.writer
    ctx = RuntimeContext(provider=w.provider if w else None, model=w.model if w else None,
                         api_key=(w.api_key or "").strip() or None if w else None)
    provider, _, key = ctx.resolve("writer")
    if not key and not collect_provider_keys(provider):
        raise HTTPException(status_code=400, detail=(
            f"The server has no {PROVIDER_LABELS[provider]} API key. Paste your own "
            f"key in the model picker to use this model, or switch back to the default."))
    return ctx


# --------------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------------- #

def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str, ensure_ascii=False)}\n\n"


class ClientGone(Exception):
    """The browser disconnected. Raised from the progress callback, which runs
    as each step starts, so an abandoned run stops at the next step."""


async def _stream_answer(request: QueryRequest, ctx: RuntimeContext) -> AsyncGenerator[str, None]:
    t0 = time.time()
    history = [{"role": t.role, "content": (t.content or "")[:1200]}
               for t in (request.chat_history or [])][-6:]

    # Show the first step at once, before the planner's model call returns.
    yield _sse({"type": "status", "stage": "planner", "label": STEP_LABELS["planner"],
                "index": 0, "total": len(_STEPS)})

    # The graph runs in a worker thread and pushes progress onto this queue.
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    cancelled = threading.Event()

    def on_step(node: str) -> None:
        if cancelled.is_set():
            raise ClientGone()
        if node in STEP_LABELS:
            loop.call_soon_threadsafe(queue.put_nowait, {
                "type": "status", "stage": node, "label": STEP_LABELS[node],
                "index": _STEPS.index(node), "total": len(_STEPS)})

    def on_step_done(node: str, detail) -> None:
        if node in STEP_LABELS:
            loop.call_soon_threadsafe(queue.put_nowait, {
                "type": "step_done", "stage": node, "detail": detail})

    task = asyncio.ensure_future(loop.run_in_executor(_executor, lambda: service.run_agent(
        request.question, chat_history=history, provider=ctx.provider, model=ctx.model,
        api_key=ctx.api_key, session_id=request.session_id or None,
        on_step=on_step, on_step_done=on_step_done)))
    # Nobody awaits an abandoned task; read its result so asyncio stays quiet.
    task.add_done_callback(lambda f: f.cancelled() or f.exception())

    try:
        while not (task.done() and queue.empty()):
            try:
                yield _sse(await asyncio.wait_for(queue.get(), timeout=0.1))
            except asyncio.TimeoutError:
                continue
        result = task.result()
    except Exception as e:
        yield _sse(error_event(e, own_key=bool(ctx.api_key)))
        yield _sse({"type": "done"})
        return
    finally:
        # Also reached when the client disconnects: stop the run.
        if not task.done():
            cancelled.set()
            task.cancel()

    meta = result["metadata"]
    yield _sse({"type": "sources", "chunks": result["chunks"], "metadata": meta})
    for chart in result["charts"]:
        yield _sse({"type": "chart", "chart": chart})

    # The answer is complete by now; it is sent in small pieces so the UI can
    # show it arriving. ponytail: not real token streaming.
    words = result["answer"].split(" ")
    for i in range(0, len(words), 6):
        yield _sse({"type": "chunk", "content": " ".join(words[i:i + 6])
                    + (" " if i + 6 < len(words) else "")})
        await asyncio.sleep(0.012)

    yield _sse({"type": "metrics", **meta, "latency": round(time.time() - t0, 3)})
    yield _sse({"type": "done"})


@app.post("/api/query")
async def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Empty question.")
    ctx = _check_writer(request)
    return StreamingResponse(_stream_answer(request, ctx), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --------------------------------------------------------------------------- #
# The built frontend, when it is served from this container
# --------------------------------------------------------------------------- #

_STATIC_DIR = Path(os.getenv("STATIC_DIR", "static")).resolve()

if (_STATIC_DIR / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=str(_STATIC_DIR / "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    def _spa_root():
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    def _spa_fallback(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404)
        # SECURITY: `full_path` is percent-decoded, so "..%2f" arrives as "../".
        # Only serve files that resolve inside STATIC_DIR.
        candidate = (_STATIC_DIR / full_path).resolve()
        if candidate.is_relative_to(_STATIC_DIR) and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_STATIC_DIR / "index.html")
