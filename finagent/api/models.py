"""Pydantic request/response models."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------------- #

class ProviderConfig(BaseModel):
    """Optional per-request override of which LLM the agent uses.

    The frontend Settings modal collects these from the user (with the API
    key stored in their browser's localStorage and posted per-request — never
    persisted server-side). When absent, the agent uses the server-default
    provider (Groq) with the keys baked into the Space's environment.
    """
    provider: Literal["groq", "gemini", "openai", "anthropic"] = "groq"
    synth_model: Optional[str] = Field(
        default=None,
        description="Model name for the synthesizer / critic. None → provider default.",
    )
    planner_model: Optional[str] = Field(
        default=None,
        description="Model name for the planner (question decomposition). "
                    "Separate from the synthesizer because the two jobs differ: "
                    "the planner emits structured retrieval keys, the "
                    "synthesizer writes prose. None → provider default.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Caller-supplied API key. None → fall back to server env var.",
    )


class ChatTurn(BaseModel):
    """One prior turn of conversation. The server keeps no chat store, so the
    client owns the thread and replays recent turns with each request.

    `content` is bounded only as a DoS guard, NOT to the length the agent
    actually uses. The client replays whole assistant answers verbatim, and a
    Deep Research report runs to many thousands of characters — capping this
    near the agent's own budget would 422 a legitimate conversation. The real
    trimming happens server-side and twice: `_agent_history` keeps 1200 chars
    per turn, and the planner prompt cuts again to 400.
    """
    role: Literal["user", "assistant"]
    content: str = Field(max_length=32_000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Opaque client-owned thread id. Not persisted — used only to "
                    "group this run's traces with the rest of the conversation.",
    )
    provider_config: Optional[ProviderConfig] = None
    chat_history: Optional[list[ChatTurn]] = Field(
        default=None,
        max_length=50,
        description="Recent turns for conversation memory; the client owns the thread. "
                    "The server keeps only the last 6 — the cap is a payload guard.",
    )
    upload_ids: Optional[list[str]] = Field(
        default=None,
        description="Ids returned by POST /api/upload. The referenced documents' "
                    "chunks are ranked against this question in memory.",
    )


class ResearchRequest(BaseModel):
    """Deep Research Mode — a multi-specialist research run (POST /api/research).

    Same provider/memory contract as QueryRequest; `upload_ids` is not
    supported (research runs over filings + live sources, not attached files).
    """
    question: str = Field(min_length=1, max_length=2000)
    session_id: Optional[str] = Field(default=None, max_length=64)
    provider_config: Optional[ProviderConfig] = None
    chat_history: Optional[list[ChatTurn]] = Field(
        default=None,
        max_length=50,
        description="Recent turns, so follow-ups like 'now research it deeply' "
                    "resolve the company under discussion.",
    )
    max_agents: Optional[int] = Field(
        default=None, ge=3, le=14,
        description="Cap on specialist agents this run. None → server default.",
    )


class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    pages: int
    tables: int
    chunks: int


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #

class HealthResponse(BaseModel):
    status: str
