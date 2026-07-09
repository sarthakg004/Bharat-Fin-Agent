"""Pydantic request/response models."""

from __future__ import annotations

from typing import Any, Literal, Optional

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
    api_key: Optional[str] = Field(
        default=None,
        description="Caller-supplied API key. None → fall back to server env var.",
    )


class ChatTurn(BaseModel):
    """One prior turn of conversation, supplied by the client for per-session
    memory when the server runs stateless (no server-side chat store)."""
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    chat_id: Optional[int] = Field(
        default=None,
        description="Existing chat to append to. If null, the server creates one.",
    )
    top_k: int = Field(default=5, ge=1, le=20)
    provider_config: Optional[ProviderConfig] = None
    chat_history: Optional[list[ChatTurn]] = Field(
        default=None,
        description="Recent turns for conversation memory. Used when the server "
                    "runs stateless (STATELESS=1); ignored otherwise.",
    )
    upload_ids: Optional[list[str]] = Field(
        default=None,
        description="Ids returned by POST /api/upload. The referenced documents' "
                    "chunks are ranked against this question in memory.",
    )


class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    pages: int
    tables: int
    chunks: int


# --------------------------------------------------------------------------- #
# Health + configs
# --------------------------------------------------------------------------- #

class HealthResponse(BaseModel):
    status: str
    collections: list[str]


class ConfigInfo(BaseModel):
    id: str
    label: str
    model: str
    description: str


class ConfigsResponse(BaseModel):
    configs: list[ConfigInfo]


# --------------------------------------------------------------------------- #
# Chats + messages
# --------------------------------------------------------------------------- #

class ChatSummary(BaseModel):
    """List-view chat row."""
    id: int
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0
    preview: Optional[str] = None


class ChatListResponse(BaseModel):
    chats: list[ChatSummary]


class CreateChatRequest(BaseModel):
    title: str = "New chat"


class RenameChatRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ChatMessage(BaseModel):
    id: int
    chat_id: int
    role: Literal["user", "assistant"]
    content: str
    chunks: list[dict] = []
    charts: list[dict] = []
    metadata: dict = {}
    latency: Optional[float] = None
    created_at: str


class ChatMessagesResponse(BaseModel):
    chat: ChatSummary
    messages: list[ChatMessage]


class DeleteResponse(BaseModel):
    deleted: int


class GenericOk(BaseModel):
    ok: bool = True
    detail: Optional[Any] = None
