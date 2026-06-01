"""Pydantic request/response models."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Market = Literal["us", "india"]


# --------------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------------- #

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    market: Market = "us"
    chat_id: Optional[int] = Field(
        default=None,
        description="Existing chat to append to. If null, the server creates one.",
    )
    top_k: int = Field(default=5, ge=1, le=20)


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
    market: str
    created_at: str
    updated_at: str
    message_count: int = 0
    preview: Optional[str] = None


class ChatListResponse(BaseModel):
    chats: list[ChatSummary]


class CreateChatRequest(BaseModel):
    title: str = "New chat"
    market: Market = "us"


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
