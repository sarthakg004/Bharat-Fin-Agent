"""Request and response shapes for the API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Provider = Literal["groq", "gemini", "openai", "anthropic"]


class WriterConfig(BaseModel):
    """The user's choice of writer model. Everything else is fixed server-side.

    `api_key` is the user's own key. It is kept in their browser, sent with each
    request, and never stored. OpenAI and Anthropic always need one.
    """
    provider: Optional[Provider] = None
    model: Optional[str] = Field(default=None, max_length=100)
    api_key: Optional[str] = Field(default=None, max_length=300)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=32_000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # The client's thread id. Not stored; it only groups the traces of one chat.
    session_id: Optional[str] = Field(default=None, max_length=64)
    writer: Optional[WriterConfig] = None
    # The server keeps no chat history. The client replays recent turns.
    chat_history: Optional[list[ChatTurn]] = Field(default=None, max_length=50)


class HealthResponse(BaseModel):
    status: str
