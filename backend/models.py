"""Pydantic request/response models."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


Market = Literal["us", "india"]
ConfigId = Literal["naive", "agentic"]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    config: ConfigId = "naive"
    market: Market = "us"
    company_filter: Optional[list[str]] = None
    top_k: int = Field(default=5, ge=1, le=20)


class HealthResponse(BaseModel):
    status: str
    collections: list[str]
    configs: list[str]


class ConfigInfo(BaseModel):
    id: str
    label: str
    model: str
    description: str


class ConfigsResponse(BaseModel):
    configs: list[ConfigInfo]


class HistoryItem(BaseModel):
    id: int
    question: str
    config: str
    market: str
    answer: str
    latency: float
    created_at: str


class HistoryItemFull(HistoryItem):
    """Single-item endpoint adds the chunks + run metadata so the UI can
    rehydrate a past chat without re-running the LLM."""
    chunks: list[dict] = []
    metadata: dict = {}


class HistoryResponse(BaseModel):
    items: list[HistoryItem]


class DeleteResponse(BaseModel):
    deleted: int
