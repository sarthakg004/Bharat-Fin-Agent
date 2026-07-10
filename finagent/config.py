"""
config.py  ·  finagent/config.py

Centralised application settings. **All environment variables are read here and
nowhere else** — every other module imports the `settings` instance instead of
calling `os.getenv()` directly. Variable *names* are unchanged from what Cloud
Run / `.env` already provide (`GROQ_API_KEY`, `STATELESS`, `ALLOWED_ORIGINS`,
`RERANKER_MODEL`, `CHROMA_DIR`, …); only the access point is centralised.

Implementation note: this uses `pydantic.BaseModel` (already a dependency) rather
than `pydantic_settings.BaseSettings` to avoid adding a new requirement — the
field loading is done explicitly in `Settings.from_env()`. `.env` is loaded via
`python-dotenv` (already used across the codebase) before fields are read.
"""

from __future__ import annotations

import os
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()  # make .env visible no matter which module imports settings first


def _as_bool(raw: Optional[str], default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _as_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


class Settings(BaseModel):
    """Typed view of the runtime environment. Construct via `Settings.from_env()`.

    Fields map 1:1 to environment variables (names in the comments). Defaults
    match the values baked into the Dockerfile / current behaviour so importing
    settings never changes how the app runs.
    """

    # --- LLM / provider keys (GROQ_API_KEY etc.). Multi-key rotation
    #     (GROQ_API_KEY2, …) is handled dynamically in finagent.llm. ----------
    groq_api_key: str = Field(default="", description="GROQ_API_KEY")
    gemini_api_key: str = Field(default="", description="GEMINI_API_KEY")
    openai_api_key: str = Field(default="", description="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", description="ANTHROPIC_API_KEY")
    tavily_api_key: str = Field(default="", description="TAVILY_API_KEY")

    # --- Runtime behaviour ---------------------------------------------------
    stateless: bool = Field(default=False, description="STATELESS")
    allowed_origins: List[str] = Field(default_factory=list, description="ALLOWED_ORIGINS (CSV)")
    force_ipv4: bool = Field(default=False, description="FORCE_IPV4")
    # Dynamic SEC fetch: persist the fetched filing into the on-disk corpus
    # (True, good locally — a self-expanding index) or use it ephemerally in
    # memory for this session only (False, for Cloud Run — no index growth, no
    # persistence, no scale-to-zero problem). Set PERSIST_DYNAMIC_FETCH=false on
    # the cloud.
    persist_dynamic_fetch: bool = Field(default=True, description="PERSIST_DYNAMIC_FETCH")

    # --- Deep Research mode ---------------------------------------------------
    # Max specialist agents per research run (bounds latency + LLM quota) and
    # how many run concurrently. Parallel stays 1 by default: the serve path
    # pins graph runs to one worker (Chroma/hnswlib + reranker thread-safety).
    research_max_agents: int = Field(default=8, description="RESEARCH_MAX_AGENTS")
    research_parallel_agents: int = Field(default=1, description="RESEARCH_PARALLEL_AGENTS")

    # --- Models --------------------------------------------------------------
    reranker_model: str = Field(default="BAAI/bge-reranker-base", description="RERANKER_MODEL")
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", description="embedding model")

    # --- Storage / collections ----------------------------------------------
    chroma_dir: str = Field(default="data/chroma", description="CHROMA_DIR")
    static_dir: str = Field(default="static", description="STATIC_DIR")
    us_collection: str = Field(default="us_filings", description="US filings collection")
    financebench_collection: str = Field(default="financebench_eval", description="eval collection")

    @classmethod
    def from_env(cls) -> "Settings":
        """Build a Settings instance from the current process environment."""
        return cls(
            groq_api_key=(os.getenv("GROQ_API_KEY") or "").strip(),
            gemini_api_key=(os.getenv("GEMINI_API_KEY") or "").strip(),
            openai_api_key=(os.getenv("OPENAI_API_KEY") or "").strip(),
            anthropic_api_key=(os.getenv("ANTHROPIC_API_KEY") or "").strip(),
            tavily_api_key=(os.getenv("TAVILY_API_KEY") or "").strip(),
            stateless=_as_bool(os.getenv("STATELESS")),
            allowed_origins=_as_list(os.getenv("ALLOWED_ORIGINS")),
            force_ipv4=_as_bool(os.getenv("FORCE_IPV4")),
            persist_dynamic_fetch=_as_bool(os.getenv("PERSIST_DYNAMIC_FETCH"), default=True),
            research_max_agents=int(os.getenv("RESEARCH_MAX_AGENTS", "8")),
            research_parallel_agents=int(os.getenv("RESEARCH_PARALLEL_AGENTS", "1")),
            reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"),
            chroma_dir=os.getenv("CHROMA_DIR", "data/chroma"),
            static_dir=os.getenv("STATIC_DIR", "static"),
            us_collection=os.getenv("US_COLLECTION", "us_filings"),
            financebench_collection=os.getenv("FINANCEBENCH_COLLECTION", "financebench_eval"),
        )


# One module-level instance, imported everywhere as `from finagent.config import settings`.
settings = Settings.from_env()

__all__ = ["Settings", "settings"]
