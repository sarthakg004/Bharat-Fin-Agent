"""
config.py  ·  finagent/config.py

Centralised application settings. **All environment variables are read here and
nowhere else** — every other module imports the `settings` instance instead of
calling `os.getenv()` directly. Variable *names* are unchanged from what Cloud
Run / `.env` already provide (`GROQ_API_KEY`, `ALLOWED_ORIGINS`,
`RERANKER_MODEL`, `QDRANT_URL`, …); only the access point is centralised.

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

# Reranker the served path uses unless RERANKER_MODEL says otherwise. Cohere's
# hosted rerank beats the local cross-encoder and costs no CPU, and a spent key
# pool degrades to `retrieval.reranker.LOCAL_FALLBACK_RERANKER` rather than
# failing. Spelled out here rather than imported from `retrieval.reranker`,
# which would be an import cycle (that module imports `finagent.llm`) and would
# drag torch into every process that merely reads settings.
DEFAULT_RERANKER = "cohere:rerank-v4.0-pro"

# The served collection. Gemini-embedded at 1536 dims, so it cannot be queried
# by a local bge model; see `vectorstore.DEFAULT_EMBED_MODEL`. The default used
# to say `us_filings_v3` while the deploy forced v4, which meant anything run
# locally without US_COLLECTION set pointed at a collection that no longer
# existed. Default and deploy now agree.
DEFAULT_US_COLLECTION = "us_filings_v5_gemini"


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

    # --- LLM / provider keys -------------------------------------------------
    # Only Tavily's lives here. The chat/embedding providers are NOT mirrored
    # into Settings: `finagent.llm.collect_provider_keys` reads the environment
    # itself because it also has to find the numbered (GROQ_API_KEY1..N) and
    # consolidated (GROQ_API_KEYS) forms, which a single field cannot hold.
    tavily_api_key: str = Field(default="", description="TAVILY_API_KEY")

    # --- Runtime behaviour ---------------------------------------------------
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
    # pins graph runs to one worker (reranker/tokenizer thread-safety).
    research_max_agents: int = Field(default=8, description="RESEARCH_MAX_AGENTS")
    research_parallel_agents: int = Field(default=1, description="RESEARCH_PARALLEL_AGENTS")

    # --- Models --------------------------------------------------------------
    # A LOCAL model here must be one the image bakes (see Dockerfile): the
    # runtime pins HF_HUB_OFFLINE=1, so an unbaked value dies on the first
    # query, not at boot. An API-served model (`cohere:`) has nothing to bake,
    # but it still degrades to the baked local cross-encoder when its key pool
    # is spent, so that one is required either way.
    reranker_model: str = Field(default=DEFAULT_RERANKER, description="RERANKER_MODEL")

    # --- Storage / collections ----------------------------------------------
    qdrant_url: str = Field(default="", description="QDRANT_URL / QDRANT_CLUSTER_ENDPOINT")
    qdrant_api_key: str = Field(default="", description="QDRANT_API_KEY")
    static_dir: str = Field(default="static", description="STATIC_DIR")
    us_collection: str = Field(default=DEFAULT_US_COLLECTION, description="US filings collection")
    financebench_collection: str = Field(default="financebench_eval", description="eval collection")

    @classmethod
    def from_env(cls) -> "Settings":
        """Build a Settings instance from the current process environment."""
        return cls(
            tavily_api_key=(os.getenv("TAVILY_API_KEY") or "").strip(),
            allowed_origins=_as_list(os.getenv("ALLOWED_ORIGINS")),
            force_ipv4=_as_bool(os.getenv("FORCE_IPV4")),
            persist_dynamic_fetch=_as_bool(os.getenv("PERSIST_DYNAMIC_FETCH"), default=True),
            research_max_agents=int(os.getenv("RESEARCH_MAX_AGENTS", "8")),
            research_parallel_agents=int(os.getenv("RESEARCH_PARALLEL_AGENTS", "1")),
            reranker_model=os.getenv("RERANKER_MODEL", DEFAULT_RERANKER),
            qdrant_url=(os.getenv("QDRANT_URL")
                        or os.getenv("QDRANT_CLUSTER_ENDPOINT") or "").strip(),
            qdrant_api_key=(os.getenv("QDRANT_API_KEY") or "").strip(),
            static_dir=os.getenv("STATIC_DIR", "static"),
            us_collection=os.getenv("US_COLLECTION", DEFAULT_US_COLLECTION),
            financebench_collection=os.getenv("FINANCEBENCH_COLLECTION", "financebench_eval"),
        )


# One module-level instance, imported everywhere as `from finagent.config import settings`.
settings = Settings.from_env()

__all__ = ["Settings", "settings"]
