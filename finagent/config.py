"""Settings read from the environment, imported everywhere as `settings`.

API keys are not here: `finagent.llm.collect_provider_keys` reads them, because
a provider can have a whole pool of keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # The Qdrant collection users search. It was embedded with Gemini at 1536
    # dimensions, so it can only be queried with the same embedder.
    us_collection: str = os.getenv("US_COLLECTION", "us_filings_v5_gemini")
    # The collection the evaluation searches (kept on a local Qdrant).
    financebench_collection: str = os.getenv("FINANCEBENCH_COLLECTION", "financebench_eval")
    # "cohere:<model>" calls Cohere's API and falls back to the local
    # cross-encoder when Cohere is unavailable.
    reranker_model: str = os.getenv("RERANKER_MODEL", "cohere:rerank-v4.0-pro")


settings = Settings()
