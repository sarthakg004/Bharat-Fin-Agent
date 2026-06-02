"""
device.py  ·  finagent/device.py

Single source of truth for which compute device the embedding + reranker
models run on. Uses the GPU when one is available (CUDA, or Apple-silicon
MPS), and falls back to CPU otherwise — so the same code runs unchanged on a
laptop, a CPU-only Hugging Face Space, or a GPU box.

Override with FINAGENT_DEVICE=cpu|cuda|mps if you ever need to pin it.
"""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def get_device() -> str:
    forced = os.getenv("FINAGENT_DEVICE")
    if forced:
        return forced.strip().lower()
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        # Apple silicon
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
