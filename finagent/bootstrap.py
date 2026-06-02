"""
bootstrap.py · finagent/bootstrap.py

Boot-time data hydration.

The prebuilt Chroma vector store is too large to live in the Space's git repo
(HF caps Space repo storage at 1 GB). Instead it lives in a separate **HF
Dataset** repo, and we download it into the persistent `/data` mount on first
boot. Because `/data` survives restarts, the download happens once.

Run as a module before starting the server (see scripts/entrypoint.sh):

    python -m finagent.bootstrap

Environment
-----------
    CHROMA_DIR          where the vector store should end up (default data/chroma)
    FINAGENT_DATA_REPO  the HF Dataset repo id (default Sarthak004/finagent-data)
    HF_TOKEN /          token for a PRIVATE dataset (public datasets need none)
    HUGGING_FACE_TOKEN
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

DEFAULT_DATA_REPO = "Sarthak004/finagent-data"
# Folder inside the dataset that holds the Chroma store (chroma.sqlite3 + segments).
CHROMA_PREFIX = "chroma"


def _token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_TOKEN") or None


def ensure_chroma() -> Path:
    """Make sure the Chroma store exists at $CHROMA_DIR, downloading it from the
    HF Dataset repo if it isn't there yet. Returns the resolved directory."""
    chroma_dir = Path(os.getenv("CHROMA_DIR", "data/chroma"))

    # Already hydrated (local dev, or a warm restart on a persistent mount).
    if (chroma_dir / "chroma.sqlite3").exists():
        print(f"[bootstrap] Chroma present at {chroma_dir} — skipping download.")
        return chroma_dir

    repo = os.getenv("FINAGENT_DATA_REPO", DEFAULT_DATA_REPO)
    print(f"[bootstrap] Hydrating Chroma from dataset '{repo}' → {chroma_dir} …")

    from huggingface_hub import snapshot_download

    staging = chroma_dir.parent / "_hf_staging"
    shutil.rmtree(staging, ignore_errors=True)
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=[f"{CHROMA_PREFIX}/**"],
        local_dir=str(staging),
        token=_token(),
    )

    src = staging / CHROMA_PREFIX
    if not (src / "chroma.sqlite3").exists():
        raise RuntimeError(
            f"Downloaded dataset '{repo}' has no '{CHROMA_PREFIX}/chroma.sqlite3'. "
            f"Upload it first with scripts/upload_data_to_hf.py."
        )

    chroma_dir.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        dest = chroma_dir / item.name
        if dest.exists():
            shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
        shutil.move(str(item), str(dest))
    shutil.rmtree(staging, ignore_errors=True)

    size = sum(f.stat().st_size for f in chroma_dir.rglob("*") if f.is_file())
    print(f"[bootstrap] Chroma ready at {chroma_dir} ({size / 1e6:.0f} MB).")
    return chroma_dir


if __name__ == "__main__":
    ensure_chroma()
