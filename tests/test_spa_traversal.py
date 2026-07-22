"""The SPA catch-all route must not read outside STATIC_DIR.

Regression test for an unauthenticated arbitrary-file-read: FastAPI decodes
percent-escapes into a `{path:path}` param AFTER Starlette normalises the raw
path, so "..%2f" reached the handler as "../" and
`GET /..%2f..%2fproc/self/environ` returned the container's environment —
every Secret Manager value injected into the Cloud Run service.
"""

from __future__ import annotations

import importlib
import os
import pathlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<html>spa</html>")
    (static / "app.js").write_text("console.log(1)")
    (tmp_path / "SECRET.txt").write_text("gsk_live_key_material")

    monkeypatch.setenv("STATIC_DIR", str(static))
    import finagent.api.main as main
    importlib.reload(main)
    yield TestClient(main.app)
    # Leave the module in its default state for the rest of the suite.
    monkeypatch.delenv("STATIC_DIR", raising=False)
    importlib.reload(main)


ESCAPES = [
    "/../SECRET.txt",
    "/..%2fSECRET.txt",
    "/%2e%2e/SECRET.txt",
    "/%2e%2e%2fSECRET.txt",
    "/a/..%2f..%2fSECRET.txt",
    "/..%2f..%2f..%2f..%2fetc/passwd",
    "/..%2f..%2f..%2f..%2fproc/self/environ",
]


@pytest.mark.parametrize("path", ESCAPES)
def test_traversal_falls_back_to_index(client, path):
    """Escaping STATIC_DIR must serve index.html, never the target file."""
    r = client.get(path)
    assert r.status_code == 200
    assert r.text == "<html>spa</html>", f"{path} escaped STATIC_DIR"


def test_real_static_files_still_served(client):
    assert client.get("/app.js").text == "console.log(1)"
    assert client.get("/some/spa/route").text == "<html>spa</html>"


def test_api_paths_are_not_swallowed(client):
    assert client.get("/api/does-not-exist").status_code == 404
    assert client.get("/api/health").json() == {"status": "ok"}
