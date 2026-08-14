"""The FinanceBench Gemini index is a ~6-day metered build, so "resume" is the
feature, not a nicety. Two things have to hold or a day's quota goes missing
without anything looking broken:

  * a batch that came back before the daily wall must be ON DISK, so tomorrow
    reads it out of the cache instead of buying it again;
  * a filing must be all-or-nothing in Qdrant, because the resumed build skips
    filings by `source_url` and a half-written one would be skipped forever.

Both failures are silent — the build "finishes", just short.
"""
from __future__ import annotations

import pytest

from finagent.vectorstore import EmbeddingQuotaExhausted, GeminiEmbeddings


def _embedder(tmp_path, monkeypatch, dim=4) -> GeminiEmbeddings:
    monkeypatch.setenv("FINAGENT_EMBED_CACHE_DIR", str(tmp_path))
    emb = GeminiEmbeddings.__new__(GeminiEmbeddings)
    emb.model, emb.dim, emb.batch_size = "test-embed", dim, 2
    emb.keys, emb._dead, emb._local = ["k"], {}, None
    return emb


def test_batches_bought_before_the_wall_survive_it(tmp_path, monkeypatch):
    """Committing the whole call at the end threw away every batch that had
    already returned when a later one hit the daily cap."""
    emb = _embedder(tmp_path, monkeypatch)
    texts = ["a", "b", "c", "d", "e", "f"]          # 3 batches of 2
    seen: list[list[str]] = []

    def fake_batch(batch, task, key_idx):
        seen.append(list(batch))
        if "e" in batch:                            # the wall, third batch
            raise EmbeddingQuotaExhausted("daily cap")
        return [[float(len(seen))] * emb.dim for _ in batch]

    monkeypatch.setattr(emb, "_embed_batch", fake_batch)
    monkeypatch.setattr("finagent.vectorstore.GEMINI_EMBED_CONCURRENCY", 1)

    with pytest.raises(EmbeddingQuotaExhausted):
        emb.embed_documents(texts)

    rows = emb._conn().execute("SELECT count(*) FROM vec").fetchone()[0]
    assert rows == 4, "the two batches that returned were not persisted"

    # Tomorrow: the cached four cost nothing and only the rest is re-requested.
    seen.clear()
    monkeypatch.setattr(
        emb, "_embed_batch",
        lambda batch, task, key_idx: (seen.append(list(batch))
                                      or [[9.0] * emb.dim for _ in batch]))
    got = emb.embed_documents(texts)
    assert seen == [["e", "f"]]
    assert len(got) == 6 and all(len(v) == emb.dim for v in got)


def test_a_filing_is_never_half_indexed(tmp_path, monkeypatch):
    """`add_documents` embeds and upserts one batch at a time, so a quota death
    mid-file used to leave the filing partly in Qdrant. Its source_url was then
    present, and the next day's `skip_if_already_indexed` pass skipped it — the
    filing stayed truncated for the life of the index.

    The whole file is embedded first, so the raise lands before any upsert.
    """
    from langchain_core.documents import Document

    from finagent.ingestion import ingest as ingest_mod

    ing = ingest_mod.CorpusIngester.__new__(ingest_mod.CorpusIngester)
    ing.embedding_model = "gemini-embedding-2"
    ing.parent_doc = True
    ing.market = "us"
    docs = [Document(page_content=f"chunk {i}", metadata={"source_url": "u"})
            for i in range(5)]
    monkeypatch.setattr(ing, "_documents_for", lambda *a, **k: docs)
    monkeypatch.setattr(ing, "_upsert_batch", lambda: 2)

    upserted: list = []

    class _Store:
        def add_documents(self, docs, ids, batch_size):
            upserted.extend(ids)

    monkeypatch.setattr(ing, "_get_vector_store", lambda: _Store())

    class _Broke:
        def embed_documents(self, texts):
            assert len(texts) == len(docs), "must pre-warm the WHOLE file"
            raise EmbeddingQuotaExhausted("daily cap")

    monkeypatch.setattr(ingest_mod, "get_embeddings", lambda m: _Broke(),
                        raising=False)
    monkeypatch.setattr("finagent.vectorstore.get_embeddings",
                        lambda m: _Broke())

    with pytest.raises(EmbeddingQuotaExhausted):
        ing.ingest_file(tmp_path / "f.htm", {"source_url": "u"})
    assert upserted == [], "points were written for a filing that never finished"


def test_local_embedders_are_not_embedded_twice(tmp_path, monkeypatch):
    """The pre-warm is quota insurance. A local encoder has no disk cache, so
    running it would just double the GPU work for every ingest."""
    from langchain_core.documents import Document

    from finagent.ingestion import ingest as ingest_mod

    ing = ingest_mod.CorpusIngester.__new__(ingest_mod.CorpusIngester)
    ing.embedding_model = "BAAI/bge-large-en-v1.5"
    ing.parent_doc = True
    ing.market = "us"
    docs = [Document(page_content="c", metadata={"source_url": "u"})]
    monkeypatch.setattr(ing, "_documents_for", lambda *a, **k: docs)
    monkeypatch.setattr(ing, "_upsert_batch", lambda: 64)
    monkeypatch.setattr(ing, "_get_vector_store",
                        lambda: type("S", (), {"add_documents":
                                               lambda s, docs, ids, batch_size: None})())

    def _boom(model):
        raise AssertionError("pre-warmed a local embedder")

    monkeypatch.setattr("finagent.vectorstore.get_embeddings", _boom)
    assert ing.ingest_file(tmp_path / "f.htm", {"source_url": "u"}) == 1


def test_resume_build_keeps_the_collection(monkeypatch):
    """`--resume-build` must not reset: the collection is days of paid quota."""
    from finagent.evaluation import evaluate_retrieval as ev

    calls = {"reset": 0, "skip": None}

    class _Ing:
        CHILD_CHUNK_SIZE = CHILD_CHUNK_OVERLAP = 0

        def __init__(self, **kw):
            pass

        def reset_collection(self):
            calls["reset"] += 1

        def ingest_all(self, manifest_path, skip_if_already_indexed):
            calls["skip"] = skip_if_already_indexed
            return type("S", (), {"total_chunks": 3, "files_processed": 1})()

    monkeypatch.setattr("finagent.ingestion.ingest.CorpusIngester", _Ing)
    monkeypatch.setattr("finagent.vectorstore.count", lambda c: 44544)

    out = ev.build_index(ev.Config(), "sweep_x", "m.json", resume=True)
    assert calls == {"reset": 0, "skip": True}
    # The row must report the whole collection, not just today's slice.
    assert out["chunks"] == 44544

    ev.build_index(ev.Config(), "sweep_x", "m.json")
    assert calls == {"reset": 1, "skip": False}


def test_the_cohere_arm_cannot_finish_as_bge(monkeypatch):
    """Production degrades a spent Cohere pool to bge-reranker-v2-m3. In an A/B
    against bge that turns the cohere row into a silent blend of both, so the
    sweep pins the bare adapter and lets an exhausted pool raise."""
    from finagent.evaluation import evaluate_retrieval as ev
    from finagent.retrieval import reranker as rr

    monkeypatch.setattr(rr, "_SHARED_RERANKERS", {})
    monkeypatch.setattr(rr.CohereReranker, "__init__",
                        lambda self, model: setattr(self, "model", model))

    ev.pin_measurable_rerankers(("BAAI/bge-reranker-v2-m3",
                                 "cohere:rerank-v4.0-pro", None))

    pinned = rr._SHARED_RERANKERS["cohere:rerank-v4.0-pro"]
    assert isinstance(pinned, rr.CohereReranker)
    assert not isinstance(pinned, rr.FallbackReranker)
    assert pinned.model == "rerank-v4.0-pro"          # prefix stripped
    # The local arms are untouched: pinning must not preload a cross-encoder.
    assert set(rr._SHARED_RERANKERS) == {"cohere:rerank-v4.0-pro"}
    # And the pin is what `cap_pool` will resolve.
    assert rr._get_shared_reranker("cohere:rerank-v4.0-pro") is pinned
