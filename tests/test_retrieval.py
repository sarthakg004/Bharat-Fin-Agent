"""Retrieval: the search that is actually issued, the filter, and what reaches the writer.

No cluster and no models: the store, the retriever and the reranker are stubs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from finagent.agent import FinAgent
from finagent.agent import retrieve as R
from finagent.retrieval import reranker
from finagent.retrieval.expansion import expand_query
from finagent.retrieval.filters import infer_filter, parse_years
from finagent.retrieval.hybrid import HybridRetriever


class StubStore:
    collection_name = "stub"

    def __init__(self):
        self.calls = []

    def similarity_search(self, query, k=None, filter=None, **kw):
        self.calls.append(("similarity_search", k))
        return []

    def max_marginal_relevance_search(self, *a, **kw):
        self.calls.append(("mmr", None))
        return []


def test_the_pool_query_is_the_fused_one_at_the_configured_depth():
    """LangChain's MMR search is dense-only: using it silently drops BM25."""
    r = HybridRetriever(StubStore(), pool_top_k=48)
    r.get_pool("total operating expenses 2023")
    assert r.store.calls == [("similarity_search", 48)]


def test_children_of_one_parent_collapse_to_a_single_candidate():
    hits = [("child a", {"local_path": "f", "parent_id": 3, "parent_text": "PARENT"}),
            ("child b", {"local_path": "f", "parent_id": 3, "parent_text": "PARENT"}),
            ("orphan", {"local_path": "f", "element_index": 9})]
    assert [t for t, _ in HybridRetriever.collapse_to_parents(hits)] == ["PARENT", "orphan"]


def test_fiscal_years_are_parsed_in_both_forms():
    assert parse_years("AMD revenue FY22 vs FY2021") == [2022, 2021]
    assert parse_years("FY98 annual report") == [1998]
    assert parse_years("grew across 22 states in 2022") == [2022]      # a bare 22 is not a year


def test_filter_picks_the_company_and_the_right_years():
    vocab, years = {"servicenow": "ServiceNow"}, {"ServiceNow": {"2022", "2023", "2025", "2026"}}
    # A named year: that year and the next (the figure is often in the next filing).
    assert infer_filter("ServiceNow revenue by segment in FY2022", vocab, years) == {
        "companies": ["ServiceNow"], "years": ["2022", "2023"]}
    # No year but "year-over-year": the two newest, never a stale one.
    assert infer_filter("How did ServiceNow's margin change year-over-year?",
                        vocab, years)["years"] == ["2025", "2026"]
    assert infer_filter("What is the weather?", vocab, years) is None


def test_a_derived_metric_gains_its_line_items():
    assert "total current liabilities" in expand_query("Verizon quick ratio FY2022")
    assert expand_query("3M total revenue 2022") == "3M total revenue 2022"


class StubRetriever:
    """Returns three passages per query, tagged with the query that found them."""

    def __init__(self):
        self.searched = []

    def search(self, query, top_k=None):
        self.searched.append((query, top_k))
        return [(f"{query} passage {i}", {"local_path": query, "parent_id": i,
                                          "company": "ACME", "year": "2022"}) for i in range(3)]


def _agent(monkeypatch, scores=None):
    agent = FinAgent(collection="stub")
    agent._retriever = StubRetriever()
    rank = type("Rank", (), {"predict": staticmethod(
        lambda pairs: [(scores or {}).get(text, 0.0) for _, text in pairs])})()
    monkeypatch.setattr(reranker, "_shared", {agent.reranker_model: rank})
    return agent


def test_retrieval_searches_the_rewrite_and_the_question_not_the_sub_queries(monkeypatch):
    agent = _agent(monkeypatch)
    state = {"question": "Is AMD liquid?", "sub_queries": ["AMD quick ratio", "AMD cash"],
             "query_routes": ["numeric", "narrative"], "retrieval_query": "AMD balance sheet cash"}
    R.retrieve(agent, state)
    # The rewrite gets the whole 8-passage budget; a narrative question gets 8 too.
    assert agent._retriever.searched == [("AMD balance sheet cash", 8), ("Is AMD liquid?", 8)]

    agent._retriever.searched.clear()
    R.retrieve(agent, {**state, "query_routes": ["numeric", "numeric"]})
    assert agent._retriever.searched[-1] == ("Is AMD liquid?", 2)      # 2 slots otherwise


def test_a_critic_retry_searches_the_failed_claims(monkeypatch):
    agent = _agent(monkeypatch)
    R.retrieve(agent, {"question": "q", "sub_queries": ["s"], "query_routes": ["narrative"],
                       "retrieval_query": "the original rewrite",
                       "retry_queries": ["supporting evidence for: claim"]})
    assert [q for q, _ in agent._retriever.searched] == ["supporting evidence for: claim", "q"]


def test_the_cap_keeps_the_best_passage_of_every_query(monkeypatch):
    """On a comparison question one company must not crowd the other out."""
    scores = {f"A passage {i}": 9 - i for i in range(3)} | {f"B passage {i}": 1 - i for i in range(3)}
    agent = _agent(monkeypatch, scores)
    agent.retrieve_cap = 2
    chunks = [{"text": t, "sub_query": t[0]} for t in scores]
    kept = [c["text"] for c in R._cap_pool(agent, {"question": "q"}, chunks)]
    assert kept == ["A passage 0", "B passage 0"]


def test_filing_search_is_skipped_when_the_company_could_not_be_fetched(monkeypatch):
    """Searching anyway would return another company's filing."""
    agent = _agent(monkeypatch)
    out = R.retrieve(agent, {"question": "q", "sub_queries": ["q"], "query_routes": ["narrative"],
                             "fetch_status": {"decision": "fetch", "status": "error"}})
    assert out == {"retrieved_chunks": []} and agent._retriever.searched == []


def test_chunker_output_matches_the_recorded_digest():
    """Point ids are derived from chunk text, and the index in Qdrant was built
    with this chunking. A changed digest means the corpus must be re-embedded."""
    pytest.importorskip("unstructured")
    from finagent.ingestion.ingest import CorpusIngester
    from finagent.vectorstore import chunk_point_id

    path = Path(__file__).parent / "fixtures" / "mini_filing.htm"
    docs = CorpusIngester("unused", "gemini-embedding-2").documents(path, {
        "company": "ACME", "ticker": "ACME", "year": "2022",
        "source_url": "fixture://mini", "filing_type": "10-K"})
    h = hashlib.sha1()
    for d in docs:
        h.update(chunk_point_id(d.metadata, d.page_content).encode())
        h.update(d.page_content.encode())
        h.update((d.metadata.get("parent_text") or "").encode())
        h.update(str(d.metadata.get("item")).encode())
        h.update(str(d.metadata.get("element_type")).encode())
    assert (len(docs), h.hexdigest()) == (16, "8bfe1cf7e5be91af77ea635d73f2cfd66c8608e2")
    assert docs[0].page_content.startswith("ACME 2022")        # the context header


def test_a_ticker_matches_only_in_capitals():
    from finagent.retrieval.filters import build_company_vocab
    vocab, years = build_company_vocab([
        {"company": "COST", "ticker": "COST", "year": "2023"},
        {"company": "ADVANCED MICRO DEVICES INC", "ticker": "AMD", "year": "2024"}])
    # "cost of sales" is not Costco (the AES question that searched Costco's filing).
    assert infer_filter("AES 2022 cost of sales and inventories", vocab, years) is None
    assert infer_filter("COST revenue in 2023", vocab, years)["companies"] == ["COST"]
    assert infer_filter("AMD revenue 2024", vocab, years)["companies"] == ["ADVANCED MICRO DEVICES INC"]
    assert infer_filter("advanced micro devices revenue", vocab, years)["companies"] == [
        "ADVANCED MICRO DEVICES INC"]
