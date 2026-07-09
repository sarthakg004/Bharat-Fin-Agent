"""Dynamic SEC fetch + hybrid retrieval nodes (v4 overrides).

Split out of `finagent.graph.agent` (methods are unchanged); mixed into
`AgenticRAGv4` ahead of `AgenticRAGv3` in the MRO.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from finagent.graph.state import AgentState, CorpusGateQuery

GATE_EXTRACT_SYSTEM = """\
You identify the single US public company a question is primarily about, so the
system can fetch its SEC filing if it isn't indexed yet. Return the company as a
ticker or name (e.g. 'CRM' or 'Salesforce'). Return an empty string if the
question names no specific company, spans many companies, or is a macro/market
question. Resolve follow-ups from the conversation context.
"""

GATE_EXTRACT_PROMPT = """\
Question: {question}

Return the one company this is about (ticker or name), or '' if none/many.
"""


class FetchNodes:
    """Dynamic SEC fetch + hybrid retrieval nodes (v4 overrides)."""

    # A question about a SPECIFIC dated event filing ("the 8-K dated 1st July
    # 2022"). These documents are never in the indexed corpus (only annual
    # reports are) and rarely on the news web — they must be pulled from EDGAR
    # by (form, date) directly.
    _FORM_REQ_RE = re.compile(r"\b(8[\s-]?K|10[\s-]?Q|6[\s-]?K|DEF\s?14A)\b", re.I)

    _MONTHS = {m: i + 1 for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"])}

    _DATE_RES = (
        re.compile(r"\b(?P<d>\d{1,2})(?:st|nd|rd|th)?\s+of\s+"
                   r"(?P<m>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+"
                   r"(?P<y>\d{4})\b", re.I),
        re.compile(r"\b(?P<d>\d{1,2})(?:st|nd|rd|th)?\s+"
                   r"(?P<m>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+"
                   r"(?P<y>\d{4})\b", re.I),
        re.compile(r"\b(?P<m>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+"
                   r"(?P<d>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<y>\d{4})\b", re.I),
        re.compile(r"\b(?P<y>\d{4})-(?P<mn>\d{2})-(?P<d>\d{2})\b"),
    )

    @classmethod
    def _dated_form_request(cls, question: str) -> Optional[tuple]:
        """(form, date) when the question names an event filing AND a date."""
        qm = cls._FORM_REQ_RE.search(question or "")
        if not qm:
            return None
        from datetime import date
        for rx in cls._DATE_RES:
            m = rx.search(question)
            if not m:
                continue
            g = m.groupdict()
            month = int(g["mn"]) if g.get("mn") else cls._MONTHS[g["m"][:3].lower()]
            try:
                dt = date(int(g["y"]), month, int(g["d"]))
            except ValueError:
                continue
            squash = re.sub(r"[\s-]", "", qm.group(1).upper())
            form = {"8K": "8-K", "10Q": "10-Q", "6K": "6-K",
                    "DEF14A": "DEF 14A"}.get(squash, squash)
            return form, dt
        return None

    def fetch_filing_node(self, state: AgentState) -> dict:
        """Corpus-membership gate + on-demand SEC fetch (Phase 5).

        Runs between routing and retrieval. It identifies the company the
        question is about and consults the gate:
          * already indexed   → nothing to do;
          * US-listed, missing → fetch the latest 10-K, ingest it into the live
            collection, and invalidate the cached hybrid retrievers so the new
            chunks are searchable on this very turn;
          * not US-listed      → leave it for the web-search branch.

        The result lands in `state["fetch_status"]` so the API/UX can show the
        "Fetching latest 10-K…" state and report what was added.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        # Eval/offline override: when the active corpus is known-complete (the
        # FinanceBench eval points at the dedicated financebench_eval collection),
        # live EDGAR fetch is pure waste and the gate would misfire on the
        # ticker/name mismatch. The served path never sets this.
        if os.getenv("DISABLE_DYNAMIC_FETCH") == "1":
            return {"fetch_status": {}}

        question = state["question"]

        # Resolve the company (with conversation context for follow-ups).
        history = state.get("chat_history") or []
        history_block = ""
        if history:
            lines = [
                f"{('User' if t.get('role') == 'user' else 'Assistant')}: "
                f"{(t.get('content') or '')[:300]}"
                for t in history[-4:]
            ]
            history_block = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

        try:
            extractor = self._get_router_llm().with_structured_output(CorpusGateQuery)
            gq: CorpusGateQuery = extractor.invoke([
                SystemMessage(content=GATE_EXTRACT_SYSTEM),
                HumanMessage(content=history_block + GATE_EXTRACT_PROMPT.format(question=question)),
            ])
            company = (gq.company or "").strip()
        except Exception as e:
            self._log(state, f"corpus-gate extract failed: {e}")
            return {"fetch_status": {}}

        if not company:
            return {"fetch_status": {}}

        # Dated event-filing request ("AMCOR's 8-K dated 1st July 2022"): pull
        # the named document straight from EDGAR by (form, date) and inject its
        # text as ephemeral chunks — `hybrid_retrieve_node` ranks them first.
        # This runs regardless of the corpus gate: the company being indexed
        # only means its ANNUAL reports are, never its event filings.
        # `fetched_chunks` is an overwrite channel; chunks seeded before this
        # node (user-uploaded documents) must be carried through every return
        # that sets the key, or a same-turn fetch would silently drop them.
        prior = state.get("fetched_chunks") or []
        extra: dict = {}
        dated = self._dated_form_request(question)
        if dated:
            form, dt = dated
            try:
                res = self.fetcher.fetch_dated_filing(company, form, dt)
            except Exception as e:
                self._log(state, f"dated filing fetch failed ({form} {dt}): {e}")
                res = {"ok": False}
            if res.get("ok") and res.get("chunks"):
                self._log(state, f"fetched {res.get('form')} filed "
                                 f"{res.get('filing_date')} from EDGAR "
                                 f"({len(res['chunks'])} chunks)")
                extra["fetched_chunks"] = prior + res["chunks"]

        try:
            gate = self.fetcher.gate(company)
        except Exception as e:
            self._log(state, f"corpus gate failed for {company!r}: {e}")
            return {"fetch_status": {}, **extra}

        # Year-aware depth: a question about FY2019 can't be answered from the
        # LATEST 10-K alone — walk back enough annual filings that the asked
        # year is covered (each 10-K carries the prior year's comparatives,
        # hence the −1). Capped to bound a one-time ingest.
        n_filings = 1
        yrs = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", question)]
        if yrs:
            from datetime import date
            # ponytail: cap at 12 filings, not 5 — a question about FY2016 asked
            # in 2026 needs ~9 filings back, and the old cap of 5 made every
            # year >5 back unreachable (the eval's single biggest miss class).
            # 12 bounds the one-time fetch; widen only if older years matter.
            n_filings = max(1, min(12, date.today().year - min(yrs) - 1))

        if gate["decision"] != "fetch":
            # "Indexed" is COMPANY-level. If the question names a year the
            # index doesn't cover (e.g. us_filings holds only FY2022-2026 but the
            # question asks about FY2016), deepen by walking back to the filing
            # that carries that year. Without this, retrieval returns nothing
            # useful and the company looks "covered" — the dominant failure mode
            # for historical numeric questions on the cloud (ephemeral) path,
            # which is exactly where this used to be skipped.
            if gate["decision"] == "already_indexed" and yrs:
                covered = self._indexed_years(gate.get("ticker") or company)
                target = min(yrs)
                digit_years = {int(y) for y in covered if y.isdigit()}
                # When we can't read indexed years (corpus keyed by name, not
                # ticker), still deepen — better a redundant fetch than a miss.
                gap = (not digit_years) or not ({target, target + 1} & digit_years)
                if gap:
                    latest = max(digit_years) if digit_years else date.today().year
                    n_deep = max(1, min(12, latest - target))
                    self._log(state, f"index covers {sorted(digit_years) or '?'} for "
                                     f"{gate.get('ticker')} but the question needs "
                                     f"{target}; fetching {n_deep} older filing(s)")
                    try:
                        if self.persist_fetch:
                            res = self.fetcher.fetch_and_ingest(
                                gate["ticker"], company=gate.get("company") or "",
                                n=n_deep)
                            if res.get("ok"):
                                self._hybrids = None    # re-index for this turn
                                self._log(state, f"deepened index with "
                                                 f"{res.get('chunks_added')} chunks")
                        else:
                            # Ephemeral (cloud): pull the older filings in memory
                            # and feed them as fetched_chunks — no index write.
                            res = self.fetcher.fetch_chunks(
                                gate["ticker"], company=gate.get("company") or "",
                                n=n_deep)
                            deep = res.get("chunks", []) if res.get("ok") else []
                            if deep:
                                extra["fetched_chunks"] = (
                                    extra.get("fetched_chunks", prior) + deep)
                                self._log(state, f"deepened in-memory with "
                                                 f"{len(deep)} chunks")
                    except Exception as e:
                        self._log(state, f"index deepening failed: {e}")
            # already_indexed → retrieval handles it; not_us_listed → web branch.
            return {"fetch_status": gate, **extra}

        self._log(state, f"dynamic fetch: pulling latest {n_filings} filing(s) "
                         f"for {gate['ticker']}…")

        # Ephemeral path (cloud / per-session): parse + chunk the filing in
        # memory and rank it against the question later — NOTHING is written to
        # the persistent index, so the corpus never grows and there's no
        # scale-to-zero persistence problem.
        if not self.persist_fetch:
            try:
                res = self.fetcher.fetch_chunks(
                    gate["ticker"], company=gate.get("company") or "", n=n_filings)
            except Exception as e:
                self._log(state, f"ephemeral fetch failed for {gate['ticker']}: {e}")
                return {"fetch_status": {**gate, "status": "error", "error": str(e)},
                        **extra}
            chunks = res.get("chunks", []) if res.get("ok") else []
            if chunks:
                self._log(state, f"fetched {len(chunks)} in-memory chunks for {gate['ticker']} "
                                 f"({res.get('form')})")
            return {
                "fetched_chunks": extra.get("fetched_chunks", prior) + chunks,
                "fetch_status": {**gate, "status": "fetched" if chunks else "error",
                                 "ephemeral": True, "form": res.get("form"),
                                 "chunks_fetched": len(chunks)},
            }

        # Persistent path (local): fetch + ingest into the live collection.
        try:
            res = self.fetcher.fetch_and_ingest(
                gate["ticker"], company=gate.get("company") or "", n=n_filings)
        except Exception as e:
            self._log(state, f"dynamic fetch failed for {gate['ticker']}: {e}")
            return {"fetch_status": {**gate, "status": "error", "error": str(e)},
                    **extra}

        if res.get("ok"):
            self._hybrids = None       # invalidate cached BM25/dense over old corpus
            self._log(state, f"ingested {res['chunks_added']} chunks for {gate['ticker']}")
        return {"fetch_status": {**gate, "status": "fetched" if res.get("ok") else "error",
                                 **res}, **extra}

    def _indexed_years(self, ticker: str) -> set[str]:
        """Metadata `year` values indexed for `ticker` (empty when unknowable —
        e.g. baseline corpora keyed by company name rather than ticker)."""
        try:
            col = self.fetcher._get_store()._collection
            res = col.get(where={"ticker": ticker},
                          include=["metadatas"], limit=2000)
            return {str(m.get("year", "")) for m in (res.get("metadatas") or [])}
        except Exception:
            return set()

    def hybrid_retrieve_node(self, state: AgentState) -> dict:
        """Persistent-corpus retrieval, plus in-memory ranking of an ephemerally
        fetched filing (cloud path) so its chunks are usable without ever being
        indexed."""
        out = super().hybrid_retrieve_node(state)
        fetched = state.get("fetched_chunks") or []
        if fetched:
            ranked = self._rank_fetched_chunks(state, fetched)
            # Fetched filing is the authoritative source for an out-of-corpus
            # company → put its chunks first.
            out["retrieved_chunks"] = ranked + (out.get("retrieved_chunks") or [])
        return out

    def _rank_fetched_chunks(self, state: AgentState, fetched: list[dict]) -> list[dict]:
        """Rank ephemerally-fetched chunks against the question IN MEMORY (BGE
        cosine pre-filter → cross-encoder rerank), returning the top few per
        sub-query — the same two-stage quality as the persistent retriever, but
        over a single filing and with nothing written to disk."""
        import numpy as np

        from finagent.retrieval.reranker import _get_shared_reranker
        from finagent.vectorstore import get_embeddings

        texts = [c.get("text", "") for c in fetched]
        if not texts:
            return []
        sub_queries = state.get("sub_queries") or [state["question"]]
        routes = state.get("query_routes") or []
        if routes and len(routes) == len(sub_queries):
            subs = [s for s, r in zip(sub_queries, routes) if r in ("narrative", "numeric")] \
                or [state["question"]]
        else:
            subs = sub_queries

        try:
            embedder = get_embeddings(self.embedding_model)
            mat = np.asarray(embedder.embed_documents(texts))          # (N, d), normalized
            reranker = _get_shared_reranker(self.reranker_model)
        except Exception as e:
            self._log(state, f"ephemeral rank failed ({e}); using fetch order")
            return [{**c, "source": self._fetched_source(c), "sub_query": subs[0]}
                    for c in fetched[: self.final_top_k]]

        pre_k = max(self.final_top_k * 4, 20)
        out: list[dict] = []
        seen: set = set()
        for sq in subs:
            qv = np.asarray(embedder.embed_query(sq))
            sims = mat @ qv
            cand = list(np.argsort(-sims)[:pre_k])
            scores = reranker.predict([(sq, texts[i]) for i in cand])
            order = [cand[j] for j in np.argsort(-np.asarray(scores))][: self.final_top_k]
            for i in order:
                key = texts[i][:80]
                if key in seen:
                    continue
                seen.add(key)
                out.append({**fetched[i], "source": self._fetched_source(fetched[i]),
                            "sub_query": sq})
        return out

    @staticmethod
    def _fetched_source(c: dict) -> str:
        if c.get("filing_type") == "uploaded":
            return f"[{c.get('filename') or c.get('company', 'uploaded document')}]"
        return f"[{c.get('company', c.get('ticker', '?'))} filing {c.get('year', '?')}]"

