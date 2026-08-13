"""Hybrid retrieval and the critic retry loop.

Adds to `base.py`: BM25 ∪ dense fusion with a cross-encoder rerank, a final
cap over the merged pool, and a critic that schedules a retry instead of just
logging. `full.py` and `agent.py` build on this; nothing here runs alone.
"""

from __future__ import annotations

import re
from typing import Optional

from finagent.graph.base import AgenticRAG
from finagent.graph.state import AgentState


# Codepoint ranges for scripts a US English filing should not be written in
# (Cyrillic, Hebrew, Arabic, Devanagari, Hiragana/Katakana, CJK, Hangul).
_NON_LATIN_RE = re.compile(
    r"[Ѐ-ӿ֐-׿؀-ۿऀ-ॿ"
    r"぀-ヿ㐀-鿿가-힯]"
)


def _mostly_non_english(text: str) -> bool:
    """True if a chunk is predominantly non-Latin script — i.e. foreign-language
    noise that has no place in an English US-filings answer. Deterministic and
    cheap (no langdetect call); ignores stray foreign words in otherwise-English
    text by requiring non-Latin to exceed 30% of the alphabetic characters."""
    if not text:
        return False
    letters = sum(c.isalpha() for c in text)
    if letters < 20:                     # too short to judge confidently
        return False
    return len(_NON_LATIN_RE.findall(text)) / letters > 0.30


from finagent.retrieval.hybrid import HybridRetriever  # noqa: E402


class AgenticRAGv2(AgenticRAG):
    """Hybrid retrieval + a critic that can schedule a retry."""

    # Slots the ORIGINAL question gets in the merged pool, versus `final_top_k`
    # for a sub-query. §13 swept it: 1 -> hit@8 64, 2 -> 66, 3 -> 66, 5 -> 67,
    # while hit@5 falls 60 -> 55 -> 53 -> 50. Two is the only setting that beats
    # the previous production on BOTH metrics.
    QUESTION_SLOTS = 2

    # …except when the planner routed the request `narrative` (§14c). Narrative
    # evidence is MD&A and risk-factor PROSE, and the §14 retrieval query is
    # terse keywords — a poor dense match for prose, since embeddings compare
    # meaning and a keyword list does not read like a paragraph. The raw question
    # is what retrieves prose well, and 2 slots is not enough room for it.
    #
    # Swept on FinanceBench (n=99, hit@8 / narrative subset):
    #     2 slots -> 71 / 16      5 slots -> 71 / 16      8 slots -> 75 / 20
    # Note the THRESHOLD: 5 changes nothing. The question needs the full budget
    # before its evidence survives into the cap. hit@5 -1 and comparison -1 are
    # both inside the +/-1 run-to-run noise this sweep demonstrated.
    NARRATIVE_QUESTION_SLOTS = 8

    def __init__(
        self,
        *args,
        reranker_model: str = HybridRetriever.DEFAULT_RERANKER,
        pool_top_k: int = 48,
        final_top_k: int = 5,
        # After merging every sub-query's hits into one pool, a final
        # cross-encoder rerank against the ORIGINAL question keeps only the
        # `retrieve_cap` best passages. Multi-hop questions decompose into
        # several sub-queries, each contributing up to `final_top_k` chunks, so
        # without this cap a single question could carry 15-25 passages into the
        # synthesizer — measured to LOWER faithfulness/precision (more material
        # to over-claim against, more noise). Capping to a tight, globally-best
        # set is the single biggest faithfulness lever in the eval. None = no cap.
        retrieve_cap: Optional[int] = 8,
        max_critic_retries: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reranker_model = reranker_model
        self.pool_top_k = pool_top_k
        self.final_top_k = final_top_k
        self.retrieve_cap = retrieve_cap
        self.max_critic_retries = max_critic_retries
        self._hybrids: Optional[list[HybridRetriever]] = None

    # ------------------------------------------------------------------ #
    # Resources
    # ------------------------------------------------------------------ #

    def _get_hybrids(self) -> list[HybridRetriever]:
        """One hybrid retriever per filings collection (fused lexical+dense, reranked).
        Retrieving over several collections and letting the reranker sort it
        out keeps every collection searchable without a toggle."""
        if self._hybrids is None:
            self._hybrids = [
                HybridRetriever.from_collection(
                    c, self.embedding_model,
                    reranker_model=self.reranker_model,
                    pool_top_k=self.pool_top_k,
                    final_top_k=self.final_top_k,
                )
                for c in self.collections
            ]
        return self._hybrids

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #

    def hybrid_retrieve_node(self, state: AgentState) -> dict:
        """BM25 + dense + cross-encoder rerank, per sub-query, across every
        configured collection, merged/deduped."""
        hybrids = self._get_hybrids()
        seen: set = set()
        chunks: list[dict] = []
        dropped_lang = 0
        # Per-sub-query routing: filing retrieval is for `narrative` sub-queries
        # (and `numeric`, since the 10-K prose is a useful fallback when XBRL
        # lacks a metric). A `market` / `external` / `cross_document` sub-query is
        # answered by its own tool (yfinance / web / EDGAR), so retrieving the
        # filings for it only injects off-topic chunks — skip those. When routes
        # are unknown (older graph, or a single bare question) retrieve for all.
        sub_queries = state.get("sub_queries") or [state["question"]]
        routes = state.get("query_routes") or []
        if routes and len(routes) == len(sub_queries):
            retrieve_subs = [s for s, r in zip(sub_queries, routes)
                             if r in ("narrative", "numeric")]
        else:
            retrieve_subs = sub_queries
        # What actually searches the filings is ONE query in the filing's own
        # vocabulary (§14), not the decomposition. The sub-queries are still what
        # route the tool lanes, gate this node, and reach the synthesizer — they
        # just stopped being retrieval keys, because decomposition buys pool
        # recall and then loses it in selection (retention 0.744 vs 0.800, hit@8
        # 67/99 vs 72/99 on FinanceBench, at 3.09 queries/question instead of 2).
        #
        # Empty `retrieval_query` → the sub-queries, unchanged. That is the path
        # on a critic retry (which deliberately searches its own hint queries),
        # when the planner call failed, and on every graph rung below v3.
        rq = (state.get("retrieval_query") or "").strip()
        search_subs = [rq] if (rq and retrieve_subs) else retrieve_subs
        # Also retrieve on the question the user actually asked. Planning
        # discards information: a sub-query is a narrow lookup key, and nothing
        # was searching for the whole question. Adding it is worth 3 of 99
        # FinanceBench questions (§13) — and 5 more on top of the rewrite (§14),
        # which is terse keywords and strips the prose a narrative question needs.
        # It gets a SMALLER slot budget than a sub-query: at the full 5 it crowds
        # the others out of the cap and costs 10 points of hit@5 for one hit@8.
        #
        # Gated on `retrieve_subs`: when the planner routed every sub-query to
        # yfinance/web/EDGAR it has decided the filings are not the source, and
        # this must not override that.
        #
        # `final_top_k` (5) is the budget for ONE OF SEVERAL sub-queries sharing
        # the cap. The §14 query is alone, so 5 + QUESTION_SLOTS = 7 would leave
        # the 8-passage cap unfilled and throw away that query's 6th-8th hits —
        # the measured 72/99 gave it the full cap. Sub-queries keep `final_top_k`
        # precisely because there are several of them competing.
        solo = self.retrieve_cap if (rq and search_subs == [rq]) else None
        queries = [(sub_q, solo) for sub_q in search_subs]
        if search_subs and state["question"] not in search_subs:
            # A narrative request needs the question searched as hard as the
            # keyword query — see NARRATIVE_QUESTION_SLOTS.
            slots = (self.NARRATIVE_QUESTION_SLOTS if "narrative" in routes
                     else self.QUESTION_SLOTS)
            queries.append((state["question"], slots))
        for sub_q, top_k in queries:
            for hyb in hybrids:
                for text, meta in hyb.search(sub_q, top_k=top_k):
                    # Prefer the parent id: since §12 every chunk STARTS with
                    # its context header, so `text[:80]` is often just
                    # "Apple 2022 10-K · Consolidated Balance Sheet" — identical
                    # across the several parents of one long statement, which
                    # silently dropped 0.3% of distinct parents as duplicates.
                    # Fall back to the text prefix only when there is no parent.
                    pid = meta.get("parent_id")
                    key = ((meta.get("local_path", ""), meta.get("page", ""), pid)
                           if pid is not None else
                           (meta.get("local_path", ""), meta.get("page", ""), text[:80]))
                    if key in seen:
                        continue
                    seen.add(key)
                    # The corpus is English US filings — drop any chunk that's
                    # predominantly non-Latin script (stray foreign-language
                    # exhibits, OCR noise) so it never reaches the synthesizer.
                    if _mostly_non_english(text):
                        dropped_lang += 1
                        continue
                    chunks.append({
                        "text": text,
                        "company": meta.get("company") or meta.get("ticker", "?"),
                        "year": meta.get("year", "?"),
                        "page": meta.get("page", "?"),
                        "source": self._citation_tag(meta),
                        "sub_query": sub_q,
                    })
        if dropped_lang:
            self._log(state, f"dropped {dropped_lang} non-English chunk(s) from retrieval")
        chunks = self._cap_pool(state, chunks)
        # Is the question's company actually in the corpus? The retriever's
        # metadata-vocab filter (no LLM) tells us: a company match means the
        # authoritative filing is indexed — answer from it instead of escalating
        # to generic web pages that only dilute faithfulness.
        in_corpus = False
        for hyb in hybrids:
            if any((hyb.infer_filter(sq) or {}).get("companies") for sq in retrieve_subs):
                in_corpus = True
                break
        return {"retrieved_chunks": chunks, "company_in_corpus": in_corpus}

    def _cap_pool(self, state: AgentState, chunks: list[dict]) -> list[dict]:
        """Trim the merged sub-query pool to the `retrieve_cap` best passages.

        The per-sub-query reranker inside each `HybridRetriever` only sees one
        sub-query at a time, so a 6-sub-query comparison question arrives here
        with up to 30 chunks and they have to be ordered against each other.

        Re-score the whole pool against the ORIGINAL question. §11 replaced this
        with the score each chunk already carried from its own sub-query, having
        measured that as better (37 -> 41 of 99). §13 re-ran the same A/B after
        chunks gained context headers and it REVERSED — 58 for the sub-query
        score versus 61 for this re-score. The §11 result was an artefact of
        chunks with no identity: re-scoring an untitled grid of numbers against
        a multi-hop question was hopeless, but once a chunk says
        "3M 2022 10-K · Consolidated Balance Sheet" the question — which is the
        only thing that knows what the user actually wants — can match it.

        The cost is one extra cross-encoder pass over ~10-30 chunks per query.
        """
        cap = self.retrieve_cap
        if not cap or len(chunks) <= cap:
            return chunks
        try:
            from finagent.retrieval.reranker import _get_shared_reranker

            reranker = _get_shared_reranker(self.reranker_model)
            scores = reranker.predict(
                [(state["question"], c["text"]) for c in chunks])
            ranked = sorted(range(len(chunks)), key=lambda i: -scores[i])
            # Per-sub-query floor: a comparison question's sub-queries each
            # cover a different (entity × year), and one entity's filing can
            # score higher across the board — ranking the merged pool purely by
            # score lets it crowd the other out entirely. Reserve each
            # sub-query's single best chunk first, then fill by score.
            kept_idx: list[int] = []
            seen_subs: set = set()
            for i in ranked:
                sub = chunks[i].get("sub_query", "")
                if sub not in seen_subs:
                    seen_subs.add(sub)
                    kept_idx.append(i)
                if len(kept_idx) >= cap:
                    break
            for i in ranked:
                if len(kept_idx) >= cap:
                    break
                if i not in kept_idx:
                    kept_idx.append(i)
            kept = [chunks[i] for i in sorted(kept_idx, key=ranked.index)]
            self._log(state, f"capped retrieval pool {len(chunks)}→{len(kept)} "
                             f"by cross-encoder rerank vs. the question")
            return self._fit_budget(state, kept)
        except Exception as e:
            # Reranking is a precision optimisation — never let it drop evidence
            # on a model/load error. Fall back to a simple head-truncation.
            self._log(state, f"pool cap rerank failed ({e}); truncating to {cap}")
            return self._fit_budget(state, chunks[:cap])

    def _fit_budget(self, state: AgentState, kept: list[dict]) -> list[dict]:
        """Bound the evidence so a shared-tier request cannot exceed 8000 tokens.

        Groq's free tier rejects an over-sized request with HTTP 413 — a
        permanent failure that no retry, no key and no wait can fix — and
        production ran at 95-98% of that ceiling, so a follow-up question's
        history was enough to break it. Unbounded before this.

        Drops whole chunks from the BOTTOM of the ranking first, because a
        complete low-ranked passage is worth less than an intact high-ranked
        one; only truncates when a single chunk is itself over budget. Callers
        with their own API key get `None` and are untouched.
        """
        from finagent.runtime import current_context

        budget = current_context().evidence_budget()
        if budget is None:
            return kept
        out, used = [], 0
        for c in kept:
            text = c["text"]
            if used + len(text) <= budget:
                out.append(c)
                used += len(text)
            elif not out:
                # The single best passage alone exceeds the budget: keep a
                # truncated head rather than answering with no evidence.
                out.append({**c, "text": text[:budget]})
                used = budget
        if len(out) < len(kept):
            self._log(state, f"evidence trimmed {len(kept)}→{len(out)} passages "
                             f"to fit the free tier's per-request limit; add an "
                             f"API key from the model picker for the full context")
        return out

    def critic_node(self, state: AgentState) -> dict:
        """Extends the parent's critic with a retrieval-retry hint.

        When the parent flags unsupported claims AND the recovery budget is
        unspent, prepare the retry. Otherwise let the flow end.

        The retry has to be DIFFERENT from the pass that just failed, or it
        just burns quota to produce the same answer. Which difference depends
        on the critic's own remedy: a `gather` retry re-searches on the flagged
        claims, while a `redraft` retry keeps the evidence and re-writes with
        the flagged claims quoted back at the synthesizer
        (`SynthesisNodes.synthesize_node` builds that block).
        """
        out = super().critic_node(state)
        crit_iter = state.get("critic_iterations", 0)

        if not out.get("needs_retry") or crit_iter >= self.max_critic_retries:
            # Either converged, or the single recovery is spent — let the
            # router decide between ending and refusing.
            out["needs_retry"] = False
            return out

        out["critic_iterations"] = crit_iter + 1
        if out.get("critic_remedy") != "gather":
            # A re-draft reuses the evidence, so leave the retrieval intent
            # alone. Overwriting `sub_queries` here used to happen on every
            # retry and replaced the planner's decomposition in the synthesis
            # prompt's "Sub-queries researched" list, for a re-draft that never
            # retrieves anything.
            return out

        # Build hint sub-queries from the unsupported-claim errors the parent
        # appended this turn. (Format set by parent: "unsupported claim: <c>".)
        new_errors = out.get("errors", [])
        hints = [
            "supporting evidence for: " + e[len("unsupported claim: "):]
            for e in new_errors[-3:]
            if e.startswith("unsupported claim: ")
        ] or [state["question"]]

        out["sub_queries"] = hints
        # These hints ARE the new retrieval intent, and leaving the §14 query
        # set would search the original question again and re-find the exact
        # evidence the critic just rejected.
        out["retrieval_query"] = ""
        return out
