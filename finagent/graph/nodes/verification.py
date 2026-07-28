"""Deterministic numeric verification, refusal/abstention, and the confidence gate.

Split out of `finagent.graph.agent` (methods are unchanged); mixed into
`AgenticRAGv4` ahead of `AgenticRAGv3` in the MRO.
"""

from __future__ import annotations

import re
from typing import Optional

from finagent.graph.state import AgentState, NumericVerification

from finagent.prompts.critic import (  # noqa: F401
    NUM_VERIFY_PROMPT,
    NUM_VERIFY_SYSTEM,
    REFUSAL_TEMPLATE,
)

# Phrases a synthesizer uses when it is *itself* conceding the evidence can't
# answer the question — the "soft refusals" buried inside otherwise-formatted
# answers. Detected so the confidence blend can zero the citation component;
# the draft is still shown (with a low-confidence caveat), never withheld.
_SOFT_REFUSAL_RE = re.compile(
    r"\b(?:"
    r"not enough info(?:rmation)?|insufficient (?:info|information|evidence|data)|"
    r"cannot (?:be )?(?:determined|calculated|computed|answered)|"
    r"could not (?:be )?(?:determined|calculated|found)|"
    r"unable to (?:determine|calculate|compute|answer|find|locate)|"
    r"do(?:es)? not (?:disclose|provide|contain|include|report)|"
    r"is not (?:disclosed|provided|available|reported)|"
    r"are not (?:disclosed|provided|available|reported)|"
    r"no (?:relevant |available )?(?:information|data|disclosure|figures?|evidence) "
    r"(?:is|are|was|were|provided|available|present|on)?"
    r")\b",
    re.I,
)


class VerificationNodes:
    """Deterministic numeric verification, refusal/abstention, and the confidence gate."""

    # Weights from the spec. Applied only over the sub-scores that are present
    # for a given question; the denominator renormalises so a missing component
    # (e.g. retrieval on a pure-XBRL numeric query) doesn't drag the blend down.
    _CONF_WEIGHTS = {
        "retrieval": 0.25,
        "verification": 0.35,
        "citation": 0.25,
        "critic": 0.15,
    }

    # A citation marker: [1] / [1, 3] and the fullwidth 【1】 form some models
    # (gpt-oss, certain Llama builds) emit despite the prompt asking for ASCII.
    # Requiring a leading digit keeps it off prose brackets like "[note]".
    _CITE_RE = re.compile(r"[\[【]\s*\d[\d,\s]*\s*[\]】]")

    # Scale words → multiplier. Single-letter B/M/K/T handled with a word
    # boundary so they don't fire on stray capitals mid-word.
    _SCALES = {
        "trillion": 1e12, "tn": 1e12, "t": 1e12,
        "billion": 1e9, "bn": 1e9, "b": 1e9,
        "million": 1e6, "mn": 1e6, "m": 1e6,
        "thousand": 1e3, "k": 1e3,
        "crore": 1e7, "lakh": 1e5,
    }

    # Trailing `(?=\b)` lookahead guards ONLY the single-letter scales (so a bare
    # "B"/"M" must end a word, e.g. "$99.8B"); `%`/`bps`/word-scales match
    # directly (a trailing `\b` here would wrongly reject "7.8%").
    _NUM_RE = re.compile(
        r"(?P<sign>[-+]?)\$?\s*"
        r"(?P<num>\d[\d,]*(?:\.\d+)?)\s*"
        r"(?P<scale>trillion|tn|billion|bn|million|mn|thousand|crore|lakh|bps|%|[bmkt](?=\b))?",
        re.IGNORECASE,
    )

    # Cap on the deduped evidence pool used for derived grounding — keeps the
    # pairwise scan (~ cap² / 2 pairs per figure) cheap.
    _DERIVE_MAX_BASE = 160

    # Powers of ten by which an exact figure may be RESTATED in prose. A filed
    # value of 135,987,000,000 is the same fact whether the answer writes it as
    # "$135.987 billion" (×1), "135,987" million (×1e-6), or a scale-free
    # "135.987" inside a worked formula (×1e-9). Expanding each exact evidence
    # value across these scales makes grounding unit-insensitive, so a correct
    # derivation isn't refused just because it dropped the "billion" word.
    _RESTATE_SCALES = (1.0, 1e-3, 1e-6, 1e-9, 1e-12)

    # Structural constants that appear in growth / CAGR / margin DERIVATIONS
    # (×100 to render a percent, the 1 in `(b/a)^(1/n) − 1`, a leading 0), plus
    # time-period constants from working-capital formulas (365 in DSO/DIO/DPO,
    # 360-day conventions, 52 weeks, 12 months, 4 quarters, 2 in an average).
    # They are math scaffolding the synthesizer writes out, never financial
    # claims — so they must count as grounded or every shown calculation
    # self-refutes.
    _MATH_CONSTANTS = (0.0, 1.0, 2.0, 100.0, 365.0, 360.0, 52.0, 12.0, 4.0)

    def _get_verifier_llm(self):
        return self._get_llm("verifier")

    def verify_numbers_node(self, state: AgentState) -> dict:
        """Phase 11 fact-checking critic — every figure traces to a source.

        Deterministically extract EVERY number in the draft, then ground each
        against XBRL facts / derived metrics (exact) and numbers parsed from the
        retrieved chunks, tables, web, and market data. The LLM verifier runs as
        a *rescue* (its matched figures supplement the evidence) and to produce
        human-readable claims. A figure grounded by neither is a hallucination;
        `_verify_router` then re-routes or refuses. Tracks the hallucination rate
        explicitly. With `strict_numeric` False this falls back to the prior
        LLM-only check.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        # Count every verify pass so `_verify_router` can bound the re-route loop
        # independently of the critic (the verifier may keep finding an
        # ungrounded figure even when the critic is happy — without this counter
        # that loops until LangGraph's recursion limit aborts the request).
        vi = {"verify_iterations": state.get("verify_iterations", 0) + 1}

        answer = state.get("draft_answer", "")
        clean = {"claims": [], "unverified": [], "score": 1.0,
                 "numbers_total": 0, "numbers_grounded": 0, "hallucination_rate": 0.0}
        if not self._has_numbers(answer):
            return {"numeric_verification": clean, **vi}

        def _run_llm_verifier() -> list[dict]:
            evidence = self._build_evidence_block(state)
            llm = self._get_verifier_llm().with_structured_output(NumericVerification)
            try:
                report: NumericVerification = llm.invoke([
                    SystemMessage(content=NUM_VERIFY_SYSTEM),
                    HumanMessage(content=NUM_VERIFY_PROMPT.format(
                        answer=answer, evidence=evidence)),
                ])
                return [c.model_dump() for c in report.claims]
            except Exception as e:
                self._log(state, f"verifier failed ({e})")
                return []

        # Legacy LLM-only path (A/B baseline).
        if not self.strict_numeric:
            claims = _run_llm_verifier()
            llm_unverified = [c for c in claims if not c.get("matched")]
            score = ((len(claims) - len(llm_unverified)) / len(claims)) if claims else 1.0
            return {"numeric_verification": {"claims": claims, "unverified": llm_unverified,
                                             "score": round(score, 3),
                                             "numbers_total": len(claims),
                                             "numbers_grounded": len(claims) - len(llm_unverified),
                                             "hallucination_rate": round(1 - score, 3)}, **vi}

        # Deterministic, exhaustive grounding — runs FIRST and without any LLM.
        draft_nums = self._extract_numbers(answer)
        if not draft_nums:
            return {"numeric_verification": dict(clean), **vi}

        by_kind = self._evidence_numbers_by_kind(state)
        evidence_mags = [m for vals in by_kind.values() for m in vals]
        derive_base = self._derivation_base(by_kind)

        def _walk(extra_mags: list[float]) -> list[dict]:
            """Walk the draft's figures in order, letting each grounded figure
            feed the pool (`chained`) so a worked calculation grounds stepwise:
            the average grounds from its two inputs, the ratio from that
            average, the final metric from the ratios. Derivation (`_derivable`)
            rescues figures the draft legitimately computed from evidence."""
            pool = evidence_mags + extra_mags
            chained: list[float] = []
            missing: list[dict] = []
            for d in draft_nums:
                ok = (
                    self._grounded(d["magnitudes"], pool)
                    or self._grounded(d["magnitudes"], chained)
                    or self._derivable(d["magnitudes"], derive_base + chained)
                )
                if ok:
                    chained.extend(d["magnitudes"])
                else:
                    missing.append({"number": d["raw"], "claim": d["ctx"]})
            return missing

        ungrounded = _walk([])
        claims: list[dict] = []
        if ungrounded:
            # LLM rescue — invoked ONLY when deterministic grounding left
            # figures unaccounted for (skipping it on the fully-grounded common
            # path saves a strong-tier LLM call per answer). Figures the
            # verifier matched count as grounded too.
            claims = _run_llm_verifier()
            rescued: list[float] = []
            for c in claims:
                if c.get("matched"):
                    for n in self._extract_numbers(
                            str(c.get("number", "")) + " " + str(c.get("evidence", ""))):
                        rescued.extend(n["magnitudes"])
            if rescued:
                ungrounded = _walk(rescued)
        ungrounded_raws = {u["number"] for u in ungrounded}
        total = len(draft_nums)
        grounded = total - len(ungrounded)
        score = grounded / total if total else 1.0
        for u in ungrounded:
            self._log(state, f"ungrounded figure: {u['number']} — {u['claim'][:100]}")

        nv = {
            "claims": claims,
            "unverified": ungrounded,              # refuse_node reads this key
            "score": round(score, 3),
            "numbers_total": total,
            "numbers_grounded": grounded,
            "hallucination_rate": round(len(ungrounded) / total, 3) if total else 0.0,
        }
        try:
            report = self._build_verification_report(
                state, draft_nums, ungrounded_raws, by_kind)
        except Exception as e:
            # The report is advisory — never let it break the refusal-bearing
            # numeric verdict.
            self._log(state, f"verification report failed ({e})")
            report = {}
        report["numeric"] = nv
        return {"numeric_verification": nv, "verification_report": report, **vi}

    def _exact_raw_values(self, state: AgentState) -> list[float]:
        """The exact, UN-expanded XBRL / calc figures (1× scale only) — used to
        tell whether a grounded draft number matched at its stated scale or only
        after a power-of-10 restatement (a possible unit slip)."""
        raw: list[float] = []
        for f in state.get("xbrl_facts", []) or []:
            v = f.get("value")
            if isinstance(v, (int, float)):
                raw.append(float(v))
        for r in state.get("calc_results", []) or []:
            for key_val in ([r.get("value")] + [s.get("value") for s in (r.get("series") or [])]
                            + [i.get("value") for i in (r.get("inputs") or []) if isinstance(i, dict)]):
                if isinstance(key_val, (int, float)):
                    raw.extend([float(key_val), float(key_val) * 100.0])
        return raw

    def _build_verification_report(self, state: AgentState, draft_nums: list[dict],
                                   ungrounded_raws: set, by_kind: dict) -> dict:
        """Assemble the financial verification report the spec (#5) asks for:
        cross-source corroboration, unit-shift flags, and source/citation
        coverage. This is a REPORT — it informs confidence and the audit trail;
        it does not itself trigger refusals (that stays with numeric grounding).
        """
        grounded_nums = [d for d in draft_nums if d["raw"] not in ungrounded_raws]
        source_kinds = [k for k in by_kind if k != "const"]

        # Cross-source: which independent lanes corroborate each grounded figure.
        details: list[dict] = []
        corroborated = single = web_only = 0
        for d in grounded_nums:
            matching = [
                k for k in source_kinds
                if any(self._num_close(m, ev) for m in d["magnitudes"] for ev in by_kind[k])
            ]
            if not matching:
                continue                            # grounded only by a constant
            if len(matching) >= 2:
                corroborated += 1
            else:
                single += 1
                if matching == ["web"]:
                    web_only += 1
            if len(details) < 12:
                details.append({"number": d["raw"], "kinds": matching})

        # Units: figures that grounded only after a scale restatement (the stated
        # magnitude didn't match an exact value at 1×) — a possible unit slip.
        exact_raw = self._exact_raw_values(state)
        scale_shifted = 0
        for d in grounded_nums:
            if any(self._num_close(m, v) for m in d["magnitudes"] for v in exact_raw):
                continue                            # matched at its stated scale
            if any(self._num_close(m, v * s) for m in d["magnitudes"]
                   for v in exact_raw for s in self._RESTATE_SCALES if s != 1.0):
                scale_shifted += 1

        # Sources: citation coverage of the structured numeric evidence items.
        ev = state.get("evidence") or []
        numeric_items = [e for e in ev if e.get("value") is not None]
        with_cit = sum(1 for e in numeric_items if (e.get("citation") or "").strip())

        return {
            "cross_source": {
                "corroborated": corroborated,       # ≥2 independent lanes agree
                "single_source": single,
                "web_only": web_only,               # weakest — web alone
                "details": details,
            },
            "units": {"scale_shifted_figures": scale_shifted},
            "sources": {
                "numeric_evidence_items": len(numeric_items),
                "with_citation": with_cit,
                "without_citation": len(numeric_items) - with_cit,
            },
        }

    def refuse_node(self, state: AgentState) -> dict:
        """Replace the final answer with an explicit refusal."""
        web_used = bool(state.get("web_results"))
        unverified = state.get("numeric_verification", {}).get("unverified", []) if isinstance(state.get("numeric_verification"), dict) else []
        detail = ""
        if unverified:
            nums = ", ".join(str(u.get("number")) for u in unverified[:3] if u.get("number"))
            if nums:
                detail = f" (unverified figures: {nums})"
        web_clause = "" if web_used else " or recent web sources"
        msg = REFUSAL_TEMPLATE.format(web_clause=web_clause, detail=detail)
        return {"final_answer": msg, "refused": True, "needs_retry": False,
                "status": "refused"}

    def _citation_score(self, state: AgentState) -> Optional[float]:
        """Fraction of material figures in the draft that carry a [N] citation.

        Returns None (component not applicable) when the draft is empty or there
        is no evidence to cite at all. For a figure-free answer it's 1.0 if any
        citation is present, else None — we don't penalise a purely qualitative
        answer for having no numbers to anchor.
        """
        answer = state.get("draft_answer", "") or ""
        if not answer.strip():
            return None
        has_evidence = any(state.get(k) for k in (
            "retrieved_chunks", "xbrl_facts", "calc_results",
            "web_results", "market_data", "edgar_results",
        ))
        if not has_evidence:
            return None

        # Positions of citation markers in the ORIGINAL answer.
        markers = [m.start() for m in self._CITE_RE.finditer(answer)]

        # Mask out tokens that look numeric but aren't financial claims, keeping
        # length (and therefore offsets) identical so `markers` stays aligned.
        def _blank(m: re.Match) -> str:
            return " " * len(m.group(0))

        masked = self._CITE_RE.sub(_blank, answer)
        masked = re.sub(r"\bFY\s?\d{2,4}\b", _blank, masked, flags=re.I)
        masked = re.sub(r"\bQ[1-4]\b", _blank, masked, flags=re.I)
        masked = re.sub(r"\b10\s?-?\s?[KQ]\b|\b8\s?-?\s?K\b", _blank, masked, flags=re.I)
        masked = re.sub(r"\bp\.?\s*\d+\b", _blank, masked, flags=re.I)

        fig_ends = [m.end() for m in self._NUM_RE.finditer(masked) if m.group("num")]
        if not fig_ends:
            return 1.0 if markers else None

        # A figure is "cited" if a marker sits just after it (or a few chars
        # before, for "[1] revenue of $X" ordering).
        WINDOW = 80
        covered = sum(
            1 for p in fig_ends
            if any(-15 <= (mk - p) <= WINDOW for mk in markers)
        )
        return round(covered / len(fig_ends), 3)

    def _confidence_components(self, state: AgentState) -> dict:
        """The sub-scores that apply to this question, each in [0,1]."""
        comps: dict[str, float] = {}

        # A draft that is itself conceding it can't answer ("No relevant evidence
        # provided…") must NOT score high: with no figures to verify and no
        # claims for the critic to refute, the critic passes vacuously and the
        # blend lands at ~1.0 — a content-free non-answer marked maximally
        # confident. Force zero so the gate routes it to an explicit abstention.
        draft = state.get("draft_answer") or state.get("final_answer") or ""
        if _SOFT_REFUSAL_RE.search(draft[:600]):
            return {}

        # Retrieval — normalise the mean grade (1-5) to [0,1]. Applicable only
        # when retrieval actually ran (graded chunks exist).
        grades = state.get("grades") or []
        avg = state.get("avg_grade")
        if grades and avg is not None:
            comps["retrieval"] = max(0.0, min(1.0, (avg - 1.0) / 4.0))
        elif state.get("retrieved_chunks"):
            # Chunks present but ungraded (e.g. ephemeral fetch on the tool path).
            comps["retrieval"] = 0.5
        elif any(r in ("narrative", "numeric") for r in (state.get("query_routes") or [])):
            # The planner said this question is answered FROM THE FILINGS and
            # retrieval delivered nothing — every chunk was off-entity, or the
            # filing could not be fetched. Omitting the sub-score (the old
            # behaviour) renormalised the blend over the three that survived, so
            # a filings question answered entirely from marketing/analyst web
            # pages scored 0.9 and was presented with no caveat at all. An
            # absent primary source is a zero, not a non-applicable.
            comps["retrieval"] = 0.0

        # Verification — only when the answer actually made numeric claims.
        nv = state.get("numeric_verification") or {}
        if isinstance(nv, dict) and nv.get("numbers_total", 0) > 0:
            comps["verification"] = max(0.0, min(1.0, float(nv.get("score", 0.0))))

        # Citation coverage.
        cs = self._citation_score(state)
        if cs is not None:
            comps["citation"] = cs

        # Critic — claim-support rate (None when the critic couldn't score).
        gs = state.get("grading_score")
        if gs is not None:
            comps["critic"] = max(0.0, min(1.0, float(gs)))

        return comps

    def confidence_node(self, state: AgentState) -> dict:
        """Blend the applicable sub-scores into a single confidence and pick a band.

        Runs once, after numeric verification has settled, on the path that the
        verifier already deemed answerable (ungrounded-figure refusals are
        handled upstream by `_verify_router`). Writes the four sub-scores plus
        the blend and the routing band into state for the gate, the audit trail,
        and the UI.
        """
        try:
            comps = self._confidence_components(state)
            if comps:
                wsum = sum(self._CONF_WEIGHTS[k] for k in comps)
                conf = sum(self._CONF_WEIGHTS[k] * v for k, v in comps.items()) / wsum
            else:
                conf = 0.0
            conf = round(conf, 3)
        except Exception as e:
            # If scoring itself fails, don't block a verified answer — fail OPEN
            # (answer, no gate) rather than refusing a good answer on a math bug.
            self._log(state, f"confidence scoring failed ({e}); answering without the gate")
            return {"confidence": None, "confidence_band": "answer",
                    "status": "answered"}

        if not self.confidence_gating:
            band = "answer"
        elif conf >= self.confidence_answer:
            band = "answer"
        elif conf >= self.confidence_warn:
            band = "warn"
        else:
            band = "refuse"

        self._log(
            state,
            f"confidence={conf} band={band} "
            f"[{', '.join(f'{k}={v:.2f}' for k, v in comps.items())}]",
        )
        return {
            "retrieval_score": comps.get("retrieval"),
            "verification_score": comps.get("verification"),
            "citation_score": comps.get("citation"),
            "critic_score": comps.get("critic"),
            "confidence": conf,
            "confidence_band": band,
            "status": "answered" if band == "answer" else state.get("status"),
        }

    def withhold_low_confidence_node(self, state: AgentState) -> dict:
        """Low-confidence band: show the draft IN FULL with the confidence
        score appended — the same presentation as the warn band, with a
        stronger caveat. (Previously this hid the draft behind a "low-
        confidence answer available" notice, which buried answers that were
        often correct; hallucinated-figure refusals are handled separately by
        `refuse_node`, so the figures here are grounded — the uncertainty is
        about completeness/corroboration, which the caveat conveys.)
        """
        ans = state.get("draft_answer", "") or state.get("final_answer", "") or ""
        if "_Confidence:" in ans:                      # idempotent on a re-entry
            return {"status": "answered_low_confidence"}

        conf = state.get("confidence")
        pct = f"{conf:.0%}" if isinstance(conf, (int, float)) else "low"
        note = (
            f"\n\n*_Confidence: {pct} — low. Parts of this answer are weakly "
            f"corroborated by the available sources; verify against the "
            f"primary filing before relying on it._*"
        )
        return {
            "final_answer": ans + note,
            "refused": False,
            "needs_retry": False,
            "status": "answered_low_confidence",
        }

    def answer_with_warning_node(self, state: AgentState) -> dict:
        """Moderate-confidence band: keep the answer but append a one-line caveat."""
        ans = state.get("final_answer", "") or state.get("draft_answer", "")
        if "_Confidence:" in ans:                      # idempotent on a re-entry
            return {"status": "answered_with_warning"}
        conf = state.get("confidence")
        pct = f"{conf:.0%}" if isinstance(conf, (int, float)) else "moderate"
        note = (
            f"\n\n*_Confidence: {pct} — moderate. Some figures or claims are only "
            f"partially corroborated by the available sources; verify against the "
            f"primary filing before relying on them._*"
        )
        return {"final_answer": ans + note, "status": "answered_with_warning"}

    @staticmethod
    def _has_numbers(text: str) -> bool:
        return bool(re.search(r"\d", text or ""))

    @classmethod
    def _extract_numbers(cls, text: str) -> list[dict]:
        """Extract EVERY material figure from `text` as {raw, magnitudes, ctx}.

        `magnitudes` is the set of plausible numeric values the figure could mean
        — e.g. "30.3%" → {30.3, 0.303}, "$394.3 billion" → {3.943e11}. Matching
        any magnitude against evidence (with tolerance) grounds the figure. We
        strip citation markers, page refs, fiscal-year tokens and bare years
        first so they aren't mistaken for financial claims.
        """
        if not text:
            return []
        # Normalise Unicode hyphens/dashes (‑ – — − …) to ASCII "-" so the
        # range/label scrubs below catch them regardless of which dash the model
        # emitted (the synth often uses a non-breaking hyphen in "FY2023‑24").
        scrubbed = text.translate({
            0x2010: 0x2d, 0x2011: 0x2d, 0x2012: 0x2d, 0x2013: 0x2d,
            0x2014: 0x2d, 0x2015: 0x2d, 0x2212: 0x2d,
        })
        # Remove things that look numeric but aren't financial claims.
        scrubbed = cls._CITE_RE.sub(" ", scrubbed)             # [1], [1, 3], 【1】
        # Fiscal-year RANGES first ("FY2023-24", "2023-2024", "FY2015 - FY2017"),
        # then single years. Without the optional second "FY" the range scrub
        # missed "FY2015 - FY2017", leaving a stray "- 3" style artifact.
        scrubbed = re.sub(r"\bFY\s?\d{4}\s?-\s?(?:FY\s?)?\d{2,4}\b", " ", scrubbed, flags=re.I)
        scrubbed = re.sub(r"\b(?:19|20)\d{2}\s?-\s?\d{2,4}\b", " ", scrubbed)
        scrubbed = re.sub(r"\bFY\s?\d{2,4}\b", " ", scrubbed, flags=re.I)
        # Window labels: "52-week", "52 wk", "200-day", "3 year average" — the
        # count names the metric/period, it isn't a reported figure.
        scrubbed = re.sub(r"\b\d{1,3}\s?-?\s?(?:week|wk|day|month|mo|year|yr|quarter|qtr)s?\b",
                          " ", scrubbed, flags=re.I)
        scrubbed = re.sub(r"\bQ[1-4]\b", " ", scrubbed, flags=re.I)
        # SEC form names in every spelling the models emit: 10-K, 10K, 8-K, 8k.
        scrubbed = re.sub(r"\b10\s?-?\s?[KQ]\b|\b8\s?-?\s?K\b", " ", scrubbed, flags=re.I)
        scrubbed = re.sub(r"\bp\.?\s*\d+\b", " ", scrubbed, flags=re.I)  # p. 12

        out: list[dict] = []
        for m in cls._NUM_RE.finditer(scrubbed):
            raw = m.group(0).strip()
            num = m.group("num").replace(",", "")
            try:
                base = float(num)
            except ValueError:
                continue
            if m.group("sign") == "-":
                # Distinguish a negative quantity (a $5.2bn loss) from a binary
                # subtraction operator inside a shown derivation ("a - b"): if the
                # text just before the '-' ends with a digit / ')' / '%', it's a
                # subtraction and the figure itself is positive. Misreading it as
                # negative would refuse a correct worked calculation.
                prefix = scrubbed[: m.start()].rstrip()
                if not (prefix and prefix[-1] in "0123456789)%"):
                    base = -base
            scale = (m.group("scale") or "").lower()

            if scale in ("%",):
                mags = {base, base / 100.0}
            elif scale == "bps":
                mags = {base, base / 100.0, base / 10000.0}
            elif scale in cls._SCALES:
                mags = {base * cls._SCALES[scale]}
            else:
                # No scale word. Skip bare integers that are almost certainly
                # years (1900-2099) — they're periods, not financial figures.
                if scale == "" and base.is_integer() and 1900 <= base <= 2099:
                    continue
                mags = {base}
            ctx = scrubbed[max(0, m.start() - 30): m.end() + 30].strip()
            out.append({"raw": raw, "magnitudes": mags, "ctx": ctx})
        return out

    @staticmethod
    def _num_close(a: float, b: float, rel_tol: float = 0.02) -> bool:
        """Scale-free closeness: within 2% relative (covers rounding like
        $394.3bn vs 394,328,000,000, or 30.3% vs 0.303)."""
        return abs(a - b) <= rel_tol * max(abs(a), abs(b), 1e-9)

    @classmethod
    def _grounded(cls, magnitudes: set, evidence_mags: list[float]) -> bool:
        return any(cls._num_close(mag, ev) for mag in magnitudes for ev in evidence_mags)

    @classmethod
    def _derivation_base(cls, by_kind: dict[str, list[float]]) -> list[float]:
        """Deduped, capped pool of evidence magnitudes for `_derivable`.
        Exact structured lanes first so they survive the cap."""
        order = ("xbrl", "calc", "filing", "market", "web", "const")
        seen: set[str] = set()
        base: list[float] = []
        for kind in order + tuple(k for k in by_kind if k not in order):
            for v in by_kind.get(kind, []):
                key = f"{v:.6g}"
                if key not in seen:
                    seen.add(key)
                    base.append(v)
                    if len(base) >= cls._DERIVE_MAX_BASE:
                        return base
        return base

    @classmethod
    def _derivable(cls, magnitudes: set, base: list[float]) -> bool:
        """A figure absent from the evidence VERBATIM may still be correct if it
        is a one-step combination of evidence values — a ratio/margin (a/b, with
        the ×100 percent and ×365 days-outstanding forms), a sum, a change
        (a−b), or a two-period average. Without this, every answer that does the
        arithmetic the question asked for (margins, averages, working-capital
        days) self-refutes: the derived figure is "ungrounded" even though all
        its inputs are in the evidence.
        """
        close = cls._num_close
        for t in magnitudes:
            n = len(base)
            for i in range(n):
                a = base[i]
                for j in range(i + 1, n):
                    b = base[j]
                    if close(t, a + b) or close(t, abs(a - b)) or close(t, (a + b) / 2.0):
                        return True
                    for num, den in ((a, b), (b, a)):
                        if den == 0.0:
                            continue
                        r = num / den
                        if close(t, r) or close(t, r * 100.0) or close(t, r * 365.0):
                            return True
        return False

    def _evidence_numbers_by_kind(self, state: AgentState) -> dict[str, list[float]]:
        """Grounding magnitudes bucketed BY SOURCE KIND, so cross-source
        validation can tell which lanes corroborate a given figure.

        Exact structured lanes (xbrl, calc) are scale-expanded — see
        `_RESTATE_SCALES`; free-text lanes (filing, table, market, web) are taken
        at face value. The flat union (`_evidence_numbers`) is identical to the
        prior behaviour, so number GROUNDING is unchanged — this only adds the
        per-kind attribution used by the verification report.
        """
        buckets: dict[str, list[float]] = {"const": list(self._MATH_CONSTANTS)}

        def add_exact(kind: str, v: float) -> None:
            b = buckets.setdefault(kind, [])
            for scale in self._RESTATE_SCALES:
                b.append(v * scale)

        def add_text(kind: str, txt: str) -> None:
            b = buckets.setdefault(kind, [])
            for n in self._extract_numbers(txt):
                b.extend(n["magnitudes"])

        # XBRL — exact filed values (ground truth).
        for f in state.get("xbrl_facts", []) or []:
            v = f.get("value")
            if isinstance(v, (int, float)):
                add_exact("xbrl", float(v))

        # Derived metrics — result, percent form, trend series, AND the exact
        # XBRL inputs the metric was computed from (what a shown derivation uses).
        for r in state.get("calc_results", []) or []:
            v = r.get("value")
            if isinstance(v, (int, float)):
                add_exact("calc", float(v)); add_exact("calc", float(v) * 100.0)
            for s in r.get("series", []) or []:
                sv = s.get("value")
                if isinstance(sv, (int, float)):
                    add_exact("calc", float(sv)); add_exact("calc", float(sv) * 100.0)
            for inp in r.get("inputs", []) or []:
                iv = inp.get("value") if isinstance(inp, dict) else None
                if isinstance(iv, (int, float)):
                    add_exact("calc", float(iv))

        # Free-text lanes — face value.
        for c in state.get("retrieved_chunks", []) or []:
            add_text("filing", c.get("text", ""))
        for mkt in state.get("market_data", []) or []:
            add_text("market", str(mkt.get("data", "")))
        for h in state.get("web_results", []) or []:
            add_text("web", h.get("content", "") or "")
        return buckets

    def _evidence_numbers(self, state: AgentState) -> list[float]:
        """Flat union of every grounding magnitude across all source kinds."""
        mags: list[float] = []
        for vals in self._evidence_numbers_by_kind(state).values():
            mags.extend(vals)
        return mags


