"""
orchestrator.py  ·  finagent/research/orchestrator.py

Deep Research Mode orchestrator.

    scope → plan → run specialists (through the production agent) → merge +
    cross-check findings → write the investment report.

Design
------
* **One execution engine.** Every specialist task runs through the injected
  `run_fn` — in production `finagent.api.rag_service.run_agentic`, i.e. the
  full `AgenticRAGv4` pipeline with retrieval, XBRL, calculator, tables,
  market data, web search, verification and citations. The orchestrator adds
  scoping, sequencing, retry, caching, merging and report writing — nothing
  the agent already does is reimplemented here.
* **Injected runner.** `run_fn: (question: str) -> {answer, chunks, metadata}`
  keeps the dependency direction clean (this module never imports the API
  layer) and makes the orchestrator fully testable without a network.
* **Citations stay auditable end to end.** Each specialist's `[N]` markers are
  local to its own run; the merger shifts them onto one global numbering that
  matches the concatenated evidence list, so every claim in the final report
  still clicks through to its source chunk.
* **Sequential by default.** The codebase serializes graph runs (Chroma/
  hnswlib + reranker thread-safety, see api/main.py); `RESEARCH_PARALLEL_AGENTS`
  raises the specialist fan-out for deployments that can afford it.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Optional

from pydantic import BaseModel, Field

from finagent.config import settings
from finagent.graph.base import AgenticRAG
from finagent.research.specialists import (
    DEFAULT_SPECIALISTS, SPECIALISTS, THESIS_LABEL, THESIS_TASK_ID,
    ResearchScope,
)

# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_CATALOG = "\n".join(
    f"- {s.id}: {s.label} — {s.template.split(':')[0].format_map({'company': 'the company', 'question': 'the user question', 'competitors': ''})}"
    for s in SPECIALISTS.values()
)

SCOPE_SYSTEM = f"""\
You are the research director of an equity research desk. A client has asked
for a deep research report. Identify the primary company, its ticker, up to 4
main competitors, restate the research objective in one sentence, and choose
which specialist analysts to assign, MOST relevant first.

Available specialists:
{_CATALOG}

Rules:
- Always include the core of a research report: company, financials, valuation, risk.
- Include 'custom' ONLY when the client asked a specific question beyond a
  general investment assessment (e.g. "how will tariffs affect Apple?").
- Include 'competitors' only when a peer comparison is relevant and you can
  name the competitors.
- 'political' and 'insider' only when ownership/trading activity is relevant
  to the objective.
"""

SCOPE_PROMPT = """\
{history_block}Client request: {question}

Return the research scope.
"""

CROSS_CHECK_SYSTEM = """\
You are a research editor checking specialist findings for consistency before
publication. List any CONTRADICTIONS between the findings — the same metric
with materially different values, opposing directional claims (growing vs
shrinking), or incompatible conclusions. Ignore mere differences of emphasis
or coverage. Return an empty list when the findings are consistent.
"""

CROSS_CHECK_PROMPT = """\
Findings:
{digest}

List the contradictions (empty list if none).
"""

REPORT_SYSTEM = """\
You are a senior equity research analyst writing an institutional-quality
investment report from your team's specialist findings.

Structure (GitHub-flavoured markdown):
- `## Executive Summary` — the answer first: 4-8 sentences a portfolio manager
  could act on.
- One `## <section>` per specialist finding provided, using the given section
  names, in the given order. Each section: a one-line summary in bold, then
  the detailed analysis. Interpret figures — never just restate them.
- `## Investment Thesis` — with `### Bull Case`, `### Bear Case`,
  `### Base Case`, `### Key Catalysts`, `### Major Risks`, and a final
  verdict line with your confidence (e.g. **Verdict: cautious buy —
  confidence 70%**).
- If an "Answer to Your Question" finding is present, keep it as its own
  section directly after the Executive Summary.

Citations and honesty:
- Cite by number only, `[N]`, using EXACTLY the citation numbers that appear
  in the findings — never renumber, never invent a number.
- Never invent figures, dates, or sources. If the findings don't support a
  claim, do not make it.
- State uncertainty explicitly: where findings are thin, contradictory, or a
  specialist failed, say so in the relevant section.
- Keep paragraphs short. Use tables for side-by-side comparisons.
"""

REPORT_PROMPT = """\
Research objective: {objective}
Company: {company}
Client's original request: {question}

Specialist findings (each with its section name; citation numbers are global
and must be reused verbatim):

{digest}
{cross_check_block}{failed_block}
---
Write the full investment report now.
"""


class CrossCheck(BaseModel):
    """Structured output of the consistency check."""

    contradictions: list[str] = Field(
        default_factory=list,
        description="One short sentence per contradiction found; empty if none.",
    )


# --------------------------------------------------------------------------- #
# Task cache — repeated research within ONE conversation reuses specialist runs.
# Keyed by the client's session id: findings are never shared between users.
# --------------------------------------------------------------------------- #

_CACHE: "OrderedDict[tuple, tuple[float, dict]]" = OrderedDict()
_CACHE_TTL = 3600.0      # seconds
_CACHE_MAX = 128
_cache_lock = threading.Lock()


def _cache_get(key: tuple) -> Optional[dict]:
    with _cache_lock:
        entry = _CACHE.get(key)
        if entry is None:
            return None
        ts, finding = entry
        if time.time() - ts > _CACHE_TTL:
            _CACHE.pop(key, None)
            return None
        _CACHE.move_to_end(key)
        return finding


def _cache_put(key: tuple, finding: dict) -> None:
    with _cache_lock:
        _CACHE[key] = (time.time(), finding)
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)


# --------------------------------------------------------------------------- #
# Citation renumbering
# --------------------------------------------------------------------------- #

_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]|【(\d+(?:\s*,\s*\d+)*)】")


def shift_citations(text: str, offset: int) -> str:
    """Shift every `[N]` / `【N】` marker by `offset` so per-specialist local
    citation numbers land on the merged global evidence numbering."""
    if not text or not offset:
        return text or ""

    def _sub(m: re.Match) -> str:
        inner = m.group(1) or m.group(2) or ""
        nums = [str(int(n) + offset) for n in re.split(r"\s*,\s*", inner)]
        return f"[{','.join(nums)}]"

    return _CITE_RE.sub(_sub, text)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

@dataclass
class Task:
    id: str
    label: str
    section: str
    question: str
    status: str = "pending"            # pending | running | done | failed
    detail: str = ""
    finding: dict = field(default_factory=dict)


class DeepResearch:
    """Plan and run a multi-specialist research workflow.

    Parameters
    ----------
    run_fn : Callable[[str], dict]
        Executes ONE research question through the production agent and
        returns `{"answer": str, "chunks": list[dict], "metadata": dict}`
        (the `rag_service.run_agentic` contract).
    provider / model / api_key
        LLM used for the orchestrator's own calls (scoping, cross-check,
        report). Defaults to the provider's synth tier.
    max_agents / parallel
        Bound the specialist fan-out; default from settings
        (RESEARCH_MAX_AGENTS / RESEARCH_PARALLEL_AGENTS).
    """

    def __init__(self, run_fn: Callable[[str], dict], provider: str = "groq",
                 model: Optional[str] = None, api_key: Optional[str] = None,
                 max_agents: Optional[int] = None,
                 parallel: Optional[int] = None,
                 session_id: Optional[str] = None):
        self.run_fn = run_fn
        # Scopes the task cache to one conversation. Without it the cache was
        # keyed on the question alone, so one user's specialist findings were
        # served to any other user asking the same thing.
        self.session_id = session_id
        self.provider = provider.lower()
        defaults = AgenticRAG.DEFAULTS.get(self.provider) or AgenticRAG.DEFAULTS["groq"]
        self.model = model or defaults["synth"]
        self.api_key = api_key
        self.max_agents = max_agents or settings.research_max_agents
        self.parallel = parallel or settings.research_parallel_agents
        self._chat = None

    # ------------------------------------------------------------------ #
    # LLM plumbing
    # ------------------------------------------------------------------ #

    def _llm(self):
        if self._chat is None:
            from finagent.llm import build_llm
            self._chat = build_llm(self.provider, self.model, self.api_key,
                                   temperature=0.0)
        return self._chat

    @staticmethod
    def _usage_of(response) -> tuple[int, int]:
        u = getattr(response, "usage_metadata", None) or {}
        return int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)

    # ------------------------------------------------------------------ #
    # Scope + plan
    # ------------------------------------------------------------------ #

    def scope(self, question: str,
              chat_history: Optional[list[dict]] = None) -> ResearchScope:
        """One structured call: company, ticker, competitors, objective, and
        the specialist ids to run. Falls back to the default specialist set
        when the call fails."""
        from langchain_core.messages import HumanMessage, SystemMessage

        history_block = ""
        if chat_history:
            lines = [f"{('User' if t.get('role') == 'user' else 'Assistant')}: "
                     f"{(t.get('content') or '')[:400]}"
                     for t in chat_history[-6:]]
            history_block = ("Recent conversation (most recent last):\n"
                             + "\n".join(lines) + "\n\n")
        try:
            llm = self._llm().with_structured_output(ResearchScope)
            out: ResearchScope = llm.invoke([
                SystemMessage(content=SCOPE_SYSTEM),
                HumanMessage(content=SCOPE_PROMPT.format(
                    history_block=history_block, question=question)),
            ])
            if out.specialists:
                return out
        except Exception as e:
            self._raise_if_exhausted(e)
            print(f"[research] scope failed ({type(e).__name__}); using defaults",
                  flush=True)
        # ponytail: fallback uses the raw request as the {company} slot — the
        # v4 planner re-decomposes each task question anyway, so a clumsy
        # phrasing still researches the right entity.
        return ResearchScope(company="", objective=question,
                             specialists=list(DEFAULT_SPECIALISTS))

    def plan(self, scope: ResearchScope, question: str) -> list[Task]:
        ids = [i for i in scope.specialists if i in SPECIALISTS]
        if not ids:
            ids = list(DEFAULT_SPECIALISTS)
        ids = ids[: self.max_agents]
        # The custom-question task runs last so the report writer places its
        # section right after the executive summary with full context around it.
        if "custom" in ids:
            ids = [i for i in ids if i != "custom"] + ["custom"]

        company = scope.company.strip() or question
        competitors = (" (" + ", ".join(scope.competitors[:4]) + ")"
                       if scope.competitors else "")
        fields = {"company": company, "question": question,
                  "competitors": competitors}
        return [Task(id=i, label=SPECIALISTS[i].label,
                     section=SPECIALISTS[i].section,
                     question=SPECIALISTS[i].template.format_map(fields))
                for i in ids]

    # ------------------------------------------------------------------ #
    # Specialist execution
    # ------------------------------------------------------------------ #

    @staticmethod
    def _raise_if_exhausted(e: Exception) -> None:
        """A fully exhausted key pool fails the WHOLE run fast — retrying the
        next specialist would just grind through the pool again."""
        from finagent.llm import AllKeysExhaustedError
        if isinstance(e, AllKeysExhaustedError):
            raise e

    def _run_task(self, task: Task) -> dict:
        key = (self.session_id, task.id, task.question.strip().lower(),
               self.provider, self.model) if self.session_id else None
        cached = _cache_get(key) if key else None
        if cached is not None:
            return {**cached, "cached": True}

        last_err: Optional[Exception] = None
        for _attempt in range(2):                    # one retry per specialist
            try:
                out = self.run_fn(task.question) or {}
                meta = out.get("metadata") or {}
                finding = {
                    "id": task.id, "label": task.label, "section": task.section,
                    "question": task.question,
                    "answer": out.get("answer") or "",
                    "chunks": out.get("chunks") or [],
                    "confidence": meta.get("confidence"),
                    "status": meta.get("answer_status"),
                    "input_tokens": meta.get("input_tokens") or 0,
                    "output_tokens": meta.get("output_tokens") or 0,
                }
                if key:
                    _cache_put(key, finding)
                return finding
            except Exception as e:
                self._raise_if_exhausted(e)
                last_err = e
        raise RuntimeError(
            f"{task.label} failed after retry: {type(last_err).__name__}")

    # ------------------------------------------------------------------ #
    # Merge + digest
    # ------------------------------------------------------------------ #

    @staticmethod
    def _merge_chunks(findings: list[dict]) -> list[dict]:
        """Concatenate every specialist's evidence onto one global numbering
        and rewrite each finding's answer citations to match. Sets
        `global_answer` on each finding; returns the merged chunk list."""
        merged: list[dict] = []
        offset = 0
        for f in findings:
            chunks = f.get("chunks") or []
            f["global_answer"] = shift_citations(f.get("answer") or "", offset)
            for c in chunks:
                c = dict(c)                       # never mutate cached entries
                c["id"] = offset + int(c.get("id", 0) or 0)
                c["agent"] = f.get("label", "")
                merged.append(c)
            offset += len(chunks)
        return merged

    @staticmethod
    def _digest(findings: list[dict], per_answer_cap: int = 3000) -> str:
        parts = []
        for f in findings:
            conf = f.get("confidence")
            conf_s = f" (confidence {conf:.0%})" if isinstance(conf, (int, float)) else ""
            parts.append(f"### {f['section']} — {f['label']}{conf_s}\n"
                         f"{(f.get('global_answer') or '')[:per_answer_cap]}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    # Cross-check + report
    # ------------------------------------------------------------------ #

    def _cross_check(self, digest: str) -> list[str]:
        from langchain_core.messages import HumanMessage, SystemMessage
        try:
            llm = self._llm().with_structured_output(CrossCheck)
            out: CrossCheck = llm.invoke([
                SystemMessage(content=CROSS_CHECK_SYSTEM),
                HumanMessage(content=CROSS_CHECK_PROMPT.format(digest=digest[:24000])),
            ])
            return [c.strip() for c in (out.contradictions or []) if c.strip()][:8]
        except Exception:
            # Best-effort — even an exhausted key pool must not fail the run
            # here: the specialists' evidence is already gathered, and the
            # report step below waits out the per-minute window itself.
            return []

    def _write_report(self, question: str, scope: ResearchScope,
                      findings: list[dict], contradictions: list[str],
                      failed: list[Task]) -> tuple[str, int, int]:
        from langchain_core.messages import HumanMessage, SystemMessage
        from finagent.llm import text_of

        digest = self._digest(findings)
        cross_block = ("\nCross-validation notes (address these explicitly):\n"
                       + "\n".join(f"- {c}" for c in contradictions) + "\n"
                       if contradictions else "")
        failed_block = ("\nSpecialists that FAILED (state these data gaps):\n"
                        + "\n".join(f"- {t.label}" for t in failed) + "\n"
                        if failed else "")
        from finagent.llm import AllKeysExhaustedError
        for attempt in range(2):
            try:
                resp = self._llm().invoke([
                    SystemMessage(content=REPORT_SYSTEM),
                    HumanMessage(content=REPORT_PROMPT.format(
                        objective=scope.objective or question,
                        company=scope.company or "(as stated in the request)",
                        question=question, digest=digest,
                        cross_check_block=cross_block, failed_block=failed_block)),
                ])
                tin, tout = self._usage_of(resp)
                report = text_of(resp)
                if report.strip():
                    return report, tin, tout
                break
            except AllKeysExhaustedError:
                # The specialists often exhaust the whole pool right before
                # this call — the ONE call that turns their minutes of work
                # into the report. Groq limits are per-minute: wait the window
                # out once instead of throwing the entire run away.
                if attempt:
                    break                            # still dry → fallback below
                print("[research] key pool exhausted at report step; "
                      "waiting 65s for the rate window", flush=True)
                time.sleep(65)
            except Exception as e:
                print(f"[research] report writer failed ({type(e).__name__}); "
                      f"assembling sections directly", flush=True)
                break
        # Deterministic fallback: the findings ARE the report, minus the thesis.
        parts = [f"# Research: {scope.company or question}",
                 "*The report writer was unavailable — below are the "
                 "specialist findings verbatim.*"]
        parts += [f"## {f['section']}\n{f.get('global_answer') or ''}"
                  for f in findings]
        return "\n\n".join(parts), 0, 0

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #

    def run(self, question: str, chat_history: Optional[list[dict]] = None,
            on_event: Optional[Callable[[dict], None]] = None) -> dict:
        """Execute the full research workflow. Emits progress dicts through
        `on_event` (SSE-ready: research_plan / agent_start / agent_done) and
        returns `{"report", "chunks", "metadata"}`."""

        def emit(event: dict) -> None:
            if on_event is None:
                return
            try:
                on_event(event)
            except Exception:
                pass                 # progress is decoration, never fatal

        t0 = time.time()
        scope = self.scope(question, chat_history)
        tasks = self.plan(scope, question)
        emit({"type": "research_plan",
              "company": scope.company, "ticker": scope.ticker,
              "objective": scope.objective or question,
              "tasks": ([{"id": t.id, "label": t.label} for t in tasks]
                        + [{"id": THESIS_TASK_ID, "label": THESIS_LABEL}])})

        findings: dict[str, dict] = {}

        def execute(task: Task) -> None:
            emit({"type": "agent_start", "id": task.id})
            try:
                f = self._run_task(task)
                task.status, task.finding = "done", f
                conf = f.get("confidence")
                detail = f"{len(f.get('chunks') or [])} evidence items"
                if isinstance(conf, (int, float)):
                    detail += f" · {conf:.0%} confidence"
                if f.get("cached"):
                    detail += " · cached"
                task.detail = detail
                findings[task.id] = f
                emit({"type": "agent_done", "id": task.id, "status": "done",
                      "detail": detail, "confidence": conf,
                      "summary": (f.get("answer") or "")[:400]})
            except RuntimeError as e:
                task.status, task.detail = "failed", str(e)
                emit({"type": "agent_done", "id": task.id, "status": "failed",
                      "detail": str(e)})

        if self.parallel > 1:
            # ponytail: opt-in only — the serve path pins graph runs to one
            # worker (Chroma/hnswlib + reranker thread-safety); raise
            # RESEARCH_PARALLEL_AGENTS only where run_fn is safe to reenter.
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.parallel) as pool:
                list(pool.map(execute, tasks))
        else:
            for task in tasks:
                execute(task)

        done = [t.finding for t in tasks if t.status == "done"]
        failed = [t for t in tasks if t.status == "failed"]
        if not done:
            raise RuntimeError("Every research specialist failed — no findings "
                               "to report on.")

        chunks = self._merge_chunks(done)

        emit({"type": "agent_start", "id": THESIS_TASK_ID})
        contradictions = self._cross_check(self._digest(done, per_answer_cap=1200)) \
            if len(done) > 1 else []
        report, rep_in, rep_out = self._write_report(
            question, scope, done, contradictions, failed)
        emit({"type": "agent_done", "id": THESIS_TASK_ID, "status": "done",
              "detail": f"{len(report.split())} words"
                        + (f" · {len(contradictions)} contradiction(s) flagged"
                           if contradictions else "")})

        confidences = [f["confidence"] for f in done
                       if isinstance(f.get("confidence"), (int, float))]
        agg_conf = round(sum(confidences) / len(confidences), 3) if confidences else None

        return {
            "report": report,
            "chunks": chunks,
            "metadata": {
                "mode": "research",
                "model": self.model,
                "company": scope.company, "ticker": scope.ticker,
                "objective": scope.objective or question,
                "confidence": agg_conf,
                "agents_run": len(done), "agents_failed": len(failed),
                "contradictions": contradictions,
                "evidence_count": len(chunks),
                "input_tokens": sum(f.get("input_tokens") or 0 for f in done) + rep_in,
                "output_tokens": sum(f.get("output_tokens") or 0 for f in done) + rep_out,
                "latency": round(time.time() - t0, 3),
                "tasks": [{"id": t.id, "label": t.label, "status": t.status,
                           "detail": t.detail} for t in tasks],
            },
        }
