"""The eval harness' own bookkeeping — the parts that silently produced wrong
numbers before, so a regression here is a wrong published metric, not a crash.

Covers: the judge context caps, the answered/evidence-recoverable subset masks,
and that a metric the judge failed on is visible in the report instead of being
averaged away.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from finagent.evaluation.financebench.parallel import _is_refusal, final_report
from finagent.evaluation.ragas import (
    CONTEXT_CHAR_CAP,
    CONTEXT_TOTAL_CHAR_CAP,
    METRIC_COLUMNS,
    _cap_contexts,
)

REFUSAL = ("I don't have enough information to answer this from the available "
           "filings. The claim-checking critic could not support the draft's "
           "claims from the gathered evidence.")


def test_cap_contexts_bounds_the_judge_prompt():
    caps = _cap_contexts(["x" * 5000] * 40)
    assert all(len(c) <= CONTEXT_CHAR_CAP for c in caps)
    assert sum(len(c) for c in caps) <= CONTEXT_TOTAL_CHAR_CAP
    # Order is retrieval order: the tail is dropped, not the head.
    kept = _cap_contexts(["a" * 100, "b" * 100])
    assert kept == ["a" * 100, "b" * 100]
    assert _cap_contexts(["", "   ", None if False else " "]) == []


def test_refusal_detection_matches_the_shipped_template():
    from finagent.prompts.critic import REFUSAL_TEMPLATE

    assert _is_refusal(REFUSAL_TEMPLATE.format(web_clause="", detail=""))
    assert _is_refusal(REFUSAL)
    assert not _is_refusal("**$1,577 million** (FY2018) [1]")
    assert not _is_refusal("")


def test_answered_and_recoverable_subsets(tmp_path, monkeypatch):
    """The two subset views must select on real signals.

    `answered` used to key off `answer_status == "answered"`, which nothing sets
    any more, and `recoverable` has to survive an absent id file.
    """
    from finagent.evaluation.financebench import runner

    outputs = [
        {"financebench_id": "a", "question": "qa", "answer": "42", "gold": "42",
         "qtype": "numeric", "retrieved_chunks": ["c"]},
        {"financebench_id": "b", "question": "qb", "answer": REFUSAL, "gold": "7",
         "qtype": "numeric", "retrieved_chunks": ["c"]},
    ]
    out_path = tmp_path / "outputs.json"
    out_path.write_text(json.dumps(outputs))
    ids_path = tmp_path / "ids.json"
    ids_path.write_text(json.dumps({"a": {}}))     # only "a" is recoverable
    monkeypatch.setattr(runner, "RECOVERABLE_IDS_PATH", str(ids_path))

    scored = pd.DataFrame([
        {"question": "qa", **{c: 1.0 for c in METRIC_COLUMNS}},
        {"question": "qb", **{c: 0.0 for c in METRIC_COLUMNS}},
        {"question": "*** MEAN ***", **{c: 0.5 for c in METRIC_COLUMNS}},
    ])
    csv = tmp_path / "scores.csv"
    scored.to_csv(csv, index=False)

    # Score without touching a judge: feed the CSV back through the aggregation.
    from finagent.evaluation import ragas as ragas_mod

    class _StubEvaluator:
        def __init__(self, *a, **k):
            pass

        def evaluate(self, *a, **k):
            return scored

    monkeypatch.setattr(ragas_mod, "RAGASEvaluator", _StubEvaluator)
    res = runner.score_answers(outputs_path=out_path, scores_csv=csv)

    assert res["n_scored"] == 2
    assert res["n_answered"] == 1                      # the refusal is excluded
    assert res["answered_only"]["faithfulness"] == 1.0
    assert res["n_recoverable"] == 1
    assert res["recoverable_only"]["faithfulness"] == 1.0
    assert res["scored_per_metric"]["faithfulness"] == 2


def test_partial_scores_are_carried_into_the_retry():
    """A rate-limited attempt must keep the metrics it already won.

    Discarding them meant the cheap context-free metric landed every attempt,
    the four context-heavy ones were refused every attempt, and the run churned
    forever without completing a single row.
    """
    # `ragas` is an eval-only extra and CI installs just the runtime deps, so
    # the assertion below that scoring makes no judge call would otherwise fail
    # on the import rather than on the behaviour it is checking. The rest of
    # this module needs no judge and keeps running.
    pytest.importorskip("ragas")

    from finagent.evaluation.ragas import RAGASEvaluator, _score_of

    prior = {"faithfulness": 0.75, "groundedness": float("nan"),
             "answer_relevancy": 0.5, "context_precision": None,
             "context_recall": "nan"}
    assert _score_of(prior, "faithfulness") == 0.75
    # nan / None / the string "nan" all mean "never scored" — not a real 0.
    for col in ("groundedness", "context_precision", "context_recall"):
        assert _score_of(prior, col) is None, col
    assert _score_of(None, "faithfulness") is None

    # With every metric already present, scoring must make NO judge call.
    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    ev.timeout, ev.max_workers = 60, 1
    full = {c: 1.0 for c in METRIC_COLUMNS}
    row = {"question": "q", "answer": "a", "gold": "g", "retrieved_chunks": ["c"]}
    exploding = object()          # any use of the judge would raise AttributeError
    out = ev._evaluate_one(row, exploding, exploding, "gold", prior=full)
    assert {c: out[c] for c in METRIC_COLUMNS} == full


def test_flush_keeps_partial_rows_it_has_not_reached_yet(tmp_path):
    """A mid-attempt flush must not erase partials from earlier attempts.

    `already_scored` holds COMPLETE rows only, so a partial row lives in `prior`
    until this attempt happens to re-reach it. Flushing without `prior` dropped
    it from the CSV; a crash before it was re-reached then lost the work, and
    the next attempt re-scored it from scratch — carry-forward silently undone.
    """
    from finagent.evaluation.ragas import RAGASEvaluator

    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    prior = {
        "done_q": {"question": "done_q", **{c: 1.0 for c in METRIC_COLUMNS}},
        "half_q": {"question": "half_q", "answer_relevancy": 0.4},   # partial
    }
    already = {"done_q": prior["done_q"]}
    fresh = [{"question": "new_q", **{c: 0.5 for c in METRIC_COLUMNS}}]

    csv = tmp_path / "out.csv"
    ev._flush_csv(csv, already, fresh, list(METRIC_COLUMNS), prior=prior)
    got = pd.read_csv(csv)
    got = got[got["question"] != "*** MEAN ***"]

    assert set(got["question"]) == {"done_q", "half_q", "new_q"}
    half = got[got["question"] == "half_q"].iloc[0]
    assert half["answer_relevancy"] == 0.4
    assert pd.isna(half["context_precision"])       # still missing, still retried

    # A row re-scored this attempt must win over its stale prior copy.
    better = [{"question": "half_q", **{c: 0.9 for c in METRIC_COLUMNS}}]
    ev._flush_csv(csv, already, better, list(METRIC_COLUMNS), prior=prior)
    got = pd.read_csv(csv)
    row = got[got["question"] == "half_q"].iloc[0]
    assert row["answer_relevancy"] == 0.9
    assert len(got[got["question"] == "half_q"]) == 1     # no duplicate row


def test_report_shows_rows_scored_per_metric(tmp_path):
    """A metric the judge only answered for half the rows must say so — the mean
    alone hid that v4's faithfulness came from 7 of 150 rows."""
    outputs = [{"financebench_id": "a", "question": "qa", "answer": "42",
                "gold": "42", "qtype": "numeric", "retrieved_chunks": ["c"]}]
    out_path = tmp_path / "outputs.json"
    out_path.write_text(json.dumps(outputs))

    md = tmp_path / "m.md"
    final_report(
        out_path,
        ragas_scores={"overall": {"faithfulness": 0.6}, "n_scored": 150,
                      "scored_per_metric": {"faithfulness": 7}},
        metrics_json=tmp_path / "m.json", metrics_md=md,
    )
    line = next(l for l in md.read_text().splitlines()
                if l.startswith("| faithfulness |"))
    assert "| 7 |" in line
