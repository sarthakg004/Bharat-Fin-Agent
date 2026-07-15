"""Self-check for the custom eval's deterministic scorers (no agent, no LLM)."""

from finagent.evaluation.custom import _period_in, score_row


def test_period_in_orderings():
    ans = "Operating margin was 11.4% in Q1 FY2026, up from 9.9% in the first quarter of 2025."
    assert _period_in("Q1 2026", ans)
    assert _period_in("Q1 2025", ans)
    assert not _period_in("FY2023", ans)
    assert _period_in("FY2022", "revenue was $394.3 billion in fiscal 2022")


def test_score_row_flags_wrong_year_answer():
    spec = {
        "gold_answer": "improved 1.5 percentage points",
        "expected_periods": ["Q1 2026", "Q1 2025"],
        "forbidden_periods": ["FY2022", "FY2023"],
        "gold_evidence": ["sales and marketing"],
    }
    # The observed bad answer: right shape, wrong years.
    bad = {"answer": "Operating margin rose 3.6 pp from 4.9% in FY2022 to 8.5% in FY2023.",
           "retrieved_chunks": ["total operating expenses were $12,084 million"]}
    s = score_row(bad, spec)
    assert s["period_match"] is False
    assert s["stale_period"] is True
    assert s["evidence_recall"] == 0.0

    good = {"answer": "Operating margin improved 1.5 percentage points, from 9.9% (Q1 2025) to 11.4% (Q1 2026).",
            "retrieved_chunks": ["Sales and marketing expenses were $1,234 million"]}
    s = score_row(good, spec)
    assert s["period_match"] is True
    assert s["stale_period"] is False
    assert s["evidence_recall"] == 1.0
    assert s["numeric_accuracy"] is True


if __name__ == "__main__":
    test_period_in_orderings()
    test_score_row_flags_wrong_year_answer()
    print("ok")
