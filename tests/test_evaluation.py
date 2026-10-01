from finagent.evaluation.answers import verdict_agreement


def test_verdict_agreement_counts_only_numeric_rows_with_both_scores():
    rows = [
        {"financebench_id": "a", "qtype": "numeric", "gold": "$1577.00", "answer": "Capex was $1,577 million."},
        {"financebench_id": "b", "qtype": "numeric", "gold": "0.66", "answer": "The ratio was 0.71."},
        {"financebench_id": "c", "qtype": "numeric", "gold": "0.66", "answer": "The ratio was 0.66."},
        {"financebench_id": "d", "qtype": "judgement", "gold": "Yes", "answer": "Yes."},
        {"financebench_id": "e", "qtype": "numeric", "gold": "12%", "answer": "12%."},  # no verdict yet
    ]
    scored = {"a": {"verdict": 1.0}, "b": {"verdict": 0.0}, "c": {"verdict": 0.0}, "d": {"verdict": 1.0}}
    # a and b agree with the figure check; c disagrees; d and e are not counted.
    assert verdict_agreement(rows, scored) == {"n": 3, "agree": 2}
