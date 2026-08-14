#!/usr/bin/env bash
# Daily RAGAS evaluation of the gemini-embed + cohere-rerank index.
#
# Run it once a day, after (or alongside) index_daily.sh. Two phases:
#
#   Phase A — ANSWER every question whose filing is now in the index. This is
#     what embeds the question (and its rewrite) as a RETRIEVAL_QUERY, caches
#     it, retrieves with the Cohere reranker, and synthesises an answer.
#     `--only-indexed` skips questions whose filing is not indexed yet (they
#     would score as a fake miss), so each day it only answers the filings the
#     latest build added. Resumable: already-answered questions carry over.
#
#   Phase B — RAGAS-SCORE the answers, alternating the two judge models
#     (gemini-3.5-flash / gemini-3.6-flash). The daily cap is per key PER MODEL,
#     so the two are separate buckets; switching when one empties drains both.
#     Resumable: each pass re-buys only the metric cells a row is still missing.
#     Stops when the day's judge quota is spent (5 dead rounds); re-run tomorrow.
#
# At the end it prints the RAGAS metrics computed on the answers evaluated SO
# FAR (a growing subset until the index is complete).
#
#   scripts/eval_daily.sh
#   nohup scripts/eval_daily.sh &
set -u
cd /home/sarthak/Documents/FinAgent
source ~/miniconda3/etc/profile.d/conda.sh && conda activate finagent

# The agent must run on the LOCAL gemini-embedding-2 index with Cohere rerank.
export QDRANT_EVAL_URL=http://localhost:6333
export FINANCEBENCH_COLLECTION=sweep_p2500_c600_gemini-embedding-2_hdr_tbl-md
export EMBEDDING_MODEL=gemini-embedding-2
export RERANKER_MODEL=cohere:rerank-v4.0-pro
export PYTHONUNBUFFERED=1
# Wait out the per-MINUTE judge limit instead of writing nan and re-buying the
# cell next pass. Nobody is watching this run.
export LLM_MAX_INLINE_WAIT_S=90
export LLM_MAX_WAIT_RETRIES=6

OUT=results/v6_partial/financebench_outputs.json
CSV=results/v6_partial/financebench_outputs_ragas.csv
LOG=logs/eval_daily.log
TOTAL_FILINGS=72
TOTAL_Q=127                       # served-HTML questions across all 72 filings
COLL=$FINANCEBENCH_COLLECTION
mkdir -p logs results/v6_partial
say(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }

bar(){
  local cur=$1 tot=$2 label=$3 width=28 fill pct b e
  (( cur > tot )) && cur=$tot
  pct=$(( tot>0 ? cur*100/tot : 0 ))
  fill=$(( tot>0 ? cur*width/tot : 0 ))
  b=$(printf '%*s' "$fill" '' | tr ' ' '#')
  e=$(printf '%*s' "$((width-fill))" '' | tr ' ' '-')
  printf '[%s%s] %3d%%  %s\n' "$b" "$e" "$pct" "$label" | tee -a "$LOG"
}

count_filings(){ python -c "from finagent.vectorstore import distinct_values; print(len(distinct_values('$COLL','source_url',limit=200000)))" 2>/dev/null || echo 0; }

# Questions whose filing is indexed right now = the answerable target.
answerable(){ python - <<'PY' 2>/dev/null || echo 0
import io, contextlib
from finagent.evaluation.financebench import parallel as P
with contextlib.redirect_stdout(io.StringIO()):
    try: n = len(P._restrict_to_indexed(P._load_all_questions()))
    except SystemExit: n = 0
print(n)
PY
}

answers(){ python -c "import json,os;print(len(json.load(open('$OUT'))) if os.path.exists('$OUT') else 0)" 2>/dev/null || echo 0; }

# Filled RAGAS cells: the honest progress/completion signal — a failed judge
# call is written as nan, not raised, so exit codes cannot be trusted.
cells(){ python - "$CSV" <<'PY' 2>/dev/null || echo 0
import os, sys, pandas as pd
from finagent.evaluation.ragas import METRIC_COLUMNS
p = sys.argv[1]
if not os.path.exists(p): print(0); raise SystemExit
d = pd.read_csv(p); d = d[d["question"] != "*** MEAN ***"]
print(int(sum(d[c].notna().sum() for c in METRIC_COLUMNS if c in d)))
PY
}

TARGET_ANS=$(answerable)
say "=== eval: $(answers)/$TARGET_ANS answered | $(cells) metric cells filled | $(count_filings)/$TOTAL_FILINGS filings indexed ==="

# ---------- Phase A: answer newly-indexed filings ----------
if [ "$(answers)" -lt "$TARGET_ANS" ]; then
  say "Phase A — answering questions whose filing is now indexed (embeds + caches the questions, Cohere rerank)…"
  python -m finagent.evaluation.financebench.parallel \
      --only-indexed --provider gemini --output "$OUT" 2>&1 | tee -a "$LOG" || true
else
  say "Phase A — all $TARGET_ANS answerable questions already answered, skipping."
fi
bar "$(answers)" "$TARGET_ANS" "Phase A: $(answers)/$TARGET_ANS answered"

# The zero-retrieval trap (fetch.py:406): if the embedding quota is dry, the
# agent catches EmbeddingQuotaExhausted and answers from ZERO chunks, and RAGAS
# would score that as a real miss. Cheap post-check (no quota): a fresh answer
# with an empty retrieved_chunks means Phase A ran on a spent embedding bucket.
ZERO=$(python -c "import json;print(sum(1 for x in json.load(open('$OUT')) if not x.get('retrieved_chunks')))" 2>/dev/null || echo 0)
if [ "$ZERO" -gt 0 ]; then
  say "!! WARNING: $ZERO answer(s) retrieved ZERO chunks — the embedding quota was"
  say "!! dry during Phase A. Run eval_daily.sh BEFORE index_daily.sh (answers need"
  say "!! ~300 embedding contents). Delete those rows and re-answer on fresh quota:"
  say "!!   python -c \"import json;p='$OUT';d=[x for x in json.load(open(p)) if x.get('retrieved_chunks')];json.dump(d,open(p,'w'),indent=2)\""
fi

# ---------- Phase B: RAGAS score, alternating judges ----------
ANS=$(answers); TARGET_CELLS=$(( ANS * 6 ))
say "Phase B — RAGAS scoring $ANS answers ($TARGET_CELLS cells), alternating gemini-3.5-flash / 3.6-flash…"
DEAD=0
for ROUND in $(seq 1 40); do
  BEFORE_ROUND=$(cells)
  for JUDGE in gemini-3.5-flash gemini-3.6-flash; do
    B=$(cells)
    say "round $ROUND, judge $JUDGE (from $B/$TARGET_CELLS)"
    python -m finagent.evaluation.financebench.parallel --score --provider gemini \
        --judge-model "$JUDGE" --no-retrieval-eval --output "$OUT" >> "$LOG" 2>&1 || true
    A=$(cells)
    bar "$A" "$TARGET_CELLS" "Phase B: $A/$TARGET_CELLS cells (last: $JUDGE +$(( A - B )))"
    [ "$A" -ge "$TARGET_CELLS" ] && { say "every metric on every answered question is filled"; break 2; }
  done
  if [ "$(cells)" -le "$BEFORE_ROUND" ]; then
    DEAD=$(( DEAD + 1 ))
    if [ "$DEAD" -ge 5 ]; then
      say "5 dead rounds at $(cells)/$TARGET_CELLS — today's judge quota is spent; re-run tomorrow"
      break
    fi
    say "no progress (dead round $DEAD/5) — backing off 90s"; sleep 90
  else
    DEAD=0
  fi
done

# ---------- metrics on whatever is evaluated so far ----------
# Computed straight from the RAGAS csv, so it is ALWAYS current — independent of
# whether today's scoring pass wrote final_metrics.md (an exhausted day may not).
say "=== RAGAS metrics so far ($(answers) answers generated) ==="
python - <<'PY' 2>&1 | tee -a "$LOG"
import os, pandas as pd
from finagent.evaluation.ragas import METRIC_COLUMNS
p = "results/v6_partial/financebench_outputs_ragas.csv"
if not os.path.exists(p):
    print("  no RAGAS csv yet — nothing scored"); raise SystemExit
d = pd.read_csv(p); d = d[d["question"] != "*** MEAN ***"]
n = len(d)
print(f"  {'metric':22} {'mean':>7}  fully-scored rows")
for c in METRIC_COLUMNS:
    if c in d:
        s = d[c].dropna()
        m = f"{s.mean():.3f}" if len(s) else "   -   "
        print(f"  {c:22} {m:>7}  {len(s)}/{n}")
print(f"  (means are DIRECTIONAL — per-metric coverage differs and judges are "
      f"mixed across rows; full report: results/v6_partial/final_metrics.md)")
PY

FIL=$(count_filings)
if [ "$FIL" -ge "$TOTAL_FILINGS" ] && [ "$(answers)" -ge "$TOTAL_Q" ] && [ "$(cells)" -ge "$(( TOTAL_Q * 6 ))" ]; then
  say "ALL CHUNKS INDEXED & RAGAS EVAL COMPLETED — $TOTAL_Q/$TOTAL_Q questions, $(( TOTAL_Q * 6 ))/$(( TOTAL_Q * 6 )) metric cells."
  exit 0
fi
say "partial ($FIL/$TOTAL_FILINGS filings, $(answers)/$TOTAL_Q answered, $(cells)/$(( $(answers) * 6 )) cells) — re-run daily until the index is complete."
exit 1
