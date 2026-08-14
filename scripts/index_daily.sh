#!/usr/bin/env bash
# Daily metered build of the gemini-embedding-2 FinanceBench index.
#
# Run it once a day. It resumes: a vector is bought once (sqlite embed cache in
# data/embed_cache, committed per batch) and a filing is ingested once (local
# Qdrant on the ~/qdrant_eval_storage bind mount). `--resume-build` keeps both
# and embeds only the chunks nobody reached yet, so re-running costs nothing for
# what is already done.
#
# ONE run spends the WHOLE day's quota. The embedder (finagent/vectorstore.py
# GeminiEmbeddings) already rotates across all keys, sleeps out per-minute
# limits, benches per-day-dead keys for 15 min, and only raises
# EmbeddingQuotaExhausted once every key is spent for the day. So a single
# invocation IS "use all keys effectively, then stop when the day is exhausted".
# When it stops, re-run tomorrow after the Pacific-midnight reset.
#
#   scripts/index_daily.sh            # today's chunk
#   nohup scripts/index_daily.sh &    # same, detached
#
# Unattended across days? Wrap it: `while :; do scripts/index_daily.sh && break
# || sleep 3600; done` (it exits 0 only when all 72 filings are in).
set -u
cd /home/sarthak/Documents/FinAgent
source ~/miniconda3/etc/profile.d/conda.sh && conda activate finagent

# The eval corpus lives on the LOCAL Qdrant, never the served cluster.
export QDRANT_EVAL_URL="${QDRANT_EVAL_URL:-http://localhost:6333}"
export PYTHONUNBUFFERED=1

MODEL=gemini-embedding-2
COLL=sweep_p2500_c600_gemini-embedding-2_hdr_tbl-md
TOTAL_FILINGS=72
TOTAL_CHUNKS=44544
PER_DAY=7400                      # observed throughput (~44.5k chunks / ~6 days)
LOG=logs/index_daily.log
mkdir -p logs
say(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }

count_pts(){ python -c "from finagent.vectorstore import count; print(count('$COLL'))" 2>/dev/null || echo 0; }
count_filings(){ python -c "from finagent.vectorstore import distinct_values; print(len(distinct_values('$COLL','source_url',limit=200000)))" 2>/dev/null || echo 0; }

# [######----------] 42%  <label>
bar(){
  local cur=$1 tot=$2 label=$3 width=28 fill pct b e
  (( cur > tot )) && cur=$tot
  pct=$(( tot>0 ? cur*100/tot : 0 ))
  fill=$(( tot>0 ? cur*width/tot : 0 ))
  b=$(printf '%*s' "$fill" '' | tr ' ' '#')
  e=$(printf '%*s' "$((width-fill))" '' | tr ' ' '-')
  printf '[%s%s] %3d%%  %s\n' "$b" "$e" "$pct" "$label" | tee -a "$LOG"
}

progress(){
  local f p remain days
  f=$(count_filings); p=$(count_pts)
  remain=$(( TOTAL_CHUNKS - p )); (( remain < 0 )) && remain=0
  days=$(( (remain + PER_DAY - 1) / PER_DAY ))
  bar "$p" "$TOTAL_CHUNKS" "$f/$TOTAL_FILINGS filings | $p/$TOTAL_CHUNKS chunks | ~${days}d left"
}

F0=$(count_filings)
say "=== index build: starting at $F0/$TOTAL_FILINGS filings ==="
progress
if [ "$F0" -ge "$TOTAL_FILINGS" ]; then
  say "ALL CHUNKS INDEXED — $(count_pts)/$TOTAL_CHUNKS chunks across $F0/$TOTAL_FILINGS filings. Nothing to do."
  exit 0
fi

say "embedding new chunks (resume from cache + qdrant)…"
python -m finagent.evaluation.evaluate_retrieval \
    --stage one --parent 2500 --child 600 --mode served \
    --embed "$MODEL" --table-format md \
    --rerankers none --resume-build --keep 2>&1 | tee -a "$LOG" || true

F1=$(count_filings)
say "=== after today's run ==="
progress
if [ "$F1" -ge "$TOTAL_FILINGS" ]; then
  say "ALL CHUNKS INDEXED — $(count_pts)/$TOTAL_CHUNKS chunks across $F1/$TOTAL_FILINGS filings. Build complete."
  exit 0
fi
say "today's embedding quota is spent (+$(( F1 - F0 )) filing(s), now $F1/$TOTAL_FILINGS). $(( TOTAL_FILINGS - F1 )) left — re-run tomorrow after the reset."
exit 1
