#!/usr/bin/env bash
# Build the gemini-embedding-2 FinanceBench index, then score it with both
# rerankers:
#
#   1. gemini-embedding-2 + markdown tables   (build + bge-reranker-v2-m3)
#   2. cohere rerank-v4.0-pro on that same index, no rebuild
#
# Only ONE index is built, deliberately. The pipe-vs-markdown A/B was settled
# on bge (pipe won, pool 0.909 vs 0.869) and the reason was bge's 1900-char
# embed cap, which Gemini's 7000-char cap removes — so markdown is the arm
# worth spending quota on, and splitting the day's budget across two builds
# would finish neither.
#
# BUDGET IN CHUNKS, NOT REQUESTS. The free tier meters the CONTENTS inside a
# request (~1000/day/key, per model), so the 44.5k-chunk corpus is a ~6-day
# build across the 7-key pool. This script is built to be re-run daily and
# resume, at three levels:
#
#   * the sqlite embed cache (data/embed_cache) — a vector is bought once, and
#     each batch is committed as it lands, so a mid-call quota death keeps it;
#   * the local Qdrant collection, which lives on a bind mount
#     (~/qdrant_eval_storage) and survives a container or machine restart;
#   * `--resume-build`, which keeps that collection and ingests only the
#     filings missing from it.
#
# So: `nohup scripts/overnight.sh &` and leave it. If it dies, or the machine
# reboots, run the same command again — it costs nothing to re-enter.
set -u

cd /home/sarthak/Documents/FinAgent
source ~/miniconda3/etc/profile.d/conda.sh
conda activate finagent

MODEL=gemini-embedding-2
ARM=gem2_md
LOGDIR=logs/retrieval; mkdir -p "$LOGDIR"
LOG="$LOGDIR/overnight.log"
say () { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }

# The eval corpus lives on the LOCAL Qdrant, never the served cluster. The
# harness sets this itself; exporting it here is what lets `progress` below use
# the same collection without importing the harness.
export QDRANT_EVAL_URL="${QDRANT_EVAL_URL:-http://localhost:6333}"

# Ask the harness for the collection name rather than re-deriving it: the tag
# encodes parent, child, embedder, headers and table format, and a copy of that
# formula here would drift and silently report on a collection nobody is
# building. Resolved once, because the import is not cheap.
COLL=$(python -c "
from finagent.evaluation.evaluate_retrieval import Config
print('sweep_' + Config(parent=2500, child=600, embed='$MODEL',
                        table_format='md').tag)") || exit 1

# Points in the sweep collection, so the log carries a progress trail across the
# days rather than just "attempt N".
progress () {
  python -c "from finagent.vectorstore import count; print(count('$COLL'))" \
      2>/dev/null || echo "?"
}

# Filings indexed, which is the number worth watching: the build advances one
# whole filing at a time (they are all-or-nothing since the pre-warm fix).
filings () {
  python -c "
from finagent.vectorstore import distinct_values
print(len(distinct_values('$COLL', 'source_url', limit=200000)))" 2>/dev/null || echo "?"
}

# Seconds until the daily embedding quota rolls over. Google free-tier daily
# counters reset at midnight Pacific, so this WAITS for the reset instead of
# polling for it.
#
# Polling was the first version and it was worse than useless. It asked for ONE
# embedding and treated a 200 as "the pool is open", but the build asks for 32
# at a time: a key with a handful of contents left answers the probe and 429s
# every real batch. The gate opened instantly on an empty pool, so the loop spun
# — three attempts in 22 minutes, the second indexing zero files — and would
# have burned all 30 attempts before dark, leaving nothing running overnight.
# A probe has to ask the question the caller actually asks, and a cheap probe
# cannot. Waiting for the reset needs no probe and spends no quota.
reset_epoch () {
  python - <<'PY'
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

now = datetime.now(ZoneInfo("America/Los_Angeles"))
# 00:10 rather than 00:00: the counters are not always readable the instant
# they roll, and ten minutes costs nothing against a six-day build.
today = now.replace(hour=0, minute=10, second=0, microsecond=0)
nxt = today if now < today else today + timedelta(days=1)
print(int(nxt.timestamp()))
PY
}

# Sleep to an ABSOLUTE deadline, re-reading the wall clock as it goes.
#
# One long `sleep $seconds` is the obvious version and it does not survive a
# laptop. `sleep` does not count time the machine spends suspended, so closing
# the lid at 10:39 with two hours to go meant the build was still waiting at
# 14:59 — the quota had been back for over two hours and nothing was using it.
# Short hops re-check the real time, so a suspend costs at most one hop.
sleep_until () {
  local target=$1 now
  while :; do
    now=$(date +%s)
    [ "$now" -ge "$target" ] && break
    sleep $(( target - now > 300 ? 300 : target - now ))
  done
}

# 1: build + score with the local cross-encoder. Each attempt embeds whatever
# the last one did not reach and pays nothing for what the cache already holds.
#
# Attempts get their OWN log, stamped with this run's start time. One truncated
# log would lose the trail across a six-day build; one appended log would make
# the QuotaExhausted grep match a stale traceback and abandon a build that had
# actually finished.
RUN=$(date '+%m%d_%H%M')
DONE=""
say "=== build starting ($RUN); $(filings)/72 filings, $(progress)/44544 points ==="
for ATTEMPT in $(seq 1 30); do
  say "build attempt $ATTEMPT: $MODEL / markdown tables"
  ALOG="${ARM}_${RUN}_a${ATTEMPT}"
  scripts/run.sh "$ALOG" --embed "$MODEL" --table-format md --resume-build || true

  # Success is a POSITIVE signal, not the absence of a quota error. Keying it on
  # "no EmbeddingQuotaExhausted" would read any other crash — a bad manifest, an
  # OOM, a Qdrant blip — as a finished build and send the cohere arm off to
  # score a half-built index.
  if grep -q "hit@8=" "$LOGDIR/$ALOG.log"; then
    say "build finished: $(filings)/72 filings, $(progress) points -> \
$(grep -o 'hit@8=[0-9.]*' "$LOGDIR/$ALOG.log" | tail -1)"
    DONE=1
    break
  fi

  if grep -q "EmbeddingQuotaExhausted" "$LOGDIR/$ALOG.log"; then
    TARGET=$(reset_epoch)
    say "daily quota gone at $(filings)/72 filings, $(progress)/44544 points \
— waiting until $(date -d "@$TARGET" '+%m-%d %H:%M') for the Pacific-midnight reset"
    sleep_until "$TARGET"
    continue
  fi

  say "attempt $ATTEMPT failed for a NON-quota reason; see $LOGDIR/$ALOG.log"
  tail -3 "$LOGDIR/$ALOG.log" | tee -a "$LOG"
  sleep 600
done

if [ -z "$DONE" ]; then
  say "still incomplete after 30 attempts ($(progress) points) — re-run to continue"
  exit 1
fi

# 2: Cohere on the SAME index (--skip-build), never on bge-large.
say "scoring the same index with cohere:rerank-v4.0-pro"
scripts/run.sh "${ARM}_cohere" --embed "$MODEL" --table-format md \
    --skip-build --rerankers cohere:rerank-v4.0-pro
say "cohere -> $(grep -o 'hit@8=[0-9.]*' "$LOGDIR/${ARM}_cohere.log" | tail -1)"
say "ALL DONE — both arms are in results/html_retrieval_sweep.md"
say "baseline to beat: bge-large served, pool 0.909 / hit@8 0.7475 (74/99)"
