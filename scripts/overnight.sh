#!/usr/bin/env bash
# Build the gemini-embedding-2 index, then score it with both rerankers.
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
# request (~1000/day/key, per model), so a ~37k-chunk corpus is a multi-day
# build across the key pool. It resumes for free off data/embed_cache, which is
# why this script loops instead of running once.
set -u

cd /home/sarthak/Documents/FinAgent
source ~/miniconda3/etc/profile.d/conda.sh
conda activate finagent

MODEL=gemini-embedding-2
LOGDIR=logs/retrieval; mkdir -p "$LOGDIR"
LOG="$LOGDIR/overnight.log"
say () { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }

# Wait for quota. Polls with ONE real embedding request every 15 minutes (~4
# contents/hour out of ~6000/day — irrelevant) over raw HTTP, because the
# google-genai SDK reports a 429 as "client has been closed".
#
# It probes EVERY key and only sleeps when they all 429: the keys are separate
# projects whose daily counters roll over at different wall-clock times, so a
# single-key probe says nothing about the pool.
wait_for_quota () {
  say "waiting for the $MODEL daily quota…"
  until python - "$MODEL" <<'PY'
import json, sys, urllib.request, urllib.error
from dotenv import load_dotenv; load_dotenv("/home/sarthak/Documents/FinAgent/.env")
from finagent.llm import collect_provider_keys
model = sys.argv[1]
body = json.dumps({"requests": [
    {"model": f"models/{model}", "content": {"parts": [{"text": "quota probe"}]},
     "taskType": "RETRIEVAL_QUERY", "outputDimensionality": 1536}]}).encode()
for k in collect_provider_keys("gemini"):
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:batchEmbedContents",
        data=body, headers={"x-goog-api-key": k, "Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=30).read()
        sys.exit(0)          # at least one key is live
    except urllib.error.HTTPError:
        continue
sys.exit(1)
PY
  do sleep 900; done
  say "quota is back"
}

# 1: build + score with the local cross-encoder. Repeats until the build gets
# all the way through — each pass embeds whatever the last one didn't reach and
# pays nothing for what the cache already holds.
for ATTEMPT in $(seq 1 20); do
  wait_for_quota
  say "build attempt $ATTEMPT: $MODEL / markdown tables"
  scripts/run.sh "gem2_md" --embed "$MODEL" --table-format md || true
  if grep -q "QuotaExhausted" "$LOGDIR/gem2_md.log"; then
    say "daily quota gone mid-build — resuming after the next reset"
    continue
  fi
  say "build finished -> $(grep -o 'hit@8=[0-9.]*' "$LOGDIR/gem2_md.log" | tail -1)"
  break
done

grep -q "QuotaExhausted" "$LOGDIR/gem2_md.log" && { say "still incomplete — stopping"; exit 1; }

# 2: Cohere on the SAME index (--skip-build), never on bge-large.
say "scoring the same index with cohere:rerank-v4.0-pro"
scripts/run.sh "gem2_md_cohere" --embed "$MODEL" --table-format md \
    --skip-build --rerankers cohere:rerank-v4.0-pro
say "cohere -> $(grep -o 'hit@8=[0-9.]*' "$LOGDIR/gem2_md_cohere.log" | tail -1)"
say "ALL DONE — compare against the bge-large baseline of 79/99"
