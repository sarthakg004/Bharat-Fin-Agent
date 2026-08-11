#!/usr/bin/env bash
# run.sh <logname> <extra args...>  — one retrieval arm, logged to logs/.
#
# Logs live in the repo (gitignored) rather than a session scratchpad: the
# previous version hardcoded one Claude session's /tmp path, which stopped
# existing the moment that session ended.
set -u
cd /home/sarthak/Documents/FinAgent
LOGDIR=logs/retrieval; mkdir -p "$LOGDIR"
LOG="$LOGDIR/$1.log"; shift
source ~/miniconda3/etc/profile.d/conda.sh
conda activate finagent
exec python -m finagent.evaluation.evaluate_retrieval \
    --stage one --parent 2500 --child 600 --mode served \
    --rerankers BAAI/bge-reranker-v2-m3 --keep "$@" > "$LOG" 2>&1
