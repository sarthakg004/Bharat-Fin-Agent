#!/usr/bin/env bash
# Ingest filings into the ONLINE (served) Qdrant index the DEPLOYED app queries.
#
# This is production. It writes to the cloud cluster's `us_filings_v5_gemini`
# collection (gemini-embedding-2, 1536 dims) — the one Cloud Run reads. Unlike
# index_daily.sh (which fills the LOCAL eval index), this grows the real corpus
# users hit. Run it occasionally, when you have embedding quota to spare and the
# eval isn't using it — the two share the same gemini-embedding-2 daily bucket.
#
# Safe to repeat: ingestion is idempotent (skips any filing whose source_url is
# already in the collection) and resumable (the data/embed_cache vectors persist
# and a mid-run quota death picks up where it stopped). It NEVER deletes.
#
#   scripts/ingest_online.sh                       # ingest the default manifest
#   scripts/ingest_online.sh path/to/manifest.json # a different set of filings
#   scripts/ingest_online.sh -y                    # skip the confirmation prompt
set -u
cd /home/sarthak/Documents/FinAgent
source ~/miniconda3/etc/profile.d/conda.sh && conda activate finagent

# CRITICAL: target the ONLINE cluster, never the local eval one. get_client()
# routes to QDRANT_EVAL_URL whenever it is set, so a stray export left over from
# index_daily.sh / eval_daily.sh would silently write production filings into
# the local eval store. Clear it so this always hits QDRANT_CLUSTER_ENDPOINT.
unset QDRANT_EVAL_URL QDRANT_EVAL_API_KEY
export PYTHONUNBUFFERED=1

COLL=us_filings_v5_gemini      # deploy US_COLLECTION / settings.us_collection
MODEL=gemini-embedding-2       # matches the collection's 1536 dims
LOG=logs/ingest_online.log
mkdir -p logs

# args: optional manifest path (positional), -y/--yes to skip confirmation
YES=0
MANIFEST=data/us/pdfs/rebuild_manifest.json
for a in "$@"; do case "$a" in -y|--yes) YES=1 ;; *) MANIFEST="$a" ;; esac; done

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

[ -f "$MANIFEST" ] || { say "manifest not found: $MANIFEST"; exit 1; }

online_pts(){ python -c "from finagent.vectorstore import count; print(count('$COLL'))" 2>/dev/null || echo 0; }
online_filings(){ python -c "from finagent.vectorstore import distinct_values; print(len(distinct_values('$COLL','source_url',limit=200000)))" 2>/dev/null || echo 0; }
manifest_files(){ python -c "import json,os;d=json.load(open('$MANIFEST'));r=d if isinstance(d,list) else d.get('records',[]);print(sum(1 for x in r if x.get('status','ok')=='ok' and os.path.exists(x.get('local_path','')))) " 2>/dev/null || echo 0; }
host(){ python -c "from urllib.parse import urlparse; from finagent.vectorstore import qdrant_url; print(urlparse(qdrant_url()).hostname or qdrant_url())" 2>/dev/null || echo "?"; }

TOTAL=$(manifest_files)
F0=$(online_filings); P0=$(online_pts)

say "=== ONLINE ingest — writes to the LIVE served index ==="
say "cluster    : $(host)"
say "collection : $COLL   ($P0 points, $F0 filing(s) present)"
say "embedding  : $MODEL"
say "manifest   : $MANIFEST   ($TOTAL filing(s) on disk)"
bar "$F0" "$TOTAL" "$F0/$TOTAL filings | $P0 points"

# Production write → confirm the target unless -y. A non-tty read (nohup) fails
# closed, which is the safe default: it will not write to production unattended
# unless you pass -y.
if [ "$YES" -ne 1 ]; then
  printf "Write to the LIVE served index above? type 'yes' to proceed: "
  read -r ans || ans=""
  [ "$ans" = yes ] || { say "aborted (no changes)."; exit 1; }
fi

say "ingesting (idempotent — skips filings already in the collection)…"
python -m finagent.ingestion.ingest \
    --manifest "$MANIFEST" --collection "$COLL" --market us \
    --embedding-model "$MODEL" 2>&1 | tee -a "$LOG" || true

F1=$(online_filings); P1=$(online_pts)
say "=== after ingest ==="
bar "$F1" "$TOTAL" "$F1/$TOTAL filings | $P1 points (+$(( P1 - P0 )))"
if [ "$F1" -ge "$TOTAL" ]; then
  say "DONE — all $TOTAL manifest filing(s) are in $COLL ($P1 points)."
  exit 0
fi
say "added $(( F1 - F0 )) filing(s) this run; $(( TOTAL - F1 )) left. If it stopped early it's the gemini-embedding-2 daily quota — re-run when it resets."
exit 1
