#!/usr/bin/env bash
# Push the Mac repo to the box (code only). Outputs must never be written into code/ on the box.
# Usage: scripts/sync.sh            -> push code
#        scripts/sync.sh pull <box-relative-path> <local-dest>   -> fetch results (e.g. runs/ident/exp1/metrics.json)
set -euo pipefail
cd "$(dirname "$0")/.."
BOX=${SEEN_BOX:-rtx}
REMOTE=/mnt/bigdrive/amirkhon/seen
if [ "${1:-}" = "pull" ]; then
  rsync -az "$BOX:$REMOTE/$2" "$3"
  exit 0
fi
LOCKDIR=/tmp/seen_sync.lock
for i in $(seq 1 120); do mkdir "$LOCKDIR" 2>/dev/null && break; sleep 1; done
trap 'rmdir "$LOCKDIR" 2>/dev/null' EXIT
rsync -az --delete \
  --exclude '.venv/' --exclude 'node_modules/' --exclude 'web/dist/' --exclude '__pycache__/' \
  --exclude '.pytest_cache/' --exclude '*.pyc' --exclude '.DS_Store' \
  ./ "$BOX:$REMOTE/code/"
echo "synced to $BOX:$REMOTE/code/"
