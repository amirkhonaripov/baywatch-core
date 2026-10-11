#!/usr/bin/env bash
# Render every episode under RENDER_DIR with Blender (run from the code dir on the box, e.g. via scripts/box.sh).
# usage: sim/blender/render_all.sh RENDER_DIR [WORKER_INDEX N_WORKERS] [ENGINE]
# Workers split the episodes round-robin; restarting skips frames that already exist.
# SEEN_BLENDER / SEEN_SIM_ASSETS override the Blender binary and asset dir (e.g. an NVMe staging copy).
set -euo pipefail
R=${SEEN_ROOT:-/mnt/bigdrive/amirkhon/seen}
BL=${SEEN_BLENDER:-$R/tools/blender-5.0.1-linux-x64/blender}
ASSETS=${SEEN_SIM_ASSETS:-$R/data/raw/sim_assets}
OUT=$1; W=${2:-0}; N=${3:-1}; ENGINE=${4:-CYCLES}
i=0
for ep in $(ls -d "$OUT"/ep*/ | sort); do
  if [ $((i % N)) -eq "$W" ]; then
    "$BL" -b --python-exit-code 1 -P sim/blender/render_bay.py -- --episode "$ep/episode.json" \
      --assets "$ASSETS" --engine "$ENGINE" 2>&1 | grep -E "\[render\]|Error|Traceback" || true
  fi
  i=$((i + 1))
done
echo "worker $W done"
