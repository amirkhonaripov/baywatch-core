#!/usr/bin/env bash
# Render the named episodes (optionally from the last frame backwards) to help the round-robin workers finish.
# usage: sim/blender/render_eps.sh RENDER_DIR REVERSE(0|1) EP [EP ...]
set -euo pipefail
R=${SEEN_ROOT:-/mnt/bigdrive/amirkhon/seen}
BL=${SEEN_BLENDER:-$R/tools/blender-5.0.1-linux-x64/blender}
ASSETS=${SEEN_SIM_ASSETS:-$R/data/raw/sim_assets}
OUT=$1; REV=$2; shift 2
EXTRA=${SIM_RENDER_EXTRA:-}  # e.g. "--every 4 --samples 16"
for ep in "$@"; do
  "$BL" -b --python-exit-code 1 -P sim/blender/render_bay.py -- --episode "$OUT/$ep/episode.json" \
    --assets "$ASSETS" --engine CYCLES --reverse "$REV" $EXTRA 2>&1 | grep -E "\[render\]|Error|Traceback" || true
done
echo "done $*"
