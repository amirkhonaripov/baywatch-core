#!/usr/bin/env bash
# One sequential copy of Blender, the sim assets and the episode definitions to the NVMe staging dir (user-approved,
# docs/LANES.md "NVMe staging", sim budget 8 GB). Final renders go back to $SEEN_ROOT/renders/ with stage_back.
# usage: sim/blender/stage_nvme.sh to|back RENDER_NAME
set -euo pipefail
R=${SEEN_ROOT:-/mnt/bigdrive/amirkhon/seen}
F=${SEEN_FAST:-/home/amirkhon/seen_fast}/sim
NAME=${2:-bay_v1}
mkdir -p "$F"
if [ "$1" = to ]; then
  rsync -a "$R/tools/blender-5.0.1-linux-x64" "$F/"
  rsync -a "$R/data/raw/sim_assets" "$F/"
  rsync -a "$R/renders/$NAME" "$F/"
else
  rsync -a "$F/$NAME/" "$R/renders/$NAME/"
fi
du -sh "$F"
