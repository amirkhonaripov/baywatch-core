#!/usr/bin/env bash
# Tiled inference for the wide PKLot frames (small cars): full frame + 3 overlapping square tiles; CNR-EXT frames (1000x750) get the full pass only.
set -u
for m in rfdetr-large+tile yolo26x+tile; do
  python -W ignore -m lanes.flow.run_detect --models $m --datasets pklot_cal,pklot_occ,cnrext_cal,cnrext_occ --batch 4 2>&1 | grep -v -E "it/s|iB/s"
  python -m lanes.flow.eval_occupancy --model $m --exp occupancy_$m 2>&1 | tail -22
done
echo OCC_TILE_DONE
