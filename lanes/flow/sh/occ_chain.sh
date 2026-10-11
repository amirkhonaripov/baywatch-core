#!/usr/bin/env bash
# Slot-occupancy detections (GPU1): PKLot + CNR-EXT calibration/test frames, zero-shot detectors, then the occupancy evaluation per model.
set -u
for m in rfdetr-large rfdetr-base yolo26x yolo11x; do
  python -W ignore -m lanes.flow.run_detect --models $m --datasets pklot_cal,pklot_occ,cnrext_cal,cnrext_occ --batch 4 2>&1 | grep -v -E "it/s|iB/s"
  python -m lanes.flow.eval_occupancy --model $m --exp occupancy_$m 2>&1 | tail -30
done
echo OCC_CHAIN_DONE
