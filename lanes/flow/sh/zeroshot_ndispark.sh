#!/usr/bin/env bash
# Zero-shot detectors on NDISPark (day = train split, night = validation split, night counts = test split). GPU1.
# One python process per model (loads weights once), skips caches that exist.
set -u
for m in rfdetr-base rfdetr-large rfdetr-medium yolo11x yolo26x; do
  python -W ignore -m lanes.flow.run_detect --models $m --datasets ndispark_train,ndispark_validation,ndispark_test --batch 4 2>&1 | grep -v -E "it/s|iB/s"
done
python -m lanes.flow.eval_detect --dataset ndispark_train --models rfdetr-base,rfdetr-medium,rfdetr-large,yolo11x,yolo26x --exp zeroshot_ndispark_day
python -m lanes.flow.eval_detect --dataset ndispark_validation --models rfdetr-base,rfdetr-medium,rfdetr-large,yolo11x,yolo26x --exp zeroshot_ndispark_night
