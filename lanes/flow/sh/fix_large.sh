#!/usr/bin/env bash
set -u
python -W ignore -m lanes.flow.run_detect --models rfdetr-large --datasets ndispark_train,ndispark_validation,ndispark_test --batch 4 2>&1 | grep -v -E "it/s|iB/s"
M=rfdetr-base,rfdetr-large,rfdetr-medium,yolo11x,yolo26x
python -m lanes.flow.eval_ndispark_gap --models $M --exp zeroshot_ndispark
python -m lanes.flow.eval_count --models $M --exp zeroshot_ndispark_count
echo FIX_LARGE_DONE
