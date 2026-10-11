#!/usr/bin/env bash
# GPU1 chain: zero-shot detectors (loaded once per model), waits for the staged data. Metrics are recomputed from the cached detections.
set -u
F=${SEEN_FAST:-/home/amirkhon/seen_fast}/flow
MODELS=rfdetr-base,rfdetr-large,rfdetr-medium,yolo11x,yolo26x
until [ -f $F/ndispark/test/ground_truth_test_counting.csv ] && [ -d $F/ndispark/validation/imgs ]; do sleep 20; done
sleep 30
for m in rfdetr-base rfdetr-large rfdetr-medium yolo11x yolo26x; do
  python -W ignore -m lanes.flow.run_detect --models $m --datasets ndispark_train,ndispark_validation,ndispark_test --batch 4 2>&1 | grep -v -E "it/s|iB/s"
done
python -m lanes.flow.eval_ndispark_gap --models $MODELS --exp zeroshot_ndispark
python -m lanes.flow.eval_count --models $MODELS --exp zeroshot_ndispark_count
until [ -f $F/detrac/test/index_s5.json ]; do sleep 30; done
for m in rfdetr-base rfdetr-large rfdetr-medium yolo11x yolo26x; do
  python -W ignore -m lanes.flow.run_detect --models $m --datasets detrac_test_s5 --batch 8 2>&1 | grep -v -E "it/s|iB/s"
done
python -m lanes.flow.eval_detect --dataset detrac_test_s5 --models $MODELS --exp zeroshot_detrac
python -W ignore -m lanes.flow.run_detect --models rfdetr-base --datasets wider_val --batch 4 2>&1 | grep -v -E "it/s|iB/s"
python -W ignore -m lanes.flow.run_detect --models rfdetr-base,rfdetr-large,yolo11x --datasets cocoval_person --batch 8 2>&1 | grep -v -E "it/s|iB/s"
echo CHAIN_GPU_DONE
