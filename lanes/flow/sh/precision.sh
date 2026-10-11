#!/usr/bin/env bash
set -u
for m in rfdetr-large:bf16 rfdetr-large:fp16; do
  python -W ignore -m lanes.flow.run_detect --models $m --datasets detrac_test_s5,ndispark_train,ndispark_validation --batch 8 2>&1 | grep -E "^saved" | cut -c1-260
done
python -m lanes.flow.eval_detect --dataset detrac_test_s5 --models rfdetr-large,rfdetr-large:bf16,rfdetr-large:fp16 --exp precision_detrac --n-boot 200 2>&1 | tail -6
python -m lanes.flow.eval_ndispark_gap --models rfdetr-large,rfdetr-large:bf16,rfdetr-large:fp16 --exp precision_ndispark --n-boot 300 2>&1 | tail -6
echo PRECISION_DONE
