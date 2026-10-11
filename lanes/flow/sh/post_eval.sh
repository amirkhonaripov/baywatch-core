#!/usr/bin/env bash
set -u
python -m lanes.flow.eval_detect --dataset cocoval_person --models rfdetr-base,rfdetr-large,yolo11x --exp zeroshot_cocoval_person --target person --n-boot 200 2>&1 | tail -6
python -m lanes.flow.make_failures --dataset detrac_test_s5 --model rfdetr-large --out $SEEN_ROOT/runs/flow/failures_det --per-cond 3 2>&1 | tail -2
python -m lanes.flow.eval_track --model rfdetr-large --exp track_rfdetr-large --n-boot 100 2>&1 | tail -3
echo POST_EVAL_DONE
