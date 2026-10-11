#!/usr/bin/env bash
set -u
python -W ignore -m lanes.flow.export_onnx detector --spec "rfdetr-large@$SEEN_ROOT/runs/flow/ft_large/checkpoint_best_total.pth" --name rfdetr-large-flow --metrics-ref docs/lanes/flow_results.json --training-data "UA-DETRAC train (54 seq) + NDISPark train (day) + COCO val2017 subset (2,000+ rank images); see docs/lanes/flow_splits.md" 2>&1 | tail -2
python -W ignore -m lanes.flow.bench detector --spec rfdetr-large-ft --onnx $SEEN_ROOT/artifacts/flow/rfdetr-large-flow/model.onnx --image /home/amirkhon/seen_fast/flow/detrac/test/MVI_39031/img00001.jpg 2>&1 | grep -E "p50_ms|runtime|stage|top5" | head -12
echo FINAL_BOX_DONE
