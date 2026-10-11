#!/usr/bin/env bash
# ONNX export + latency (GPU1, shared with other jobs: utilisation is recorded in the notes). Real 960x540 DETRAC frame.
set -u
IMG=/home/amirkhon/seen_fast/flow/detrac/test/MVI_39031/img00001.jpg
A=$SEEN_ROOT/artifacts/flow
[ -f $A/rfdetr-large-coco/model.onnx ] || python -W ignore -m lanes.flow.export_onnx detector --spec rfdetr-large --name rfdetr-large-coco 2>&1 | tail -2
python -W ignore -m lanes.flow.bench detector --spec rfdetr-base --onnx $A/rfdetr-base-coco/model.onnx --image $IMG 2>&1 | tail -40
python -W ignore -m lanes.flow.bench detector --spec rfdetr-large --onnx $A/rfdetr-large-coco/model.onnx --image $IMG 2>&1 | tail -40
python -W ignore -m lanes.flow.bench detector --spec yolo26x --image $IMG 2>&1 | tail -20
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo BENCH_DONE
