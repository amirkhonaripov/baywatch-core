#!/usr/bin/env bash
set -u
IMG=/home/amirkhon/seen_fast/flow/detrac/test/MVI_39031/img00001.jpg
A=$SEEN_ROOT/artifacts/flow
python -W ignore -m lanes.flow.export_onnx router --model google/siglip2-so400m-patch14-384 2>&1 | tail -2
python -W ignore -m lanes.flow.bench router --model google/siglip2-so400m-patch14-384 --onnx $A/router/encoder.onnx --image $IMG 2>&1 | tail -60
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo BENCH_ROUTER_DONE
