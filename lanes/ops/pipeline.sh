#!/bin/bash
# Lane C end-to-end (box). Reproduces every reported number after the raw data is in place:
#   data/raw/ops_k700 (lanes.ops.acquire.k700), CrossTask/COIN frames (lanes.ops.prep frames_howto_ff + howto_subsample,
#   run where the videos are), stills (lanes.ops.prep stills / yolo), CCTV frames (lanes.ops.acquire.cctv sample).
# Usage: bash lanes/ops/pipeline.sh <exp> [stage...]   stages: table detect embed teacher_items cctv_detect vlm_cctv
#        vlm_teacher train export cctv  (default: all, in this order)
set -euo pipefail
R=/mnt/bigdrive/amirkhon/seen
F=/home/amirkhon/seen_fast
EXP=${1:-heads_v1}; shift || true
STAGES=${*:-"table detect parts embed teacher_items cctv_detect vlm_cctv vlm_teacher train export cctv handeval parity report"}
cd $R/code
source $F/env/main/bin/activate
export SEEN_ROOT=$R SEEN_FAST=$F PYTHONPATH=$R/code HF_HOME=$R/tools/hf-cache SEEN_PHASH_CACHE=$F/ops/phash_cache.jsonl
VLM_MODEL=${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}
VLM_HF=$F/ops/hf   # VLM weights live on the NVMe (HDD writes stalled)
mkdir -p $R/runs/ops/$EXP $R/runs/ops/vlm
for st in $STAGES; do
  echo "== $st $(date +%T)"
  case $st in
    table) python -m lanes.ops.prep table ;;
    detect) CUDA_VISIBLE_DEVICES=1 python -m lanes.ops.featurize detect ;;
    embed) CUDA_VISIBLE_DEVICES=1 python -m lanes.ops.featurize embed --backbone dinov2l ;;
    parts) CUDA_VISIBLE_DEVICES=1 python -m lanes.ops.featurize parts ;;
    handcheck) python -m lanes.ops.teacher sheets && python -m lanes.ops.eval_handcheck --exp $EXP ;;
    report) python -m lanes.ops.report --exp $EXP ;;
    parity) CUDA_VISIBLE_DEVICES=1 python -m lanes.ops.parity --exp $EXP ;;
    handeval) python -m lanes.ops.eval_handcheck --exp $EXP ;;
    teacher_items) python -m lanes.ops.teacher items ;;
    cctv_detect) CUDA_VISIBLE_DEVICES=1 python -m lanes.ops.eval_cctv detect && python -m lanes.ops.eval_cctv vlm_items ;;
    vlm_cctv) CUDA_VISIBLE_DEVICES=1 HF_HOME=$VLM_HF python -m lanes.ops.vlm --engine hf --model $VLM_MODEL --crop --batch 8 \
                --timing-n 30 --items $F/ops/cctv/vlm_items.parquet --out $R/runs/ops/vlm/cctv_vlm.parquet ;;
    vlm_teacher) CUDA_VISIBLE_DEVICES=1 HF_HOME=$VLM_HF python -m lanes.ops.vlm --engine hf --model $VLM_MODEL --batch 16 --crop \
                --items $F/ops/teacher_items.parquet --out $F/ops/feat/vlm_teacher.parquet
                 CUDA_VISIBLE_DEVICES=1 HF_HOME=$VLM_HF python -m lanes.ops.vlm --engine hf --model $VLM_MODEL --batch 16 --crop \
                --items $F/ops/handcheck_items.parquet --out $R/runs/ops/vlm/handcheck_vlm.parquet ;;
    train) T=""; [ -f $F/ops/feat/vlm_teacher.parquet ] && [ "${NO_TEACHER:-0}" != 1 ] && T="--teacher $F/ops/feat/vlm_teacher.parquet"
           CUDA_VISIBLE_DEVICES= python -m lanes.ops.train --exp $EXP --backbone dinov2l $T ;;
    export) CUDA_VISIBLE_DEVICES=1 python -m lanes.ops.export --exp $EXP --dest runs --speed ;;
    cctv) CUDA_VISIBLE_DEVICES=1 python -m lanes.ops.eval_cctv predict --exp $EXP && \
          python -m lanes.ops.eval_cctv metrics --exp $EXP --vlm $R/runs/ops/vlm/cctv_vlm.parquet ;;
    *) echo "unknown stage $st"; exit 2 ;;
  esac
done
echo "PIPELINE_DONE $EXP"
