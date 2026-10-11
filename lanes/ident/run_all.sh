#!/usr/bin/env bash
# Regenerates every number reported in docs/lanes/ident_notes.md / ident_results.json (run on the box from the synced code dir):
#   scripts/box.sh --gpu 1 -- bash lanes/ident/run_all.sh <stage>
# Stages: data | baselines | train | export | eval | all-eval
# Data preparation and training are separate long jobs; see the commands below (each writes into runs/ident/<exp>/).
set -euo pipefail
R=${SEEN_ROOT:-/mnt/bigdrive/amirkhon/seen}
PRE=$R/data/raw/ident_pretrained
stage=${1:-all-eval}

data() {
  python -m lanes.ident.data.prep_autoria --cc kz kg ru                      # AUTO.RIA packs (NVMe)
  bash lanes/ident/data/stage_uz_rf.sh                                       # Roboflow Uzbek_Plates -> NVMe, splits, crops
  python -m lanes.ident.synth.gen_ocr --name test_synth_uz --n 3000 --seed 7 --native
  python -m lanes.ident.synth.gen_ocr --name synth_uz_val --n 3000 --seed 2000 --exclude-texts test_synth_uz.texts.txt
  python -m lanes.ident.synth.gen_ocr --name synth_uz_train --n 60000 --seed 1000 --exclude-texts test_synth_uz.texts.txt
  python -m lanes.ident.data.fetch_ccpd --want db:all weather:all challenge:1200 base:1800
  python -m lanes.ident.data.stage_det_eval --n 1000 --sets rodosol_test
  python -m lanes.ident.data.stage_photos
  python -m lanes.ident.synth.make_composites --rodosol 2500 --ccpd 2400 --photos
  python -m lanes.ident.data.build_det_dataset --variant uz_only
  python -m lanes.ident.data.build_det_dataset --variant full --ccpd 1200 --rodosol 700
  python -m lanes.ident.labels_uz build                                      # real-UZ test pack from lanes/ident/labels/*.json
}
baselines() {
  for m in cct_s_v2_global global_mobile_vit_v2_ocr; do
    python -m lanes.ident.eval_ocr --exp baseline_$m --model-dir $PRE --model $m --sets test_autoria_kz test_autoria_kg test_autoria_ru test_synth_uz --modes argmax
    python -m lanes.ident.eval_ocr --exp baseline_$m --model-dir $PRE --model $m --sets test_uz_real --modes argmax --tag _upright
    python -m lanes.ident.eval_ocr --exp baseline_$m --model-dir $PRE --model $m --sets test_uz_real_raw --modes argmax --orient --tag _orient_search
  done
  python -m lanes.ident.eval_det --exp baseline_fastalpr --detector fastalpr --sets uz_real_test rodosol_test ccpd_db ccpd_weather ccpd_challenge --n 1000
  python -m lanes.ident.eval_e2e --exp baseline_e2e --det fastalpr --ocr-model-dir $PRE --ocr-model cct_s_v2_global --mode argmax
}
train() {  # GPU0 (no flock when the integrator allows it), ~1-2 h each on the shared box
  python -m lanes.ident.train_ocr --name ocr_main --variant main --epochs 10 --batch 128 --lr 5e-4 --workers 8
  python -m lanes.ident.train_ocr --name ocr_real_only --variant real_only --epochs 10 --batch 128 --lr 5e-4 --workers 6
  python -m lanes.ident.train_det --variant full --model small --epochs 10 --batch 8 --res 512 --workers 6
  python -m lanes.ident.train_det --variant uz_only --model small --epochs 10 --batch 8 --res 512 --workers 6
}
export_all() {
  python -m lanes.ident.export_ocr --run ocr_main --out plate_ocr --static-batch 1 --version 0.1.0
  python -m lanes.ident.export_ocr --run ocr_real_only --out plate_ocr_real_only --static-batch 1 --version 0.1.0-ablation
  python -m lanes.ident.export_det --run det_full_small --out plate_det --res 512 --version 0.1.0
  python -m lanes.ident.export_det --run det_uz_only_small --out plate_det_uz_only --res 512 --version 0.1.0-ablation
}
evals() {
  A=$R/artifacts/ident
  python -m lanes.ident.eval_ocr --exp final --model-dir $A/plate_ocr --model model --sets test_autoria_kz test_autoria_kg test_autoria_ru test_synth_uz test_uz_real --modes argmax constrained auto
  python -m lanes.ident.eval_ocr --exp final --model-dir $A/plate_ocr --model model --sets test_uz_real_raw --modes argmax constrained auto --orient --tag _orient_search
  python -m lanes.ident.eval_ocr --exp final --model-dir $A/plate_ocr --model model --sets test_uz_real_raw --modes argmax --tag _no_orient_search
  python -m lanes.ident.eval_ocr --exp real_only --model-dir $A/plate_ocr_real_only --model model --sets test_autoria_kz test_autoria_kg test_autoria_ru test_synth_uz test_uz_real --modes argmax constrained
  python -m lanes.ident.eval_voting --exp final --model-dir $A/plate_ocr --model model --tracks 400
  for d in plate_det plate_det_uz_only; do
    python -m lanes.ident.eval_det --exp ours_$d --detector onnx:$A/$d --sets uz_real_test rodosol_test ccpd_db ccpd_weather ccpd_challenge --n 1000
  done
  python -m lanes.ident.eval_e2e --exp ours_e2e --det onnx:$A/plate_det --ocr-dir $A/plate_ocr --mode auto --orient
  python -m lanes.ident.bench_latency --exp final
  python -m lanes.ident.merge_results
}
case $stage in data) data;; baselines) baselines;; train) train;; export) export_all;; eval|all-eval) evals;; *) echo "unknown stage";; esac
