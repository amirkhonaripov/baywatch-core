# Lane D (inspect): light inspection notes

## Current state / next steps
Status: DONE (wrapped up after the usage-limit pause). No box jobs running. Everything is regenerable with `lanes/inspect/RUN.md`.
- Artifacts (box, `/mnt/bigdrive/amirkhon/seen/artifacts/inspect/`): `damage_seg` (model B, 1024 px), `parts_seg` (800 px), `tyre_cond` (DINOv2 + binary head + experimental 3-way head), `dot_chars`; `dot_region` was exported but is NOT shipped (see DOT). All FP32 ONNX; no FP16 variant was tried (precision policy: ship FP32).
- Weights: `runs/inspect/{dmg_A,dmg_B,parts,dot_chars}_yolo/weights/best.pt`. Metrics: `docs/lanes/inspect_results.json` (31 ComponentResults). Predictor: `lanes/inspect/predict.py::get_predictor()` (`role="inspection"`, `name="seen-inspect"`, `version="0.1.0"`).
- Not done: RF-DETR-Seg (Apache-2.0 alternative to AGPL Ultralytics); tread-depth fine-tuning; FP16/TensorRT; GPU ONNX latency (onnxruntime CUDA provider failed to load on the box: needs cuDNN 9 / CUDA 13).
- Next commands: see `lanes/inspect/RUN.md`.

## What was built
Four capabilities, all measured on public data with held-out sources (splits and dedup: `inspect_splits.md`; licences: `inspect_sources.md`):
1. Exterior damage instance segmentation (YOLO11m-seg, 7 `DAMAGE_TYPES` classes). Model A = VehiDE + HITL (no CarDD); model B = A + CarDD train (shipped). Both trained 1.5 h on a shared GPU (13 epochs).
2. Car parts segmentation (20 unified side-agnostic parts) and damage->part assignment (max mask overlap).
3. Tyre condition (good/defective), experimental 3-way, tread depth (mm), DOT date-code reading with week/year validity check.
4. ONNX export + predictor.

## Data hygiene findings (worth showing)
- CarDamagePanel: 238 of 565 images are exact duplicates (pHash distance 0) of Humans-in-the-Loop parts images; removed from the test side (327 remain).
- Roboflow `jijo/car-damage-new`: 104 of 2,427 images are near-duplicates of CarDD/VehiDE training images; removed.
- Leakage audit with Lane G's `eval.manifest.leakage_audit` against our held-out manifests: PASS for damage A and B, parts, tyre, DOT (`inspect_leakage_audit.json`). `cardd_trainpart` is a test set for model A only.
- Several test sets contain near-duplicate groups (DOT test 558 images = 304 clusters; SYNDCAR 231 = 184). All CIs below resample near-duplicate clusters (`inspect_cluster_stats.json`).
- SYNDCAR is real photos of ONE vehicle (Citroen Ami, two vehicle IDs) with digitally generated damage (our reading of the images; method undocumented), so every SYNDCAR result is tagged `data_kind: synthetic`.

## Results (95% CIs in `inspect_results.json`; domain = phone/web photos, no fixed-camera damage data exists)
Damage, box AP50 (mask AP50 in the JSON), model B / model A:
| Test set | n | B mAP50 | A mAP50 | kind |
|---|---|---|---|---|
| CarDD held-out 20% | 543 | 0.699 | 0.413 (cross-source for A) | in-source for B |
| VehiDE official val | 2322 | 0.433 | 0.486 | in-source |
| HITL damage test | 82 | 0.137 | 0.161 | in-source (tiny, heterogeneous classes) |
| SYNDCAR (synthetic damage, one car) | 231 | 0.177 | 0.184 | cross-source |
| Coco Car Damage, any-damage AP50 | 69 | 0.218 | 0.200 | cross-source |
| Roboflow jijo (dent/scratch, 416 px stretched) | 2323 | 0.177 | 0.145 | cross-source |
Model B on CarDD test: dent 0.57, scratch 0.57, crack 0.28, glass 0.99, lamp 0.87, flat tyre 0.92. Recall (any damage) at 0.5 false positives/image: 0.60 on CarDD test, 0.33 on VehiDE val, 0.12 on SYNDCAR. Adding CarDD raised CarDD AP but did not help SYNDCAR: cross-source damage detection is weak, and cross-source numbers are the ones to quote.

Parts (mask AP50): in-source carparts test 0.948 (n=548), HITL parts 0.909 (n=168); cross-source CarDamagePanel (deduped) 0.869 (n=327); SYNDCAR at coarse-family level 0.611 (n=231; front/rear end and windshield are weak on this micro-car).
Damage->part on SYNDCAR (family level, reference part derived from GT part polygons, synthetic damage): with GT damage boxes + predicted parts 70.2% (n=1295 instances, image-independence assumed in the CI); end to end only 10% of GT damage is detected at conf >= 0.25, so its 75% (n=128) is not meaningful.

Tyre good/defective (TyreNet-trained): in-source 98.9% (n=357, AUROC 1.0; optimistic, likely easy sessions); warcoder cross-dataset (TyreNet near-duplicates removed, only 2) 87.1% [85.4, 88.5], AUROC 0.972, defective recall 0.77; sameer cross-dataset (NEW+SERVICEABLE = good, UNUSABLE = defective) 56.7%, AUROC 0.70: it does NOT transfer to that source (kart-like slick tyres labelled unusable, dirty serviceable tread). 3-way head (good/worn/defective, `worn` from sameer's README definition): 85.0% in 5-fold CV inside sameer only (majority baseline 0.56); as a good-vs-defective view on warcoder 83.0%. Shipped default is the binary head; `emit_worn=True` is experimental.
Tread depth (TyreCV, 508 stereo pairs, grouped 5-fold by pair, DINOv2 + ridge, mean of two views): MAE 0.62 mm [0.58, 0.66], within 1 mm 81%, vs 1.06 mm for the median baseline. Physical-tyre identity is unknown, so CV may be optimistic; lab stereo camera, not a phone.
DOT: character reader on TyreAI test (304 clusters): full-string exact match 69.5% [64.0, 75.1], character error rate 7.1%; date code exact on digit-serial codes (n=145) 89.7% [80.0, 97.0]. The week/year validator accepts 100% of correct readings but rejects only a third of wrong ones (wrong readings are usually still a plausible week/year). **DOT region detector failed**: on an independent source (97 images) AP50 = 0.0 (confidences <= 0.03), so full-photo DOT localisation does not work; the predictor reads DOT only from close-ups framed on the code (region model is opt-in and not shipped).

Speed / ONNX: PyTorch eval on GPU1 (shared, loaded box) 12-37 ms per image including pre/post for the 1024 px seg model. ONNX parity vs PyTorch: damage 98.1% of detections found, parts 98.4%, DOT chars 100%, tyre max diff 1e-7. The onnxruntime CUDA provider did not load, so the ONNX predictor ran on CPU: p50 1.36 s per 1024x768 frame for damage+parts (p95 1.66 s); GPU ONNX latency is NOT measured. Predictor check (`inspect_predictor_check.json`): tyre probabilities match the training pipeline within 0.01 on average; through the predictor 124 of 145 digit-serial date codes are recovered.

## Failure cases (`docs/lanes/failures/inspect/`, 12 images)
SYNDCAR damage misses/false positives/wrong class (plates blurred); tyre misses (kart-style slick tyres labelled unusable, sidewall-lettering close-ups labelled defective, dirty tread flagged defective); DOT misreads (mirrored augmentation, curved two-line codes, X read as K, size text 235 read as 225). Web-photo galleries from CarDD/VehiDE were generated but not kept because we did not check them for faces. Categories asked for in the brief: small scratches are the main miss type (scratch AP50 0.57 at best); reflections and dirt were not separately labelled in any dataset, so no number is claimed for them.

## Known limits
- All damage data are web/phone photos; nothing fixed-camera, wet or dirty. Cross-source damage detection is poor (AP50 about 0.15-0.22).
- Models were trained 13 epochs on a shared GPU; they are baselines, not tuned.
- Vocabulary: `part` names are ours (see Requests); tyre `worn` is experimental; `tire_flat` comes only from CarDD.
- Confidence calibration was not evaluated for damage.

## For the deck (must-mention caveats)
- Damage detection is validated on phone/web photos only; cross-source AP50 is about 0.15-0.22 (SYNDCAR, Coco, Roboflow) versus 0.70 in-source on CarDD. Say "works on photos like its training data", not "works at the gate".
- SYNDCAR = one vehicle model with digitally generated damage: its numbers are synthetic and weak evidence.
- CarDD licence: research use only, prior consent from the PIC Lab, no redistribution, "testing commercial systems" counts as commercial. We got it via a Hugging Face mirror carrying Voxel51's signed form, not ours. Model B must not go into a commercial pilot without that authorisation; model A (no CarDD) is the fallback (VehiDE licence is also unclear; HITL is CC0).
- Ultralytics code/weights are AGPL-3.0: a commercial deployment needs a licence or an RF-DETR retrain.
- Test-set contamination was found and removed (CarDamagePanel 42% duplicates of HITL; Roboflow jijo vs CarDD).
- Tyre: in-source 98.9% is optimistic; independent set 87%; one other source 57%. Tread depth is from a lab stereo rig.
- DOT works on close-ups framed on the code only (reader 90% on digit-serial date codes); region detection failed.
- Faces/plates: public datasets contain plates; gallery images are plate-blurred or plate-free.

## Requests to integrator
- Add `PART_NAMES` to `seen/vocab.py`: front_bumper, back_bumper, hood, trunk, front_door, back_door, fender, quarter_panel, headlight, tail_light, mirror, windshield, back_windshield, front_window, back_window, wheel, roof, grille, rocker_panel, license_plate (and `part:<name>` labels if the UI wants them; `emit_parts=True`).
- Adopt `docs/lanes/inspect_manifests/*.jsonl` into `eval/manifests/` (G).
- Fix the onnxruntime CUDA provider (cuDNN 9 / CUDA 13) if GPU ONNX latency is needed; our ONNX latency is CPU only.
- NVMe: `/home/amirkhon/seen_fast/inspect` (about 5 GB) can be deleted once nobody needs retraining; the raw HDD downloads under `data/raw/inspect_*` (~25 GB) are re-fetchable by `lanes/inspect/fetch*.py`.
