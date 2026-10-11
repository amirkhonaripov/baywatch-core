# Lane B (flow + router + face blur): notes

## Current state / next steps

**All box jobs finished; nothing is running.** Lane complete pending integrator wiring.
* Shipped detector: `artifacts/flow/rfdetr-large-flow/` (RF-DETR Large fine-tuned, `model.onnx` + `checkpoint.pth` + `model_card.json`, FP32, role "vehicle"); zero-shot ONNX `rfdetr-{base,large}-coco/`. `get_predictor()` defaults to it. Router: `get_router()` (role "router"), probe `lanes/flow/assets/router_probe.*` fitted without the 10 jury near-duplicates. Blur: `seen/blur.py` (YuNet, model in `lanes/flow/assets/`). Zones: `seen/zones.py`.
* Results merged in `docs/lanes/flow_results.json` (316 components, 15 speed rows); regenerate: pull `runs/flow/` (rsync command in `lanes/flow/collect.py`), `python -m lanes.flow.collect <dir>`.
* If resumed: only optional work is a TensorRT BF16 test (TensorRT not installed on the box).

## What was done and what the numbers say

All CIs are 95% bootstrap over the independent unit named (sequence, image, day-image) unless Wilson is stated. Real data throughout (`data_kind` real).

### 1. Vehicle detection, zero-shot, per condition (UA-DETRAC test, 40 sequences, 11,268 frames at 5 fps, 136,700 GT boxes)
Class-agnostic "vehicle", ignored regions removed, AP all-point interpolated. AP50 / AP75:

| model | all | sunny (8 seq) | cloudy (11) | rainy (9) | night (12) |
|---|---|---|---|---|---|
| RF-DETR Base | 0.848 / 0.746 | 0.860 / 0.760 | 0.842 / 0.738 | 0.816 / 0.707 | 0.884 / 0.800 |
| RF-DETR Medium | 0.851 / 0.748 | 0.866 / 0.771 | 0.840 / 0.737 | 0.818 / 0.705 | 0.888 / 0.807 |
| **RF-DETR Large** | **0.854 [0.83,0.88] / 0.759** | 0.867 / 0.775 | 0.849 / 0.755 | 0.822 / 0.715 | 0.889 / 0.811 |
| YOLO11x (AGPL) | 0.834 / 0.747 | 0.860 / 0.776 | 0.834 / 0.750 | 0.785 / 0.693 | 0.878 / 0.795 |
| YOLO26x (AGPL) | 0.845 / 0.757 | 0.864 / 0.781 | 0.844 / 0.760 | 0.803 / 0.705 | 0.884 / 0.807 |

* **Night is not harder than clear day on DETRAC** (AP50 clear minus night = -0.034 [-0.085, +0.030] for RF-DETR Large): the CI includes 0. Rain is the hardest condition (AP50 -0.032 [-0.038, +0.089] vs clear). Caveat: conditions are confounded with camera locations (each sequence is one fixed camera), 8-12 sequences per condition, so the CIs are wide and differences between models are inside the noise.
* The remaining AP gap is largely annotation convention: DETRAC labels an articulated bus as several boxes and labels heavily occluded far vehicles; see `failures/flow/detrac_*`.
* **NDISPark** (7 parking cameras, ODC-By). Correction of the dataset description: the Zenodo *validation* split is **daytime** (mean luminance 103.8 = train 103.8; test 85.9 and visibly night). Boxes exist only for train (112) + validation (30) = day; the **night test split has counts only**. Day AP50/AP75 on 142 images: Large 0.903 [0.888,0.919] / 0.447, Base 0.868 / 0.371, Medium 0.874 / 0.404, YOLO11x 0.860 / 0.458, YOLO26x 0.869 / 0.495.
* **Night counting (117 images, threshold fixed on the day train split)**, MAE in vehicles per image (mean GT 12.0): Base 3.00, Large 3.09, Medium 2.84, YOLO11x 2.09, YOLO26x 1.91 (relative error 16-26%); the same detectors on the day validation images: relative error 13-23%. Night-minus-day relative error is small and positive for most models; the absolute MAE is lower at night only because there are fewer cars. Detectors over-count slightly at night (mean signed error +0.4 to +2.6).
* Person detection check (COCO val, 2,000 images): AP50 Base 0.702, Large 0.714, YOLO11x 0.701 (CIs about +-0.035).
* **MIO-TCD: cut** (3.7 GB, the shared HDD was IO-saturated; stopped at 1.6 GB). Reason recorded in `flow_sources.md`.
* **Precision parity (integrator request)**: RF-DETR Large autocast BF16 and FP16 vs FP32, same frames: DETRAC AP50 0.854 / 0.854 / 0.855, AP75 0.759 / 0.759 / 0.759; NDISPark day AP50 0.903 / 0.905 / 0.904. Both pass parity (identical within 0.002). Latency in eager PyTorch autocast is not better (15.5 / 15.0 ms per image at batch 8 vs 14.8 fp32 on a shared GPU), so the shipping default stays FP32; the PyTorch autocast option is `model="rfdetr-large:bf16"` in `run_detect`. TensorRT is not installed on the box, so TRT BF16 was not tested.
* **Fine-tuning (RF-DETR Large, 8 epochs, DETRAC train 54 seq + NDISPark day train + COCO subset; GPU0, AMP)**, zero-shot -> fine-tuned on held-out data:
  UA-DETRAC test AP50 / AP75: all 0.854 -> **0.887 [0.86,0.91]** / 0.759 -> **0.806**; sunny 0.867 -> 0.886; cloudy 0.849 -> 0.900; rainy 0.822 -> 0.868; night 0.889 -> 0.896 (AP75 0.811 -> 0.835). NDISPark *validation* (30 held-out day images): AP50 0.921 -> 0.982 [0.968,0.992], AP75 0.472 -> 0.856. NDISPark night counting MAE (117 images, counts never used in training): 3.09 -> **0.58** vehicles/image (day validation 3.97 -> 0.53). Cost: COCO person AP50 0.714 -> 0.690 [0.66,0.73] (small forgetting; the COCO subset in the mixture limited it). Tracking at 5 fps with FT detections: OC-SORT HOTA 0.731 / IDF1 0.841 (without the low-score association 0.744 / 0.859). Caveat: DETRAC train and test come from the same camera families and NDISPark train/validation/test share the same 7 cameras, so the fine-tuned gains are in-domain; they do not transfer to a service bay. **Shipped: fine-tuned FP32** (BF16/FP16 autocast parity was shown on the zero-shot Large; not re-run for the FT model, so FP32 is shipped). FT latency (shared GPU, 960x540, batch 1): PyTorch 7.8 ms, ONNX Runtime CUDA 9.5 ms (5.9 ms forward only).
  (The `ft_ndispark` pooled day AP contains the training images and is not reported.)

### 2. Slot occupancy from detector boxes + slot polygons (`seen/zones.py`, `eval_occupancy.py`)
Slot is occupied if the best vehicle box covers >= tau of the slot polygon (and >= 20% of that box lies in it). (conf, tau) calibrated on calibration days of one dataset, applied to test days of both. Accuracy [95% CI, bootstrap over images]:

| detector | PKLot sunny / cloudy / rainy | PKLot all | CNR-EXT sunny / overcast / rainy | cross: calibrated on PKLot, tested on CNR-EXT | cross: CNR-EXT -> PKLot |
|---|---|---|---|---|---|
| RF-DETR Large, full frame | - / - / 0.668 | 0.814 | - / - / 0.964 | 0.949 | 0.811 |
| **RF-DETR Large + tiling** | 0.996 / 0.994 / **0.997** | **0.996 [0.995,0.996]** | 0.983 / 0.989 / 0.984 | 0.983 [0.979,0.988] | 0.993 [0.992,0.994] |
| YOLO26x + tiling | 0.984 / 0.988 / 0.975 | 0.983 | 0.983 / 0.986 / 0.984 | 0.983 | 0.981 |
| YOLO26x / YOLO11x, full frame | | 0.704 / 0.695 | | 0.893 / 0.875 | |

* Without tiling the small cars in the 1280x720 PKLot frames are missed (recall 0.64, precision 0.99; rainy 0.67 accuracy). Full frame + 3 overlapping 720x720 tiles fixes it. Test: 1,114 PKLot images (63,048 slots), 351 CNR-EXT images (12,480 slots), splits by capture day.
* This proves the zone logic on outdoor lots, not service bays. Tables: `runs/flow/occupancy_*/table.md`.

### 3. Tracking (UA-DETRAC test, RF-DETR Base detections at score >= 0.35)
HOTA / IDF1 / ID switches per 1000 GT boxes (sequence bootstrap in `runs/flow/track_rfdetr-base/table.md`; Large in `track_rfdetr-large`):
5 fps: OC-SORT (own re-implementation, `tracker.py`) HOTA 0.672 IDF1 0.782 IDSW 7.7; greedy IoU 0.668 / 0.777 / 21.1; ByteTrack 0.647 / 0.756 / 11.7.
1 fps: OC-SORT 0.527 / 0.521; IoU 0.513 / 0.497; ByteTrack 0.477 / 0.522. 0.5 fps: 0.515 / 0.506 (OC-SORT), IoU 0.512 / 0.498.
In the slowest tertile of sequences (closest to a bay) OC-SORT keeps HOTA 0.69 at 5 fps and 0.61 at 1 fps. At 0.5-1 fps on moving highway traffic every tracker degrades heavily: for the bay (parked cars) track on IoU of stationary boxes; do not extrapolate from DETRAC highways. MOT17 was cut.

### 4. Scene router (`seen/router.py::decide`, `lanes/flow/predict.py::get_router`)
SigLIP2 so400m-patch14-384, frozen. Pool 3,742 images, 7 scenes, 33 sources (Commons hand-curated, COCO, Stanford Cars, Roboflow, public workshop CCTV frames, fixed-camera frames), split by source (`flow_splits.md`). Held-out = 1,371 images from 15 sources never used for fitting/threshold/blend selection.

| head (prompt set v2) | held-out accuracy [Wilson] | macro-F1 | ECE |
|---|---|---|---|
| zero-shot | **0.858 [0.840,0.874]** | 0.852 | 0.042 |
| linear probe only | 0.703 [0.681,0.725] | 0.698 | 0.105 |
| blend 0.5 probe + 0.5 zero-shot (weight chosen by leave-one-source-out) = **deployed** | 0.848 [0.830,0.865] | 0.844 | 0.061 |

Prompt set v1 (first draft): zero-shot 0.808. Excluding the rule-labelled `coco_street` source (labels come from a box-count rule, not hand-checked; includes parks with cars far away): zero-shot 0.971, blend 0.953, probe 0.771. The probe over-fits source style and does *not* beat zero-shot on unseen sources; it is kept in a 50/50 blend because leave-one-source-out preferred it, held-out says zero-shot is equal or slightly better (`get_router(mode="zeroshot")` is available). Main error: street scenes with buildings/people predicted "unknown" (197 of 670 held-out street images). Unknown threshold 0.46 (leave-one-source-out, answered accuracy 0.90 at coverage 0.94); `decide` now reports a meaningful confidence (never 1.0) and a reason.
**Lane G jury-style set (96 images, external, 10 near-duplicates removed from training)**: strict accuracy zero-shot 0.607 [0.503,0.702], blend 0.629 [0.525,0.722]; lenient (prediction in `acceptable_scenes`) 0.876 / 0.888 [0.805,0.938]; 0.90 of images answered (rest "unknown"). The jury set is harder than the pool (hard cases, Tashkent scenes).
Latency (GPU1, shared): router encoder fp32 51 ms PyTorch vs 23.7 ms ONNX Runtime CUDA (cosine to torch 0.9999999); see `runs/flow/bench`.

### 5. Face blur (`seen/blur.py`)
YuNet (OpenCV Zoo, MIT) at score 0.5: full frame at 1x and 2x + upscaled head-and-shoulders crops for every detected person (RF-DETR Base boxes) + heuristic box (upper 30% of the person) when no face is found; 25% padding; mosaic + blur. WIDER FACE val, every 2nd image: 1,611 images. "Covered" = at least 80% of a face box lies inside the blur regions (95% CI bootstrap over images):

| mode | covered, all faces | height < 16 px | 16-31 px | 32-63 px | >= 64 px | blurred area |
|---|---|---|---|---|---|---|
| old heuristic (upper 30% of person box) | 0.345 [0.30,0.40] | 0.126 | 0.396 | 0.669 | 0.585 | 17.2% |
| YuNet full frame (2 scales) | 0.797 [0.75,0.84] | 0.673 | 0.859 | 0.917 | 0.948 | 12.3% |
| + person head crops | 0.802 [0.76,0.84] | 0.677 | 0.867 | 0.921 | 0.951 | 12.4% |
| **final (+ heuristic fallback)** | **0.818 [0.77,0.86]** | 0.690 [0.60,0.77] | 0.888 | 0.942 | 0.960 | 14.6% |

YuNet detector recall at IoU 0.5: 0.716 overall (0.54 under 16 px). Roughly 1 in 5 WIDER faces (crowds, tiny faces under 16 px, profile) is still not covered, so evidence frames of crowded scenes should not be shown without review; CCTV faces at bay distance are mostly in the 16-64 px range where coverage is 0.89-0.94. Cost on the Mac CPU about 2.9 s for all four modes per image (final mode about 1.5 s per 1 Mpx frame with 2 scales).

### 6. Predictor and speed
`lanes/flow/predict.py`: `get_predictor(model="rfdetr-large"|"rfdetr-base"|"yolo26x"|<artifact folder>, tracker="ocsort"|"iou"|"bytetrack", runtime="torch"|"onnx")` (role "vehicle", real names/versions) and `get_router()` (role "router"). Latency (earlier, contended GPU) on one real 960x540 frame, batch 1, shared RTX PRO 5000 (other jobs ran at ~100% utilisation, treat as upper bounds): RF-DETR Base PyTorch eager 21.8 ms end-to-end, ONNX Runtime CUDA 12.2 ms (7.8 ms forward only), ONNX parity top-5 score difference 0.0009; RF-DETR Large PyTorch 21.5 ms, ONNX 17.2 ms (10.8 forward); YOLO26x PyTorch 25.3 ms. Batched throughput on the full test: Base 17.5 ms/img, Large 14.8 ms/img at batch 8 (includes shared GPU). Tracker cost 1 ms per frame order of magnitude (`tracker_ms_per_frame` in results).

## Failure cases
`docs/lanes/failures/flow/`: router (10 images, faces blurred), detection (UA-DETRAC worst frames per weather: articulated buses labelled as several boxes, occluded far vehicles in rain, glare at night), occupancy (untiled vs tiled on the same PKLot rainy image). Faces never appear unblurred.

## Known limits
* No service-bay or gate camera data: gate class = 69 toll-plaza photos, bay class = 90 images (public workshop CCTV frames from 5 videos + Commons lifts); both small, CIs wide.
* DETRAC conditions confounded with camera location; night is not harder there. NDISPark has no night boxes.
* Router labels for Commons street categories are not hand-cleaned (~10-15% noise); Roboflow washing-car images are augmented triples of the same scenes.
* Occupancy proves outdoor lots only. Tracking numbers come from highways.

## For the deck
* "Detector: RF-DETR Large (Apache-2.0). Zero-shot COCO weights AP50 0.85 / AP75 0.76 on UA-DETRAC test (public traffic CCTV, not a service centre); fine-tuned on public traffic/parking data AP50 0.89 / AP75 0.81; no drop at night in that data, rain is the weak spot; night parking counting error 0.6 vehicles per image after fine-tuning (in-domain cameras)."
* "Parking-slot occupancy 99.6% on PKLot, 98.3-98.5% on CNR-EXT (rainy 98.4-99.7%) with tiled inference; without tiling 81%: resolution matters."
* "Scene router: 85.8% on unseen public sources (97% excluding one rule-labelled source), 61-63% strict / 88-89% lenient on a 96-image jury-style set; unknown inputs are rejected with a stated reason."
* "Face blur: YuNet + person-box fallback covers 82% of faces in WIDER FACE crowd photos (94-96% for faces over 32 px); not measured on CCTV."
* Must-mention caveats: trained/evaluated on public outdoor traffic and parking data and web photos, not service-centre footage; the NDISPark night test has counts only.
