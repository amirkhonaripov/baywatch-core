# Lane D (inspect) — datasets and pretrained models

Columns: name, URL, licence (exact or "not stated"), what we used it for, faces/plates present. "Roboflow licence" is the uploader's
label; image provenance is usually undocumented. Raw data lives on the box under `data/raw/inspect_*` (never committed).

## Training data

| Name | URL | Licence | Used for | Faces / plates |
|---|---|---|---|---|
| VehiDE (Kaggle copy of the official 13,945-image release, JSON train 11,621 + validation 2,324) | https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection ; paper doi via https://doaj.org/article/6e0eb600059440b390431a42dfe8d138 | Kaggle page label "Apache 2.0" (uploader); the paper states no licence. **Treat as unclear.** | damage: train/val (train JSON) and in-source test (official validation) | unknown, plates likely |
| Humans in the Loop "Car Parts and Car Damages" | https://www.kaggle.com/datasets/humansintheloop/car-parts-and-car-damages ; https://humansintheloop.org/resources/datasets/car-parts-and-car-damages-dataset/ | CC0 1.0 | damage half (814 imgs): train/val/test; parts half (998 imgs): parts train/val/test | plates visible (`License-plate` is a class) |
| CarDD (via the Hugging Face FiftyOne mirror `harpreetsahota/CarDD`, 2,816 images = the official training portion) | https://huggingface.co/datasets/harpreetsahota/CarDD ; original https://cardd-ustc.github.io/ ; paper Wang, Li, Wu, IEEE T-ITS 24(7), 2023, doi 10.1109/TITS.2023.3258480 | **Research use only; prior consent from the PIC Lab (CAS) required; any commercial use incl. "testing commercial systems" needs authorisation; no transfer/distribution to third parties without authorisation** (CarDD_license.pdf shipped in the mirror, signed by Voxel51, not by us). Obtained via the mirror at the integrator's instruction. | damage: train (model B only) + in-source test (`cardd_test`) + cross-source test for model A (`cardd_trainpart`) | faces/plates mosaicked or removed (paper) |
| Ultralytics carparts-seg (origin Roboflow `car-seg`, Gianmarco Russo) | https://docs.ultralytics.com/datasets/segment/carparts-seg/ | CC BY 4.0 (docs page) | parts: train/val/test (pooled and re-split by pHash cluster) | not checked |
| TyreNet (Mendeley 32b5vfj6tc) | https://data.mendeley.com/datasets/32b5vfj6tc | CC BY 4.0 | tyre condition: train/val/in-source test | tyre close-ups, low risk |
| TyreAI OCR (Roboflow `tyreai/ocr-yphpv` v3, 4,054 images, 39 character classes) | https://universe.roboflow.com/tyreai/ocr-yphpv | CC BY 4.0 (Roboflow label) | DOT character reader: train/val/test (pHash-cluster split) | none |
| Roboflow `dots-tracking/tire-dot-finder` v5 (83) and `kumbhar-saket/tyre-sidewall-text-label` v5 ('dot' class only, train split) | https://universe.roboflow.com/dots-tracking/tire-dot-finder ; https://universe.roboflow.com/kumbhar-saket/tyre-sidewall-text-label | CC BY 4.0 (Roboflow label) | DOT region detector: train/val | none |

## Held-out / cross-source test data (never used in training)

| Name | URL | Licence | Used for | Faces / plates |
|---|---|---|---|---|
| SYNDCAR | https://data.mendeley.com/datasets/hzpj48krdt | CC BY 4.0 | damage (4 classes, boxes) and parts (28 classes) cross-source test; damage->part assignment. **Photos of ONE vehicle model (Citroen Ami, two vehicle IDs) with digitally generated damage (our reading; the page does not describe the method)** | plates blurred in the photos we looked at; `license_plate` is a parts class |
| Kaggle Coco Car Damage | https://www.kaggle.com/datasets/lplenka/coco-car-damage-detection-dataset | CC0 | class-agnostic damage test (70 annotated images; the 8 'test' images ship without annotations) | not checked |
| Roboflow `jijo/car-damage-new` v2 (HF export `JijoJS/car-damage-new1`) | https://universe.roboflow.com/jijo/car-damage-new | CC BY 4.0 (Roboflow label); provenance unknown | dent/scratch cross-source test (416x416 aspect-stretched) | plates possible |
| CarDamagePanel (CMU 24-679 project, 565 imgs, 21 panels) | https://huggingface.co/datasets/kadireks/cardamagepanel-originals | "other" (not specified) | parts cross-source test | `license-plate` is a class |
| Kaggle `warcoder/tyre-quality-classification` | https://www.kaggle.com/datasets/warcoder/tyre-quality-classification | CC BY 4.0 (Kaggle label) | tyre condition cross-dataset test (after removing TyreNet near-duplicates) | none seen |
| Kaggle `sameersambhare1/tyre-condition-classification-dataset` | https://www.kaggle.com/datasets/sameersambhare1/tyre-condition-classification-dataset | CC BY-SA 4.0 | tyre condition cross-dataset test (NEW + SERVICEABLE = good, UNUSABLE = defective, our mapping) | none seen |
| Roboflow `subhams-workspace-l47ut/dot-code-6shg0` (97) | https://universe.roboflow.com/subhams-workspace-l47ut/dot-code-6shg0 | CC BY 4.0 (Roboflow label) | DOT region detector cross-source test | none |

## Considered, not used

| Name | Why not |
|---|---|
| Roboflow `khpi/car-segmentation-ga8mc` (9,011 parts images) | the project exposes no generated version through the API (`versions: []`), so it cannot be exported |
| HF `tugberkkalay/autodamageiq-vehicle-damage-dataset` | merge of CarDD + VehiDE + HITL with GPT-4o labels: contaminated by construction |
| Kaggle CarDD copies, other HF CarDD mirrors | only `harpreetsahota/CarDD` was used, as instructed; others carry the same licence problem |
| DSMLR Car-Parts-Segmentation | no licence file; not needed once carparts-seg + HITL parts were pooled |
| TyreCV (Kaggle `maxfinkenbeiner/tyrecv-two-views-are-all-you-need`) | see `inspect_notes.md` (tread depth section) |
| Undercarriage data | out of scope for this lane |

## Pretrained models

| Name | Source | Licence | Used for |
|---|---|---|---|
| YOLO11m-seg / YOLO11s / YOLO26 COCO-pretrained weights (Ultralytics) | https://github.com/ultralytics/assets/releases | **AGPL-3.0** (code and weights) | initialisation + training framework for damage, parts, DOT region and DOT characters. Commercial deployment needs an Ultralytics licence or a re-train on an Apache-2.0 detector (RF-DETR) |
| DINOv2 ViT-B/14 (`timm/vit_base_patch14_dinov2.lvd142m`) | https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m | Apache-2.0 (weights and code, Meta AI) | frozen feature extractor of the tyre condition classifier |
