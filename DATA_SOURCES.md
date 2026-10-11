# Data sources and licences

Every dataset, pretrained model and third-party asset used in this project, collected from each lane's notes (`docs/lanes/<lane>_sources.md`).

Rules we followed:
- Non-commercial and research-only data was used for this prototype only. A paid pilot needs re-licensing or our own pilot data.
- Faces are blurred in all evidence images.
- Synthetic data is declared with its generation method.

Some datasets were obtained through public mirrors rather than the original request forms:
- RodoSol-ALPR via the Kaggle mirror `tewjinglai/robosol`.
- CarDD via the Hugging Face mirror `harpreetsahota/CarDD`.
- The original licences apply to both.

## Plate identification (`ident`)


Columns: name, URL, licence (exact as found, or "not stated"), what we used it for, faces/plates present.
"Train/val/test" refer to our own splits (see `ident_splits.md`). All plates in real datasets belong to real vehicles of people who did not consent to this project; nothing from these sets is shown in the web UI or the deck (per `docs/LANES.md` rule 6, public sets are allowed for training and testing).

### Datasets

| Name | URL | Licence | Used for | Faces / plates |
|---|---|---|---|---|
| AUTO.RIA OCR crops: Kazakhstan `autoriaNumberplateOcrKz-2022-11-29` (9,922 crops), Kyrgyzstan `...Kg-2022-11-30` (31,917), Russia `...Ru-2021-09-01` (57,120) | https://nomeroff.net.ua/datasets/ (Nomeroff-Net project, `github.com/ria-com/nomeroff-net`) | **not stated** on the dataset page or in the zips (code repo is GPL-3.0) | OCR train + val (train/val splits of each zip), OCR test (native `test` split of each zip: KZ 279, KG 1,068, RU 2,845) | plates yes (real marketplace photos), faces no |
| AUTO.RIA OCR SU `...Su-2023-03-10` | same | not stated | downloaded, **not used** (labels are mostly Cyrillic; deleted) | plates yes |
| Roboflow Universe **Uzbek_Plates** (`suxrobs-workspace/uzbek_plates`, version 1) | https://universe.roboflow.com/suxrobs-workspace/uzbek_plates | CC BY 4.0 (the `roboflow` licence field of the export) | detector train/val/test (boxes), real-UZ OCR test (301 original images, text labels made by us, see notes), pseudo-labelled OCR training crops (train split only) | plates yes (real Uzbek cars at an inspection-station-like site), faces no (a few people at the edge of some frames) |
| Mendeley Data "Dataset Of Central Asia vehicle number plate with Annotations" (v2) | https://data.mendeley.com/datasets/4c5z833m6m/2 (DOI 10.17632/4c5z833m6m.2) | CC BY 4.0 | **background photos only** for plate compositing: the zip has 502 JPEGs in per-country folders without any annotation file, and in the Uzbek folder the plates are blurred by the seller, so it cannot serve as plate ground truth | plates blurred |
| Wikimedia Commons, Uzbekistan car/plate categories + keyword searches (per-file licences in `data/raw/ident_uz/commons/meta.jsonl` on the box) | https://commons.wikimedia.org (fetcher: `lanes/ident/tools/fetch_commons_uz.py`) | per file: CC BY-SA 4.0 / CC BY 4.0 / CC BY-SA 3.0 / CC0 / public domain / CC BY 2.0 etc. (counted in the notes) | background photos for compositing; qualitative checks | some plates visible |
| **CCPD2019** (Chinese City Parking Dataset) subsets `ccpd_db`, `ccpd_weather`, `ccpd_challenge`, `ccpd_base` | https://github.com/detectRecog/CCPD ; fetched from the HF mirror `htkien95/ccpd2019` (`CCPD2019.tar.xz`, stream-extracted, only these subsets kept) | MIT (GitHub licence of the CCPD repo); the HF mirror states no licence | detection robustness: night/bright/uneven illumination (DB), weather, challenge; Chinese plate formats (wrong script), so used for detection and compositing backgrounds only, never OCR | plates yes, faces no |
| **RodoSol-ALPR** (Kaggle mirror `tewjinglai/robosol`: only `cars-br` and `cars-me`, 5,001 + 5,000 images; the motorcycle subsets are not in this mirror) | original: https://github.com/raysonlaroca/rodosol-alpr-dataset ; mirror: https://www.kaggle.com/datasets/tewjinglai/robosol | academic/non-commercial research use under the authors' licence agreement (`pdfs/license-agreement.pdf` in the original repo); the Kaggle mirror itself states no licence | detection + OCR test (block-split test part), night/rain subset by measured luminance, composite backgrounds (train part) | plates yes, drivers' faces blurred by the authors |
| Unidata licence-plate sample (Kaggle `unidpro/license-plate-detection-dataset`, 3.7 MB) | https://www.kaggle.com/datasets/unidpro/license-plate-detection-dataset | commercial dataset sample, licence not read in detail | downloaded, has no Uzbek images (Argentina, France, Mexico, Portugal only); **not used**, deleted | plates yes |
| Kaggle `valixonovilyosbek/cars-that-are-common-in-uzbekistan` | https://www.kaggle.com/datasets/valixonovilyosbek/cars-that-are-common-in-uzbekistan | MIT (uploader) | inspected via file listing only: it is a Cityscapes-based Roboflow export, not Uzbek plates; **not used** | n/a |
| GitHub `sardor-asatillaev/uzlpr`, `bekhzod-olimov/Synthethic-Uzbek-Car-Registration-Plates-Generator`, `IzzatulloOne/plate-recognition-uz` | github.com/... | NOASSERTION / none stated / MIT | read only: plate-format rules (`plate_rules.py` regexes), series statistics sanity check; nothing from platesmania.com was downloaded; no code or glyph images copied | n/a |

### Pretrained models

| Name | URL | Licence | Used for |
|---|---|---|---|
| fast-plate-ocr `cct-s-v2-global-model` (ONNX + Keras weights `cct_s_v2_global.keras`), `global-plates-mobile-vit-v2-model` | https://github.com/ankandrew/fast-plate-ocr (releases `arg-plates`) | MIT | baseline numbers; `cct_s_v2_global.keras` is the initialisation of our fine-tuning |
| fast-alpr detector `yolo-v9-t-384-license-plate-end2end` (open-image-models) | https://github.com/ankandrew/open-image-models | MIT (training data of the weights not stated) | baseline detector numbers |
| RF-DETR (Nano/Small) COCO-pretrained backbone | https://github.com/roboflow/rf-detr | Apache-2.0 | initialisation of the plate detector (see notes) |
| Qwen3-VL-4B-Instruct | https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct | Apache-2.0 | second independent reader for pre-labelling the real-UZ OCR test set (labels then verified by Claude, an AI, pending human spot-check) |

### Fonts (synthetic plate renderer, `lanes/ident/synth/fonts/`)

No freely licensed digitisation of the real FE-Schrift plate font with a verifiable licence was found (searched GitHub/Google Fonts; "Euro Plate" by Andy Hoppe and similar have unverified terms). The renderer uses these DIN-like faces from https://github.com/google/fonts instead, all **SIL Open Font License 1.1** (texts in `fonts/OFL-*.txt`): Barlow Bold, Barlow Semi Condensed Bold/SemiBold, Barlow Condensed SemiBold, Saira Semi Condensed Bold, Share Tech Regular, with random width/weight jitter. Glyph details of the real plates (for example the slotted zero) are therefore only approximated.

## Vehicle detection, zones, router, face blur (`flow`)


Faces / plates column = whether real faces / readable plates of private people appear in the data. Public datasets with faces/plates are allowed for
training and testing (user decision); nothing from them is shown in the UI, and evidence images go through `seen.blur.blur_faces`.

### Datasets

| Name | URL | Licence (as stated) | Used for | Faces / plates |
|---|---|---|---|---|
| UA-DETRAC (HF reformat `abhineet123/ua_detrac`, frames + boxes) | https://huggingface.co/datasets/abhineet123/ua_detrac | Original licence **not found** (official site is down). The uploader tagged the HF repo `cc-by-4.0`, which does not override the original terms. Treat as **evaluation / research only**. | test (40 official test sequences: detection AP per weather, tracking), train (60 sequences, fine-tuning) | plates readable (Chinese traffic) / faces: no |
| UA-DETRAC original XML annotations (weather label per sequence, vehicle type, ignored regions, speed) | read from the original archive in HF repo `ShantyCam/ua-detrac` (`ua-detrac-orig.zip`, 100 XML files only, via HTTP range requests) | the uploader's note says CC-BY-NC-SA and that it is a private mirror; original licence not found | weather labels (`sence_weather`), vehicle types for fine-tuning classes, ignored regions, speeds | as above |
| NDISPark | https://zenodo.org/records/6560823 | ODC-By 1.0 (record + COCO json) | day (train split, 112 images) vs night (validation split, 30 images with boxes; test split 117 images with vehicle counts only); fine-tune on day split | parking lots; not stated |
| PKLot | https://web.inf.ufpr.br/vri/databases/parking-lot-database/ | CC BY 4.0 | slot occupancy test and threshold calibration (PUCPR, UFPR04, UFPR05; sunny/cloudy/rainy), frames for router "street" | plates unlikely readable / faces: no |
| CNRPark+EXT (CNR-EXT full frames, slot boxes, labels) | http://www.cnrpark.it/ | ODbL 1.0 (project page) | slot occupancy cross-dataset test and calibration (sunny/overcast/rainy), router "street" frames | frames downsampled to 1000x750 "due to privacy issues" |
| MIO-TCD localization | https://tcd.miovision.com/ | CC BY-NC-SA 4.0 | **cut**: download (3.7 GB) was too slow on the shared drive and stopped at 1.6 GB, see notes | not stated |
| COCO val2017 | https://cocodataset.org | annotations CC BY 4.0; images are Flickr photos with mixed CC licences | person-AP regression check (zero-shot vs fine-tuned), router "street"/"motorcycle"/"unknown" images (never used to train a detector) | faces: yes (web photos) |
| WIDER FACE (val, via HF `CUHK-CSE/wider_face`) | http://shuoyang1213.me/WIDERFACE/ | non-commercial research (HF card: cc-by-nc-nd-4.0) | face-blur recall (test only) | faces: yes (that is the data) |
| Wikimedia Commons categories (repair shops, lifts, tyre shops, garages, car washes, boom barriers, car-park entrances, gatehouses, checkpoints, Ducati, scooters, car hoods/wheels/engines, a few non-car searches) | https://commons.wikimedia.org | per file: CC BY / CC BY-SA / CC0 / public domain etc. The exact licence, author and page URL of every image are in `data/prep/flow/router/manifest.jsonl` | router training + held-out evaluation | people and plates can appear in some images |
| Stanford Cars (HF `tanganke/stanford_cars`) | https://huggingface.co/datasets/tanganke/stanford_cars | Stanford Cars: research use, no explicit licence on the mirror | router "closeup" (whole-car photos) | plates sometimes readable |

### Pretrained models

| Name | URL | Licence | Used for |
|---|---|---|---|
| RF-DETR Nano/Small/Medium/Base/Large (COCO) | https://github.com/roboflow/rf-detr | Apache-2.0 for these sizes (the XLarge/2XLarge sizes are under a separate licence and are not used) | vehicle + person detector (zero-shot and fine-tuned) |
| Ultralytics YOLO11x, YOLO26x (COCO) | https://github.com/ultralytics/ultralytics | AGPL-3.0 | zero-shot comparison only, not shipped |
| SigLIP2 so400m-patch14-384, base-patch16-384 | https://huggingface.co/google/siglip2-so400m-patch14-384 | Apache-2.0 | scene router (zero-shot + linear probe on frozen embeddings) |
| YuNet `face_detection_yunet_2023mar.onnx` | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet | MIT | face blur |
| OC-SORT (algorithm: Cao et al., CVPR 2023) | https://github.com/noahcao/OC_SORT | MIT; `lanes/flow/tracker.py` is our own re-implementation | tracker |
| ByteTrack (via `supervision`) | https://github.com/roboflow/supervision | MIT | tracker baseline |

## Bay states and operations (`ops`)


Every item used by `lanes/ops/*`. "Faces/plates" = whether the source contains identifiable faces or plates.

| Name | URL | Licence | Used for | Faces / plates |
|---|---|---|---|---|
| Kinetics-700-2020 (clips via HF mirror `atalaydenknalbant/Kinetics-700`, annotation CSVs from S3) | https://huggingface.co/datasets/atalaydenknalbant/Kinetics-700 ; https://s3.amazonaws.com/kinetics/700_2020/annotations/train.csv , val.csv | Annotations CC BY 4.0 (secondary sources; official statement not found); videos are YouTube uploads (mirror tag "other") | train (train split) / test (K700 validation split) for `work`; teacher-labelled frames for the other states | y / y |
| CrossTask (Change a Tire, Jack Up a Car, Add Oil to Your Car + 5 related car tasks) | https://www.di.ens.fr/~dzhukov/crosstask/crosstask_release.zip ; https://github.com/DmZhukov/CrossTask | Code BSD-3-Clause; no data licence stated; videos YouTube | train/test split by video: `raised`, `wheel_off` (derived from step spans), `work` proxy (inside/outside steps), `hood_open` (Add Oil steps) | y / y |
| COIN (ChangeCarTire, UseJack, ReplaceTyreValveStem, ReplaceCarFuse, PolishCar, ReplaceAWiperHead) | https://github.com/coin-dataset/annotations | Annotations CC BY-NC 4.0; videos YouTube | same as CrossTask | y / y |
| Roboflow `carhoodopen/car-hood-open` v4 | https://universe.roboflow.com/carhoodopen/car-hood-open | CC BY 4.0 (uploader label) | train (train+valid) / test (RF test split) for `hood_open`; hood open/closed detector | not systematically checked / y |
| Roboflow `vehicles-ztmmv/car-open-hood` | https://universe.roboflow.com/vehicles-ztmmv/car-open-hood | CC BY 4.0 (uploader label) | train / test (RF test split) for `hood_open` | not checked |
| Roboflow `deep-learning-ocos/car-condition-detection` | https://universe.roboflow.com/deep-learning-ocos/car-condition-detection | CC BY 4.0 (uploader label) | test only (cross-project `hood_open`) | not checked |
| Roboflow `um-84bee/washing-car` | https://universe.roboflow.com/um-84bee/washing-car | CC BY 4.0 (uploader label) | train for `washing`, `hood_open` (fixed wash-yard cameras) | y (one readable plate seen in preview) |
| Ultralytics Carparts-Seg (from Roboflow car-seg by G. Russo) | https://docs.ultralytics.com/datasets/segment/carparts-seg/ | Docs page: none stated; Roboflow source CC BY 4.0 | wheel part detector; parked-car negatives | not checked |
| Wikimedia Commons "Stationary automobile service lifts" (41 files, 24 kept after hand screening) | https://commons.wikimedia.org/wiki/Category:Stationary_automobile_service_lifts | Per file: CC BY-SA 4.0/3.0/2.0, CC BY 3.0/2.0, public domain (see `lanes/ops/labels/commons_labels.csv`) | train for `raised` (hand-labelled); 7 files are jury items and are removed by `eval.manifest.filter_train` | some / some |
| Public workshop CCTV / timelapse clips: `e4O3X9gli1s` (Vianor, CC BY), `-OJPETY9Bis` (Revolution Automotive, CC BY), `JIEX2ruxZKM`, `wWSNERw5oME`, `2mhugD9KYjI`, `BZsVY9FUUOo` (standard YouTube licence) | https://www.youtube.com/watch?v=<id> | as listed | TEST ONLY (hand-labelled bay ROIs); never trained on; videos kept on the box, not in the repo | y / some |
| `Bh4IqT2Wq-s` (Williams Auto Group) | https://www.youtube.com/watch?v=Bh4IqT2Wq-s | standard YouTube licence | downloaded, not used (close tripod shot, not a bay view) | y / ? |
| DINOv2 ViT-L/14 (`facebook/dinov2-large`) | https://huggingface.co/facebook/dinov2-large | Apache-2.0 | frozen crop-embedding backbone (image-only pretraining, no Kinetics) | n/a |
| RF-DETR Base (COCO) via `seen.baseline.PretrainedDetector` | https://github.com/roboflow/rf-detr | Apache-2.0 | vehicle/person context boxes for training features and CCTV eval | n/a |
| Ultralytics YOLO11s (COCO-pretrained init) | https://github.com/ultralytics/ultralytics | AGPL-3.0 | init for wheel / hood part detectors (if trained) | n/a |
| Qwen3-VL-8B-Instruct | https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct | Apache-2.0 | zero-shot baseline on the CCTV test frames; teacher labels for training frames that have no human/derived label | n/a |

## Light inspection: damage, parts, tyres (`inspect`)


Columns: name, URL, licence (exact or "not stated"), what we used it for, faces/plates present. "Roboflow licence" is the uploader's
label; image provenance is usually undocumented. Raw data lives on the box under `data/raw/inspect_*` (never committed).

### Training data

| Name | URL | Licence | Used for | Faces / plates |
|---|---|---|---|---|
| VehiDE (Kaggle copy of the official 13,945-image release, JSON train 11,621 + validation 2,324) | https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection ; paper doi via https://doaj.org/article/6e0eb600059440b390431a42dfe8d138 | Kaggle page label "Apache 2.0" (uploader); the paper states no licence. **Treat as unclear.** | damage: train/val (train JSON) and in-source test (official validation) | unknown, plates likely |
| Humans in the Loop "Car Parts and Car Damages" | https://www.kaggle.com/datasets/humansintheloop/car-parts-and-car-damages ; https://humansintheloop.org/resources/datasets/car-parts-and-car-damages-dataset/ | CC0 1.0 | damage half (814 imgs): train/val/test; parts half (998 imgs): parts train/val/test | plates visible (`License-plate` is a class) |
| CarDD (via the Hugging Face FiftyOne mirror `harpreetsahota/CarDD`, 2,816 images = the official training portion) | https://huggingface.co/datasets/harpreetsahota/CarDD ; original https://cardd-ustc.github.io/ ; paper Wang, Li, Wu, IEEE T-ITS 24(7), 2023, doi 10.1109/TITS.2023.3258480 | **Research use only; prior consent from the PIC Lab (CAS) required; any commercial use incl. "testing commercial systems" needs authorisation; no transfer/distribution to third parties without authorisation** (CarDD_license.pdf shipped in the mirror, signed by Voxel51, not by us). Obtained via the mirror at the integrator's instruction. | damage: train (model B only) + in-source test (`cardd_test`) + cross-source test for model A (`cardd_trainpart`) | faces/plates mosaicked or removed (paper) |
| Ultralytics carparts-seg (origin Roboflow `car-seg`, Gianmarco Russo) | https://docs.ultralytics.com/datasets/segment/carparts-seg/ | CC BY 4.0 (docs page) | parts: train/val/test (pooled and re-split by pHash cluster) | not checked |
| TyreNet (Mendeley 32b5vfj6tc) | https://data.mendeley.com/datasets/32b5vfj6tc | CC BY 4.0 | tyre condition: train/val/in-source test | tyre close-ups, low risk |
| TyreAI OCR (Roboflow `tyreai/ocr-yphpv` v3, 4,054 images, 39 character classes) | https://universe.roboflow.com/tyreai/ocr-yphpv | CC BY 4.0 (Roboflow label) | DOT character reader: train/val/test (pHash-cluster split) | none |
| Roboflow `dots-tracking/tire-dot-finder` v5 (83) and `kumbhar-saket/tyre-sidewall-text-label` v5 ('dot' class only, train split) | https://universe.roboflow.com/dots-tracking/tire-dot-finder ; https://universe.roboflow.com/kumbhar-saket/tyre-sidewall-text-label | CC BY 4.0 (Roboflow label) | DOT region detector: train/val | none |

### Held-out / cross-source test data (never used in training)

| Name | URL | Licence | Used for | Faces / plates |
|---|---|---|---|---|
| SYNDCAR | https://data.mendeley.com/datasets/hzpj48krdt | CC BY 4.0 | damage (4 classes, boxes) and parts (28 classes) cross-source test; damage->part assignment. **Photos of ONE vehicle model (Citroen Ami, two vehicle IDs) with digitally generated damage (our reading; the page does not describe the method)** | plates blurred in the photos we looked at; `license_plate` is a parts class |
| Kaggle Coco Car Damage | https://www.kaggle.com/datasets/lplenka/coco-car-damage-detection-dataset | CC0 | class-agnostic damage test (70 annotated images; the 8 'test' images ship without annotations) | not checked |
| Roboflow `jijo/car-damage-new` v2 (HF export `JijoJS/car-damage-new1`) | https://universe.roboflow.com/jijo/car-damage-new | CC BY 4.0 (Roboflow label); provenance unknown | dent/scratch cross-source test (416x416 aspect-stretched) | plates possible |
| CarDamagePanel (CMU 24-679 project, 565 imgs, 21 panels) | https://huggingface.co/datasets/kadireks/cardamagepanel-originals | "other" (not specified) | parts cross-source test | `license-plate` is a class |
| Kaggle `warcoder/tyre-quality-classification` | https://www.kaggle.com/datasets/warcoder/tyre-quality-classification | CC BY 4.0 (Kaggle label) | tyre condition cross-dataset test (after removing TyreNet near-duplicates) | none seen |
| Kaggle `sameersambhare1/tyre-condition-classification-dataset` | https://www.kaggle.com/datasets/sameersambhare1/tyre-condition-classification-dataset | CC BY-SA 4.0 | tyre condition cross-dataset test (NEW + SERVICEABLE = good, UNUSABLE = defective, our mapping) | none seen |
| Roboflow `subhams-workspace-l47ut/dot-code-6shg0` (97) | https://universe.roboflow.com/subhams-workspace-l47ut/dot-code-6shg0 | CC BY 4.0 (Roboflow label) | DOT region detector cross-source test | none |

### Considered, not used

| Name | Why not |
|---|---|
| Roboflow `khpi/car-segmentation-ga8mc` (9,011 parts images) | the project exposes no generated version through the API (`versions: []`), so it cannot be exported |
| HF `tugberkkalay/autodamageiq-vehicle-damage-dataset` | merge of CarDD + VehiDE + HITL with GPT-4o labels: contaminated by construction |
| Kaggle CarDD copies, other HF CarDD mirrors | only `harpreetsahota/CarDD` was used, as instructed; others carry the same licence problem |
| DSMLR Car-Parts-Segmentation | no licence file; not needed once carparts-seg + HITL parts were pooled |
| TyreCV (Kaggle `maxfinkenbeiner/tyrecv-two-views-are-all-you-need`) | see `inspect_notes.md` (tread depth section) |
| Undercarriage data | out of scope for this lane |

### Pretrained models

| Name | Source | Licence | Used for |
|---|---|---|---|
| YOLO11m-seg / YOLO11s / YOLO26 COCO-pretrained weights (Ultralytics) | https://github.com/ultralytics/assets/releases | **AGPL-3.0** (code and weights) | initialisation + training framework for damage, parts, DOT region and DOT characters. Commercial deployment needs an Ultralytics licence or a re-train on an Apache-2.0 detector (RF-DETR) |
| DINOv2 ViT-B/14 (`timm/vit_base_patch14_dinov2.lvd142m`) | https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m | Apache-2.0 (weights and code, Meta AI) | frozen feature extractor of the tyre condition classifier |

## Synthetic data: scenario engine and Blender (`sim`)


| Name | URL | Licence | Used for | Faces / plates |
|---|---|---|---|---|
| Kenney Car Kit | https://kenney.nl/assets/car-kit | CC0 1.0 | synthetic renders (cars, wheels): train + test | n / n (toy models, no plates) |
| Kenney Mini Characters | https://kenney.nl/assets/mini-characters | CC0 1.0 | synthetic renders (proxy mechanics): train + test | stylised cartoon faces, not real people / n |
| Poly Haven HDRIs (auto_service, autoshop_01, garage, skylit_garage, machine_shop_02, empty_workshop) | https://polyhaven.com/hdris | CC0 1.0 | lighting and background of renders; skylit_garage held out for test | n / tiny distant background cars, plates not readable at 1k |
| Poly Haven textures (concrete_floor_02, concrete_floor_painted, concrete_floor_worn_001, painted_plaster_wall) | https://polyhaven.com/textures | CC0 1.0 | floor and wall materials | n / n |
| Blender 5.0.1 | https://www.blender.org | GPL-3.0 (outputs are ours) | rendering | – |
| Scenario engine (own code, `sim/scenario.py`) | this repo | project licence | all synthetic timelines, work orders, noise test | n / n (random synthetic plate strings in Uzbek format) |

The duration priors in `sim/scenario.py` are assumptions, loosely anchored on public figures quoted in
`research/service_ops_kpis.md`: Toyota TSM's "about 50 % of servicing time is the car stagnating", ~1 h express
service, J.D. Power's 22 min average handover, and ~20 % workshop idle. No dataset is used to fit them.

Pretrained models: none trained by this lane. `sim/eval_synthetic.py` runs whatever `seen.pipeline.default_pipeline()`
deploys at the time; licences are listed by the lanes that own those models (currently RF-DETR COCO and fast-alpr,
from `seen/baseline.py`).

## Evaluation: jury-style test set and cost sources (`eval`)


| name | URL | licence | used for | faces / plates |
|---|---|---|---|---|
| Seen jury set v1 (96 inputs: 82 photos, 7 videos, 7 frames cut from those videos) | Wikimedia Commons; one file page per item in `eval/jury/jury.jsonl` | per file, recorded in `eval/jury/jury.jsonl`: CC BY 2.0/3.0/4.0, CC BY-SA 2.0/3.0/4.0, CC0, public domain | test only (jury-style end-to-end test); held out in `eval/manifests/jury.jsonl` | yes / yes (some plates were blurred by the uploaders) |
| RF-DETR base, COCO-pretrained (via `seen.baseline`, rfdetr 1.11.2) | https://github.com/roboflow/rf-detr | Apache-2.0 (code and weights, per repo) | speed benchmark (PyTorch, ONNX Runtime, TensorRT); baseline pipeline under test | n/a |
| fast-alpr 0.4.0: `yolo-v9-t-384-license-plate-end2end` detector + `cct-xs-v2-global-model` OCR | https://github.com/ankandrew/fast-alpr | library MIT; model weights: licence not stated on the hub page (not checked) | speed benchmark; baseline pipeline under test | n/a |
| OpenCV YuNet face detector `face_detection_yunet_2023mar.onnx` | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet | MIT | extra face blur on every failure-gallery image | n/a |
| OpenCV Haar cascades (frontal, profile), bundled with opencv-python | https://github.com/opencv/opencv/tree/4.x/data/haarcascades | Intel open-source licence (BSD-style), per the XML headers | extra face blur on every failure-gallery image | n/a |
| NVIDIA TensorRT 10.15 libraries (`tensorrt_cu13_libs` wheel, only libnvinfer/plugin/onnxparser + sm120/ptx builder resources kept, on the NVMe at `seen_fast/eval/trt` (1.2 GB)) | https://pypi.nvidia.com/tensorrt-cu13-libs/ | NVIDIA TensorRT licence (proprietary, redistribution restricted) | speed table only (ONNX Runtime TensorRT EP, FP16) | n/a |

### Price and spec sources used by `eval/cost.py`

| item | value | source (read 2026-10-10) |
|---|---|---|
| Modal L4 GPU | $0.000222 per second | https://modal.com/pricing |
| Modal CPU / memory | $0.0000131 per physical core per second; $0.00000222 per GiB per second | https://modal.com/pricing |
| Modal idle window | 60 s by default, and idle time is billed | https://modal.com/docs/guide/cold-start |
| Electricity, legal entities, Uzbekistan | 1,100 UZS per kWh from 1 Jun 2026 | https://kun.uz/en/news/2026/05/15/uzbekistan-to-raise-electricity-and-natural-gas-tariffs-from-june-1 (Interfax, 15 May 2026, agrees) |
| UZS/USD | 11,778.45 | Central Bank of Uzbekistan official rate for 6 Oct 2026 (reported by zamin.uz) |
| RTX 4060 Ti PC in Tashkent | about $925–1,345 (used); about $1,020 (new i5-13400F build) | `research/competitors_legal.md` (OLX.uz, 10 Oct 2026) |
| RTX 4060 Ti | 22.06 TFLOPS FP32 (derived: 4,352 cores × 2 × 2.535 GHz); 288 GB/s; 160–165 W | technical.city, Puget Systems |
| NVIDIA L4 | 30.3 TFLOPS FP32; 300 GB/s; 72 W | NVIDIA L4 datasheet |
| RTX PRO 5000 Blackwell | about 66.9 TFLOPS FP32 (derived: 14,080 cores × 2 × 2.377 GHz); 1,344 GB/s | PNY/NVIDIA datasheet (bandwidth), Puget Systems (cores) |

## Web app sample inputs (`app`)


Lane F trains nothing. It serves whatever `seen.pipeline.default_pipeline()` loads and ships a few public demo inputs.

### Demo inputs shipped in `web/public/samples/` (shown in the web UI)

| Name | URL | Licence | Used for | Faces / plates |
|---|---|---|---|---|
| Uzbek_Plates v1, image `AAF0119214_car-front_…` (Damas van, front) | https://universe.roboflow.com/suxrobs-workspace/uzbek_plates | CC BY 4.0 (dataset README on the box: `data/raw/ident_uz/roboflow_uzbek_plates/README.dataset.txt`) | demo input "Car with an Uzbek plate"; recorded backup run | faces: n (checked visually); plates: y (public dataset) |
| "Chery A1 - service shop in Ukraine (7)", ANT Berezhnyi | https://commons.wikimedia.org/wiki/File:Chery_A1_-_service_shop_in_Ukraine_(7).jpg | CC BY 3.0 (from `data/raw/ops_commons/meta.csv`) | demo input "Service bay: car on a lift" | faces: n; plates: n |
| "Classic Mustang at Garage, West Miami Florida August 2024", Phillip Pessar | https://commons.wikimedia.org/wiki/File:Classic_Mustang_at_Garage,_West_Miami_Florida_August_2024.jpg | CC BY 2.0 (from `data/raw/ops_commons/meta.csv`) | demo input "Garage bay, car raised" | faces: n; plates: y (small, US) |
| UA-DETRAC test sequence MVI_39051, frames 1–300, re-encoded to a 12 s H.264 mp4 (25 fps, 960×540) | https://detrac-db.rit.albany.edu/ | CC BY-NC-SA 3.0 | demo video for the in-browser sampler; recorded backup run | faces: not identifiable at this resolution (traffic camera); plates: not legible |
| `flower.jpg` from scikit-learn sample images (vultilion on Flickr) | https://www.flickr.com/photos/vultilion/6056698931/ | CC BY 2.0 (scikit-learn `datasets/images/README.txt`) | demo "not a car" input (unknown path) | n / n |

The Uzbek_Plates and Commons images are also used by other lanes (ident / ops). They appear in the demo only as inputs; Lane F reports no metrics on them.

### Pretrained models the API serves today (via `seen/baseline.py`, owned by the integrator)

| Name | URL | Licence | Used for | Faces / plates |
|---|---|---|---|---|
| RF-DETR Base (COCO), `rf-detr-base.pth` | https://github.com/roboflow/rf-detr | Apache-2.0 | vehicle + person detection in the served pipeline | trained on COCO (faces y) |
| fast-alpr default plate detector `yolo-v9-t-384-license-plates-end2end` (open-image-models) | https://github.com/ankandrew/open-image-models | MIT (repo) | plate detection in the served pipeline | plates y |
| fast-plate-ocr `cct-xs-v2-global-model` | https://github.com/ankandrew/fast-plate-ocr | MIT (repo) | plate OCR in the served pipeline (not Uzbek-tuned) | plates y |

When lanes replace the baseline predictors, their weights reach Modal through `serve/upload_artifacts.py` (Volume `seen-artifacts`); their licences are recorded in each lane's own `*_sources.md`.

### Web dependencies

React 18.3.1 (MIT), Vite 5.4.11 (MIT), TypeScript 5.6.3 (Apache-2.0), Vitest 2.1.8 (MIT). No runtime CDN; no analytics.

## Demo inputs (`demo_inputs/`) and held-out samples on the service floor (`web/public/stages/heldout/`)
None of these files were used to train any model. Per-file credits are in `demo_inputs/README.md` and `demo_inputs/videos/CREDITS.json`.

| File(s) | Source | Licence |
|---|---|---|
| uz_van.jpg | Roboflow Universe Uzbek_Plates, held-out split | CC BY 4.0 |
| uz_truck.jpg, rain.jpg, ducati.jpg, mustang.jpg, wash_station.jpg, wash_building_exterior.jpg, videos/gate_car_park.mp4; heldout/wash_selfservice.jpg, heldout/wash_entrance.jpg | Wikimedia Commons (our robustness test set; author and licence per file in `eval/manifests/jury.jsonl`) | CC BY / CC BY-SA / CC0, per file |
| damage_side_dent.jpg, damage_bumper_scratch.jpg; heldout/inspect_front.jpg, heldout/inspect_crash.jpg | CarDD, its own test split (HF mirror `harpreetsahota/CarDD`) | research use, see the CarDD row above |
| tyre_cracked_closeup.jpg, tyre_good_closeup.jpg; heldout/inspect_tyre.jpg | TyreNet, test split | CC BY 4.0 |
| videos/bay_lift_cctv.mp4, videos/bay_tyre_shop_cctv.mp4 | YouTube -OJPETY9Bis (Revolution Automotive) and e4O3X9gli1s (Vianor), cut to 25–30 s, faces blurred, intro cards removed | CC BY (YouTube) |
