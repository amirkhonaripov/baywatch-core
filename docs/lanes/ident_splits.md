# Lane A (ident): train / val / test splits

`eval/manifests/` had no plate manifests when this lane ran, so every split below is ours, made by **source** (zip, car, block, file-name hash), never by crop or frame. Lane G: the held-out items are listed in `lanes/ident/splits/` (machine-readable, manifest-style) and in the per-set rules here; please fold them into `eval/manifests/plate_ocr.jsonl` / `plate_det.jsonl`.

## OCR

| Set | Source | Rule | Counts |
|---|---|---|---|
| AUTO.RIA KZ test | `autoriaNumberplateOcrKz-2022-11-29/test` | the dataset's own held-out `test` part (split by source zip) | 279 |
| AUTO.RIA KG test | `...Kg-2022-11-30/test` | same | 1,068 |
| AUTO.RIA RU test | `...Ru-2021-09-01/test` | same | 2,845 |
| AUTO.RIA train / val | native `train` / `val` of the same zips | **leakage guard**: KZ/KG crops whose ad id (file-name prefix) also occurs in the test part are dropped (KZ: 497 train + 57 val; KG: 263 train + 1 val); RU file names are the plate text, so crops whose text equals a test text are dropped (10 train + 1 val). RU train/val are further sub-sampled at random (25,000 of 49,372; 1,500 of 4,892) for the NVMe budget | KZ 8,145 / 944, KG 28,941 / 1,644, RU 25,000 / 1,500 |
| Synthetic UZ test | `lanes/ident/synth/gen_ocr.py --native --seed 7` | texts are invented; the train/val generators reject any text that occurs in the test set | 3,000 |
| Synthetic UZ train / val | same generator, seeds 1000 / 2000 | disjoint from the test texts | 60,000 / 3,000 |
| Real UZ test | Roboflow Uzbek_Plates **valid** originals | see below | 301 crops (289 cars) |

## Real Uzbek images (Roboflow Uzbek_Plates v1, 8,297 images exported)

The export holds 301 untouched originals in `valid` and 7,996 images in `train` that are Roboflow augmentations (up to 3 copies per source photo: horizontal flip, crop, shear). A source photo = one car id + camera (`car-front` / `car-rear`), and a car appears with a front and a rear photo (same plate text), so the split unit is the **car id**.

| Split | Rule | Images | Cars |
|---|---|---|---|
| test | all 301 `valid` originals | 301 | 289 |
| quarantine (never trained on) | `train`-split images of any car that also occurs in test (627), plus pHash near-duplicates (Hamming <= 8) of a test image (2 images, already inside the 627 or counted separately) | 627 | 200 |
| val | about 8 % of the remaining cars, chosen by a seeded hash of the car id | 571 | 136 |
| train | the rest | 6,798 | 1,535 |

Notes: crops/boxes are stretched to 640 x 640 by Roboflow's preprocessing, so plate aspect ratios are distorted by the original photo's aspect ratio. About 40 % of the photos are upside-down (phone rotated), a few plates are vertical.

## Detection

| Set | Rule | n |
|---|---|---|
| UZ real test | the same 301 `valid` originals | 301 images |
| RodoSol test | `cars-br` + `cars-me` (Kaggle mirror, 10,001 images), block split: image index // 25 hashed with a fixed seed, 40 % test / 20 % val / 40 % train; 1,000 random test images evaluated | 1,000 |
| CCPD-DB / Weather / Challenge | CCPD2019 images from the HF mirror `zenitsu09/ccpd-subset-30k`; file-name hash < 0.5 = eval half, otherwise train half (used for detector training and compositing) | DB 1,035 total, Weather 1,021, Challenge 1,200 fetched; the eval halves are evaluated |

Detector training data never contains: Roboflow test/quarantine, RodoSol test/val blocks, CCPD eval-half files.

## Real-UZ OCR test labels

The 301 test crops were labelled by reading every crop (Claude, an AI) on contact sheets, with two independent models as cross-checks (see `ident_notes.md`). They are **pending human spot-check**: `docs/lanes/ident_uz_review.html` shows every crop with its label.
