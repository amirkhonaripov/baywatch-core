# Lane A (ident): datasets and pretrained models used

Columns: name, URL, licence (exact as found, or "not stated"), what we used it for, faces/plates present.
"Train/val/test" refer to our own splits (see `ident_splits.md`). All plates in real datasets belong to real vehicles of people who did not consent to this project; nothing from these sets is shown in the web UI or the deck (per `docs/LANES.md` rule 6, public sets are allowed for training and testing).

## Datasets

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

## Pretrained models

| Name | URL | Licence | Used for |
|---|---|---|---|
| fast-plate-ocr `cct-s-v2-global-model` (ONNX + Keras weights `cct_s_v2_global.keras`), `global-plates-mobile-vit-v2-model` | https://github.com/ankandrew/fast-plate-ocr (releases `arg-plates`) | MIT | baseline numbers; `cct_s_v2_global.keras` is the initialisation of our fine-tuning |
| fast-alpr detector `yolo-v9-t-384-license-plate-end2end` (open-image-models) | https://github.com/ankandrew/open-image-models | MIT (training data of the weights not stated) | baseline detector numbers |
| RF-DETR (Nano/Small) COCO-pretrained backbone | https://github.com/roboflow/rf-detr | Apache-2.0 | initialisation of the plate detector (see notes) |
| Qwen3-VL-4B-Instruct | https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct | Apache-2.0 | second independent reader for pre-labelling the real-UZ OCR test set (labels then verified by Claude, an AI, pending human spot-check) |

## Fonts (synthetic plate renderer, `lanes/ident/synth/fonts/`)

No freely licensed digitisation of the real FE-Schrift plate font with a verifiable licence was found (searched GitHub/Google Fonts; "Euro Plate" by Andy Hoppe and similar have unverified terms). The renderer uses these DIN-like faces from https://github.com/google/fonts instead, all **SIL Open Font License 1.1** (texts in `fonts/OFL-*.txt`): Barlow Bold, Barlow Semi Condensed Bold/SemiBold, Barlow Condensed SemiBold, Saira Semi Condensed Bold, Share Tech Regular, with random width/weight jitter. Glyph details of the real plates (for example the slotted zero) are therefore only approximated.
