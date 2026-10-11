# Lane F (app + serving): data and model sources

Lane F trains nothing. It serves whatever `seen.pipeline.default_pipeline()` loads and ships a few public demo inputs.

## Demo inputs shipped in `web/public/samples/` (shown in the web UI)

| Name | URL | Licence | Used for | Faces / plates |
|---|---|---|---|---|
| Uzbek_Plates v1, image `AAF0119214_car-front_…` (Damas van, front) | https://universe.roboflow.com/suxrobs-workspace/uzbek_plates | CC BY 4.0 (dataset README on the box: `data/raw/ident_uz/roboflow_uzbek_plates/README.dataset.txt`) | demo input "Car with an Uzbek plate"; recorded backup run | faces: n (checked visually); plates: y (public dataset) |
| "Chery A1 - service shop in Ukraine (7)", ANT Berezhnyi | https://commons.wikimedia.org/wiki/File:Chery_A1_-_service_shop_in_Ukraine_(7).jpg | CC BY 3.0 (from `data/raw/ops_commons/meta.csv`) | demo input "Service bay: car on a lift" | faces: n; plates: n |
| "Classic Mustang at Garage, West Miami Florida August 2024", Phillip Pessar | https://commons.wikimedia.org/wiki/File:Classic_Mustang_at_Garage,_West_Miami_Florida_August_2024.jpg | CC BY 2.0 (from `data/raw/ops_commons/meta.csv`) | demo input "Garage bay, car raised" | faces: n; plates: y (small, US) |
| UA-DETRAC test sequence MVI_39051, frames 1–300, re-encoded to a 12 s H.264 mp4 (25 fps, 960×540) | https://detrac-db.rit.albany.edu/ | CC BY-NC-SA 3.0 | demo video for the in-browser sampler; recorded backup run | faces: not identifiable at this resolution (traffic camera); plates: not legible |
| `flower.jpg` from scikit-learn sample images (vultilion on Flickr) | https://www.flickr.com/photos/vultilion/6056698931/ | CC BY 2.0 (scikit-learn `datasets/images/README.txt`) | demo "not a car" input (unknown path) | n / n |

The Uzbek_Plates and Commons images are also used by other lanes (ident / ops). They appear in the demo only as inputs; Lane F reports no metrics on them.

## Pretrained models the API serves today (via `seen/baseline.py`, owned by the integrator)

| Name | URL | Licence | Used for | Faces / plates |
|---|---|---|---|---|
| RF-DETR Base (COCO), `rf-detr-base.pth` | https://github.com/roboflow/rf-detr | Apache-2.0 | vehicle + person detection in the served pipeline | trained on COCO (faces y) |
| fast-alpr default plate detector `yolo-v9-t-384-license-plates-end2end` (open-image-models) | https://github.com/ankandrew/open-image-models | MIT (repo) | plate detection in the served pipeline | plates y |
| fast-plate-ocr `cct-xs-v2-global-model` | https://github.com/ankandrew/fast-plate-ocr | MIT (repo) | plate OCR in the served pipeline (not Uzbek-tuned) | plates y |

When lanes replace the baseline predictors, their weights reach Modal through `serve/upload_artifacts.py` (Volume `seen-artifacts`); their licences are recorded in each lane's own `*_sources.md`.

## Web dependencies

React 18.3.1 (MIT), Vite 5.4.11 (MIT), TypeScript 5.6.3 (Apache-2.0), Vitest 2.1.8 (MIT). No runtime CDN; no analytics.
