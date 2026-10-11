# Lane C (ops / bay states): datasets and pretrained models

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
