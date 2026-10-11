# Lane B (flow): datasets and pretrained models used

Faces / plates column = whether real faces / readable plates of private people appear in the data. Public datasets with faces/plates are allowed for
training and testing (user decision); nothing from them is shown in the UI, and evidence images go through `seen.blur.blur_faces`.

## Datasets

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

## Pretrained models

| Name | URL | Licence | Used for |
|---|---|---|---|
| RF-DETR Nano/Small/Medium/Base/Large (COCO) | https://github.com/roboflow/rf-detr | Apache-2.0 for these sizes (the XLarge/2XLarge sizes are under a separate licence and are not used) | vehicle + person detector (zero-shot and fine-tuned) |
| Ultralytics YOLO11x, YOLO26x (COCO) | https://github.com/ultralytics/ultralytics | AGPL-3.0 | zero-shot comparison only, not shipped |
| SigLIP2 so400m-patch14-384, base-patch16-384 | https://huggingface.co/google/siglip2-so400m-patch14-384 | Apache-2.0 | scene router (zero-shot + linear probe on frozen embeddings) |
| YuNet `face_detection_yunet_2023mar.onnx` | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet | MIT | face blur |
| OC-SORT (algorithm: Cao et al., CVPR 2023) | https://github.com/noahcao/OC_SORT | MIT; `lanes/flow/tracker.py` is our own re-implementation | tracker |
| ByteTrack (via `supervision`) | https://github.com/roboflow/supervision | MIT | tracker baseline |
