# Lane B (flow): splits (always by source: sequence, camera, day or image source, never by frame)

Until Lane G publishes `eval/manifests/<component>.jsonl` for these components, these are the splits. Only `eval/manifests/jury.jsonl` existed while this lane
ran; every router image was audited against it (see "Router").

## Vehicle detection (zero-shot comparison + fine-tuning)

| Set | Role | Unit of independence | Contents |
|---|---|---|---|
| UA-DETRAC **official test** (HF mirror `abhineet123/ua_detrac`, 40 sequences MVI_39031 ... MVI_40905) | test for every detector, never trained on | sequence (40) | stride 5 = 5 fps: 11,268 frames. Weather per sequence from the original XML: sunny 8, cloudy 11, rainy 9, night 12 sequences |
| UA-DETRAC official train (60 sequences) | fine-tuning only | sequence | stride 10; 6 whole sequences are held out as the fine-tuning validation set: MVI_20011, MVI_20051, MVI_40192, MVI_40752, MVI_40991, MVI_63553 (2 night, 1 rainy, 2 sunny, 1 cloudy) at stride 25; the other 54 sequences -> 7,343 training frames |
| NDISPark train (112 images, 7 cameras) | zero-shot day AP; fine-tuning (repeated x4) | image | all daytime |
| NDISPark validation (30 images) | zero-shot day AP; **held out from fine-tuning** | image | **daytime** (the Zenodo description says night; it is not: mean luminance equals the train split, see notes) |
| NDISPark test (117 images) | night counting MAE only (no boxes exist); held out | image | night; counts from `ground_truth_test_counting.csv` |
| COCO val2017, first 2,000 images of the seeded shuffle (`data_extra.cocoval_items`) | person-AP regression check; held out | image | persons incl. crowd regions as ignore |
| COCO val2017, images ranked 2,000+ of the same shuffle (2,500 with a vehicle/person annotation; 1,761 after the fetch limit) | fine-tuning regulariser (keeps the person class alive) | image | person/car/motorcycle/bus/truck boxes, no crowd boxes |

Training list for the audit: `runs/flow/ft_large/train_files.txt` (UA-DETRAC train frames, NDISPark train, COCO val2017 subset). Nothing in it comes from the DETRAC
test split, the NDISPark validation/test splits or the 2,000 COCO person-check images.

## Slot occupancy

* PKLot (PUCPR, UFPR04, UFPR05; sunny/cloudy/rainy) and CNR-EXT (sunny/overcast/rainy; 9 cameras). Capture **days** are split with `md5(day) % 3 == 0` -> calibration days
  (about one third), the rest -> test days; a day never appears on both sides. Per weather cell, up to 150 test images (spread over as many days as possible) and 100 calibration images.
  PKLot: 1,114 test images (63,048 slots), 726 calibration images. CNR-EXT: 351 test images (12,480 slots), 235 calibration images.
* (conf, tau) is chosen on the calibration images of ONE dataset and applied unchanged to the test images of both datasets (the cross-dataset rows).
* PKLot and CNR-EXT frames were subsampled once while copying to the NVMe (PKLot 1 in 4, CNR-EXT 1 in 2, md5 of the file path).

## Tracking

UA-DETRAC test only (the 40 sequences above). 5 fps = every 5th source frame, 1 fps = every 25th, 0.5 fps = every 50th. Sequences are also split into speed tertiles
(slow / mid / fast) by the median GT speed in box-widths per second, computed from the GT tracks.

## Scene router

* Pool: `data/prep/flow/router/manifest.jsonl` (3,742 images, 7 scenes, 33 sources). Every record carries source, dataset, licence and URL.
* **Held-out sources**: for each scene whole sources are held out (sorted by `md5(source)`) until at least 35% of the scene is held out, never more than about 55%, always keeping
  one source for training. Held out (15 sources, 1,371 images): `coco_street`, `rf_gate_counter` and `pklot_cam` (street); `commons_ducati` (motorcycle); `commons_engines`, `commons_hoods`,
  `commons_wheels` (closeup); `commons_lifts` (bay); `commons_search_{animals,doc,food,landscape,office}` (unknown); `gate2_toll_plazas` (gate); `rf_washing_car` (wash).
  The probe, its regularisation C, the probe/zero-shot blend weight and the "unknown" threshold are all chosen by leave-one-source-out on the *training* sources only.
  The prompt set (v1 vs v2) was also chosen from the zero-shot accuracy on the training sources only.
* **Lane G audit**: 10 images that are pHash near-duplicates of `eval/manifests/jury.jsonl` items (jury_023, 028, 041, 043, 044, 046, 047, 049, 051, 063) were removed from the pool before
  any fitting (`eval.manifest.filter_train`, list in `runs/flow/.../heldout_dropped.json`); the final probe is fitted without them. The jury set is used only as an external test
  (`router_jury_eval.py`). Training list for G: `data/prep/flow/router/train_files.txt` (Mac) .
* Hand curation (by eye, contact sheets): Commons gate/bay/wash categories contained mostly irrelevant images, so only the kept ids in `lanes/flow/curation.json` are used;
  Roboflow "gate" sets turned out to be highway CCTV and were relabelled `street` (counter) or dropped (augmented set).

## Face blur

WIDER FACE validation (3,222 images with at least one valid face, 39,112 faces) is used for testing only; YuNet was trained by its authors on WIDER FACE train.
