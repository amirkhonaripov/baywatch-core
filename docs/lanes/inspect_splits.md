# Lane D (inspect) — splits and held-out sets

`eval/manifests/` had no manifest for the inspect components when this lane was built, so splits are made **by source and by pHash
cluster** (rule from `docs/LANES.md` hard rule 7). Held-out lists are written by the prep scripts to
`data/prep/inspect/*/splits.json` on the box and copied to `docs/lanes/inspect_manifests/` (one JSONL per component, in the
`eval/manifests` schema) for Lane G to adopt. Training file lists for the leakage audit are in
`runs/inspect/<exp>/train_files.txt` (original image paths, hashed by `eval.manifest.leakage_audit`).

## Near-duplicate rule (all sets)
`eval.manifest` 64-bit DCT pHash, Hamming distance <= 8, computed on the image **and its mirror** (Roboflow re-uploads and
augmentations are often flipped). Clusters = connected components of that relation; a cluster is never split across
train / val / test. Whenever a test item and a training item are near-duplicates:
* in-source tests (carved out of a training source): the **training** twin is dropped, the test set stays intact;
* cross-source tests (a source never used in training): the **test** twin is dropped and the count is reported.

## Damage (7 classes = `seen.vocab.DAMAGE_TYPES`)
| Set | Role | Rule |
|---|---|---|
| VehiDE train JSON (11,621) | train / val | 95 % / 5 % by cluster |
| VehiDE official validation (2,324) | **in-source test** `vehide_val` | never touched |
| HITL damage (814) | train / val / **in-source test** `hitl_test` | 80 / 10 / 10 by cluster |
| CarDD HF mirror (2,816) | model B: train / val; **test** `cardd_test` | 75 / 5 / 20 by cluster |
| CarDD train+val portion | `cardd_trainpart`: **cross-source test for model A only** | model A never sees CarDD |
| SYNDCAR (245) | **cross-source test** | all images, test only |
| Kaggle Coco Car Damage (70 annotated) | **cross-source test**, class-agnostic | all, test only |
| Roboflow jijo/car-damage-new v2 | **cross-source test**, dent / scratch | all, test only |

Two training variants share one image pool: **A** = VehiDE + HITL (no CarDD), **B** = A + CarDD train. Model A gives clean
cross-source numbers on CarDD, SYNDCAR, Coco and Jijo; model B is the shipped model (its held-out tests are `cardd_test`,
`vehide_val`, `hitl_test` in-source and SYNDCAR / Coco / Jijo cross-source).

## Parts (20 unified classes, `lanes/inspect/common.py::PART_CLASSES`)
| Set | Role | Rule |
|---|---|---|
| Ultralytics carparts-seg (all 4,193 images pooled) + HITL parts (998) | train / val / in-source tests `carparts_test`, `hitl_parts_test` | re-split 80 / 5 / 15 by pHash cluster (carparts-seg ships Roboflow augmentation copies of the same photo in several of its own splits) |
| CarDamagePanel (565) | **cross-source test** | test only |
| SYNDCAR parts (245) | **cross-source test** at coarse part-family level | test only |

## Tyres
| Set | Role | Rule |
|---|---|---|
| TyreNet (1,698) | train / val / in-source test | 70 / 10 / 20 by cluster |
| Kaggle warcoder tyre-quality | **cross-dataset test** | images with a pHash twin in TyreNet removed (overlap reported) |
| Kaggle sameersambhare1 | **cross-dataset test** | same removal; NEW + SERVICEABLE = good, UNUSABLE = defective |

## DOT
| Set | Role | Rule |
|---|---|---|
| Roboflow TyreAI OCR (4,054 train+valid pooled) | character reader train / val / in-source test | 80 / 5 / 15 by cluster |
| tire-dot-finder + tyre-sidewall-text-label ('dot' class, train) | DOT region detector train / val | 88 / 12 by cluster |
| dot-code-6shg0 (97) | **cross-source test** of the region detector | test only; near-duplicates removed from training |
