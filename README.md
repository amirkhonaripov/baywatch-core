# BayWatch core: models, pipeline and API

Built by team DNA during the hackathon final (10–11 Oct 2026).

BayWatch turns a dealer's existing service-centre cameras into answers to two questions about every service visit:
- **Where and when:** from the plate at the gate, through the bays (car present, lift raised, wheel off, bonnet open, people working or idle), to wash, ready and handover. Idle bays are flagged and the customer can be told the status.
- **What was done:** the operations seen on camera are matched to the work order. Each paid line gets **evidence found**, **not observed** or **not camera-verifiable**. It never claims a job is "done".

This repository holds the models, the pipeline, the API, the evaluation and the data documentation.

The website lives in **[baywatch](https://github.com/amirkhonaripov/baywatch)** (live: https://amirkhonaripov.github.io/baywatch/).

## Quick start (Linux, NVIDIA GPU)

Requirements:
- **Python and tools:** Python 3.12 and [uv](https://docs.astral.sh/uv/).
- **NVIDIA driver:** ≥ 580 (CUDA 13); see below for older drivers.
- **GPU memory:** 24 GB with the vision-language model, about 8 GB without it.
- **Disk:** about 25 GB (about 6 GB without the VLM).

```bash
git clone https://github.com/amirkhonaripov/baywatch-core && cd baywatch-core
scripts/setup_env.sh                 # .venv with pinned packages (TORCH_CUDA=cu128 scripts/setup_env.sh on older drivers)
source .venv/bin/activate
scripts/download_weights.sh          # our weights (1.9 GB, Google Drive, sha256-checked) + SigLIP2 + Qwen3-VL-8B
                                     # add --no-vlm to skip Qwen3-VL (17 GB); the fast models still run
source baywatch_data/env.sh
python -m serve.local --port 8008    # the API, models loaded once (about 1 minute)
```

Check it, from a second terminal:
```bash
curl -s -F file=@demo_inputs/photos/uz_truck.jpg http://localhost:8008/analyze/image | python -m json.tool | grep -E '"scene"|"text"'
# expected: scene "closeup", plate text "01U418QC"
```

Then open the website against your API:
- **Locally:** `npm ci && npm run dev` in the baywatch repo.
- **Hosted:** `https://amirkhonaripov.github.io/baywatch/?api=http://localhost:8008#/console`

`demo_inputs/` has photos and short videos to upload to each camera. Its README lists the result we measured for every file, and none of these files was used in training.

If the script can't reach Google Drive, download the archive by hand from the [baywatch-weights folder](https://drive.google.com/drive/folders/1R-s7FKONv_iGv3URnUctRn7nQCSA2C7y?usp=sharing), then run `scripts/download_weights.sh --archive <file>`.

**What `download_weights.sh` gets:**
- **Our trained weights:** 25 files in one archive, listed with their sha256 in `weights/manifest.json`. Each file is checked after unpacking.
- **Worldwide plate detector cache:** linked into `~/.cache/open-image-models`.
- **Public base models from Hugging Face** (no account needed): `google/siglip2-so400m-patch14-384`, plus `Qwen/Qwen3-VL-8B-Instruct` unless you pass `--no-vlm`.

We tested this clean: an empty home folder, only the archive, then the pipeline on the demo photos. Same plate and tyre results as the deployed API.

## Architecture
```
photo | video (the website samples 16 frames per clip) | camera stream
 → scene router: SigLIP2-so400m zero-shot + our linear probe (blend), "unknown" below a calibrated threshold
 → detectors: RF-DETR-L fine-tuned by us (vehicles, people)
              + plates: our detector + our OCR (Uzbek, KZ/KG/RU formats) + a worldwide detector; "never guess" rule
 → face blur (YuNet) on every stored or shown frame
 → bay states: frozen DINOv2-L features + 24 geometry cues + our bonnet/part detectors → calibrated heads (every frame)
               └ cascade: Qwen3-VL-8B re-checks only uncertain frames + one frame per 15 s (max 8 calls per clip)
 → inspection on close-ups: our damage and car-part segmentation (damage → part), our tyre-condition head, DOT code reader
 → visit engine: HMM/Viterbi over the visit grammar (priors from our scenario engine), hysteresis fallback
 → work-order matcher → evidence found / not observed / not camera-verifiable
```
Every event carries its source (observed_real / observed_synthetic / simulated), its confidence, and the model and version that produced it.

### What we trained, and how it scores on data it never saw
| Model | Trained on | Held-out result |
|---|---|---|
| Plate OCR (Uzbek + CIS) | 62k Kazakh/Kyrgyz/Russian crops + **60,000 synthetic Uzbek plates from our generator** (never a labelled real Uzbek plate) | 99.0% exact on 299 real Uzbek crops; 86.6% without the synthetic plates; best pretrained 33.4% |
| Plate pipeline, photo → text | the OCR above + our detector (Uzbek_Plates, CCPD, RodoSol, synthetic composites) | 98.7% on 299 real Uzbek photos (off-the-shelf fast-alpr: 15.7%) |
| Vehicle/person detector | RF-DETR-L fine-tuned on UA-DETRAC + NDISPark + COCO subset | UA-DETRAC test AP50 0.887 (night 0.896, rain 0.868) |
| Scene router | linear probe on SigLIP2 features, 3,742 images from 33 sources | 0.85 accuracy on held-out sources |
| Bay states | heads on DINOv2-L: 5,034 Kinetics-700 clips, CrossTask/COIN, 6,785 frames labelled by Qwen3-VL (teacher) | AUROC on 458 real workshop CCTV crops we labelled: lift 0.93, bonnet 0.93, car 0.88, work 0.71 |
| Cascade (fast heads + VLM on uncertain frames) | — | same CCTV set, 30% of frames escalated: car present 0.96, bonnet 0.95, wheel off 0.86 accuracy |
| Damage / car parts / tyre | YOLO-seg damage + parts; DINOv2 tyre head | CarDD test AP50 0.70; parts mask AP50 0.87; tyre cross-dataset 87.1%. Damage does not transfer across sources (AP50 0.15–0.22) |
| Visit decoding (synthetic) | HMM with priors from our scenario engine | events within ±10 s at F1 0.89 when frames are 90% right (120 synthetic visits); thresholding alone F1 ≤ 0.03 |

Everything is in `docs/eval_report.md` (generated by `eval/run_all.py`), with confidence intervals, per-sample predictions and failure galleries.

The method, splits and caveats for each component are in `docs/lanes/<lane>_notes.md`. Real and synthetic results are always reported separately.

**Known failures:**
- No service-bay CCTV of our own: the bay-state models learned from handheld/YouTube video. Work-vs-idle drops from AUROC 0.89 to 0.71 on CCTV.
- Damage detection does not transfer across photo sources.
- Gate and street scenes get confused by the router.
- Toy and model cars create false visits.
- Foreign plates are detected but not read ("not readable").

## Repository layout
| Path | Contents |
|---|---|
| `seen/` | Core: predictor interface, pipeline, router, blur, zones, HMM, visit engine, work-order matcher (`seen` is the code name) |
| `lanes/ident` | Plates: synthetic Uzbek plate generator, detector, OCR, evaluation |
| `lanes/flow` | Vehicle/person detector, tracker, slot occupancy, scene router |
| `lanes/ops` | Bay-state heads, VLM teacher (`vlm.py`), cascade (`cascade.py`), CCTV test labels |
| `lanes/inspect` | Damage, parts, tyre condition, DOT reading |
| `sim/` | Scenario engine, noise test, Blender synthetic visits (`sim/blender/README.md`), the simulated day (`day.py`) |
| `eval/` | Independent evaluation: test manifests, leakage audit, statistics, robustness set, benchmarks, cost model |
| `serve/` | FastAPI app (`app.py`), local server (`local.py`), Modal app (`modal_app.py`), weight upload to Modal |
| `contracts/` | JSON Schemas; `scripts/codegen.sh` generates `seen/contracts/` and the website's `src/contracts.ts` |
| `weights/manifest.json` | The weight archive: Google Drive link, sha256 of the archive and of every file |
| `demo_inputs/` | Photos and videos for live demos (held out from training) |
| `research/` | Market, review and data research behind the product decisions |

## Re-training and re-evaluation
```bash
scripts/setup_env.sh --eval    # adds the evaluation / training packages
```
Each lane documents its data download, preparation, training and evaluation commands in `docs/lanes/<lane>_notes.md`. Datasets are not in git; every source, with link and licence, is in `DATA_SOURCES.md`.

`python -m eval.run_all` regenerates `docs/eval_report.{md,json}`. Copy the JSON to the website repo as `public/eval_report.json`.

Paths default to our GPU box. Elsewhere, set `SEEN_ROOT` (data, runs) and `SEEN_ARTIFACTS` (weights).

## Deploy (public API on Modal)
- `.github/workflows/modal.yml` runs `modal deploy serve/modal_app.py` on every push to `main` that touches the code.
  - Repository secrets needed: `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`.
  - The deployment history is in the Actions tab.
- Weights are not deployed from git. They live in the Modal volume `seen-artifacts`, pushed once with `python -m serve.upload_artifacts`.
- Two GPU classes on L4, both scale to zero:
  - `Analyzer`: all the fast models.
  - `VLM`: Qwen3-VL-8B.
- For the live defence we run the same API on our own GPU (`scripts/demo_rtx.sh`): no cold start, about 2× faster.

## Data, licences, privacy
- `DATA_SOURCES.md` lists every dataset, pretrained model and asset, with its licence and how it is used. Several are research-only; a commercial pilot retrains without them.
- Synthetic data (Uzbek plates, Blender visits, scenario timelines) is declared with its generation method.
- No face recognition. Faces are blurred on every stored or shown frame.
- Metrics are per bay and per job, never per employee.
- On site, video stays on the PC; only events leave.
