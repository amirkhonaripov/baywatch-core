# Lane conventions (read fully before starting)

Project: **Seen**. Cameras at a car service centre answer *where/when* (visit flow, status) and *what was done* (operations matched to work-order lines). Partner mode: Royal Motors dealer group (Toyota, BYD, Ducati) in Uzbekistan. Plan: `/Users/diznect/.claude/plans/ethereal-fluttering-alpaca.md`. Research: `research/*.md` (especially `data_inventory_bay.md`, `data_inventory_inspection.md`, `stage_data.md`, `data_models.md`).

## Hard rules
1. **Honesty.** No invented metrics, no hard-coded answers, no mock outputs shown as real. Every number must be reproducible by a script in the repo. Report real and synthetic results separately, never blended. If something fails, say so.
2. **Ownership.** Only edit files in your lane's directories (listed in your brief). `seen/*` (except files explicitly assigned to you), `contracts/*`, `scripts/*` and `docs/deck_notes.md` belong to the integrator. If you need a change there, put it in your final report under "Requests to integrator".
3. **Contracts.** Code against `seen/predictor.py` (`Frame`, `Det`, `FramePred`, `Predictor`), `seen/vocab.py` (label strings), and the generated models in `seen/contracts/` (from `contracts/*.schema.json`). Do not invent new label strings; request them.
4. **No git.** Repo is not under git yet. Do not run `git init`.
5. **Licences.** Non-commercial or research-only datasets are allowed. Record every dataset and pretrained model you use in `docs/lanes/<lane>_sources.md` with: name, URL, licence (exact, or "not stated"), what you used it for (train/val/test/demo), faces/plates present (y/n).
6. **Faces and plates.** Public datasets containing them are OK for training and testing. Never put identifiable faces into evidence images or the web UI. Evidence images go through `seen.blur.blur_faces`.
7. **Eval independence.** Lane G owns the held-out test manifests in `eval/manifests/`. Trainers must exclude every item listed there (use `eval/manifests/<component>.jsonl` ids; if G hasn't produced one yet, create your own split by **source**, not by frame, write it to `docs/lanes/<lane>_splits.md`, and tell G).

## Machines
- **Mac (source of truth for code):** `/Users/diznect/Downloads/hackathon_final/`.
- **Box (GPU):** ssh alias `rtx`. Workspace `/mnt/bigdrive/amirkhon/seen/` (do not use "hackathon" in any box path; do not write outside this workspace; do not touch other users' folders — in particular never read or write `/mnt/bigdrive/sanjar` or `/mnt/bigdrive/jowillianto`).
  - 2× RTX PRO 5000 Blackwell 48 GB (sm_120), 32 threads, 60 GB RAM (shared machine: keep RAM < 16 GB per job, dataloader workers ≤ 8).
  - Env: `/mnt/bigdrive/amirkhon/seen/env/main` (torch 2.14 cu130, ultralytics, rfdetr, onnxruntime-gpu, transformers, timm, open_clip, fast-plate-ocr, fast-alpr, torchcodec, PyNvVideoCodec, remotezip, yt-dlp, supervision, imagehash). Install extra packages with `scripts/box.sh -- flock /mnt/bigdrive/amirkhon/seen/locks/pip.lock uv pip install <pkg>` (this goes into the active env; see the env note below). **Never upgrade/downgrade torch.** If you need conflicting deps, create a separate venv under `env/<lane>-<name>` and say so.
  - Blender 5.0.1: `/mnt/bigdrive/amirkhon/seen/tools/blender-5.0.1-linux-x64/blender`.
  - `/usr/local/cuda` = CUDA 13.2 (nvcc not on PATH; use full path). TensorRT Python libs are not installed yet (ORT falls back to CUDA EP; pass providers=['CUDAExecutionProvider','CPUExecutionProvider']).
- **Workflow:** edit on Mac → `scripts/sync.sh` (rsync Mac→box `code/`, with --delete) → run with `scripts/box.sh`:
  - `scripts/box.sh --gpu 1 -- python -m lanes.flow.eval_ndispark` (foreground; sets env, PYTHONPATH, SEEN_ROOT, HF_HOME)
  - `scripts/box.sh --gpu 0 --lock --bg ident_train -- python -m lanes.ident.train ...` (detached; log at `logs/ident_train.log`; `--lock` queues on GPU0)
  - Fetch results: `scripts/sync.sh pull runs/<lane>/<exp>/metrics.json docs/lanes/`
  - **Never write outputs into `code/` on the box** (they get deleted by the next sync).
- **GPU policy:** GPU0 = training (always `--lock`). GPU1 = evaluation, inference, pre-labelling, Blender (shared; keep each job ≤ 20 GB VRAM).
- **Disk budget on `/mnt/bigdrive` (shared drive, 478 GB free):** ident 30 GB, flow 40 GB, ops 40 GB, inspect 30 GB, sim 50 GB, eval 10 GB. Check with `du -sh` before big downloads; skip a dataset rather than exceed the budget, and report it.
- Paths: `seen/paths.py` (`RAW`, `PREP`, `RUNS`, `ARTIFACTS`). Raw data → `data/raw/<dataset>/`; prepared → `data/prep/<lane>/`; runs → `runs/<lane>/<exp>/`; deployable models → `artifacts/<lane>/<model>/` with `model.onnx` (or weights) + `model_card.json` {name, version, classes, input_size, metrics_ref, training_data, licence_notes}.

## Deliverables every lane produces
1. `lanes/<lane>/predict.py` with `get_predictor(**kw) -> Predictor` (loads from `artifacts/`), plus `lanes/<lane>/test_smoke.py` (pytest, CPU-runnable on a tiny input; skip GPU parts if no CUDA).
2. Training/eval scripts that regenerate every number you report: write metrics as JSON matching `contracts/metrics.schema.json#/$defs/ComponentResult` to `runs/<lane>/<exp>/component_results.json` (list), and copy them to `docs/lanes/<lane>_results.json` on the Mac.
3. `docs/lanes/<lane>_notes.md`: what you did, datasets, splits, metrics with n and 95% CIs, **failure cases** (save 5–15 example images, face-blurred, to `docs/lanes/failures/<lane>/`), known limits, and a section **"For the deck"** with must-mention caveats (e.g. "trained on handheld YouTube video, not CCTV").
4. Final reply (≤ 500 words): what works, key numbers, what failed/was cut, requests to integrator.

## Timebox
Prefer a working, measured baseline early, then improve. Report progress in your notes file as you go. If a dataset download fails or is too slow, fall back to the next option listed in your brief and document it.

## Shared eval tooling (from Lane G, ready now)
- Stats: `from eval.stats import prop_metric, boot_metric, mae_metric, detection_ap, confusion, ece, accuracy_coverage`. `prop_metric(name, k, n)` gives a Wilson CI. Bootstrap resamples images or videos, not boxes or frames. These return dicts in the shape of `metrics.schema.json` Metric.
- Exclusion: `from eval.manifest import excluded_ids, filter_train`.
  - `kept, dropped = filter_train(train_paths, components=[...])` drops held-out items by path, id or near-duplicate image (pHash).
  - Manifests live at `eval/manifests/<component>.jsonl`, one JSON object per line: {id:"<dataset>:<relpath>", path (relative to SEEN_DATA), source, dataset, split, phash, label}.
  - Until G publishes a component's manifest, split by source with `eval.manifest.split_by_source` and document the split in `docs/lanes/<lane>_splits.md`.
- Leakage audit: write your training file list (one path per line) to `runs/<lane>/<exp>/train_files.txt`. G runs `python -m eval.manifest audit --train-list ... --components ...`.
- Failure gallery: optionally add `docs/lanes/failures/<lane>/failures.json` [{id, component, file, description, category}]. G re-blurs faces.
- Box extras: the `kaggle` CLI (authenticated at user level — never read the credentials), `kagglehub` and `roboflow` are installed in env/main. `ROBOFLOW_API_KEY` will appear in `/mnt/bigdrive/amirkhon/seen/.env` once the user adds it.

## NVMe staging (user-approved: 60 GB total), because the HDD is IO-saturated
- The shared HDD `/mnt/bigdrive` hit about 90% "full" IO pressure, and the GPUs sat idle at 0%.
- **Prepared/active training and eval data now goes to the NVMe** at `/home/amirkhon/seen_fast/<lane>/` (`seen.paths.FAST` / `fast_dir(lane)`; env `SEEN_FAST` is set by `scripts/box.sh`).
- Per-lane budgets: ident 10 GB, flow 12 GB, ops 15 GB, inspect 12 GB, sim 8 GB, eval 3 GB. Check with `du -sh /home/amirkhon/seen_fast/<lane>` before writing. Never exceed your budget, and never write anywhere else on the NVMe.
- These stay on the HDD workspace: raw downloads/archives, runs/weights, artifacts, final renders.
- IO hygiene:
  - Copy to the NVMe with one sequential pass (resize/re-encode while copying) instead of random reads in parallel jobs.
  - At most one IO-heavy prep job per lane at a time.
  - Dataloaders read from the NVMe copy.
  - Hash and dedup on the NVMe copy.
- Cleanup: at the end, each lane lists what in its fast dir is still needed. The integrator deletes the rest.

## Model naming in events (integrator, after the Lane F review)
- Give your predictor class a `role` attribute: "vehicle", "plate", "router", "states" or "inspection".
- Use a meaningful `name` and a `version` such as the artifact/model-card version.
- The pipeline maps roles to the `model` and `model_version` on every event's evidence. "plate@0" must never appear in the UI.

## Python env moved to the NVMe (integrator)
- `scripts/box.sh` now activates **`/home/amirkhon/seen_fast/env/main`**: the same packages, rebuilt on the NVMe because imports from the HDD env were stalling for minutes.
- The old HDD env (`/mnt/bigdrive/amirkhon/seen/env/main`) still exists as a fallback. Jobs already running keep using it.
- If a package you installed into the old env is missing, install it again with the pip lock command above.
- Separate venvs (VLM etc.): create them under `/home/amirkhon/seen_fast/env/<lane>-<name>`. They count toward your lane's NVMe budget.

## Precision policy (integrator)
- **Training:** use mixed precision (AMP; bf16 preferred on these Blackwell cards). Ultralytics defaults to `amp=True`, so don't disable it. For Keras (fast-plate-ocr), set `keras.mixed_precision.set_global_policy("mixed_bfloat16")` unless it hurts accuracy.
- **Inference / export:** FP16 or BF16 is allowed **only after a parity check** against FP32 on your test set: same metric within noise. Report both precisions.
  - Known failure: RF-DETR under TensorRT FP16 drops to AP50 0.18–0.25. Test BF16 for RF-DETR instead.
  - Anything that fails parity ships in FP32 (ONNX Runtime CUDA).
