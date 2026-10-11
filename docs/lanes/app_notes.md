# Lane F: web app, serving, deployment (notes)

Status 2026-10-10: web app builds and works end-to-end against `serve.local` on the box (GPU1) and against the deployed Modal API.
Everything below was measured in this session; scripts and commands are listed so each number can be regenerated.

## What exists

| Piece | Path | Notes |
|---|---|---|
| Web app (Vite + React + TS) | `web/` | Analyze / Result / Twin / Metrics / About; EN default, RU, UZ; hash routing (Pages-safe); dark/light |
| In-browser video sampler | `web/src/lib/sampler.ts` (+ `sampler.test.ts`) | 2 fps, long side ≤ 720 px, ≤ 60 s, ≤ 120 frames, JPEG q=0.82; video never uploaded |
| Photo prep | `web/src/lib/image.ts` | EXIF orientation baked in via `createImageBitmap(imageOrientation:"from-image")`, long side ≤ 1920, JPEG |
| API (FastAPI) | `serve/app.py`, `serve/core.py` | implements `contracts/api.openapi.yaml`; Backend / JobStore / EvidenceStore abstractions; limits |
| Local runner | `serve/local.py` | `scripts/box.sh --gpu 1 -- python -m serve.local --port 8008` + `ssh -N -L 8008:127.0.0.1:8008 rtx` |
| Modal app | `serve/modal_app.py` | CPU `@asgi_app` web function + L4 GPU class; `spawn()`/`FunctionCall.from_id().get(timeout=0)`; `modal.Dict` "seen-jobs"; Volumes "seen-artifacts", "seen-evidence" |
| Weights upload | `serve/upload_artifacts.py` | box → Volume "seen-artifacts" (`--dry-run` lists 5 files, 0.76 GB today) |
| Recorded backup runs | `serve/record_run.py` → `web/public/samples/recorded/` | real API output stored unchanged + evidence JPEGs; UI labels them "recorded, not live" |
| API tests (CPU) | `serve/test_app.py` | 6 tests: health/catalogue, photo→visit+evidence, non-car→unknown, frames job, limits (413/400/404), kill switch |
| Workflows | `.github/workflows/pages.yml`, `.github/workflows/modal.yml` | inactive until the repo exists |
| Placeholders | `web/public/eval_report.json`, `web/public/data_sources.json` | `"placeholder": true`, no numbers; the UI shows a warning banner |

### API details
- Limits (`serve/core.py`): ≤ 120 frames, ≤ 10 MB per image, ≤ 80 MB per request, last timestamp ≤ 60 s (+0.5 s tolerance), server-side downscale of the long side to 1920 px (warning added to `warnings` when it happens).
- Errors: 400 bad input, 413 too large, 404 unknown job/evidence, 503 kill switch (`SEEN_DISABLED=1`) or GPU cold start > 140 s, 500 pipeline failure (message passed through).
- Evidence URIs are relative (`evidence/<input_id>_<frame>.jpg`); the web resolves them against the API base (live) or the recorded-run folder. Images are written only by `Pipeline.analyze`'s evidence writer, i.e. after `seen.blur.blur_faces`.
- `GET /health?warm=1` (extension, not in the OpenAPI file) asks Modal to start a GPU container in the background, at most once per 60 s; the Analyze page calls it on open so the GPU warms while the user picks a file.
- CORS: any `https://*.github.io` origin and localhost, plus `SEEN_CORS_ORIGINS`.

### Web details
- Honesty in the UI: every event shows a source badge (observed · real input / observed · rendered input / simulated), confidence, `model@version`; simulated events are hidden by default, dashed, ghosted and tagged "SIMULATED"; the twin draws the simulated path dashed with a ghost car; recorded runs carry a purple "recorded, not live" banner; scene "unknown" shows "Seen does not guess"; bays without data say "–"/"no camera data" (not "free").
- Cold start: after the upload finishes, if no answer within 5 s the UI says "Warming up GPU… the first request after idle can take up to a minute." Network errors offer Retry and "Open recorded backup run".
- Config: `VITE_API_BASE` (build time), `VITE_BASE` (Pages sub-path; default `./`), and `?api=https://…` at runtime overrides the API.
- Zone editor: tap 4 corners on the first frame, choose a zone type (bay/gate/wash/ready/…); default = whole frame. The twin places the car from the zone type; with a whole-frame zone it uses an "unmapped camera" box. It never guesses a location, except that raised / wheel_off / person_under_car put the car in Bay 1 with the label "place inferred from the bay state".

## Verification done (this session)

### Build and unit tests
- `npm run build`: passes. Bundle: JS 227.8 kB (74.6 kB gzip), CSS 15.7 kB (4.3 kB gzip), index.html 0.7 kB. Runtime dependencies are React and React DOM only. `dist/` is 5.2 MB, of which 5.0 MB is `samples/` (loaded only when tapped: thumbnails ≈ 20 kB each, the 12 s demo clip 1.9 MB, recorded runs 2.3 MB).
- `npx vitest run`: 7/7 (sampleTimes, targetSize, monotonic).
- `pytest serve/test_app.py` on the box: 6/6.

### In-browser sampler (Chromium, built-in browser pane)
- 12 s UA-DETRAC clip (960×540, 25 fps): 24 frames, timestamps exactly 0.0…11.5 s in 0.5 s steps, 720×406, ≈ 87–93 kB per frame, 2.0–2.1 MB total, 1.5 s to sample on the Mac.
- Rotation: the same clip remuxed with `-display_rotation 90` (what an iPhone portrait video carries) was reported by the browser as 540×960 and sampled to 406×720 portrait with the frame rotated, with no double rotation and no squashing (`app_screens/08_sampler_rotation_test.png`). Safari on a real iPhone was **not** tested (no device available); the code path relies on browsers applying the track matrix, which current Safari, Chrome and Firefox do.
- Bug found and fixed: the first sampled frame could come out black when `currentTime` was already 0 (no real seek happened). The sampler now always seeks, waits for `seeked` plus `requestVideoFrameCallback`, and re-draws frame 0 if it is fully black.

### End-to-end against `serve.local` (box GPU1, RTX PRO 5000, baseline pretrained models)
| Input | Result |
|---|---|
| (a) `uz_plate_van.jpg` (Uzbek plate "40 B 472 RA") | scene street, truck 0.70, plate read **"C0B472RA"** (conf 0.87; the global OCR is not Uzbek-tuned, which is an honest miss), visit created, work order: oil change not_observed, wash not_observed, cabin filter not_camera_verifiable |
| (a) `bay_lift_chery.jpg`, `bay_lift_mustang.jpg` | visit created; plates read "AA2483IP" (0.90) and "IN9904" (1.00); `bay_states` skipped: "module not deployed" |
| (b) 12 s video sampled in the browser + gate zone + 3-line work order + simulation on | 24 frames, visit with arrived / state_started car_present / entered_zone (zone gate), 38–39 simulated events returned by the scenario engine, server time 2.3 s |
| (c) `non_car_flower.jpg` | scene **unknown**, no visit, warning "No vehicle detected: no visit created.", skipped modules listed with reasons |
- Warm latency per photo on the box: 52–58 ms server-side (`server_total`). The first requests after start were very slow (116 s, then 10 s): the box disk was at 80–94 % I/O stall from other lanes' downloads, and lazy imports were paying for it.
- Same flows through the UI at 375×812 (mobile emulation): screenshots `app_screens/01…07`. No horizontal overflow (`scrollWidth == 375`) after a fix for long file names.

### Deployed on Modal (`modal deploy` from the box, profile "sanjar")
- URL: **https://sanjar--seen-api.modal.run** (public, unauthenticated). GPU L4, `min_containers=0`, `max_containers=2` (GPU) / 2 (web), `scaledown_window=120 s`, CPU memory snapshot on (imports only), timeout 300 s.
- Smoke (`modal run serve/modal_app.py`, first ever cold start incl. RF-DETR weight download into the Volume): 68.7 s wall; `torch 2.14.1+cu130`, CUDA available on L4.
- After deploy (end-to-end = curl from the box in Tashkent to Modal, `uz_plate_van.jpg`, 44 kB). `timings_ms.modal_roundtrip` is measured in the web function around the GPU call:
  | Version | Call | End-to-end | GPU-side `server_total` | `modal_roundtrip` |
  |---|---|---|---|---|
  | v1 (commit before return) | cold (scale-from-zero) | 42.4 s, 43.5 s | 0.25 s | n/a (container `pipeline_load_s` 24.6–24.9 s) |
  | v1 | warm | 1.8–2.4 s | 70–90 ms | 0.9–1.3 s (of which Volume commit 0.4–0.8 s) |
  | v2 (background commit) | cold, after 330 s idle | 29.4 s | 0.24 s | 28.4 s (`pipeline_load_s` 23.2 s) |
  | v2 | second cold, after another 330 s idle | 26.1 s | 0.23 s | 25.1 s (`pipeline_load_s` 21.1 s) |
  | v2 | warm (4 calls) | 1.3–1.7 s | 75–83 ms | 0.34–0.39 s |
  - What the cold start is made of: ≈ 21–23 s is `pipeline_load_s` (RF-DETR 373 MB from the Volume, ORT sessions, one warm-up inference); the CPU memory snapshot only covers imports. Modal's experimental GPU snapshot was not tried (not validated, so not enabled).
  - One outlier on v1: a call made while the old container was scaling down took 119.7 s (a new container started; its `pipeline_load_s` was 28.3 s). It stayed under the 140 s wait, but it shows that a cold start can be much slower than 30–45 s.
  - Evidence JPEG fetched right after the result (v2, background commit + up to 3.5 s retry in `/evidence`): 200 in 1.2–1.9 s.
  - frames job, 24 frames (720p) + work order + simulate (v1, warm): 13.8 s from upload to `done` (`server_total` 2.7 s).
  - non-car image: 3.7 s, scene unknown, no visit.
  - Browser → Modal (Chromium, mobile viewport, `?api=https://sanjar--seen-api.modal.run`, CORS from localhost): 12 s clip sampled in the browser (24 frames) + simulation → result page in 6.6 s (GPU warm, `total` 1.5 s); evidence images and timeline thumbnails loaded from the Volume.
  - `/health` and `/catalogue`: 0.7 s warm; first web-function cold start 6.4 s.
  - GPU warm-up policy: the page never warms the GPU on load (bots would keep an L4 running); it calls `/health?warm=1` when the user picks a file, so the container starts while the browser samples or prepares the input. The server rate-limits warm calls to one per 60 s.
- Bug found and fixed on Modal: `/health` returned 500 because the cached GPU info held a `torch.__version__` object that the torch-free web image could not unpickle. Dict values are now JSON round-tripped.

## Known limits / not done
- 3D twin (react-three-fiber): not built (optional; it is on the plan's cut list). The 2D SVG twin is complete.
- iPhone Safari not tested on a device (see above). HEIC: converted to JPEG in the browser where the browser can decode it (iOS Safari yes; desktop Chrome no → clear error).
- Volume evidence has no retention or cleanup yet; jobs in `modal.Dict` expire per Modal's Dict TTL. No per-IP rate limit; spend is bounded by `max_containers=2` on L4 and scale-to-zero, plus the `SEEN_DISABLED=1` kill switch (redeploy).
- `GET /health?warm=1` is an extension of the OpenAPI contract (integrator may want to add it).
- Model names in evidence are only as good as what `seen/pipeline.py` passes (see requests below).

## Requests to integrator
1. `seen/pipeline.py`: the `model_versions` dict passed to the visit engine has no `plate`/`plate_version`/`states` keys, so evidence shows `plate@0` for plate reads and `vehicle-detector+hmm-fitted@0` for state events instead of `fast-alpr-global@pretrained` etc. Please add per-role names and versions (honesty: every result must carry its real model + version).
2. `seen/pipeline.py`: `PretrainedALPR` runs as a detector on every frame, so for a non-car image the route says `plate` was skipped ("no vehicle detected") while `timings_ms.plate` shows it ran. Either run ALPR after the router or stop listing it as skipped.
3. `contracts/api.openapi.yaml`: add `GET /health?warm=1`, the 400/413/503 responses, and "evidence URIs are relative to the API base". The `timings_ms.modal_roundtrip` key only appears on Modal.
4. `seen/router.py`: the heuristic gives scene "unknown" with confidence 1.0; the UI shows "unknown · confidence 100%". Consider reporting the max non-unknown probability, or omitting confidence for unknown.
5. `seen/baseline.py`: RF-DETR warns "Model is not optimized for inference" (`optimize_for_inference()` / fp16 would cut L4 latency).
6. Please generate `web/public/data_sources.json` from `DATA_SOURCES.md` (shape in the placeholder file), and ask Lane G to write `web/public/eval_report.json` (schema `contracts/metrics.schema.json`; drop `"placeholder": true`).
7. `scripts/box.sh --bg`: the ssh session stays open until the remote process exits (I ran it with run_in_background). While you were editing it, I once saw `scripts/box.sh: line 30: lse: command not found`. Concurrent `scripts/sync.sh` runs from several lanes race on `--delete` (one rsync error seen; a retry fixed it).
8. When lanes ship weights: `scripts/box.sh -- python -m serve.upload_artifacts` (Volume `seen-artifacts`), update `default_pipeline()`, then `modal deploy serve/modal_app.py` from the box. Add any new pip packages to `GPU_DEPS` in `serve/modal_app.py` (pinned to the box env).

## For the deck
- "Video never leaves the phone: the browser samples 2 frames/s at ≤ 720p; only JPEG frames are uploaded."
- Latency (L4 on Modal, baseline pretrained models, measured from Tashkent): cold start ≈ 30–45 s after scale-to-zero (once 120 s), warm photo ≈ 1.4 s end-to-end (≈ 0.08 s on the GPU); 24-frame clip ≈ 14 s end-to-end. Box RTX PRO 5000: ≈ 55 ms per photo server-side. These are baseline-model numbers, not the final lane models.
- Every result in the UI shows source, confidence, model and version; simulated continuation is off by default and dashed; recorded runs are labelled; "unknown" is a first-class answer.
- The plate OCR in the demo today is the pretrained global model: it misreads the Uzbek "40" region code (e.g. "C0B472RA"). Lane A's tuned OCR replaces it.
- Demo samples are public: Roboflow Uzbek_Plates (CC BY 4.0), Wikimedia Commons (CC BY 3.0 / 2.0), UA-DETRAC (CC BY-NC-SA 3.0), scikit-learn flower (CC BY 2.0). See `app_sources.md`.

## How to reproduce
```
# web
cd web && npm ci && npm test && npm run build                 # bundle sizes above
VITE_API_BASE=http://localhost:8008 npm run dev               # with the tunnel below
# local API on the box
scripts/box.sh --gpu 1 --bg serve_local -- python -m serve.local --port 8008
ssh -N -L 8008:127.0.0.1:8008 rtx
curl -F file=@web/public/samples/uz_plate_van.jpg http://localhost:8008/analyze/image
# recorded backup runs
python3 -m serve.record_run --api http://localhost:8008 --image web/public/samples/uz_plate_van.jpg --id uz_plate_van \
  --title "Car photo with an Uzbek plate" --work-order oil_filter_change,wash,cabin_filter --simulate
# Modal (on the box)
scripts/box.sh -- modal run serve/modal_app.py                # smoke
scripts/box.sh -- modal deploy serve/modal_app.py             # SEEN_MODAL_GPU / SEEN_MODAL_MAX_GPU / SEEN_MODAL_SCALEDOWN / SEEN_DISABLED
scripts/box.sh -- python -m serve.upload_artifacts --dry-run
```
