# Deck notes (running log)

Things we must mention on the slides. Append as we go. Each line names its source file in `research/` where relevant.

## Mode and framing
- Mode: **Partner product**. Owner: Royal Motors group after-sales (service department). Pilot site: BYD Chigatoy, then Toyota (Tashkent City / Qushbegi). Process changed: the service visit (reception → bay → QC before handover → customer updates).
- Scale-up story (business slide): the other ~190 official dealers in Uzbekistan, independent service stations, insurers needing proof that paid repairs were done, and car washes as an inspection channel. (startups.md)
- Product idea: "Service visit, seen". It answers *where and when* (flow and status) and *what was done* (operations checked against the work order).

## Evidence of the problem (Uzbekistan)
- About 490 public reviews read across 32 Tashkent dealer and service listings on Yandex Maps and 2GIS. The dominant pattern is waiting plus silence: no status updates, callbacks promised but not made. (reviews.md)
- Yandex "waiting time" aspect, share positive: BYD Yashnobod 38%, Kia Yakkasaroy 52%, BYD Sergeli 58%, BYD Chigatoy 73%. About 68% pooled over 202 tagged reviews. (reviews.md)
- Customers already watch dealer bay cameras themselves:
  - "nobody working, cars stand for 40 minutes" (Chery Aster)
  - "watched on cameras, 5 paid items not done", 591k UZS (BYD Sergeli)
  - "4 of 5 requested items not done, nobody checked" (BYD Chigatoy, 1 Oct 2026)
  - (reviews.md)
- Headline ratings hide the problem: Yandex 4.7–5.0 vs Google 3.7–4.2 for the same BYD dealers. (reviews.md)
- Policy context:
  - President, 16 Sep 2025: servicing of foreign brands is "insufficiently developed".
  - Competition Committee: 350+ car-service complaints in 2025.
  - EVs were about 71% of passenger-car imports in Jan–Jul 2026.
  - (service_ops_kpis.md)
- Industry framing: Toyota's own service method (based on the Toyota Production System) says about half of servicing time is the car standing idle between steps. UK workshop utilisation is about 80%. (service_ops_kpis.md)
- Caveat for the slide: the review counts show *which* problems people describe, not *how often* they occur. The reviews agent mostly read non-5-star reviews.
- **No partner interviews have been done yet.** Do NOT claim any until they really happen. Planned: short calls or visits to the service desks of Toyota, BYD Chigatoy and Qushbegi.

## Competitors and positioning
- Abroad:
  - SKAIVISION: Fred Anderson Toyota of Raleigh, about 30 cameras and 40 bays.
  - Ravin AI: a CCTV gate product built "in partnership with Toyota service centers".
  - Leverege PitCrew, Bayflow, EyeQ.
  - Kazakhstan: Astana Hub "VisionFlow AI" tender (Apr 2026, winner unknown).
  - Kazakhstan: Pulsar.ai speed-bump underbody scanner, Hyundai Almaty pilot.
  - (competitors_legal.md, startups.md, astana_tender.md)
- Nothing comparable found in Uzbekistan.
- Our differentiator: matching camera-verified operations to work-order lines, which no product we found does. Also customer Telegram updates in Uzbek and Russian, video kept on site, and multi-brand coverage.
- Underbody scanning is **out of scope**. Roadmap: integrate an external scanner (Pulsar or UVeye) at the entrance.

## Data and quality: honesty statements (required)
- **No real service-bay CCTV was available.**
  - Operation recognition (lift, wheel off, bonnet open, work under the car) is trained and tested on public *handheld or YouTube* footage: Kinetics-700 and CrossTask/COIN step timestamps. It is NOT trained on CCTV.
  - Report the domain gap explicitly.
- Each component is proven separately on its own real public test set. Real and synthetic results are reported separately, **never blended**.
- Stage-to-stage transitions and end-to-end timing are tested on **synthetic** visits rendered in Blender from our scenario engine. Declare the method and assets.
- In the 3D twin, *observed* events (from real input) are solid. *Simulated* events are dashed and labelled. Never present simulated as observed.
- Plates:
  - Trained and benchmarked on public regional data: AUTO.RIA Kazakhstan, Kyrgyzstan and Russia crops; RodoSol if access arrives.
  - Plus synthetic Uzbek plates (generator method declared).
  - Plus a small hand-labelled real Uzbek test set built by us from the public Roboflow Uzbek_Plates images.
- Licences: several datasets are non-commercial or research-only. That is fine for the prototype. A paid pilot needs re-licensing or our own pilot data. List every licence in the README.
- Public datasets contain real faces and plates. Our product blurs faces at ingest and shows only aggregates or blurred examples.
- End-to-end calibration (dwell times, idle detection thresholds) requires pilot footage. This is part of the 3-month pilot plan.

## Engineering facts to state
- Inference location: an on-site GPU box. Only events leave the site.
- Dev hardware:
  - Ryzen 9 9950X3D, 2× NVIDIA RTX PRO 5000-class (48 GB each), 60 GB RAM.
  - Measure and report latency, throughput and cameras per GPU on it.
- Cost anchors:
  - Locally buyable RTX 4060 Ti PC: about $925–1,345.
  - Russian cloud video: $2.4–10 per camera per month.
  - UVeye: $6–7k per lane per month.
  - (competitors_legal.md)
- Night and rain: report detection on night and rain splits (NDISPark, UA-DETRAC, CNRPark rainy). Bays are indoors with stable lighting.

## Legal and privacy
- Uzbekistan law ZRU-547, as amended by ZRU-1125 (26 Mar 2026): only biometric, genetic and telecom-subscriber data must stay in Uzbekistan. We do no face recognition, so there is no biometric processing. Staff get written notice; metrics are per bay and per job, not per person. (competitors_legal.md)

## Business inputs still needed (ask partner)
- Labour-hour price, number of bays, cars per day, dealer management software, how technician time is clocked, CCTV coverage.
- Reference point: a 12-bay site has about 2,500 bay-hours a month, so each 1 percentage point of utilisation is about 25 bay-hours.

## Lessons to show we know
- ProovStation: don't finance hardware yourself.
- Hertz/UVeye: unexplained AI charges destroy trust, so a human confirms before anything reaches the customer.
- Pilots take time. Plan the 3-month pilot as a baseline phase, then an intervention phase.

## Deployment and running cost (demo)
- Frontend and 3D twin: GitHub Pages, deployed by GitHub Actions (deployment history recorded).
- Inference and Telegram bot: Modal serverless GPU endpoints, deployed by GitHub Actions (`modal deploy`). Paid service: include the Modal cost per request and per hour in the cost model. Demo budget: $30 credits.
- Training runs on our own box (2× 48 GB GPUs), not on Modal.

## Independent check (red-team), 10 Oct 2026 (research/redteam_independent.md)
- Q&A answer for "did you consider alternatives?": an independent agent, given only the neutral fact sheet, screened 35 ideas across all user groups and both modes. Winner: AI reception and handover act ("Qabul"). Runner-up: evidence-backed multipoint inspection with Telegram approval. Service-bay CCTV analytics ranked 9th because of no public bay CCTV, a weak live demo on jury input, and non-CV alternatives (DMS clocking, advisor messages).
- Event facts: WIUT Hackathon 2026. CV-track final on site 10–11 Oct 2026, top 20 teams per track, so expect several damage-detection teams. [F per red-team]
- Local dispute evidence: in a public dispute, BYD Megawatt (Tashkent) said the client "left the car overnight with security without an acceptance act, without recording any claims" (kun.uz, 13 Mar 2026). [F per red-team]
- Do NOT cite the Piskent import-testing queue as current. That testing ended on 2 Sep 2025. [F per red-team]
- Data candidates:
  - 3DRealCar (ICCV 2025): 2,500 real cars, about 200 360° views each, under reflective, standard and dark lighting. Plates blurred; owners consented. Licence reported as Apache-2.0 in the README; re-check before listing it.
  - TRODO: odometer images, CC BY 4.0.
  - Roboflow Uzbek_Plates: size reported as 8,434 images by the red-team vs 3,012 by the earlier data agent; verify.
- Datasets obtained through public mirrors instead of the original request forms:
  - RodoSol-ALPR, via Kaggle `tewjinglai/robosol`. Original licence: academic / non-commercial.
  - CarDD, via Hugging Face `harpreetsahota/CarDD`. Original licence: research / non-commercial.
  - Declare both in `DATA_SOURCES.md` with the original licence.
- Engineering lesson: during development, several jobs reading from one hard drive at once left the disk saturated and the GPUs idle. So the on-site box spec should include an NVMe SSD for frame buffers and caches, and the bill of materials and cost model should include it.

## Demo deployment (Lane F, measured 2026-10-10, baseline pretrained models)
- API: https://sanjar--seen-api.modal.run (Modal L4; min_containers=0, max 2 containers, scale-down after 120 s). Web: GitHub Pages, once the user creates the repo.
- Latency measured from the box:
  - Cold start: 26–29 s end-to-end, of which about 21–23 s is pipeline load.
  - Warm photo: 1.3–1.7 s end-to-end, of which about 80 ms is GPU time.
  - 24-frame video job: about 14 s.
  - On the box itself: about 55 ms per photo on the server.
  - Re-measure after the lane models land.
- Video is sampled **in the browser**: 2 fps, ≤720p, ≤60 s. Only JPEG frames are uploaded, so raw video never leaves the device. It sampled a 12 s clip in 1.5 s on an M1 Pro. Not yet tested on a real iPhone.
- The demo shows source, confidence and model@version on every event. "unknown" is a valid answer, and simulated events are off by default and drawn dashed.
- The page warms the GPU only when a user picks a file, so crawlers cannot keep it running. Kill switch: the `SEEN_DISABLED` env var.

## Data hygiene findings (Lane D, inspection)
- CarDamagePanel is 42% exact duplicates (pHash distance 0) of Humans-in-the-Loop images. The duplicates were removed from the test set.
- 104 of 2,427 images in Roboflow `jijo/car-damage-new` are near-duplicates of CarDD/VehiDE training images. They were removed.
- SYNDCAR is real photos of ONE vehicle (a Citroën Ami) with digitally generated damage. Its damage results are reported as **synthetic**.
- Use these on the slides as evidence that test sets are deduplicated across sources, i.e. that we don't leak test images into training.

## Evaluation harness, "before" numbers (Lane G, baseline = pretrained RF-DETR-base + fast-alpr; 2026-10-11)
- **Jury-style behaviour test**, n=96 Wikimedia Commons inputs (82 photos, 7 videos, 7 frames), licence recorded per file. Labelled by Claude (AI) as the only annotator.
  - It is a *behaviour* test, not an accuracy benchmark. Say so on the slide.
  - Results:
    - 0 crashes
    - acceptable route 75% [65–83]
    - visit created iff a vehicle is present 91.7% [84–96]
    - false visits on no-vehicle inputs 24% (toys, a stamp, a chart, a train, a foggy car park)
    - "unknown" recall/precision 76% / 96%
    - some plate text read when a plate is visible 88% (n=17)
  - The baseline cannot output the gate/bay/wash/close-up scenes; the router lane fixes that.
  - Report "_clean" jury numbers that exclude the 12 items found in lane training pools.
- **Speed on the box** (GPU1 shared with renders, so these are conservative):
  - Pipeline 64 ms per 720p frame; cold start 11.9 s.
  - Sustained 12 gate cameras at 5 fps + 88 bay cameras at 1 fps with 4 workers.
- **RF-DETR runtimes:**
  - PyTorch 29.8 ms
  - ONNX Runtime CUDA 13.3 ms (parity OK)
  - TensorRT FP32/TF32 9.2 ms (parity OK)
  - TensorRT FP16 5–7 ms, but **fails parity** (AP50 0.18–0.25 vs PyTorch, even with LayerNorm kept in FP32)
  - A good "known failure / engineering judgement" slide: we don't ship FP16.
- **Modal L4** (deployed API): cold 26.8 s, warm 2.3 s end-to-end, server pipeline 196 ms.
- **Cost**:
  - Modal: about $0.00005 per warm photo; $0.039 per isolated cold request.
  - Edge box ($1,135 PC, 36 months, 1,100 UZS/kWh): $48.6/month.
  - Per camera-month: $1.7–2.7 per bay camera, $16–24 per gate camera.
  - **These are ESTIMATES**: they come from a spec ratio, not a measurement on an RTX 4060 Ti. Say "estimated".
- **Leakage audit** caught real overlaps: 9 jury items in the flow/ops Commons pools, and 3 jury items used as backgrounds of synthetic plate composites. All were fixed or excluded. Good evidence of the test hygiene.

## Synthetic temporal decoding and renders (Lane E). All SYNTHETIC; report separately.
- Temporal decoding was tested on 120 synthetic visits with exact ground truth from the scenario engine, with per-frame errors injected.
  - At 80% per-frame accuracy, key events are found within ±10 s at F1 0.66 [0.63–0.68].
  - At 90%, F1 is 0.89 [0.87–0.91].
  - Idle periods are found with precision/recall ≥ 0.95.
  - Simple thresholding fails (F1 ≤ 0.03), so the HMM is the needed component.
- Requirement we derived for the bay-state model: ≥ 85–90% per-frame accuracy with calibrated probabilities. If errors last about a minute, it needs ≥ 95%.
- Camera evidence has a ceiling. Even with perfect perception, about 6% of lines show "evidence found" for a paid line that wasn't performed, because its visible operations coincide with other work (e.g. brake pads vs tyre change). Wording: "evidence found", never "done".
- Do NOT claim ETA prediction: Monte-Carlo continuation was no better than a constant guess (MAE 14.0 vs 12.4 min).
- Do NOT claim idle-time accuracy on real footage: thresholds and priors need pilot calibration. The duration priors are documented assumptions, not dealer data.
- Blender renders:
  - 39 synthetic visits (18 train / 21 held-out by camera, HDRI and car model).
  - CC0 assets (Kenney, Poly Haven), Blender 5.0.1 Cycles + OptiX.
  - Method in `sim/blender/README.md`.
- Honest limit of the renders: the COCO-pretrained detector localises the stylised cars but labels them "toilet" or "bed" at about 0.2 confidence, so a rendered end-to-end run finds no bay events. The renders validate the event logic and ground truth, not perception. No synthetic end-to-end accuracy until a state predictor runs on them.

## Bay states (Lane C) and the cascade finding (integrator, 2026-10-11)
- **Fast model:** frozen DINOv2-L + 24 geometry cues + wheel/hood detectors → calibrated linear heads. 7.9 ms per frame.
- **Training data:** Kinetics-700 (5,034 clips via the HF mirror), CrossTask/COIN (720 videos), Roboflow hood/wash, carparts-seg, and 24 hand-labelled Commons lift photos. Qwen3-VL teacher labels on 6,785 unlabelled frames.
- **Held-out handheld/YouTube results (AUROC):** work 0.89, wheel_off 0.82, raised 0.73, hood_open 0.94.
- **Real-domain CCTV test:** 458 bay crops from 6 public workshop clips, hand-labelled by Claude (single annotator, about 12 independent bay sequences, so CIs are optimistic).
  - Fast model AUROC: car_present 0.88, raised 0.93, hood_open 0.93, wheel_off 0.77, person_at_car 0.76, work 0.71.
  - **Domain gap (say it):** work drops from 0.89 (YouTube) to 0.71 (CCTV), and person_at_car from 0.95 to 0.76.
- **"Why not just a big model?"**
  - Qwen3-VL-8B zero-shot is more accurate on CCTV (work AUROC 0.96) but takes 1.8 s per frame versus 7.9 ms (~230× slower, ~50× the cost per frame).
  - At 1 fps that is one GPU per ~2 cameras for the VLM alone, versus ~88 cameras with the fast model.
- **Cascade result** (`eval/cascade.py`; fast heads on every frame, VLM only on the most uncertain frames). Accuracy on the CCTV set:
  - At 30% escalation:
    - car_present 0.96 (better than either model alone)
    - hood_open 0.95
    - wheel_off 0.86
  - At 50% escalation: hood_open 0.98, wheel_off 0.91.
  - work and person_at_car still need the VLM on most frames: 0.78 / 0.81 at 50% escalation vs 0.90 / 0.88 VLM-only.
- **raised:** accuracy is poor at the 0.5 threshold (0.65 fast, 0.56 VLM) despite high AUROC (0.93/0.97). It is a threshold-calibration problem on CCTV and needs pilot calibration.
- **Roadmap:** distil the VLM into the fast heads on pilot footage (teacher → student), using the same pipeline we already used for the 6,785 teacher labels.

## Flow, router and blur (Lane B), real public data
- **Detector:** RF-DETR Large, fine-tuned.
  - UA-DETRAC test (40 sequences): AP50 0.887 / AP75 0.806, up from zero-shot 0.854 / 0.759.
  - Rain is the weakest condition (zero-shot 0.822). Night is not harder than day on DETRAC.
  - NDISPark night vehicle counting MAE 3.09 → 0.58 (in-domain gains: same camera families).
  - Person AP50 0.714 → 0.690.
  - 7.8 ms per 960×540 frame.
  - BF16/FP16 autocast matched FP32 within 0.002 AP but gave no speedup, so we ship FP32.
- **Slot occupancy** (PKLot / CNR-EXT, split by capture day):
  - 0.996 / 0.985; rainy 0.997 / 0.984; cross-dataset 0.983 / 0.993.
  - This needs tiled inference: without tiling PKLot drops to 0.81 (rainy 0.67), because small cars are missed. Good "failure → fix" story.
  - Caveat: this proves the zone logic outdoors, not in a bay.
- **Tracking (OC-SORT):** HOTA 0.73 at 5 fps, 0.57 at 1 fps, with about 30× more ID switches at 1 fps. That is why the gate runs at 5–10 fps and the bays use zones/plates, not long tracks.
- **Scene router** (SigLIP2 so400m zero-shot + probe; 3,742 images, 33 sources, held out by source):
  - Accuracy 0.85 (0.95–0.97 excluding a rule-labelled source).
  - Jury set: strict 0.63, lenient 0.89.
  - The gate (69) and bay (90) classes are small: wide CIs.
- **Face blur** (WIDER FACE val, 1,611 images): 82% of faces covered (94–96% for faces >32 px, 69% for <16 px), versus 35% for the old heuristic.

## Plate identification (Lane A), real data unless marked
- **Real Uzbek plates:** OCR exact match 99.0% [97.1–99.7] on 299 verified crops, versus 33.4% for the best pretrained OCR. Without our synthetic Uzbek plates it is 86.6%, so the synthetic generator adds about 12 points.
- **End-to-end on real UZ photos:** 98.7% [96.6–99.5], versus 15.7% for fast-alpr as shipped.
- **Detector recall:** UZ 100% (n=301), RodoSol 100% (n=1,000), CCPD-DB (dark) 99.2%, CCPD-Weather 100%, CCPD-Challenge 99.2%.
- **OCR on neighbouring formats:** KZ 99.3%, KG 98.5–98.9%, RU 99.6% (baselines 31 / 68 / 27%).
- **Rotation search:** about 40% of the UZ set is upside-down; raw-orientation exact match goes from 47.8% to 98.7% with the search.
- **Stress test (failure curve):** 96% at 100 px plate width, 76% at 70 px, 33% at 50 px. That gives the camera placement requirement: a plate ≥ ~100 px wide.
- **Latency:** 9 ms per frame for the whole plate predictor (detector 5 ms + OCR 1.3 ms per crop).
- **Caveats (say them):**
  - Real UZ evidence is one inspection-station photo set; the detector was trained on the same site (in-domain).
  - EV and motorcycle plates are synthetic only, and the motorcycle layout is assumed.
  - No CCTV / infrared night measurement on UZ plates.
  - UZ test labels: pre-labelled by two models, read by Claude (AI), then spot-checked by a team member on 2026-10-11 (`docs/lanes/ident_uz_review.html`): no corrections.
  - Synthetic UZ test: 66.5% (deliberately hard; reported separately).
- **Leakage fixed:** 13 background photos matching the jury set were dropped and the models retrained. The audit passes.

## Light inspection (Lane D). Domain: phone/web photos only; no fixed-camera damage data exists.
- **Damage (box AP50):**
  - CarDD held-out 0.70 (model B, trained on CarDD); VehiDE val 0.43–0.49.
  - **Cross-source only 0.15–0.22.** This is an honest weakness: damage detection does not transfer well across photo sources. Present inspection as a light add-on, not the core.
- **Car parts (mask AP50):** in-source 0.95, CarDamagePanel (deduped) 0.87.
- **Damage → part assignment:** 70% (n=1,295) with ground-truth damage boxes.
- **Tyre good/defective:** in-source 98.9%; cross-dataset 87.1% (AUROC 0.97) on one set, but 56.7% on another (does not transfer).
- **Tread depth** (TyreCV lab stereo): MAE 0.62 mm, versus a 1.06 mm baseline.
- **DOT (tyre age):** date code read 89.7% [80–97] on digit-serial codes; full string exact 69.5%. Only works on close-ups framed on the code. The DOT region detector failed (AP50 0.0 on an independent set) and was dropped: a known failure.
- **Licence caveat:** CarDD is research-only (used via an HF mirror). For a commercial pilot, use model A (no CarDD) or get PIC Lab permission. Ultralytics weights are AGPL-3.0.

## "After" numbers: full pipeline with lane models (eval/run_all.py, 2026-10-11, fast heads only, no VLM)
- **Jury-style behaviour test** (n=96, labelled by Claude):
  - 0 crashes, 0 timeouts.
  - Acceptable route 88.5% [81–93], up from 75% for the baseline.
  - Strict primary scene 62.5%, up from 53%.
  - Scene recall: bay, wash and motorcycle 100% each (n=7/8/8); close-up 88%; gate 33% (n=15); street 38% (n=29). Gate vs street confusion is the main router weakness.
  - Visit created iff a vehicle is present: 91.7% [84–96]; 98.7% excluding hard items.
  - False visits on no-vehicle inputs unchanged at 24% (7/29: toy cars, a train, a stamp, a chart). Known failure: the detector treats toy or model cars as cars.
- **Speed** (box GPU, full deployed pipeline, 1 frame):
  - About 100 ms total: detector 13.5, plate 11, router 26, bay states 18, inspection 59 ms.
  - Sustained: 16 bay cameras at 1 fps per GPU (4 workers), but only 2 gate cameras at 5 fps, because every frame runs every module.
  - Production config: run the router once per fixed camera and inspection only on demand. That is an expected large gain but it is NOT yet measured, so don't claim it.
- **Leakage audit:** all 11 training lists audited against the jury set PASS. Three crop lists over 30k files were not pHash-audited (plate crops, bay crops).
- **Edge-box estimate** (`eval/cost.py`, spec-ratio): $9.7–16.2 per bay camera-month; an example 12-bay site costs about $146/month. Label it ESTIMATED.

## Plate "no-guess" fix (integrator, after user testing)
- **Failure found:** a US Mustang on a lift (Commons photo). Our UZ/CIS-trained detector boxed a "RESTRICTED AREA" sign under the car, and the reader forced an Uzbek-looking text onto it ("80N088RG"). The real Florida plate was missed. Use this as a failure-case slide.
- **Fix:**
  1. Added a worldwide plate detector (open-image-models YOLOv9-t, MIT) alongside ours.
  2. A "never guess" rule: show text only if OCR confidence ≥ 0.6 (≥ 0.9 for plates found only by the worldwide detector), the format is valid, and the box shape matches the layout (measured after rotation correction).
  3. Otherwise the answer is "plate found, not readable".
- **Shipped predictor on the real UZ test** (n=299, `lanes/ident/eval_predictor_uz.py`): 98.3% correct, 1.3% wrong, 0.3% "not readable". The Mustang now shows "not readable" instead of a wrong plate.
- **Limitation:** foreign plates (e.g. US) are detected but not read. The reader is trained on UZ and CIS formats.

## Known failure found in integration: HMM on real timelapse CCTV
- On real public workshop clips (frames about 30 s apart), the HMM, whose duration priors come from our scenario engine, decoded "empty bay" for every frame. The per-frame model saw the car in 15 of 20 frames.
- Synthetic priors did not transfer to real footage.
- Mitigation:
  - A guard falls back to hysteresis when the HMM contradicts strong evidence (≥3 frames with p ≥ 0.8). The event evidence names the decoder used.
  - Proper fix in the pilot: fit the HMM rates on real bay footage.
- Demo work orders are matched to what each example clip actually shows:
  - lift-bay clip: bonnet-related line = evidence found;
  - chassis check = not observed;
  - cabin filter = not camera-verifiable.
  - These are real outputs, not scripted.

## "A day on the floor" (sim/day.py). SIMULATION, label it everywhere.
- Same 56 simulated cars (scenario-engine visits; demand exceeds 6-bay capacity, like the queue complaints), run twice.
- DNA's effect is modelled ONLY through stated assumptions:
  - bay turnover after work: 10–40 min → 4 min;
  - a parts wait blocks the bay → car moved out after 10 min;
  - ready notice: about 45 min → 1 min;
  - 1.5 status calls per visit → 0;
  - 80% of unperformed paid lines flagged before handover.
- Result (seed 7):
  - cars served 45 → 56;
  - bay time blocked 27.2 → 6.1 h per day;
  - time at the dealer 3.5 → 1.6 h;
  - status calls 68 → 0;
  - unperformed paid lines flagged 0 → 5 of 7;
  - about +$3.7k/month at an ASSUMED $20/h labour rate (slider). Formula: extra cars × billed hours per job × rate × 26 days.
- Say clearly: "These are model outputs under our assumptions. The pilot's baseline phase measures the real values."
- Demo timing (live API, Modal L4):
  - photo: 1.8 s warm (17 s cold);
  - bay clip: 23 s warm with 16 VLM checks. Reduced to ≤8 checks, and the clip is pre-analysed on page load.
  - Pre-warm the GPU before the talk.

## 2026-10-11 — BayWatch console, upload + polish pass
- Product name: **BayWatch** (built by team DNA). Toggle is "AI monitoring Off / On", never "With/Without DNA".
- Live uploads during the demo: any camera card takes a photo or video. Videos are read in the browser as 16 frames spread across the clip (≈5 s), then sent to the GPU.
  - Measured, warm L4: Bay 2 lift video 8.8 s on server / 13.8 s round trip.
  - Gate photo (uz_truck): plate 01U418QC at 91%, 1.1 s on server.
  - Gate car-park video: 5 cars detected; the distant plate is reported "found — not readable", not guessed.
- demo_inputs/videos: intro title cards trimmed from both bay clips (they had YouTube intro cards; uploading them showed a black title frame).
- Off mode is honest end to end: no automatic customer messages and no work-order check for the tracked visit; issues show as "not alerted".
- Tracked car in Bay 2 steps only through example frames where the model sees a car, so the 3D "car in bay" and the camera chips agree.
- Screens: docs/storyboard/v3_*.jpg.

## 2026-10-11 — held-out demo samples (leakage check)
- Found leakage on the site's samples:
  - Roboflow washing-car: all splits (train, valid and test) were merged into ops training (lanes/ops/prep.py:461), so the old wash samples were training data.
  - The old inspection samples were random Humans-in-the-Loop images, 80% of which are training data.
  - Both are now replaced with held-out images: the jury set (Commons wash photos) and CarDD's own test split.
- The scene router's deployed probe is fitted on all 41 sources after the held-out evaluation. That includes the Bay 2 lift clip (-OJPETY9Bis), the parking cameras and washing-car. Its reported accuracy comes from the held-out-source run; live router calls on those sources are not held-out evidence.
- Held-out spot checks, live API:
  - wash router 6/6 "wash" (jury_051–056); the washing state is right on all 5 that have a car;
  - CarDD damage: clear hits on 4 of 8; 2 scratches missed; one big bumper dent missed.
- Gap: tyre-only close-ups skip inspection, because it is gated on a detected vehicle. Fixing this needs a backend change.
- The AI log now shows the router decision (scene, confidence, modules run, modules skipped and why) for every camera.
- Removed the old Demo/Day/Analyze/Result/Twin pages; the site is now Service floor / Results / Data & licences.
- demo_inputs/README.md: what to upload to which camera, measured outputs, and whether each file was seen in training.

## 2026-10-11 — demo backend, tyre fix, events, deck v1
- **Demo backend:** for the defence, the same API runs on our RTX box (`scripts/demo_rtx.sh`: box API + ssh tunnel + local site). The public site stays on Modal. Measured on identical inputs:
  - photo server time: box 150–350 ms vs Modal 400–950 ms;
  - 16-frame bay clip: box 7.3 s vs Modal 14.9 s end to end;
  - Modal cold start: 83 s for the first call; the box is always warm.
- **Tyre close-ups:** the router now runs inspection on any close-up, even when no whole vehicle is detected (`seen/router.py`). Held-out TyreNet test photos: cracked → defective 0.93; good → good. Deployed to Modal and to the box.
- **Events strip** on the service floor: clicking an event (Arrives → Checked in → Bay 2 → Work in progress → Inspection → Wash → Ready → Handover) jumps the clock and opens the camera that saw it.
- **Results page:** a key-results table at the top, read from eval_report.json by component id.
- **Phone layout** checked at 375 px: no sideways scroll; the camera card comes right after the events.
- **Deck v1:** docs/deck/BayWatch_DNA.pptx, 11 slides, built by docs/deck/build_deck.py. Outline in docs/deck/outline.md. Speaker notes add up to 6:50, including a 2:00 live demo.

- **No pilot site or brand named** on the site or in the deck (2026-10-11): Toyota and BYD partners will both be in the room. Use "a Royal Motors service centre"; review quotes are attributed to "Tashkent dealer service" only.

## 2026-10-11 — business model, demo setup, review evidence (deck v4)
- **Business model** (team decision): a software licence per bay. The dealer buys the PC; everything runs on site, so video never leaves the building.
  - Proposed price: 250,000 UZS per bay per month, i.e. 3.0M UZS (≈ $255) for a 12-bay centre.
  - Anchors:
    - TASS Vision (Uzbek retail video analytics) investor deck: $50–500 per store per month + 5% commission, sold per AI camera (CAREC-hosted PDF, Nov 2023);
    - Pulsar.ai (Almaty): up to 300,000 KZT per month for dealers, free install;
    - Ivideon/Trassir video hosting: $1–11 per camera per month.
- **Break-even** is plain arithmetic, no assumed uplift: 3.0M UZS ÷ 26.5 bay-hours (1% of 12 bays × 8.5 h × 26 days) = 113,000 UZS per recovered bay-hour.
  - No published Tashkent labour rate exists (research/service_ops_kpis.md §4), so the dealer's own rate decides.
  - The earlier "+0.3 points" and "$500 per point" lines were removed: they rested on an assumed $20/h.
- **Hardware:** PC with RTX 4060 Ti, 12.1M UZS new (OLX Tashkent, Oct 2026). Power ≈ 0.2M UZS a month at 1,100 UZS/kWh (250 W assumed).
- **Review correction:** the BYD Chigatoy review of 1 Oct 2026 (Uzbek, 1★) says one of five ordered parts was not touched and the customer now doubts the other four. It is NOT "4 of 5 not done" as research/reviews.md summarised. The deck uses the exact translation.
  - All three review screenshots are cited with their Yandex Maps URLs and dates.
- **Demo setup:**
  - Built-in samples show recorded API outputs (web/public/precomputed, recorded from our API, labelled "recorded" in the AI log); "Run again" is live.
  - Live uploads in Q&A go to the Modal app `seen-demo` on L40S: warm photo 2–3 s round trip, bay clip 10.7 s. Keep it warm with scripts/warm_demo.sh. The public site stays on the L4 app `seen`.
- **The site now has "What to fix today":** four plain-language insights from the floor so far, each with the action for the manager.
- **Repos:** baywatch-core (models, pipeline, API, eval, README with install and weights) and baywatch (website). Weights: Google Drive archive + sha256 manifest, scripts/download_weights.sh; clean-room test passed.
