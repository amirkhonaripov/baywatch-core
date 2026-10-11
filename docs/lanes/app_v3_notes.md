# App v3 notes (DNA demo rebuild)

Product renamed Seen -> **DNA** ("DNA — every service visit, seen."): title, og/meta tags, header, footer, i18n strings.

## Navigation
Demo (default, `#/demo`) · A day on the floor (`#/day`, new) · Results (Metrics) · Data & licences (About) · small link **Under the hood** (`#/analyze`, `#/result`, `#/twin`; was "Advanced").

## Demo page (`web/src/pages/Demo.tsx`)
- Landing: one value line, **Follow a car** / **A day on the floor**, source legend (Your input · Real public footage (different car) · Synthetic render · Simulated), "AI warming up" badge.
- On load: `GET /health?warm=1`, then `warmStages()` (`web/src/demo/stages.ts`) analyses in the background and caches in memory: bay clip (`-OJPETY9Bis`, 12 frames, `/analyze/frames`, `simulate=true`, work_order coolant_check / chassis_diagnostics / cabin_filter), inspect, tyre, wash, ready (`/analyze/image`). Other bay clips (`e4O3X9gli1s`, synthetic) are analysed on first selection; own bay footage can be uploaded at the Bay stage (then the 3D car is badged Observed).
- Clip frames: title cards are skipped (`SKIP_LEAD`/`SKIP_TAIL` in stages.ts) so only footage frames are sent.
- The only click-time analysis is the gate (upload / take photo / sample photo or video). The clock starts at once (camera flies to the gate CCTV) and holds on the scan animation until the API answers.
- Stages Gate · Bay · Inspection · Wash · Ready · Handover, 66 s auto-advance with play/pause/scrub (`timeline.ts`). Every camera frame carries a source badge, a scan sweep, then live boxes; plate crop zooms and types out; data chips fly into the **Visit record** (plate, vehicle, booking (demo data), bay states with clip timestamps, findings, tyres, wash state, cars in lot, work-order evidence). If a stage's analysis is not ready the clock holds and a spinner shows; nothing scripted ever replaces a model result.
- Handover: work-order card with the three statuses from the bay analysis (evidence found / not observed / not camera-verifiable), evidence thumbnails, and the manager "Before handover" alert.
- 3D (`Scene3D.tsx`): cinematic camera flies between CCTV mounts (cross-fades into the Camera view); six bays, grey simulated cars/staff. 3D is always badged Simulated unless the user uploaded own bay footage.
- Verified: uz_plate_van sample -> plate 40B472RA, booking WO-DEMO-0001, work order Coolant = evidence found, Chassis = not observed, Cabin filter = not camera-verifiable (live API).

## Honest caveats
- Tyre photos (`stages/tyre`) are tread close-ups; the API finds no vehicle and returns nothing for them. The tyre condition shown at Inspection therefore comes from the tyre classifier's output on the inspection photo (labelled "from the inspection photo"). If a tyre photo ever returns a label it is used instead.
- "Cars in view" at Ready counts detector hits (>=0.5) on one PKLot frame; there is no slot map, so no occupancy ratio.
- Vehicle colour is a pixel estimate (labelled "estimated"). Booking/VIN come from the small demo list.
- Bay clip states vary by clip; the record shows only states the model reported (>=50%) up to the current frame.

## A day on the floor (`web/src/pages/Day.tsx`, `demo/day.css`)
- Data: `public/day/day.json` (SIMULATION). Two synchronized **2D top-down canvases** (chosen over 3D: crisper and cheap for 56 cars x 2 sides): Without DNA vs With DNA, 6 bays, arrivals queue, parts-wait (orange glow), blocked bay (red), wash/parts yard, ready lot (amber = customer not told, green = told), pickup. 80 s for the whole day, play/pause/scrub, 1x/2x/4x, day clock.
- Live counters (computed from the schedule): cars served, bay time blocked (h), avg time at dealer, status calls (derived from calls_per_visit_without: alternating 2 and 1 per car), missed paid work flagged (flagged / not performed). End values match `metrics` in day.json exactly (45/27.2/3.46/68/0/5 and 56/6.1/1.63/0/5/7).
- Results band: +cars/day, bay-hours freed, +$/month with labour-rate and working-days sliders; formula shown (`extra_cars x avg_bay_job_hours x rate x days`; the exact implied job length reproduces $3,674 at defaults). Collapsible "Assumptions" lists every key of `assumptions` verbatim. Big SIMULATION badge and a sentence that field events come from the camera pipeline of Follow a car.

## Files
New: `src/demo/stages.ts`, `src/demo/day.css`, `src/pages/Day.tsx`. Rewritten: `Demo.tsx`, `demo/parts.tsx`, `demo/Scene3D.tsx`, `demo/SceneView.tsx`, `demo/timeline.ts`, `demo/demo.css`. Edited: `App.tsx`, `router.ts`, `index.html`, `i18n.ts`, `demo/data.ts` (two new source kinds).
`npm run build` passes.
