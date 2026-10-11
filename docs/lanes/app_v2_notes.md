# App v2: guided demo front page (notes)

Default route `#/demo` (`web/src/pages/Demo.tsx`, helpers in `web/src/demo/`). Nav: Demo, Results (Metrics), Data & licences (About), Advanced (Analyze / Result / Twin, unchanged).

## Flow
1. Gate: photo upload or sample -> `POST /analyze/image` (simulate=true). Plate card (reads "Plate found — not readable" when text missing, conf < 0.6, or `plate_format_valid=false` and format not foreign). "No vehicle found" shows the router/skip reason.
2. Booking lookup is simulated from 3 fixed demo bookings (`demo/data.ts`), up to 2 characters of OCR tolerance, labelled "demo booking data"; VIN labelled "VIN from dealer record (demo)". No match -> explicit "Continue with a demo booking".
3. 3D car (Kenney, CC0; lazy chunk `Scene3D`, 2D fallback when WebGL fails) drives Gate -> Reception -> Bay 2.
4. Bay: real CCTV (default) / synthetic render / own photo or clip -> `POST /analyze/frames` + `/jobs/{id}` with the demo booking's lines. Lift, wheel (debris-tire), bonnet tag and mechanic follow the per-frame `states` (threshold 0.5). Chips show current-frame probability; a tick = >= 50% in at least one frame; tapping jumps to the best frame.
5. Wash / Ready / Handover: simulated, ghosted, "Simulated" badge; uses `simulated_continuation` when it contains wash + ready + handed_over events, else a scripted animation (the drawer says which).
6. Work order card: statuses straight from `visit.verifications`; tap for note and evidence frame (nearest frame to `evidence.video_time_s`).
7. "How this was computed" drawer: router decision, timings, model versions, source of every stage.

## Limits and decisions
- Synthetic episodes span 1104 s; the API accepts <= 60 s, so timestamps are scaled to 59 s (shown in the drawer). The chosen real clip spans 57 s and is sent unscaled.
- `public/models/Textures/colormap.png` does not match the car UVs (bodies rendered black), so the 3D scene paints parts with plain colours by name. Cars are therefore generic slate, not Kenney colours.
- With the deployed backend the real tyre-shop clip gives "Not observed" for seasonal tyre change and chassis diagnostics (per-frame wheel_off/raised spikes do not persist into events) and "Not camera-verifiable" for cabin filter. Shown as returned, not adjusted.
- Demo UI text is English only; language switcher is hidden on the Demo route.
- Cold start: "Starting the AI… ~1 min on first use" while the request is warming/queued; demo page calls `/health?warm=1` on open.
- Not tested: real phone camera capture, Safari.

## v2 (spec update): one upload drives the whole visit
- Single entry: photo or short video of a car entering (or sample). Plate card, vehicle type, colour estimated from the vehicle box (median of lower-middle crop, labelled "estimated from the photo"), then the demo booking card.
- 3D workshop is the main stage: 3 simulated background cars (grey, translucent; Bay 1 raised with a mechanic, 2 in the waiting lot). Your car = Kenney model by vehicle type, tinted to the estimated colour, floating plate tag. A 60 s demo-time timeline (`demo/timeline.ts`, pure `poseAt(t)`) drives it: Gate 0-5, Reception 5-13, Bay 13-38, Wash 38-46, Ready 46-52, Handover 52-60; play/pause/scrub; stepper buttons seek.
- "How the AI sees this stage" panel: example footage (public CCTV default, or synthetic) is sent live to the API and played in sync with the bay window, labelled "Example footage — different car". The 3D bay work (lift, wheel, bonnet, mechanic) is a SCRIPTED sequence and badged simulated.
- "Use my own bay footage" (photo or clip): the bay becomes observed; the 3D car follows the actual per-frame states (>= 50%), badge "Your input"; the timeline holds at the bay entrance until the analysis returns.
- Customer phone and Service manager panels follow the timeline. Manager bay board, counters and the other cars are simulated background (labelled). "Before handover" alert lists work-order lines whose status is not_observed; it is based on the example footage unless own footage is used.
- Wash/Ready/Handover remain simulated.
