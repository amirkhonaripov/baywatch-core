# Demo inputs: upload these live during the defence

Select a camera on the left of the BayWatch service floor. Then click **Upload footage to …** (on the gate: **Upload a car**).
Results were measured on 2026-10-11 against the deployed API (warm L4).

| File | Upload to | What the models return | Seen in training? |
|---|---|---|---|
| photos/uz_truck.jpg | Gate | plate 01U418QC, 91%, ~1 s; walk-in | no (held-out test set) |
| photos/uz_van.jpg | Gate | plate 40B472RA, 88%; booking found → visit drives to Bay 2 | no (Uzbek_Plates held-out split) |
| photos/rain.jpg, ducati.jpg, mustang.jpg | Gate | night/rain, motorcycle, US plate | no (held-out test set) |
| videos/gate_car_park.mp4 | Gate | 5 cars; distant plate "found — not readable" (no guess) | no (held-out test set) |
| videos/bay_lift_cctv.mp4 | Bay 2 | car in bay 100%, lift raised 68%, mechanic 44%; ~14 s | bay-state models: no (real-domain test clip). Scene router: yes, frames of this clip are in its "bay" class |
| videos/bay_tyre_shop_cctv.mp4 | Bay 4 | tyre-shop CCTV: car, people at car | no (real-domain test clip) |
| photos/wash_station.jpg | Wash | router "wash" 95%; 4 cars; washing 100% | no (held-out test set) |
| photos/wash_building_exterior.jpg | Wash | router "wash" 97%; 3 cars; washing 0% (correct: nobody washing) | no (held-out test set) |
| photos/damage_side_dent.jpg | Inspection | dent 0.51 | no (CarDD test split) |
| photos/damage_bumper_scratch.jpg | Inspection | scratch 0.49 | no (CarDD test split) |

**Known misses, to say out loud if asked:**
- Tyre-only close-ups get no inspection, because inspection only runs when a whole vehicle is detected.
- Some scratches on held-out CarDD images are missed (2 of 8 in our spot check).
- An overhead wash camera with a foam-covered car can miss the car entirely.

**On the day:**
- Videos are read in the browser as 16 frames spread over the clip. Keep the browser window visible: Chrome does not load video in hidden tabs.
- Pre-warm the API about 10 minutes before the defence: the first call after idle takes 17–100 s.

Sources and credits: videos/CREDITS.json and DATA_SOURCES.md.
