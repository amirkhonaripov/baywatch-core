# Synthetic service-bay visits (Blender)

**What this is:** rendered, compressed-time CCTV-style clips of two service bays, with exact ground truth.
Every visit, its timing and its work order come from the scenario engine (`sim/scenario.py`). Nothing is
filmed, and there are no real people or plates. The method must be declared on the slides as **synthetic**.

## Generation method

1. **Scenario engine** (`sim/scenario.py`). Visits follow the stage grammar
   gate → reception → waiting → bay → [wash] → ready → handover.
   - In the bay, the job is planned from work-order lines: express maintenance (ICE or EV), tyre change or rotation,
     brake pads, chassis diagnostics on the lift, and repair with a parts wait.
   - The planner inserts the physical steps between lines: raise and lower the lift, wheels off and on, bonnet open.
   - It adds realistic gaps: short pauses, occasional long technician absences, and parts waits.
   - Durations are lognormal priors. They are assumptions, documented in the code. They are not measured at a dealer.
2. **Episodes** (`sim/blender/make_episodes.py`, seed 3000):
   - 10 episodes in total. Each episode is one camera over two bays.
   - Each bay serves 1–2 visits back to back, with random empty gaps.
   - The result is 29 visits and 14,472 frames, at **one frame per 4 simulated seconds**.
   - On integrator request, 3 more held-out test episodes (ep10–ep12, `--n-test 7`, 10 visits) were added. They are rendered at every 4th frame (one per 16 simulated s) with 16 samples, to halve render time. The test split is 21 visits in total.
   - Ground truth (`gt_frames.jsonl`) is the exact per-frame bay state, taken from the scenario pieces.
3. **Rendering** (`sim/blender/render_bay.py`, Blender 5.0.1 Cycles + OptiX, 32 samples + OptiX denoiser, 1280×720 JPEG q80, GPU1 on the box).
   - The scene is built by script:
     - concrete floor, three walls with door openings, yellow bay lines
     - two-post lifts: posts, carriages and arms move with the lift height
     - benches, toolboxes, area lights
     - a Poly Haven HDRI for ambient light
   - Each frame's state is set from the scenario pieces:
     - Cars drive in and out.
     - The lift rises over 25 s.
     - Wheels are removed one by one and laid on the floor (2 wheels for brake jobs, 4 otherwise).
     - The bonnet opens.
     - The mechanic stands at the car or under it, or is at the bench or away. A working pose means work; an idle pose means no work.
   - Outputs per episode:
     - `frames/NNNNNN.jpg`
     - `labels.jsonl`: projected 2D boxes for the car, the person and the removed wheels, per bay
     - `rois.json`: each bay's image region, used to crop per-bay clips
4. **Held-out split** (by source, not by frame):
   - train: cameras A/B; HDRIs auto_service, autoshop_01, garage, machine_shop_02, empty_workshop; cars sedan, sedan-sports, suv, van, truck, hatchback-sports
   - test: camera C, HDRI skylit_garage, cars suv-luxury and taxi (none of these appear in train)

Commands (box):
```
python -m sim.blender.fetch_assets                                   # CC0 assets -> data/raw/sim_assets
python -m sim.blender.make_episodes --out $SEEN_FAST/sim/bay_v1      # scenario JSON + ground truth
SEEN_SIM_ASSETS=$SEEN_FAST/sim/sim_assets bash sim/blender/render_all.sh $SEEN_FAST/sim/bay_v1 0 1
bash sim/blender/stage_nvme.sh back bay_v1                           # final copy to $SEEN_ROOT/renders/bay_v1
python -m sim.eval_synthetic --renders $SEEN_ROOT/renders/bay_v1 --split test
```

## Assets and licences

| Asset | Source | Licence | Used for |
|---|---|---|---|
| Kenney Car Kit (sedan, sedan-sports, suv, suv-luxury, van, truck, hatchback-sports, taxi) | https://kenney.nl/assets/car-kit | CC0 1.0 (asset page "License" row) | cars, separate wheel meshes |
| Kenney Mini Characters (rigged, with idle / interact / crouch / holding actions) | https://kenney.nl/assets/mini-characters | CC0 1.0 | proxy mechanics |
| Poly Haven HDRIs: auto_service, autoshop_01, garage, skylit_garage, machine_shop_02, empty_workshop (1k) | https://polyhaven.com/hdris | CC0 1.0 | ambient light, background through doors |
| Poly Haven textures: concrete_floor_02, concrete_floor_painted, concrete_floor_worn_001, painted_plaster_wall (1k) | https://polyhaven.com/textures | CC0 1.0 | floor, walls |
| Blender 5.0.1 | https://www.blender.org | GPL (renders are ours) | rendering |

Each download has a `SOURCE.json` next to it (url, licence, sha256). The HDRIs contain tiny distant background
cars; at 1k resolution, seen only through door openings, no plate is readable.

## What synthetic results can and cannot claim

- **Can:** the scenario-to-events logic (temporal decoding, idle detection, work-order matching) works end to end
  on visits with exact timestamps. Numbers are reproducible from this folder.
- **Cannot:** accuracy on real dealer CCTV. The cars are low-poly toy models. The bonnet is a proxy panel. The
  people are stylised proxies. Lighting is clean, with no motion blur, rain or real occlusion. Time is
  compressed (one frame per 4 s). Ground truth during lift motion is ambiguous for about 25 s: the label switches
  at the start of the raise step, while the car is still near the floor.
- Durations and job mix are assumptions. Idle and ETA numbers describe the generator, not a dealership.
