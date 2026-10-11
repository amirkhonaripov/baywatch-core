"""Build 2-bay render episodes from scenario-engine visits, with exact per-frame ground truth.

    python -m sim.blender.make_episodes --out $SEEN_ROOT/renders/bay_v1     (default: 10 episodes, 1 frame / 4 s)

An episode = one camera over two bays for a few simulated hours. Each bay serves 1-2 visits back to back
(scenario-engine bay segments) with random empty gaps. Splits are held out by car model, HDRI and camera
(train: cameras A/B, 5 HDRIs, 6 car models; test: camera C, skylit_garage HDRI, 2 unseen car models).

Writes per episode: episode.json (scene parameters, per-bay visit schedule with the scenario pieces, work orders,
ground-truth events) and gt_frames.jsonl (per frame: t_s and the exact BayState labels of each bay).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from seen.vocab import BAY_STATES
from sim.scenario import generate_visits

TRAIN = {"cameras": ["cam_A", "cam_B"], "hdris": ["auto_service", "autoshop_01", "garage", "machine_shop_02", "empty_workshop"],
         "cars": ["sedan", "sedan-sports", "suv", "van", "truck", "hatchback-sports"]}
TEST = {"cameras": ["cam_C"], "hdris": ["skylit_garage"], "cars": ["suv-luxury", "taxi"]}
FLOORS = ["concrete_floor_02", "concrete_floor_painted", "concrete_floor_worn_001"]
CHARACTERS = ["character-male-a", "character-male-b", "character-male-c", "character-female-a", "character-female-b"]


def build(n_train: int, n_test: int, seed: int, frame_dt: float, out: Path) -> list[dict]:
    rng = np.random.default_rng(seed)
    pool = generate_visits(400, seed)
    # Keep render time bounded: skip extreme bay stays (> 3 h).
    pool = [p for p in pool if (p[2].span("bay")[1] - p[2].span("bay")[0]) <= 3 * 3600]
    used = 0
    episodes = []
    for e in range(n_train + n_test):
        split = "train" if e < n_train else "test"
        S = TRAIN if split == "train" else TEST
        ep = {"episode_id": f"ep{e:02d}_{split}", "split": split, "seed": int(rng.integers(1 << 30)),
              "camera": str(rng.choice(S["cameras"])), "hdri": str(rng.choice(S["hdris"])),
              "hdri_rotation_deg": float(rng.uniform(0, 360)), "hdri_strength": float(rng.uniform(0.6, 1.4)),
              "floor": str(rng.choice(FLOORS)), "frame_dt_s": frame_dt, "resolution": [1280, 720], "bays": []}
        for b, bay_id in enumerate(("bay1", "bay2")):
            t = float(rng.uniform(60, 900))  # empty bay before the first car
            visits = []
            for k in range(2 if rng.random() < 0.4 else 1):
                visit, wo, tl = pool[used]
                used += 1
                a, z = tl.span("bay")
                pieces = [dict(p, start_s=p["start_s"] - a) for p in tl.meta["pieces"] if p["loc"] == "bay"]
                visits.append({"visit_id": tl.visit_id, "offset_s": t, "duration_s": z - a, "bay_span_s": [a, z],
                               "t0": tl.t0.isoformat(), "car_model": str(rng.choice(S["cars"])),
                               "character": str(rng.choice(CHARACTERS)), "job_type": tl.meta["job_type"],
                               "lines_performed": tl.meta["lines_performed"], "phantom_lines": tl.meta["phantom_lines"],
                               "work_order": wo.model_dump(), "pieces": pieces,
                               "gt_events": [json.loads(ev.model_dump_json()) for ev in visit.events
                                             if ev.zone is not None and ev.zone.kind.value == "bay"]})
                t += (z - a) + float(rng.uniform(120, 600))
            ep["bays"].append({"bay_id": bay_id, "visits": visits, "end_s": t})
        ep["duration_s"] = max(b["end_s"] for b in ep["bays"])
        ep["n_frames"] = int(ep["duration_s"] // frame_dt) + 1
        episodes.append(ep)
        d = out / ep["episode_id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "episode.json").write_text(json.dumps(ep, indent=1))
        with open(d / "gt_frames.jsonl", "w") as f:
            for i in range(ep["n_frames"]):
                ts = i * frame_dt
                f.write(json.dumps({"frame": i, "t_s": ts, "bays": {b["bay_id"]: frame_labels(b, ts) for b in ep["bays"]}}) + "\n")
    return episodes


def frame_labels(bay: dict, ts: float) -> dict:
    """Exact ground-truth labels of a bay at episode time ts (from the scenario pieces)."""
    lab = {s: 0 for s in BAY_STATES}
    lab["visit_id"] = None
    for v in bay["visits"]:
        u = ts - v["offset_s"]
        if 0 <= u < v["duration_s"]:
            for p in v["pieces"]:
                if p["start_s"] <= u < p["start_s"] + p["dur_s"]:
                    lab.update(car_present=1, raised=p["raised"], wheel_off=p["wheel"], hood_open=p["hood"],
                               person_at_car=int(p["person"] == 1), person_under_car=int(p["person"] == 2),
                               work=p["work"], idle=int(not p["work"]), washing=0)
                    lab["visit_id"], lab["step"] = v["visit_id"], p["step"]
                    break
    return lab


def main() -> None:
    from seen.paths import RENDERS

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RENDERS / "bay_v1"))
    ap.add_argument("--n-train", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--frame-dt", type=float, default=4.0)
    a = ap.parse_args()
    eps = build(a.n_train, a.n_test, a.seed, a.frame_dt, Path(a.out))
    n_vis = sum(len(b["visits"]) for e in eps for b in e["bays"])
    print(f"{len(eps)} episodes, {n_vis} visits, {sum(e['n_frames'] for e in eps)} frames -> {a.out}")
    for e in eps:
        print(e["episode_id"], e["camera"], e["hdri"], e["n_frames"], [(v["car_model"], v["job_type"]) for b in e["bays"] for v in b["visits"]])


if __name__ == "__main__":
    main()
