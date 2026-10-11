"""Synthetic end-to-end evaluation on the Blender renders (Lane E). Re-runnable as lane predictors land.

    python -m sim.eval_synthetic --renders RENDER_DIR [--split test|train|all] [--state-predictor mod:factory ...]
                                 [--out RUN_DIR] [--device cuda:0] [--max-frames N]

For every episode and bay: the bay's image region (rois.json) is cropped from each rendered frame (FramePred.states
is per frame, not per zone, so each bay is analysed as its own camera) and the bay's zone polygon (projected footprint,
sim/blender/scene_spec.py) is passed with the zone, so vehicles of the neighbouring bay do not count; the frames go through
``seen.pipeline.default_pipeline()`` in chunks (detectors, router, plus any ``--state-predictor``), and the merged
per-frame predictions are decoded once per bay with ``seen.visit_engine.build_visit`` (zone kind "bay").

Ground truth: exact per-second labels rebuilt from the scenario pieces in episode.json, turned into events with the
same ``label_events`` function the visit engine uses. Scores (data_kind "synthetic"): key-event F1 at +-10/30 s and
onset error, the same restricted to the states the deployed predictors actually report, idle-span P/R, and
work-order line status per visit (decoded events inside the visit's bay window vs oracle; shipped verify and the
zone/state-keyed variant), and the per-frame accuracy of the car-present cue (vehicle box in the zone polygon), which
places the renders on the noise-test curve. Time is compressed: one frame per ``frame_dt_s`` simulated seconds (4 s).
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from eval.stats import boot_metric, metric, prop_metric
from seen import visit_engine
from seen.contracts.events import EventType, Source, Zone, ZoneKind
from seen.contracts.workorder import WorkOrder
from seen.predictor import Det, Frame, FramePred
from seen.vocab import BAY_STATES
from sim import metrics as M

T_EP = datetime(2026, 10, 12, 8, 0, tzinfo=timezone(timedelta(hours=5)))
KI = {s: i for i, s in enumerate(BAY_STATES)}


def bay_label_grid(bay: dict, duration_s: float) -> np.ndarray:
    """Exact per-second labels of one bay over the whole episode (scenario pieces)."""
    T = int(duration_s) + 2
    lab = np.zeros((T, len(BAY_STATES)), dtype=bool)
    for v in bay["visits"]:
        for p in v["pieces"]:
            a = int(round(v["offset_s"] + p["start_s"]))
            b = min(T, a + int(p["dur_s"]))
            row = np.zeros(len(BAY_STATES), dtype=bool)
            row[KI["car_present"]] = True
            row[KI["raised"]], row[KI["wheel_off"]], row[KI["hood_open"]] = p["raised"], p["wheel"], p["hood"]
            row[KI["person_at_car"]], row[KI["person_under_car"]] = p["person"] == 1, p["person"] == 2
            row[KI["work"]] = bool(p["work"])
            row[KI["idle"]] = not p["work"]
            lab[a:b] = row
    return lab


def gt_events(bay: dict, duration_s: float, zone: Zone) -> list:
    lab = bay_label_grid(bay, duration_s)
    times = np.arange(len(lab), dtype=float)
    observed = np.array([s != "washing" for s in BAY_STATES])
    return visit_engine.label_events(times, lab, lab.astype(float), observed, zone=zone, visit_id=bay["bay_id"],
                                     ts=lambda i: T_EP + timedelta(seconds=float(times[i])), source=Source.simulated)


def load_pipeline(device: str, state_specs: list[str]):
    from seen import pipeline as P

    pipe = P.default_pipeline(device=device)
    for spec in state_specs:
        mod, fn = spec.split(":")
        pred = getattr(importlib.import_module(mod), fn)()
        pred.load(device)
        pipe.state_predictors.append(pred)
    return pipe


def zone_in_crop(ep: dict, bay_id: str, roi) -> list[list[float]]:
    """The bay's zone polygon (scene_spec projection of its footprint) in the crop's normalised coordinates."""
    from sim.blender.scene_spec import bay_zone_polygon

    poly = bay_zone_polygon(ep["episode_id"], ep["camera"], bay_id)
    w, h = roi[2] - roi[0], roi[3] - roi[1]
    return [[float((u - roi[0]) / w), float((v - roi[1]) / h)] for u, v in poly]


def run_bay(pipe, ep_dir: Path, ep: dict, bay: dict, roi, max_frames: int | None, chunk: int = 64):
    from PIL import Image

    files = sorted((ep_dir / "frames").glob("*.jpg"))
    if max_frames:
        files = files[:max_frames]
    zone = Zone(id=bay["bay_id"], kind=ZoneKind.bay, polygon=zone_in_crop(ep, bay["bay_id"], roi))
    preds: list[FramePred] = []
    timings: dict[str, float] = {}
    for k in range(0, len(files), chunk):
        frames = []
        for f in files[k:k + chunk]:
            im = Image.open(f).convert("RGB")
            W, H = im.size
            crop = im.crop((int(roi[0] * W), int(roi[1] * H), int(roi[2] * W), int(roi[3] * H)))
            i = int(f.stem)
            frames.append(Frame(index=i, t_s=i * ep["frame_dt_s"], image=np.asarray(crop)))
        res = pipe.analyze(frames, zones=[zone], source=Source.observed_synthetic, start_time=T_EP)
        for fr in res.frames:
            preds.append(FramePred(index=fr.frame_index, t_s=fr.t_s, states=dict(fr.states or {}),
                                   detections=[Det(cls=d.cls, confidence=d.confidence, bbox=tuple(d.bbox), text=d.text)
                                               for d in fr.detections]))
        for key, v in res.timings_ms.items():
            timings[key] = timings.get(key, 0.0) + v
    visit = visit_engine.build_visit(preds, start_time=T_EP, zones=[zone], source=Source.observed_synthetic,
                                     model_versions=pipe.model_versions, visit_id=f"{ep['episode_id']}_{bay['bay_id']}")
    observed_states = sorted({s for p in preds for s in p.states})
    # Per-frame car_present cue (vehicle box inside the zone polygon, conf >= 0.5) vs ground truth.
    gt = {json.loads(l)["frame"]: json.loads(l) for l in open(ep_dir / "gt_frames.jsonl")}
    frame_cmp = []
    for p in preds:
        c = visit_engine._vehicle_conf(p, zone.polygon)
        frame_cmp.append({"frame": p.index, "pred": bool(c is not None and c >= 0.5), "conf": c,
                          "gt": bool(gt[p.index]["bays"][bay["bay_id"]]["car_present"]),
                          "raised": bool(gt[p.index]["bays"][bay["bay_id"]]["raised"])})
    return visit, preds, timings, observed_states, len(files), frame_cmp


def score_bay(ep: dict, bay: dict, visit, observed_states: list[str], last_t: float) -> dict:
    zone = Zone(id=bay["bay_id"], kind=ZoneKind.bay)
    gt = [e for e in gt_events(bay, ep["duration_s"], zone) if (e.t - T_EP).total_seconds() <= last_t]
    pr = [] if visit is None else [e for e in visit.events if e.type not in (EventType.arrived, EventType.plate_read)]
    avail = set(observed_states) | {"car_present"}  # car_present falls back to the vehicle detector

    def covered(e):
        if e.type in (EventType.state_started, EventType.state_ended):
            return e.state.value in avail
        if e.type in (EventType.idle_started, EventType.idle_ended):
            return "work" in avail or "idle" in avail
        return True

    rec = {"events": M.event_counts(gt, pr, T_EP), "events_covered": M.event_counts([e for e in gt if covered(e)], pr, T_EP),
           "idle": M.idle_counts(M.idle_spans(gt, T_EP, last_t), M.idle_spans(pr, T_EP, last_t)), "lines": []}
    for v in bay["visits"]:
        a, b = v["offset_s"] - 60, v["offset_s"] + v["duration_s"] + 60
        if v["offset_s"] + v["duration_s"] > last_t:
            continue  # visit not fully rendered / evaluated
        inside = lambda evs: [e for e in evs if a <= (e.t - T_EP).total_seconds() <= b]  # noqa: E731
        rec["lines"] += M.line_records(WorkOrder.model_validate(v["work_order"]), inside(gt), inside(pr), v["lines_performed"])
    return rec


def main() -> None:
    from seen.paths import RENDERS, RUNS

    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", default=str(RENDERS / "bay_v1"))
    ap.add_argument("--split", default="test", choices=["test", "train", "all"])
    ap.add_argument("--state-predictor", action="append", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--tag", default="default_pipeline")
    a = ap.parse_args()
    out = Path(a.out) if a.out else RUNS / "sim" / "eval_synthetic" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline(a.device, a.state_predictor)
    recs, per_bay, total_frames, t0 = [], [], 0, time.time()
    frame_cmps: list[dict] = []
    timings: dict[str, float] = {}
    observed_all: set[str] = set()
    for ep_dir in sorted(Path(a.renders).glob("ep*")):
        ep = json.loads((ep_dir / "episode.json").read_text())
        if a.split != "all" and ep["split"] != a.split:
            continue
        if not (ep_dir / "rois.json").exists() or not any((ep_dir / "frames").glob("*.jpg")):
            print("skip (not rendered)", ep_dir.name)
            continue
        rois = json.loads((ep_dir / "rois.json").read_text())["rois"]
        for bay in ep["bays"]:
            visit, preds, tm, obs, n, fc = run_bay(pipe, ep_dir, ep, bay, rois[bay["bay_id"]], a.max_frames)
            rec_fc = [dict(x, episode=ep["episode_id"], bay=bay["bay_id"]) for x in fc]
            frame_cmps.extend(rec_fc)
            last_t = max(p.t_s for p in preds) if preds else 0.0
            rec = score_bay(ep, bay, visit, obs, last_t)
            recs.append(rec)
            observed_all |= set(obs)
            total_frames += n
            for k, v in tm.items():
                timings[k] = timings.get(k, 0.0) + v
            per_bay.append({"episode": ep["episode_id"], "bay": bay["bay_id"], "camera": ep["camera"], "hdri": ep["hdri"],
                            "frames": n, "observed_states": obs, "n_pred_events": 0 if visit is None else len(visit.events),
                            "key_f1_10s_covered": M.event_f1([rec["events_covered"]], 10),
                            "pred_events": [] if visit is None else json.loads(visit.model_dump_json())["events"]})
            print(ep["episode_id"], bay["bay_id"], n, "frames; states:", obs, "covered key F1@10s",
                  round(M.event_f1([rec["events_covered"]], 10), 3), flush=True)
    if not recs:
        raise SystemExit("no rendered episodes found for this split")
    ev = lambda key: [r[key] for r in recs]  # noqa: E731
    lines = [r["lines"] for r in recs]
    n_lines = sum(len(x) for x in lines)
    ip = M.idle_pr([r["idle"] for r in recs])
    ms = [metric("frames", total_frames, unit="count"),
          boot_metric("key_event_f1_at_10s_all_states", recs, lambda rs: M.event_f1([r["events"] for r in rs], 10), "fraction", n_boot=500),
          boot_metric("key_event_f1_at_10s_covered_states", recs, lambda rs: M.event_f1([r["events_covered"] for r in rs], 10), "fraction", n_boot=500),
          boot_metric("key_event_f1_at_30s_covered_states", recs, lambda rs: M.event_f1([r["events_covered"] for r in rs], 30), "fraction", n_boot=500)]
    med = M.median_abs_err(ev("events_covered"))
    if med == med:
        ms.append(boot_metric("key_event_median_abs_onset_error_covered", recs,
                              lambda rs: M.median_abs_err([r["events_covered"] for r in rs]), "s", n_boot=500))
    for k in ("seg_precision", "seg_recall"):
        if ip[k] == ip[k]:
            ms.append(metric(f"idle_{k}_iou50", ip[k], unit="fraction"))
    k_fr = sum(x["pred"] == x["gt"] for x in frame_cmps)
    ms.append(prop_metric("car_present_cue_frame_accuracy", k_fr, len(frame_cmps)))
    raised = [x for x in frame_cmps if x["raised"]]
    if raised:
        ms.append(prop_metric("car_present_cue_recall_when_raised", sum(x["pred"] for x in raised), len(raised)))
    if n_lines:
        k_gt = sum(r["pred"] == r["gt"] for x in lines for r in x)
        k_key = sum(r["pred_keyed"] == r["gt_keyed"] for x in lines for r in x)
        ms += [prop_metric("workorder_line_status_accuracy_vs_oracle", k_gt, n_lines),
               prop_metric("workorder_line_status_accuracy_vs_oracle_keyed_spans", k_key, n_lines)]
    sec = time.time() - t0
    result = {"id": f"e2e.synthetic.{a.tag}.{a.split}", "name": f"Synthetic end-to-end ({a.tag}, {a.split} split)",
              "lane": "e2e", "data_kind": "synthetic", "domain": "rendered (Blender 5.0.1 Cycles, Kenney CC0 assets, CCTV-like high camera)",
              "dataset": f"{Path(a.renders).name}: {len(recs)} bay clips", "split": a.split, "n": len(recs),
              "model": " + ".join(f"{k}:{v}" for k, v in pipe.model_versions.items()) + " + visit_engine:" + visit_engine.DECODER,
              "metrics": ms,
              "notes": (f"States reported by the deployed predictors: {sorted(observed_all) or 'none'} (car_present from the vehicle "
                        "detector when no state predictor runs). 'covered' = GT events of those states only. One frame per "
                        f"4 simulated seconds; {n_lines} camera-verifiable work-order lines; CIs = bootstrap over bay clips. "
                        f"Wall time {sec:.0f}s for {total_frames} frames. Synthetic only; not evidence of real-CCTV accuracy.")}
    (out / "component_results.json").write_text(json.dumps([result], indent=1))
    (out / "per_bay.json").write_text(json.dumps(per_bay, indent=1))
    (out / "frame_cue.jsonl").write_text("".join(json.dumps(x) + "\n" for x in frame_cmps))
    (out / "timings_ms.json").write_text(json.dumps({k: v / max(1, total_frames) for k, v in timings.items()}, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
