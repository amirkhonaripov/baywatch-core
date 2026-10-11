"""Run the full pipeline on the jury-style set and score the behaviour (Lane G).

    scripts/box.sh --gpu 1 -- python -m eval.jury_run --out /mnt/bigdrive/amirkhon/seen/runs/eval/latest

Per input we check: route scene (against the acceptable scenes), visit created iff a vehicle is present, 'unknown'
when nothing relevant is in view, plate read when a plate is visible, expected modules ran (only modules that are
deployed count), crashes, and latency against the plan's limits (60 s per video, we use 10 s per photo).
Labels are in eval/jury/jury.jsonl (single annotator: the Lane G agent, from the images).

Outputs (under --out): docs/lanes/eval_jury.json (JurySet), docs/lanes/eval_results.json (ComponentResult list,
lane e2e/router, real data), docs/lanes/failures/eval/*.jpg + failures.json (face-blurred), and the raw outcome
table runs/eval/jury/<stamp>/outcomes.{json,csv}.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from eval.jury.common import SCENES, expected_modules, jury_items, local_path
from eval.stats import accuracy_coverage, confusion, metric, percentiles, prop_metric, wilson_ci

REPO = Path(__file__).resolve().parents[1]
VIDEO_FPS = 2.0  # browser samples phone video at 2 fps (plan)
VIDEO_MAX_S = 60.0
FRAME_MAX = (1280, 720)
LIMIT_S = {"image": 10.0, "video_frame": 10.0, "video": 60.0}
DEPLOYABLE = ("plate", "visit_engine", "bay_states", "inspection")


def load_image(p: Path) -> np.ndarray:
    import cv2

    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        from PIL import Image

        return np.asarray(Image.open(p).convert("RGB"))
    return im[:, :, ::-1].copy()


def sample_video(p: Path, fps: float = VIDEO_FPS, max_s: float = VIDEO_MAX_S) -> list[tuple[float, np.ndarray]]:
    """Decode with PyAV (VP9/webm safe), keep one frame every 1/fps s for the first max_s s, fit into 1280x720."""
    import av
    import cv2

    out, next_t = [], 0.0
    with av.open(str(p)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for fr in c.decode(s):
            t = float(fr.time) if fr.time is not None else len(out) / fps
            if t > max_s:
                break
            if t + 1e-6 < next_t:
                continue
            img = fr.to_ndarray(format="rgb24")
            h, w = img.shape[:2]
            sc = min(FRAME_MAX[0] / w, FRAME_MAX[1] / h, 1.0)
            if sc < 1.0:
                img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
            out.append((t, img))
            next_t += 1.0 / fps
    return out


def run_item(pipe, it: dict, store: dict) -> tuple[dict, list]:
    from seen.predictor import Frame

    p = local_path(it)
    if it["kind"] == "video":
        fr = sample_video(p)
        frames = [Frame(index=i, t_s=t, image=img) for i, (t, img) in enumerate(fr)]
    else:
        frames = [Frame(index=0, t_s=0.0, image=load_image(p))]
    if not frames:
        raise RuntimeError("no frames decoded")

    def writer(name: str, img: np.ndarray) -> str:
        store[name] = img  # already passed through seen.blur.blur_faces by the pipeline
        return f"mem://{name}"

    t0 = time.perf_counter()
    res = pipe.analyze(frames, evidence_writer=writer)
    return {"latency_ms": (time.perf_counter() - t0) * 1000, "result": res, "n_frames": len(frames)}, frames


def seen_by_lanes() -> dict[str, list[str]]:
    """Jury ids that appear in another lane's data pool or training list (eval/run_all.py pools + audit)."""
    from seen.paths import RUNS

    seen: dict[str, list[str]] = {}
    d = RUNS / "eval" / "leakage"
    p = d / "jury_pool_overlap.json"
    if p.exists():
        for r in json.loads(p.read_text()):
            for j in r["jury_ids"]:
                seen.setdefault(j, []).append(r["pool"])
    for f in d.glob("*.json"):
        if f.name == "jury_pool_overlap.json":
            continue
        try:
            rep = json.loads(f.read_text())
            for j in rep.get("components", {}).get("jury", {}).get("leaked_loose_test_ids", []):
                seen.setdefault(j.split(":", 1)[-1], []).append(f.stem)
        except Exception:
            pass
    return seen


def score(it: dict, run: dict | None, err: str | None) -> dict:
    exp = it["expected"]
    o = {"id": it["id"], "category": it["category"], "kind": it["kind"], "hard": bool(exp.get("hard")),
         "exp_scene": exp["scene"], "acceptable": "|".join(exp["acceptable_scenes"]), "exp_vehicle": bool(exp["vehicle"]),
         "exp_plate": bool(exp.get("plate_visible")), "licence": it.get("licence"), "title": it.get("title")}
    exp_mods = expected_modules(exp)
    if run is None:
        o.update({"status": "crash", "error": (err or "")[:300], "route_ok": False, "visit_ok": False})
        return o
    r = run["result"]
    got = r.route.scene.value if hasattr(r.route.scene, "value") else str(r.route.scene)
    ran = list(r.route.ran)
    not_deployed = [s.module for s in r.route.skipped if s.reason == "module not deployed"]
    dets = Counter(d.cls for f in r.frames for d in f.detections)
    plates = [d.text for f in r.frames for d in f.detections if d.cls == "plate" and d.text]
    visit = r.visit is not None
    limit = LIMIT_S[it["kind"]]
    o.update({
        "status": "ok" if run["latency_ms"] <= limit * 1000 else "timeout",
        "n_frames": run["n_frames"], "got_scene": got, "route_conf": round(float(r.route.confidence), 3),
        "visit": visit, "plate_read": bool(plates), "plate_texts": "|".join(sorted(set(plates)))[:80],
        "ran": "|".join(ran), "not_deployed": "|".join(not_deployed),
        "dets": " ".join(f"{k}:{v}" for k, v in sorted(dets.items())),
        "latency_ms": round(run["latency_ms"], 1),
        "route_ok": got in exp["acceptable_scenes"],
        "route_primary_ok": got == exp["scene"],
        "visit_ok": visit == bool(exp["vehicle"]),
    })
    if exp["scene"] == "unknown":
        o["unknown_ok"] = got == "unknown"
    if exp.get("plate_visible"):
        o["plate_ok"] = bool(plates)
    deployed_exp = [m for m in exp_mods if m in DEPLOYABLE and m not in not_deployed]
    missing = [m for m in deployed_exp if m not in ran]
    extra = [m for m in ran if m in DEPLOYABLE and m not in exp_mods]
    o["exp_modules"] = "|".join(exp_mods)
    o["missing_modules"] = "|".join(missing)
    o["extra_modules"] = "|".join(extra)
    o["modules_ok"] = not missing and not extra
    o["exp_modules_not_deployed"] = "|".join(m for m in exp_mods if m in not_deployed)
    return o


def _rate(outs, key, cond=lambda o: True):
    sel = [o for o in outs if cond(o) and key in o]
    k = sum(bool(o[key]) for o in sel)
    return k, len(sel)


def summarise(outs: list[dict]) -> tuple[dict, list[dict], dict]:
    s: dict[str, float] = {"n": len(outs)}
    s["crashes"] = sum(o["status"] == "crash" for o in outs)
    s["timeouts"] = sum(o["status"] == "timeout" for o in outs)
    metrics = []

    def add(name, key, cond=lambda o: True, label=None):
        k, n = _rate(outs, key, cond)
        if n == 0:
            return
        p, lo, hi = wilson_ci(k, n)
        s[name] = round(p, 4)
        s[f"{name}_ci_low"] = round(lo, 4)
        s[f"{name}_ci_high"] = round(hi, 4)
        s[f"{name}_n"] = n
        metrics.append(prop_metric(label or name, k, n))

    ran = lambda o: o["status"] != "crash"  # noqa: E731
    add("route_accuracy", "route_ok", ran)
    add("route_accuracy_excl_hard", "route_ok", lambda o: ran(o) and not o["hard"])
    add("route_primary_scene_accuracy", "route_primary_ok", ran)
    for sc in SCENES:  # per-scene recall against the primary label (the baseline cannot output gate/bay/wash/closeup)
        sel = [o for o in outs if ran(o) and o["exp_scene"] == sc]
        if sel:
            k = sum(o.get("route_primary_ok", False) for o in sel)
            s[f"scene_recall[{sc}]"] = round(k / len(sel), 4)
            s[f"scene_recall_n[{sc}]"] = len(sel)
            metrics.append(prop_metric(f"scene_recall_{sc}", k, len(sel)))
    add("visit_iff_vehicle_accuracy", "visit_ok", ran)
    add("visit_iff_vehicle_accuracy_excl_hard", "visit_ok", lambda o: ran(o) and not o["hard"])
    # false visits on inputs without a vehicle; missed visits on inputs with one
    k = sum(o.get("visit", False) for o in outs if ran(o) and not o["exp_vehicle"])
    n = sum(1 for o in outs if ran(o) and not o["exp_vehicle"])
    if n:
        p, lo, hi = wilson_ci(k, n)
        s.update({"false_visit_rate_no_vehicle": round(p, 4), "false_visit_rate_no_vehicle_ci_low": round(lo, 4),
                  "false_visit_rate_no_vehicle_ci_high": round(hi, 4), "false_visit_rate_no_vehicle_n": n})
        metrics.append(prop_metric("false_visit_rate_on_no_vehicle_inputs", k, n))
    k = sum(not o.get("visit", False) for o in outs if ran(o) and o["exp_vehicle"])
    n = sum(1 for o in outs if ran(o) and o["exp_vehicle"])
    if n:
        p, lo, hi = wilson_ci(k, n)
        s.update({"missed_visit_rate_vehicle": round(p, 4), "missed_visit_rate_vehicle_ci_low": round(lo, 4),
                  "missed_visit_rate_vehicle_ci_high": round(hi, 4), "missed_visit_rate_vehicle_n": n})
        metrics.append(prop_metric("missed_visit_rate_on_vehicle_inputs", k, n))
    add("unknown_recall", "unknown_ok", ran)
    # unknown precision: of inputs routed 'unknown', how many accept 'unknown'
    sel = [o for o in outs if ran(o) and o.get("got_scene") == "unknown"]
    if sel:
        k = sum("unknown" in o["acceptable"].split("|") for o in sel)
        p, lo, hi = wilson_ci(k, len(sel))
        s.update({"unknown_precision": round(p, 4), "unknown_precision_ci_low": round(lo, 4),
                  "unknown_precision_ci_high": round(hi, 4), "unknown_precision_n": len(sel)})
        metrics.append(prop_metric("unknown_precision", k, len(sel)))
    add("route_accuracy_clean", "route_ok", lambda o: ran(o) and not o.get("seen_by_lane"))
    add("visit_iff_vehicle_accuracy_clean", "visit_ok", lambda o: ran(o) and not o.get("seen_by_lane"))
    s["n_seen_by_a_lane"] = sum(bool(o.get("seen_by_lane")) for o in outs)
    add("plate_read_rate_when_visible", "plate_ok", ran)
    add("modules_as_expected", "modules_ok", ran)
    for kind in ("image", "video_frame", "video"):
        lat = [o["latency_ms"] for o in outs if o["kind"] == kind and "latency_ms" in o]
        if lat:
            pc = percentiles(lat)
            s[f"latency_{kind}_p50_ms"] = round(pc["p50"], 1)
            s[f"latency_{kind}_p95_ms"] = round(pc["p95"], 1)
            metrics.append(metric(f"latency_{kind}_p50", pc["p50"], unit="ms"))
    rr = [o for o in outs if ran(o) and "route_conf" in o]
    if rr:  # accuracy when only the most confident routes are answered (others would become 'unknown')
        for pt in accuracy_coverage([o["route_conf"] for o in rr], [o["route_ok"] for o in rr], (1.0, 0.8, 0.6, 0.4)):
            s[f"route_accuracy@coverage{pt['coverage']:.1f}"] = round(pt["accuracy"], 4)
            s[f"route_conf_threshold@coverage{pt['coverage']:.1f}"] = round(pt["threshold"], 4)
    by_cat = defaultdict(list)
    for o in outs:
        if ran(o):
            by_cat[o["category"]].append(o)
    for c, os_ in sorted(by_cat.items()):
        s[f"route_accuracy[{c}]"] = round(sum(o["route_ok"] for o in os_) / len(os_), 4)
        s[f"visit_ok[{c}]"] = round(sum(o["visit_ok"] for o in os_) / len(os_), 4)
        s[f"n[{c}]"] = len(os_)
    conf = confusion([o["exp_scene"] for o in outs if ran(o)], [o.get("got_scene", "unknown") for o in outs if ran(o)], list(SCENES))
    return s, metrics, conf


def failure_reason(o: dict) -> str | None:
    if o["status"] == "crash":
        return f"crash: {o.get('error', '')[:120]}"
    bits = []
    if not o["route_ok"]:
        bits.append(f"routed '{o['got_scene']}' (expected {o['acceptable'].replace('|', '/')})")
    if not o["visit_ok"]:
        bits.append("visit created without a vehicle" if o["visit"] else "no visit although a vehicle is present")
    if o.get("plate_ok") is False:
        bits.append("visible plate not read")
    if o["status"] == "timeout":
        bits.append(f"slow: {o['latency_ms'] / 1000:.1f}s")
    return "; ".join(bits) if bits else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--detectors", default="", help="comma list of module:factory; default = seen.pipeline.default_pipeline()")
    ap.add_argument("--states", default="", help="comma list of module:factory for state predictors")
    ap.add_argument("--inspection", default="", help="comma list of module:factory for inspection predictors")
    ap.add_argument("--tag", default="", help="label for this pipeline variant (output goes to docs/lanes/eval_jury<_tag>.json)")
    a = ap.parse_args(argv)
    from seen.paths import RUNS
    from seen.pipeline import default_pipeline

    out_root = Path(a.out) if a.out else RUNS / "eval" / "latest"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS / "eval" / "jury" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    items = jury_items()
    if a.ids:
        want = set(a.ids.split(","))
        items = [i for i in items if i["id"] in want]
    if a.limit:
        items = items[: a.limit]
    missing = [i["id"] for i in items if not local_path(i).exists()]
    items = [i for i in items if local_path(i).exists()]
    t_load = time.perf_counter()
    if a.detectors:
        from seen.pipeline import Pipeline, _load

        specs = lambda x: [_load(sp) for sp in x.split(",") if sp]  # noqa: E731
        pipe = Pipeline(specs(a.detectors), specs(a.states), specs(a.inspection), device=a.device)
    else:
        pipe = default_pipeline(a.device)
    load_s = time.perf_counter() - t_load
    print(f"pipeline loaded in {load_s:.1f}s: {pipe.model_versions}; {len(items)} inputs, {len(missing)} missing media", flush=True)
    from eval.failures import save_gallery_image

    fail_dir = out_root / "docs" / "lanes" / "failures" / ("eval" + (f"_{a.tag}" if a.tag else ""))
    fail_dir.mkdir(parents=True, exist_ok=True)
    for old in fail_dir.glob("jury_*.jpg"):
        old.unlink()
    seen = seen_by_lanes()
    outs, fails = [], []
    for it in items:
        store: dict = {}
        try:
            run, frames = run_item(pipe, it, store)
            o = score(it, run, None)
        except Exception as e:
            run, frames = None, []
            o = score(it, None, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-600:]}")
        o["seen_by_lane"] = "|".join(sorted(set(seen.get(it["id"], []))))
        outs.append(o)
        why = failure_reason(o)
        print(f"{it['id']} {it['category']:<12} {o.get('got_scene', '-'):<10} visit={o.get('visit')} "
              f"{o.get('latency_ms', 0):7.0f}ms {'FAIL: ' + why if why else 'ok'}", flush=True)
        if why and run is not None:
            r = run["result"]
            k = len(r.frames) // 2
            fr = frames[k]
            dets = []
            from seen.predictor import Det

            for d in r.frames[k].detections:
                dets.append(Det(cls=d.cls, confidence=d.confidence, bbox=tuple(d.bbox)))
            dst = save_gallery_image(fr.image, fail_dir / it["id"], dets)
            fails.append({"id": f"jury:{it['id']}", "component": "e2e jury set", "file": dst.name,
                          "description": f"{it['expected']['notes']}: {why}. Licence {it.get('licence')}, {it.get('page')}",
                          "category": it["category"]})
    (fail_dir / "failures.json").write_text(json.dumps(fails, indent=2, ensure_ascii=False))
    summ, metrics, conf = summarise(outs)
    summ["pipeline_load_s"] = round(load_s, 2)
    summ["missing_media"] = len(missing)
    mv = ", ".join(f"{k}={v}" for k, v in pipe.model_versions.items())
    jury = {"n": len(outs), "outcomes": outs, "summary": summ}
    cols = sorted({k for o in outs for k in o})
    with (run_dir / "outcomes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(outs)
    (run_dir / "outcomes.json").write_text(json.dumps(jury, indent=2, ensure_ascii=False))
    comp_common = {"data_kind": "real", "domain": "mixed web photos and short videos (Wikimedia Commons), not CCTV",
                   "dataset": f"Seen jury set v1 ({len(outs)} inputs, Wikimedia Commons CC/PD)",
                   "split": "held out: eval/manifests/jury.jsonl", "model": f"seen.pipeline.default_pipeline [{mv}]",
                   "predictions_csv": f"runs/eval/jury/{stamp}/outcomes.csv"}
    route_m = [m for m in metrics if m["name"].startswith(("route_", "unknown_", "scene_recall"))]
    visit_m = [m for m in metrics if m["name"].startswith(("visit_iff", "false_visit", "missed_visit"))]
    other_m = [m for m in metrics if m not in route_m and m not in visit_m]
    comps = [
        {"id": "e2e.jury.routing", "name": "Scene routing on jury-style inputs", "lane": "router", "n": len(outs),
         "metrics": route_m, "confusion": conf,
         "notes": "Scene label = single annotator (Lane G agent) with a list of acceptable scenes per input; "
                  "route_accuracy accepts any listed acceptable scene; route_primary_scene_accuracy and scene_recall_* "
                  "use the single primary label (confusion matrix). "
                  "'hard' = toys, drawings, trains, tiny or partial vehicles. '_clean' metrics exclude inputs that appear in "
                  "another lane's data pool or training list (Commons overlap, see eval_jury_pool_overlap.json)."
                  , **comp_common},
        {"id": "e2e.jury.visit", "name": "Visit created iff a vehicle is present (jury set)", "lane": "e2e", "n": len(outs),
         "metrics": visit_m, "notes": "A visit must be created only when a real road vehicle is in view.", **comp_common},
        {"id": "e2e.jury.behaviour", "name": "Plate reading, module selection and latency (jury set)", "lane": "e2e",
         "n": len(outs), "metrics": other_m,
         "notes": "plate_read = any plate text returned when a plate is visible (correctness of text not scored: no text labels). "
                  "modules_as_expected counts only modules that are deployed.", **comp_common},
    ]
    d = out_root / "docs" / "lanes"
    d.mkdir(parents=True, exist_ok=True)
    suffix = f"_{a.tag}" if a.tag else ""
    if a.tag:
        for c in comps:
            c["id"] += f".{a.tag}"
            c["name"] += f" [{a.tag}]"
    (d / f"eval_jury{suffix}.json").write_text(json.dumps(jury, indent=2, ensure_ascii=False))
    (d / f"eval{suffix}_results.json").write_text(json.dumps(comps, indent=2, ensure_ascii=False))
    print(json.dumps(summ, indent=2))
    print(f"wrote {run_dir} and {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
