"""Tracking on UA-DETRAC test at the frame rates we will use: gate (5 fps) and bay (1 and 0.5 fps).

    python -m lanes.flow.eval_track --model rfdetr-base --exp track_rfdetr-base

Detections come from the cached zero-shot (or fine-tuned) detector run on every 5th frame (5 fps); 1 fps / 0.5 fps keep every 5th / 10th of
those frames, so the tracker really sees a 1 s / 2 s gap. Trackers: greedy-IoU baseline, ByteTrack (supervision), OC-SORT (lanes/flow/tracker.py).
Metrics: HOTA / DetA / AssA / IDF1 / MOTA / ID-switches pooled over the 40 test sequences, 95% CI = bootstrap over sequences. Sequences are also
split into slow / mid / fast tertiles by GT speed (box-widths per second), because a service bay looks like the slow tertile, not like a highway.
"""
from __future__ import annotations

import argparse
import pickle
import time
from collections import defaultdict

import numpy as np

from eval.stats import metric
from lanes.flow import mot_metrics as MM
from lanes.flow.common import component, save_results
from lanes.flow.data import detrac_items
from lanes.flow.eval_detect import filter_ignored, vehicle_preds
from lanes.flow.run_detect import cache_path
from lanes.flow.tracker import make_tracker

FPS_STEPS = {5.0: 1, 1.0: 5, 0.5: 10}  # fps -> how many of the cached 5-fps frames to skip


def load_sequences(model: str, ign: bool = True):
    items = detrac_items("test", 5)
    dets = pickle.load(open(cache_path("detrac_test_s5", model), "rb"))["dets"]
    seqs: dict[str, list] = defaultdict(list)
    for it in items:
        d = vehicle_preds(dets[it.id], "vehicle")
        if ign:
            d = filter_ignored(d, it.ignore)
        tid = np.asarray(it.extra["track_id"], dtype=int)
        keep = tid >= 0
        seqs[it.cluster].append({"frame": it.extra["frame"], "gt": it.gt[keep], "gt_ids": tid[keep], "dets": d})
    for s in seqs.values():
        s.sort(key=lambda r: r["frame"])
    return seqs


def gt_speed(frames: list[dict]) -> float:
    """Median over consecutive-frame GT track steps of centre displacement / box width, per second (frames are 0.2 s apart)."""
    last: dict[int, np.ndarray] = {}
    sp = []
    for r in frames:
        cur = {}
        for b, i in zip(r["gt"], r["gt_ids"]):
            c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(b[2] - b[0], 1.0)])
            cur[int(i)] = c
            if int(i) in last:
                sp.append(np.linalg.norm(c[:2] - last[int(i)][:2]) / last[int(i)][2] / 0.2)
        last = cur
    return float(np.median(sp)) if sp else 0.0


def run_seq(frames: list[dict], tracker, step: int):
    sub = frames[::step]
    tracker.reset()
    gt, pr = {}, {}
    t_total = 0.0
    for r in sub:
        t0 = time.perf_counter()
        out = tracker.update(r["dets"][r["dets"][:, 4] >= 0.1] if len(r["dets"]) else np.zeros((0, 5)), r["frame"] / 25.0)
        t_total += time.perf_counter() - t0
        gt[r["frame"]] = (r["gt_ids"], r["gt"])
        pr[r["frame"]] = (out[:, 4].astype(int) if len(out) else np.zeros(0, int), out[:, :4] if len(out) else np.zeros((0, 4)))
    return MM.seq_stats(gt, pr), t_total / max(len(sub), 1) * 1000


def run(model: str, exp: str, det_thresh: float = 0.35, n_boot: int = 300, trackers=("iou", "bytetrack", "ocsort", "ocsort_nb")):
    seqs = load_sequences(model)
    speeds = {s: gt_speed(f) for s, f in seqs.items()}
    order = sorted(speeds, key=speeds.get)
    third = len(order) // 3
    groups = {"slow": order[:third], "mid": order[third:2 * third], "fast": order[2 * third:], "all": order}
    print("speed tertile cut (box-widths/s):", speeds[order[third]], speeds[order[2 * third]])
    results, rows = [], []
    for fps, step in FPS_STEPS.items():
        for tn in trackers:
            per_seq, ms_upd = {}, []
            for s, fr in seqs.items():
                trk = make_tracker(tn.replace("_nb", ""), fps=fps, det_thresh=det_thresh, max_age_s=3.0 if fps <= 1 else 2.0, **({"use_byte": False} if tn == "ocsort_nb" else {}))
                per_seq[s], t = run_seq(fr, trk, step)
                ms_upd.append(t)
            for g, names in groups.items():
                bs = MM.bootstrap({n: per_seq[n] for n in names}, n_boot=n_boot)
                pooled = MM.pool([per_seq[n] for n in names])
                ms = [metric(k, v[0], v[1], v[2], "fraction") for k, v in bs.items()]
                ms += [metric("IDSW_per_1k_gt", pooled["IDSW_per_1k_gt"], unit="per 1000 GT boxes"), metric("IDSW", pooled["IDSW"], unit="count"),
                       metric("LocA", pooled["LocA"], unit="fraction"), metric("tracker_ms_per_frame", float(np.mean(ms_upd)), unit="ms")]
                results.append(component(f"flow.track.{model}.{tn}.{fps:g}fps.{g}", f"Tracking {tn} @ {fps:g} fps ({g} sequences)", "UA-DETRAC test",
                                         int(pooled["n_gt"]), ms, split=g, model=f"{model}+{tn}", domain="fixed CCTV",
                                         notes=f"{len(names)} sequences, detections {model} (score>={det_thresh}); {fps:g} fps = every {int(25 / fps)}th source frame; "
                                               f"speed tertile {g}"))
                rows.append((fps, tn, g, bs, pooled))
    p = save_results(exp, results)
    md = "| fps | tracker | sequences | HOTA | IDF1 | MOTA | AssA | IDSW/1k GT |\n|---|---|---|---|---|---|---|---|\n"
    for fps, tn, g, bs, pooled in rows:
        f = lambda k: f"{bs[k][0]:.3f} [{bs[k][1]:.2f},{bs[k][2]:.2f}]"  # noqa: E731
        md += f"| {fps:g} | {tn} | {g} | {f('HOTA')} | {f('IDF1')} | {f('MOTA')} | {f('AssA')} | {pooled['IDSW_per_1k_gt']:.1f} |\n"
    (p.parent / "table.md").write_text(md)
    (p.parent / "sequence_speed.json").write_text(__import__("json").dumps(speeds, indent=1))
    print(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--det-thresh", type=float, default=0.35)
    ap.add_argument("--n-boot", type=int, default=300)
    a = ap.parse_args()
    run(a.model, a.exp, a.det_thresh, a.n_boot)
