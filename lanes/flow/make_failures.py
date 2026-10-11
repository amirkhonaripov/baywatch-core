"""Failure-case gallery for detection: worst frames per condition (most missed GT vehicles at score>=conf), drawn with GT (green), hits (cyan),
misses (red, thick) and false positives (yellow).

    python -m lanes.flow.make_failures --dataset detrac_test_s5 --model rfdetr-base --out runs/flow/failures_det --per-cond 3
    python -m lanes.flow.make_failures --dataset ndispark_validation --model rfdetr-base --out ... --per-cond 4
Writes <out>/<cond>_<k>.jpg and failures.json ([{id, component, file, description, category}]). Faces are not visible in these datasets; every
image still passes through seen.blur.blur_faces with the person boxes the detector found (a no-op for vehicles-only scenes).
"""
import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

from lanes.flow.common import run_dir
from lanes.flow.eval_detect import filter_ignored, vehicle_preds
from lanes.flow.run_detect import cache_path, get_items
from lanes.flow.tracker import iou_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cond", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.4)
    a = ap.parse_args()
    items = get_items(a.dataset)
    dets = pickle.load(open(cache_path(a.dataset, a.model), "rb"))["dets"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = []
    for it in items:
        d = filter_ignored(vehicle_preds(dets[it.id]), it.ignore)
        d = d[d[:, 4] >= a.conf]
        iou = iou_batch(it.gt, d[:, :4]) if len(d) and len(it.gt) else np.zeros((len(it.gt), len(d)))
        hit = iou.max(1) >= 0.5 if len(d) and len(it.gt) else np.zeros(len(it.gt), bool)
        fp = (iou.max(0) < 0.5) if len(d) and len(it.gt) else np.ones(len(d), bool)
        stats.append((it, d, hit, fp))
    meta, by_cond = [], {}
    for s in stats:
        by_cond.setdefault(s[0].cond, []).append(s)
    for cond, lst in sorted(by_cond.items()):
        lst.sort(key=lambda s: -(~s[2]).sum())
        picked, seen_clusters = [], set()
        for s in lst:  # at most one frame per sequence/camera so the gallery shows different scenes
            if s[0].cluster in seen_clusters:
                continue
            seen_clusters.add(s[0].cluster)
            picked.append(s)
            if len(picked) >= a.per_cond:
                break
        for k, (it, d, hit, fp) in enumerate(picked):
            im = cv2.imread(it.path)
            sc = 1000 / max(im.shape[:2]) if max(im.shape[:2]) > 1000 else 1.0
            if sc < 1:
                im = cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            for r in it.ignore * sc:
                cv2.rectangle(im, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (90, 90, 90), 1)
            for b, h in zip(it.gt * sc, hit):
                cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 0) if h else (0, 0, 255), 1 if h else 3)
            for b, f in zip(d[:, :4] * sc, fp):
                if f:
                    cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 1)
            txt = f"{cond}  {it.cluster}  GT {len(it.gt)}  missed {(~hit).sum()}  FP {fp.sum()}  (conf>={a.conf})"
            cv2.putText(im, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(im, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            fn = f"{a.dataset}_{a.model}_{cond}_{k}.jpg"
            cv2.imwrite(str(out / fn), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
            meta.append({"id": f"flow_det_{a.dataset}_{cond}_{k}", "component": f"flow.det.{a.dataset}.{a.model}.{cond}", "file": fn,
                         "description": f"{a.model} on {it.cluster} ({cond}): {int((~hit).sum())} of {len(it.gt)} GT vehicles missed at IoU 0.5 / score >= {a.conf}, {int(fp.sum())} false positives. "
                                        "Red = missed GT, cyan = detected GT, yellow = false positive, grey = ignored region.",
                         "category": f"{cond} misses"})
    (out / "failures.json").write_text(json.dumps(meta, indent=1))
    print(len(meta), "images in", out)


if __name__ == "__main__":
    main()
