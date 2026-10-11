"""Why the default pipeline sees no cars in the renders: COCO RF-DETR on rendered frames at a very low threshold,
matched to the projected ground-truth car boxes (labels.jsonl).

    python -m sim.detector_probe RENDER_DIR [--out runs/sim/detector_probe/out.json]

Reports, per episode, how many ground-truth cars get any box with IoU > 0.3 (threshold 0.05), how many of those
reach the pipeline threshold 0.4, and which COCO classes the boxes carry.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from PIL import Image

from eval.stats import iou_matrix


def main() -> None:
    from rfdetr import RFDETRBase
    from rfdetr.assets.coco_classes import COCO_CLASSES

    from seen.paths import RUNS

    ap = argparse.ArgumentParser()
    ap.add_argument("renders")
    ap.add_argument("--episodes", default="ep00_train,ep02_train,ep06_test,ep08_test")
    ap.add_argument("--per-episode", type=int, default=25)
    ap.add_argument("--out", default=str(RUNS / "sim" / "detector_probe" / "out.json"))
    a = ap.parse_args()
    m = RFDETRBase()
    R = Path(a.renders)
    res = {}
    for ep in a.episodes.split(","):
        labels = {}
        for line in open(R / ep / "labels.jsonl"):
            r = json.loads(line)
            labels[r["frame"]] = r
        frames = sorted(labels)[:: max(1, len(labels) // a.per_episode)][: a.per_episode]
        hits, n = [], 0
        for f in frames:
            im = np.asarray(Image.open(R / ep / "frames" / f"{f:06d}.jpg").convert("RGB"))
            H, W = im.shape[:2]
            det = m.predict(im, threshold=0.05)
            boxes = np.array([[x1 / W, y1 / H, x2 / W, y2 / H] for x1, y1, x2, y2 in det.xyxy]).reshape(-1, 4)
            for d in labels[f]["bays"].values():
                if not d["car"]:
                    continue
                n += 1
                if len(boxes):
                    ious = iou_matrix(np.array([d["car"]]), boxes)[0]
                    j = int(np.argmax(ious))
                    if ious[j] > 0.3:
                        hits.append((COCO_CLASSES.get(int(det.class_id[j]), "?"), float(det.confidence[j])))
        res[ep] = {"gt_cars": n, "localised_iou30": len(hits), "conf_ge_0.4": sum(c >= 0.4 for _, c in hits),
                   "vehicle_class_ge_0.4": sum(c >= 0.4 and k in ("car", "truck", "bus", "motorcycle") for k, c in hits),
                   "classes": collections.Counter(k for k, _ in hits).most_common(6),
                   "median_conf": float(np.median([c for _, c in hits])) if hits else None}
        print(ep, res[ep], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
