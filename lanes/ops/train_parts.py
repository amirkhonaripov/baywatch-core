"""Part detectors used as cues: wheel (Ultralytics carparts-seg, boxes from polygons) and hood open/closed
(Roboflow car-hood-open). GPU0 with --lock.

  python -m lanes.ops.train_parts --which parts|hood --epochs 60
"""
from __future__ import annotations

import argparse
import json

import yaml

from seen.paths import FAST, PREP, RAW, RUNS


def _yaml(which: str) -> str:
    out = PREP / "ops" / f"{which}_data.yaml"
    root = FAST / "ops" / "yolo"  # NVMe copy made by `python -m lanes.ops.prep yolo`
    if which == "parts":
        names = yaml.safe_load((RAW / "ops_carparts" / "carparts-seg.yaml").read_text())["names"]
        d = dict(path=str(root / "carparts"), train="images/train", val="images/val", test="images/test", names=names)
    else:
        d = dict(path=str(root / "hood"), train="images/train", val="images/val", test="images/test",
                 names={0: "hood_closed", 1: "hood_open"})
    out.write_text(yaml.safe_dump(d))
    return str(out)


def main() -> None:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["parts", "hood"], required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--model", default="yolo11s.pt")
    a = ap.parse_args()
    import os
    import shutil

    work = FAST / "ops" / "yolo_runs"  # NVMe: per-epoch checkpoints never touch the IO-bound HDD
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)  # pretrained weights are downloaded into the cwd
    data = _yaml(a.which)
    m = YOLO(a.model)
    m.train(data=data, epochs=a.epochs, imgsz=640, batch=32, workers=8, device=0, project=str(work),
            name=f"{a.which}_yolo", exist_ok=True, patience=15, plots=False, task="detect", single_cls=False, amp=True)
    best_p = work / f"{a.which}_yolo" / "weights" / "best.pt"
    final = RUNS / "ops" / f"{a.which}_yolo" / "weights"
    final.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_p, final / "best.pt")  # one write to the HDD at the end
    best = YOLO(str(best_p))
    res = {}
    for split in ("val", "test"):
        r = best.val(data=data, split=split, imgsz=640, batch=32, device=0, plots=False)
        names = r.names
        res[split] = {"map50": float(r.box.map50), "map50_95": float(r.box.map),
                      "per_class_ap50": {names[int(c)]: float(ap50) for c, ap50 in zip(r.box.ap_class_index, r.box.ap50)}}
    (RUNS / "ops" / f"{a.which}_yolo" / "metrics.json").write_text(json.dumps(res, indent=1))
    shutil.rmtree(work / f"{a.which}_yolo", ignore_errors=True)
    print("PARTS_TRAIN_DONE", json.dumps(res)[:500])


if __name__ == "__main__":
    main()
