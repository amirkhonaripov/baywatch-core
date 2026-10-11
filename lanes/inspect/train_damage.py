"""Train an Ultralytics YOLO-seg damage model on the prepared YOLO dataset.

    scripts/box.sh --gpu 0 --lock --bg dmg_A -- python -P -m lanes.inspect.train_damage --exp dmg_A --data <data.yaml>

Licence note (for the deck): Ultralytics code and weights are AGPL-3.0. Fine for this research prototype; a commercial
deployment needs an Ultralytics licence or the RF-DETR-Seg (Apache-2.0) alternative trained by lanes.inspect.train_damage_rfdetr.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from lanes.inspect import common as C
from seen.paths import ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default=str(ROOT / "tools/weights/yolo11m-seg.pt"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--train-files", default="")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--hours", type=float, default=0.0, help="time budget in hours (Ultralytics `time=`; overrides --epochs, LR schedule adapts)")
    ap.add_argument("--aug", default="{}", help="JSON of extra Ultralytics train() overrides, e.g. '{\"fliplr\": 0.0}'")
    a = ap.parse_args()

    from ultralytics import YOLO

    out = C.ensure(C.RUNS_I / a.exp)
    if a.train_files:
        shutil.copy(a.train_files, out / "train_files.txt")  # for eval.manifest leakage audit
    model = YOLO(a.weights)
    extra = json.loads(a.aug)
    if a.hours > 0:
        extra["time"] = a.hours
    model.train(data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, workers=a.workers, device=0,
                project=str(C.RUNS_I), name=a.exp + "_yolo", exist_ok=True, patience=a.patience, cos_lr=True,
                close_mosaic=5, amp=True, plots=True, seed=0, fraction=a.fraction, cache=False, **extra)
    print("done", out)


if __name__ == "__main__":
    main()
