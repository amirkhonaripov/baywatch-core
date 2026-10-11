"""Fine-tune an RF-DETR (Apache-2.0) plate detector on the assembled YOLO dataset (data/prep -> $SEEN_FAST/ident/det/<variant>).

  lanes/ident/tools/bgbox.sh det_full --gpu 0 --lock -- python -m lanes.ident.train_det --variant full --model small --epochs 25

Variants (ablation): uz_only (real Uzbek train split) / full (+ real foreign plates from CCPD+RodoSol train parts + synthetic composites).
Outputs: runs/ident/det_<variant>_<model>/ (checkpoints, logs), exported ONNX in .../onnx/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from seen.paths import RUNS, fast_dir

FAST = fast_dir("ident")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["uz_only", "full", "smoke"], default="full")
    ap.add_argument("--model", choices=["nano", "small", "medium"], default="small")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--res", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    name = a.name or f"det_{a.variant}_{a.model}"
    out = RUNS / "ident" / name
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)  # RF-DETR drops the pretrained .pth in the CWD
    import rfdetr

    cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium}[a.model]
    model = cls(resolution=a.res) if a.res else cls()
    ds = FAST / "det" / a.variant
    model.train(dataset_dir=str(ds), epochs=a.epochs, batch_size=a.batch, grad_accum_steps=max(1, 16 // a.batch), lr=a.lr,
                output_dir=str(out / "rfdetr"), num_workers=a.workers, device="cuda:0", early_stopping=True, early_stopping_patience=8)
    (out / "train_args.json").write_text(json.dumps(vars(a)))
    print("DONE", out)


if __name__ == "__main__":
    main()
