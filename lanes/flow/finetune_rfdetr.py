"""Fine-tune RF-DETR on the lane's mixture (UA-DETRAC train + NDISPark day + COCO persons) built by prep_finetune.py.

    scripts/box.sh --gpu 0 --lock --bg flow_ft -- python -m lanes.flow.finetune_rfdetr --size base --epochs 10 --name ft_base
Writes runs/flow/<name>/{checkpoint_best_total.pth, classes.json, train_files.txt} (train_files.txt feeds Lane G's leakage audit).
"""
import argparse
import json
import shutil
from pathlib import Path

from lanes.flow.common import CLASSES, artifact_dir, fast_prep, run_dir
from lanes.flow.detectors import RF_WEIGHTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base", choices=["nano", "small", "medium", "base", "large"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--data", default="ft")
    ap.add_argument("--name", default="ft_base")
    a = ap.parse_args()
    import rfdetr

    data = fast_prep(a.data)
    out = run_dir(a.name)
    cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium, "base": rfdetr.RFDETRBase, "large": rfdetr.RFDETRLarge}[a.size]
    model = cls(num_classes=len(CLASSES), pretrain_weights=str(artifact_dir("pretrained") / RF_WEIGHTS[a.size]))
    (out / "classes.json").write_text(json.dumps(list(CLASSES)))
    shutil.copy(data / "train_files.txt", out / "train_files.txt")
    shutil.copy(data / "summary.json", out / "data_summary.json")
    model.train(dataset_dir=str(data), epochs=a.epochs, batch_size=a.batch, grad_accum_steps=a.accum, lr=a.lr, output_dir=str(out),
                num_workers=a.workers, early_stopping=False)
    print("done", sorted(p.name for p in Path(out).glob("*.pth")))


if __name__ == "__main__":
    main()
