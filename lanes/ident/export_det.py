"""Export a trained RF-DETR plate detector run to artifacts/ident/plate_det/ (ONNX + model_card.json).

  python -m lanes.ident.export_det --run det_full_small [--version 0.1.0] [--res 640]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from seen.paths import ARTIFACTS, RUNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="plate_det")
    ap.add_argument("--model", default="small")
    ap.add_argument("--res", type=int, default=640)
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--metrics-ref", default="docs/lanes/ident_results.json")
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()
    import rfdetr

    run = RUNS / "ident" / a.run
    ck = Path(a.ckpt) if a.ckpt else next(iter(sorted((run / "rfdetr").glob("checkpoint_best_total.pth"))), None)
    assert ck and ck.exists(), f"no best checkpoint under {run}/rfdetr"
    cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium}[a.model]
    model = cls(pretrain_weights=str(ck), resolution=a.res)
    tmp = run / "onnx"
    tmp.mkdir(exist_ok=True)
    path = model.export(output_dir=str(tmp), simplify=False) if "simplify" in model.export.__code__.co_varnames else model.export(output_dir=str(tmp))
    onnx = Path(path) if path else next(iter(sorted(tmp.glob("*.onnx"))))
    out = ARTIFACTS / "ident" / a.out
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx, out / "model.onnx")
    args = json.loads((run / "train_args.json").read_text()) if (run / "train_args.json").exists() else {}

    card = {"name": "ident-plate-det", "version": a.version, "arch": "rfdetr", "rfdetr_variant": a.model, "classes": ["plate"], "class_index": 0,
            "input_size": a.res, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "input": "float32 NCHW RGB, plain resize to input_size x input_size (no letter-boxing), ImageNet mean/std",
            "outputs": {"dets": "(1, Q, 4) cx,cy,w,h normalised to the input", "labels": "(1, Q, C) class logits; score = sigmoid(logit[:, class_index])"},
            "metrics_ref": a.metrics_ref, "training_args": args,
            "training_data": "see docs/lanes/ident_notes.md: real Uzbek (Roboflow Uzbek_Plates train split) [+ CCPD/RodoSol train parts + synthetic composites for the 'full' variant]",
            "licence_notes": "RF-DETR Apache-2.0 (code + COCO-pretrained initialisation). Training images: Roboflow Uzbek_Plates CC BY 4.0; CCPD (MIT); RodoSol-ALPR (academic, non-commercial); "
                             "synthetic composites built on those photos. Research artefact: RodoSol terms restrict commercial use of derived weights.",
            "source_run": str(run)}
    (out / "model_card.json").write_text(json.dumps(card, indent=1))
    import onnxruntime as ort
    s = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])
    print("exported", out, [(i.name, i.shape) for i in s.get_inputs()], [(o.name, o.shape) for o in s.get_outputs()])


if __name__ == "__main__":
    main()
