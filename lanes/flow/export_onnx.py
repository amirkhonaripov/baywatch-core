"""ONNX export into artifacts/flow/<name>/ with model_card.json.

    python -m lanes.flow.export_onnx detector --spec rfdetr-base --name rfdetr-base-coco
    python -m lanes.flow.export_onnx detector --spec rfdetr-base@/path/checkpoint_best_total.pth --name rfdetr-base-flow --metrics-ref docs/lanes/flow_results.json
    python -m lanes.flow.export_onnx router --model google/siglip2-so400m-patch14-384
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from lanes.flow.common import CLASSES, artifact_dir
from lanes.flow.detectors import RFDETR, get_detector


def export_detector(spec: str, name: str, metrics_ref: str = "", training_data: str = "", licence_notes: str = "", device: str = "cuda:0"):
    det = get_detector(spec)
    assert isinstance(det, RFDETR), "ONNX export is implemented for RF-DETR (Apache-2.0); YOLO weights are AGPL and not shipped"
    det.load(device)
    out = artifact_dir(name)
    tmp = out / "export_tmp"
    res = det.model.model_config.resolution
    det.model.export(output_dir=str(tmp), shape=(res, res), batch_size=1, opset_version=17)
    cands = sorted(tmp.glob("*.onnx"))
    assert cands, f"no onnx produced in {tmp}: {list(tmp.iterdir())}"
    shutil.copy(cands[0], out / "model.onnx")
    shutil.rmtree(tmp, ignore_errors=True)
    classes = json.loads(Path(det.checkpoint).with_name("classes.json").read_text()) if det.checkpoint and Path(det.checkpoint).with_name("classes.json").exists() \
        else ["coco-91 ids; mapped to " + ",".join(CLASSES)]
    card = {"name": name, "version": "1", "classes": classes, "input_size": [res, res], "metrics_ref": metrics_ref, "training_data": training_data or "COCO (pretrained, not fine-tuned by us)",
            "licence_notes": licence_notes or "RF-DETR code and weights Apache-2.0", "weights": "model.onnx", "size": det.size, "resolution": res, "spec": spec}
    (out / "model_card.json").write_text(json.dumps(card, indent=1))
    if det.checkpoint:
        shutil.copy(det.checkpoint, out / "checkpoint.pth")
    print("exported", out / "model.onnx", (out / "model.onnx").stat().st_size / 1e6, "MB")


def export_router(model: str, device: str = "cuda:0", opset: int = 17):
    import torch

    from lanes.flow.router_model import SigLIP2Encoder

    enc = SigLIP2Encoder(model, device, dtype="float32")
    m = enc.model

    class ImageFeat(torch.nn.Module):
        def __init__(self, mm):
            super().__init__()
            self.m = mm

        def forward(self, pixel_values):
            f = self.m.get_image_features(pixel_values=pixel_values)
            if not torch.is_tensor(f):
                f = getattr(f, "pooler_output", None) if getattr(f, "pooler_output", None) is not None else f[0]
            return torch.nn.functional.normalize(f, dim=-1)

    size = enc.proc.image_processor.size
    h = size.get("height", 384) if isinstance(size, dict) else 384
    x = torch.randn(1, 3, h, h, device=device)
    out = artifact_dir("router")
    path = out / "encoder.onnx"
    torch.onnx.export(ImageFeat(m).eval(), (x,), str(path), input_names=["pixel_values"], output_names=["embedding"], opset_version=opset,
                      dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}}, dynamo=False)
    print("exported", path, path.stat().st_size / 1e6, "MB; input", tuple(x.shape))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["detector", "router"])
    ap.add_argument("--spec", default="rfdetr-base")
    ap.add_argument("--name", default="rfdetr-base-coco")
    ap.add_argument("--model", default="google/siglip2-so400m-patch14-384")
    ap.add_argument("--metrics-ref", default="")
    ap.add_argument("--training-data", default="")
    a = ap.parse_args()
    if a.what == "detector":
        export_detector(a.spec, a.name, a.metrics_ref, a.training_data)
    else:
        export_router(a.model)
