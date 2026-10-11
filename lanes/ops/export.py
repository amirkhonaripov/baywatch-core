"""Package a trained head + backbone (+ optional part detectors) into artifacts/ops/states and measure latency.

  python -m lanes.ops.export --exp heads_v1 [--onnx-backbone] [--dest runs|artifacts]
Writes model.onnx (= head), head.npz, backbone_hf/, optional backbone.onnx, part detectors, model_card.json,
speed.json (GPU latency per frame; run on GPU1).
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from lanes.ops.backbones import BACKBONES, load_backbone
from lanes.ops.geometry import GEOM_NAMES
from lanes.ops.heads import HeadModel, to_onnx
from seen.paths import ARTIFACTS, RUNS
from seen.vocab import BAY_STATES


def _percentiles(v):
    v = np.asarray(v)
    return dict(p50_ms=float(np.percentile(v, 50)), p95_ms=float(np.percentile(v, 95)), n=int(len(v)))


def export(exp: str, dest: Path, onnx_backbone: bool, link_backbone: bool = False) -> None:
    run = RUNS / "ops" / exp
    dest.mkdir(parents=True, exist_ok=True)
    m = json.loads((run / "metrics.json").read_text())
    head = HeadModel.load(run / "head.npz")
    shutil.copy(run / "head.npz", dest / "head.npz")
    to_onnx(head, dest / "head.onnx")
    shutil.copy(dest / "head.onnx", dest / "model.onnx")
    # backbone weights copied next to the head so the artifact is self-contained
    from huggingface_hub import snapshot_download

    src = Path(snapshot_download(BACKBONES[m["backbone"]]["hf"], allow_patterns=["*.json", "*.safetensors"]))
    bdst = dest / "backbone_hf"
    if not bdst.exists():
        if link_backbone:  # working copies under runs/: symlink to the HF snapshot (the HDD is IO-bound)
            bdst.symlink_to(src, target_is_directory=True)
        else:  # deliverable under artifacts/: real copy so the artifact is self-contained
            shutil.copytree(src, bdst, symlinks=False)
    parts = {}
    for key, name in (("parts_detector", "parts_yolo"), ("hood_detector", "hood_yolo")):
        w = RUNS / "ops" / name / "weights" / "best.pt"
        if w.exists() and m.get("used_parts", False):
            shutil.copy(w, dest / f"{name}.pt")
            parts[key] = f"{name}.pt"
    card = dict(
        name="ops-bay-states", version="0.1", classes=list(BAY_STATES), head_states=head.states,
        backbone=m["backbone"], input_size=BACKBONES[m["backbone"]]["size"],
        features=f"[{m['backbone']} CLS+meanpatch embedding of the car crop (box +25 %), {len(GEOM_NAMES)} geometry cues]",
        geometry_features=GEOM_NAMES,
        thresholds={s: 0.5 for s in BAY_STATES},
        derived={"car_present": "detector confidence of the primary vehicle (bay ROI if given)",
                 "idle": "car_present * (1 - work)",
                 "person_under_car": "head probability gated by raised (x min(1, raised/0.5))"},
        calibration="Platt scaling per state on a held-out handheld/YouTube validation split (by video)",
        metrics_ref=f"docs/lanes/ops_results.json ; runs/ops/{exp}/",
        training_data="Kinetics-700 service + car-context classes (HF mirror clips), CrossTask/COIN car-task videos "
                      "(step-span labels), Roboflow car-hood-open / car-open-hood / washing-car, carparts-seg, Wikimedia "
                      "Commons lift photos (hand-labelled), Qwen3-VL-8B teacher labels where no human/derived label exists",
        licence_notes="Kinetics/CrossTask/COIN videos remain YouTube content (research use); COIN annotations CC BY-NC 4.0; "
                      "Roboflow sets CC BY 4.0 (uploader label); Commons CC BY-SA / PD per file; DINOv2 Apache-2.0. "
                      "Non-commercial research prototype.",
        domain_caveat="trained on handheld/YouTube video and web photos, not on service-bay CCTV",
        **parts,
    )
    (dest / "model_card.json").write_text(json.dumps(card, indent=1))
    if onnx_backbone:
        bb = load_backbone(m["backbone"], "cpu", dtype=torch.float32, path=str(bdst))

        class Wrap(torch.nn.Module):
            def __init__(self, mod):
                super().__init__()
                self.mod = mod

            def forward(self, x):
                o = self.mod(pixel_values=x).last_hidden_state
                return torch.cat([o[:, 0], o[:, 1:].mean(1)], dim=1)

        w = Wrap(bb.model).eval()
        x = torch.randn(1, 3, bb.size, bb.size)
        torch.onnx.export(w, x, str(dest / "backbone.onnx"), input_names=["pixels"], output_names=["embedding"],
                          dynamic_axes={"pixels": {0: "n"}, "embedding": {0: "n"}}, opset_version=17, dynamo=False)
        import onnxruntime as ort

        s = ort.InferenceSession(str(dest / "backbone.onnx"), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            ref = w(x).numpy()
        got = s.run(None, {"pixels": x.numpy()})[0]
        cos = float((ref * got).sum() / np.linalg.norm(ref) / np.linalg.norm(got))
        card["backbone_onnx"] = {"file": "backbone.onnx", "parity_cosine_vs_torch": cos}
        (dest / "model_card.json").write_text(json.dumps(card, indent=1))
        print("backbone onnx parity cosine", cos)
    print("EXPORTED", dest)


def speed(dest: Path, n: int = 200) -> None:
    """Latency per frame on the current GPU: backbone (torch bf16, batch 1 and 16), head (ONNX), backbone ONNX (CUDA EP)."""
    card = json.loads((dest / "model_card.json").read_text())
    dev = "cuda:0"
    bb = load_backbone(card["backbone"], dev, path=str(dest / "backbone_hf"))
    size = bb.size
    out = []
    for bs in (1, 16):
        x = torch.randn(bs, 3, size, size, device=dev)
        with torch.inference_mode():
            for _ in range(10):
                bb.forward(x)
            torch.cuda.synchronize()
            ts = []
            for _ in range(n // bs + 5):
                t0 = time.perf_counter()
                bb.forward(x)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000 / bs)
        out.append(dict(stage=f"ops backbone {card['backbone']} (per frame)", runtime="torch bf16", batch=bs, **_percentiles(ts)))
    import onnxruntime as ort

    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    s = ort.InferenceSession(str(dest / "head.onnx"), providers=prov)
    D = s.get_inputs()[0].shape[1]
    xh = np.random.randn(1, D).astype(np.float32)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        s.run(None, {"features": xh})
        ts.append((time.perf_counter() - t0) * 1000)
    out.append(dict(stage="ops linear head (ONNX)", runtime=f"onnxruntime {s.get_providers()[0]}", batch=1, **_percentiles(ts)))
    if (dest / "backbone.onnx").exists():
        sb = ort.InferenceSession(str(dest / "backbone.onnx"), providers=prov)
        xb = np.random.randn(1, 3, size, size).astype(np.float32)
        for _ in range(5):
            sb.run(None, {"pixels": xb})
        ts = []
        for _ in range(50):
            t0 = time.perf_counter()
            sb.run(None, {"pixels": xb})
            ts.append((time.perf_counter() - t0) * 1000)
        out.append(dict(stage="ops backbone (ONNX fp32)", runtime=f"onnxruntime {sb.get_providers()[0]}", batch=1, **_percentiles(ts)))
    hw = torch.cuda.get_device_name(0)
    for o in out:
        o["hardware"] = hw
    (dest / "speed.json").write_text(json.dumps(out, indent=1))
    for o in out:
        print(o)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--dest", choices=["runs", "artifacts"], default="runs")
    ap.add_argument("--onnx-backbone", action="store_true")
    ap.add_argument("--speed", action="store_true")
    a = ap.parse_args()
    dest = (RUNS / "ops" / a.exp / "artifact") if a.dest == "runs" else (ARTIFACTS / "ops" / "states")
    export(a.exp, dest, a.onnx_backbone, link_backbone=a.dest == "runs")
    if a.speed:
        speed(dest)
