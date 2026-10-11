"""Export lane-D models to ONNX in artifacts/inspect/<name>/ with a model_card.json, a parity check and latency numbers.

  python -P -m lanes.inspect.export yolo --name damage_seg --weights <best.pt> --imgsz 1024 --task seg \
        --metrics-ref docs/lanes/inspect_results.json --training-data "..." --licence "..." --parity-set <eval set dir>
  python -P -m lanes.inspect.export tyre --head runs/inspect/tyre_dino_lr/head.npz
  python -P -m lanes.inspect.export latency --names damage_seg,parts_seg,tyre_cond,dot_region,dot_chars     # run on GPU1

Parity: ONNX (our decode in onnx_yolo.py) vs the Ultralytics predictor on the same images (conf 0.25): fraction of PyTorch
detections found by ONNX at box IoU >= 0.7 with the same class, and mean box IoU of the matches.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from eval.stats import iou_matrix, percentiles
from lanes.inspect import common as C
from lanes.inspect.onnx_yolo import YoloOnnx
from seen.paths import ARTIFACTS

ART = ARTIFACTS / "inspect"


def _card(d: Path, **kw) -> None:
    (d / "model_card.json").write_text(json.dumps(kw, indent=1))


def export_yolo(a) -> None:
    from ultralytics import YOLO
    m = YOLO(a.weights)
    names = [m.names[i] for i in sorted(m.names)]
    if a.names_yaml:  # class names by index from the dataset yaml (unquoted 'Y'/'N' were once parsed as booleans in weights)
        import yaml
        nm = yaml.safe_load(Path(a.names_yaml).read_text())["names"]
        names = [str(nm[i]) for i in sorted(nm)] if isinstance(nm, dict) else [str(n) for n in nm]
    d = C.ensure(ART / a.name)
    p = m.export(format="onnx", imgsz=a.imgsz, opset=17, simplify=True, dynamic=False, half=False, device=0)
    shutil.copy(p, d / "model.onnx")
    shutil.copy(a.weights, d / "weights.pt")
    parity = {}
    if a.parity_dir:
        parity = parity_check(m, d / "model.onnx", names, a.imgsz, a.task, Path(a.parity_dir), a.parity_n)
    _card(d, name=a.name, version="0.1.0", classes=names, input_size=a.imgsz, task=a.task, runtime="onnxruntime (CUDA EP, opset 17)",
          metrics_ref=a.metrics_ref, training_data=a.training_data, licence_notes=a.licence, parity_vs_pytorch=parity,
          source_weights=str(a.weights), preprocessing="letterbox to input_size (pad 114), RGB, /255; postprocessing in lanes/inspect/onnx_yolo.py")
    print("exported", d, parity)


def parity_check(model, onnx_path: Path, names, imgsz, task, img_dir: Path, n: int) -> dict:
    sess = YoloOnnx(onnx_path, names, imgsz, task=task)
    files = sorted(img_dir.glob("*.jpg"))[:n]
    found = tot = 0
    ious = []
    for f in files:
        img = C.read_rgb(f)
        r = model.predict(img[:, :, ::-1], imgsz=imgsz, conf=0.25, iou=0.6, device=0, verbose=False)[0]
        dets, _ = sess(img, conf=0.25, iou=0.6, mask_canvas=None)
        if r.boxes is None or not len(r.boxes):
            continue
        pb, pc = r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)
        ob = np.array([d["box"] for d in dets]).reshape(-1, 4)
        oc = np.array([d["cls"] for d in dets])
        iou = iou_matrix(pb, ob) if len(ob) else np.zeros((len(pb), 0))
        for i in range(len(pb)):
            tot += 1
            ok = [(iou[i, j]) for j in range(iou.shape[1]) if oc[j] == pc[i] and iou[i, j] >= 0.7]
            if ok:
                found += 1
                ious.append(max(ok))
    return {"images": len(files), "pytorch_detections": tot, "found_by_onnx_iou0.7_same_class": found,
            "fraction_found": round(found / max(tot, 1), 4), "mean_box_iou_of_matches": round(float(np.mean(ious)) if ious else 0.0, 4)}


def export_tyre(a) -> None:
    """One ONNX with a shared DINOv2 backbone and two heads: p_defective (binary, TyreNet-trained, the validated default) and
    probs3 (good / worn / defective, TyreNet + sameer, experimental)."""
    import torch
    from lanes.inspect import tyre_features as TF
    h = np.load(a.head)
    h3 = np.load(a.head3) if a.head3 else None
    size = int(h["size"])

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bb = TF.make_backbone("cpu", size)
            self.register_buffer("mu", torch.tensor(h["mu"]))
            self.register_buffer("sd", torch.tensor(h["sd"]))
            self.register_buffer("w", torch.tensor(h["w"]))
            self.register_buffer("b", torch.tensor(float(h["b"])))
            if h3 is not None:
                self.register_buffer("mu3", torch.tensor(h3["mu"]))
                self.register_buffer("sd3", torch.tensor(h3["sd"]))
                self.register_buffer("w3", torch.tensor(h3["w"]).T.contiguous())  # [D, 3]
                self.register_buffer("b3", torch.tensor(h3["b"]))

        def forward(self, x):
            f = TF.embed(self.bb, x)
            p2 = torch.sigmoid(((f - self.mu) / self.sd) @ self.w + self.b).reshape(-1, 1)
            if h3 is None:
                return p2
            p3 = torch.softmax(((f - self.mu3) / self.sd3) @ self.w3 + self.b3, dim=1)
            return p2, p3

    net = Net().eval()
    d = C.ensure(ART / "tyre_cond")
    x = torch.randn(1, 3, size, size)
    outs = ["p_defective"] + (["probs3"] if h3 is not None else [])
    torch.onnx.export(net, x, str(d / "model.onnx"), opset_version=17, input_names=["input"], output_names=outs,
                      dynamic_axes={"input": {0: "batch"}, **{o: {0: "batch"} for o in outs}}, dynamo=False)
    import onnxruntime as ort
    s = ort.InferenceSession(str(d / "model.onnx"), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = net(x)
    ref = [r.numpy() for r in (ref if isinstance(ref, tuple) else (ref,))]
    got = s.run(None, {"input": x.numpy()})
    parity = {"max_abs_diff_vs_torch": [float(np.abs(r - g).max()) for r, g in zip(ref, got)]}
    _card(d, name="tyre_cond", version="0.1.0", classes=["good", "defective"], classes3=["good", "worn", "defective"] if h3 is not None else None,
          outputs=outs, input_size=size, task="classification", runtime="onnxruntime", metrics_ref=a.metrics_ref, parity_vs_pytorch=parity,
          training_data="p_defective: TyreNet (Mendeley 32b5vfj6tc, CC BY 4.0) train+val; probs3: TyreNet + Kaggle sameersambhare1 (CC BY-SA 4.0), 'worn' only from sameer",
          licence_notes="DINOv2 ViT-B/14 weights Apache-2.0; TyreNet CC BY 4.0; sameer CC BY-SA 4.0",
          preprocessing=f"shorter side -> {size} (bicubic), centre crop {size}x{size}, RGB, ImageNet mean/std")
    print("exported tyre_cond", parity)


def latency(a) -> None:
    import onnxruntime as ort
    from lanes.inspect.tyre_features import MEAN, STD  # noqa: F401
    out = []
    rng = np.random.default_rng(0)
    img = (rng.random((768, 1024, 3)) * 255).astype(np.uint8)
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    for name in a.names.split(","):
        card = json.loads((ART / name / "model_card.json").read_text())
        if card["task"] == "classification":
            s = ort.InferenceSession(str(ART / name / "model.onnx"), providers=prov)
            x = rng.standard_normal((1, 3, card["input_size"], card["input_size"])).astype(np.float32)
            fn = lambda: s.run(None, {"input": x})  # noqa: E731
        else:
            y = YoloOnnx(ART / name / "model.onnx", card["classes"], card["input_size"], prov, card["task"])
            fn = lambda: y(img, conf=0.25, iou=0.6, mask_canvas=384 if card["task"] == "seg" else None)  # noqa: E731
        for _ in range(a.warmup):
            fn()
        ts = []
        for _ in range(a.iters):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000)
        pct = percentiles(ts)
        out.append({"stage": f"inspect.{name}", "hardware": a.hardware, "runtime": f"onnxruntime {ort.__version__} CUDA EP (providers {prov[0]})",
                    "batch": 1, "p50_ms": pct["p50"], "p95_ms": pct["p95"], "fps": 1000.0 / pct["p50"],
                    "notes": f"per image incl. letterbox + decode + mask reconstruction; input {card['input_size']}; n={a.iters} after {a.warmup} warm-up; "
                             f"box shared with other lanes (load not controlled)"})
        print(out[-1])
    C.write_json(out, C.RUNS_I / "speed_results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    y = sub.add_parser("yolo")
    y.add_argument("--name", required=True); y.add_argument("--weights", required=True); y.add_argument("--imgsz", type=int, required=True)
    y.add_argument("--task", choices=["seg", "det"], required=True); y.add_argument("--metrics-ref", default="docs/lanes/inspect_results.json")
    y.add_argument("--training-data", default=""); y.add_argument("--licence", default="")
    y.add_argument("--names-yaml", default=""); y.add_argument("--parity-dir", default=""); y.add_argument("--parity-n", type=int, default=60)
    t = sub.add_parser("tyre")
    t.add_argument("--head", required=True); t.add_argument("--head3", default=""); t.add_argument("--metrics-ref", default="docs/lanes/inspect_results.json")
    l = sub.add_parser("latency")
    l.add_argument("--names", required=True); l.add_argument("--iters", type=int, default=100); l.add_argument("--warmup", type=int, default=10)
    l.add_argument("--hardware", default="RTX PRO 5000 Blackwell (GPU1, shared box)")
    a = ap.parse_args()
    {"yolo": export_yolo, "tyre": export_tyre, "latency": latency}[a.cmd](a)
