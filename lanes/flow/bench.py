"""Latency on GPU1: PyTorch vs ONNX Runtime (CUDA EP) for the detector and the router encoder.

    scripts/box.sh --gpu 1 -- python -m lanes.flow.bench detector --spec rfdetr-base --onnx artifacts/flow/rfdetr-base-coco/model.onnx --image <jpg>
    scripts/box.sh --gpu 1 -- python -m lanes.flow.bench router --model google/siglip2-so400m-patch14-384

Writes runs/flow/bench/speed_results.json (``SpeedResult`` entries of contracts/metrics.schema.json). Protocol: batch 1, one real 960x540 frame,
30 warm-up + 200 timed iterations, CUDA synchronised; "end-to-end" includes pre-processing, forward and post-processing (NMS), image already decoded.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from lanes.flow.common import run_dir
from lanes.flow.data import load_rgb


def timeit(fn, n=200, warm=30, sync=None):
    for _ in range(warm):
        fn()
    if sync:
        sync()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)), "fps": float(1000 / np.mean(ts))}


def gpu_name():
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def bench_detector(spec: str, onnx: str | None, image: str, device: str):
    import torch

    from lanes.flow.detectors import get_detector
    from lanes.flow.onnx_runtime import OnnxRFDETR

    im = load_rgb(image)
    out = []
    sync = torch.cuda.synchronize if torch.cuda.is_available() else None
    d = get_detector(spec)
    d.load(device)
    r = timeit(lambda: d.infer([im], 0.3), sync=sync)
    out.append({"stage": f"vehicle_detector:{spec}", "hardware": gpu_name(), "runtime": ("pytorch eager (ultralytics, fp16)" if spec.startswith("yolo") else "pytorch eager (rfdetr)"), "batch": 1, **r,
                "notes": f"end-to-end incl. pre/post-processing, input {im.shape[1]}x{im.shape[0]}"})
    if onnx:
        o = OnnxRFDETR(onnx)
        t0 = time.perf_counter()
        o.load(device)
        cold = time.perf_counter() - t0
        r = timeit(lambda: o.infer([im], 0.3), sync=None)
        out.append({"stage": f"vehicle_detector:{spec}", "hardware": gpu_name(), "runtime": f"onnxruntime {o.active[0]}", "batch": 1, **r, "cold_start_s": cold,
                    "notes": "end-to-end incl. numpy pre/post-processing"})
        x = o.preprocess(im)
        r = timeit(lambda: o.run_raw(x), sync=None)
        out.append({"stage": f"vehicle_detector:{spec}:forward-only", "hardware": gpu_name(), "runtime": f"onnxruntime {o.active[0]}", "batch": 1, **r, "notes": "session.run only"})
        # parity: same detections as PyTorch?
        a = d.infer([im], 0.3)[0]
        b = o.infer([im], 0.3)[0]
        out.append({"stage": "parity", "hardware": gpu_name(), "runtime": "onnx vs torch", "p50_ms": 0.0,
                    "notes": json.dumps({"n_torch": int(len(a)), "n_onnx": int(len(b)), "top5_score_diff": float(np.abs(np.sort(a[:, 4])[::-1][:5] - np.sort(b[:, 4])[::-1][:5]).max()) if len(a) >= 5 and len(b) >= 5 else None})})
    return out


def bench_router(model: str, image: str, device: str, onnx: str | None = None):
    import torch

    from lanes.flow.onnx_runtime import providers
    from lanes.flow.router_model import SigLIP2Encoder

    im = load_rgb(image)
    out = []
    for dtype in ("float16", "float32"):
        enc = SigLIP2Encoder(model, device, dtype=dtype)
        r = timeit(lambda: enc.encode_images([im]), sync=torch.cuda.synchronize)
        out.append({"stage": "router", "hardware": gpu_name(), "runtime": f"pytorch {dtype}, end-to-end (HF processor + encoder)", "batch": 1, **r, "notes": f"{model}, image already decoded"})
        inp = enc.proc(images=[im], return_tensors="pt")["pixel_values"].to(device, enc.dtype)
        with torch.inference_mode():
            r = timeit(lambda: enc.model.get_image_features(pixel_values=inp), sync=torch.cuda.synchronize)
        out.append({"stage": "router:forward-only", "hardware": gpu_name(), "runtime": f"pytorch {dtype}", "batch": 1, **r, "notes": "encoder only"})
        if dtype == "float32":
            ref = enc.encode_images([im])[0]
            pv = inp.float().cpu().numpy()
        del enc
    if onnx:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(onnx, so, providers=providers(device))
        r = timeit(lambda: sess.run(None, {"pixel_values": pv}), sync=None)
        out.append({"stage": "router:forward-only", "hardware": gpu_name(), "runtime": f"onnxruntime {sess.get_providers()[0]} fp32", "batch": 1, **r, "notes": "session.run only"})
        e = sess.run(None, {"pixel_values": pv})[0][0]
        cos = float(np.dot(ref, e) / (np.linalg.norm(ref) * np.linalg.norm(e)))
        out.append({"stage": "parity", "hardware": gpu_name(), "runtime": "onnx vs torch (router)", "p50_ms": 0.0, "notes": json.dumps({"cosine": cos})})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["detector", "router"])
    ap.add_argument("--spec", default="rfdetr-base")
    ap.add_argument("--onnx")  # detector: model.onnx; router: encoder.onnx
    ap.add_argument("--model", default="google/siglip2-so400m-patch14-384")
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--exp", default="bench")
    a = ap.parse_args()
    res = bench_detector(a.spec, a.onnx, a.image, a.device) if a.what == "detector" else bench_router(a.model, a.image, a.device, a.onnx)
    p = run_dir(a.exp) / f"speed_{a.what}_{a.spec.replace('/', '_') if a.what == 'detector' else 'router'}.json"
    p.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
