"""Latency of the exported models (ONNX Runtime, CUDA EP with CPU fallback): per-crop OCR (batch 1), per-image detection, whole predictor per frame.

  scripts/box.sh --gpu 1 -- python -m lanes.ident.bench_latency --exp final --n 300
Writes SpeedResult entries (contracts/metrics.schema.json) to runs/ident/<exp>/speed_results.json. Wall-clock around the ORT call including
host pre/post-processing; warm-up excluded; the box is shared, so numbers are indicative (GPU and CPU were not idle).
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import cv2
import numpy as np

from lanes.ident.data.pack import load_test_pack
from lanes.ident.ocr_onnx import PlateOcrOnnx, default_providers
from lanes.ident.predict import DET_DIR, OCR_DIR, OnnxPlateDetector, get_predictor
from lanes.ident.reader import PlateReader
from seen.paths import RUNS, fast_dir
from seen.predictor import Frame


def pct(v):
    v = np.asarray(v)
    return float(np.percentile(v, 50)), float(np.percentile(v, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="final")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out = []
    hw = f"{platform.node()} {'GPU1 (RTX PRO 5000 Blackwell, shared)' if a.device.startswith('cuda') else 'CPU'}"
    imgs, _, _, _ = load_test_pack("test_uz_real")
    imgs = imgs[: a.n]
    for dev in (a.device, "cpu"):
        ocr = PlateOcrOnnx(OCR_DIR / "model.onnx", OCR_DIR / "plate_config.yaml", providers=default_providers(dev))
        for mode, orient in (("argmax", False), ("auto", True)):
            rd = PlateReader(ocr, mode=mode, orient=orient)
            for c in imgs[:10]:
                rd.read([c])
            lat = []
            for c in imgs:
                t = time.perf_counter()
                rd.read([c])
                lat.append((time.perf_counter() - t) * 1000)
            p50, p95 = pct(lat)
            out.append({"stage": f"plate OCR per crop (decode={mode}, rotation search={orient})", "hardware": hw if dev != "cpu" else "CPU (shared box, 32 threads)", "runtime": f"onnxruntime {'CUDA EP' if dev != 'cpu' else 'CPU EP'}",
                        "batch": 1, "p50_ms": p50, "p95_ms": p95, "fps": 1000 / p50, "notes": f"{len(imgs)} real-UZ test crops; ONNX has a static batch of {ocr.static_b}"})
    ims = [cv2.imread(str(p)) for p in sorted((fast_dir("ident") / "uz_rf" / "valid" / "images").glob("*.jpg"))[: a.n]]
    for dev in (a.device, "cpu"):
        det = OnnxPlateDetector(DET_DIR, device=dev, conf=0.25)
        for im in ims[:5]:
            det.detect([im])
        lat = []
        for im in ims:
            t = time.perf_counter()
            det.detect([im])
            lat.append((time.perf_counter() - t) * 1000)
        p50, p95 = pct(lat)
        out.append({"stage": f"plate detector per image ({det.card.get('rfdetr_variant', '')} {det.size}px)", "hardware": hw if dev != "cpu" else "CPU (shared box, 32 threads)",
                    "runtime": f"onnxruntime {'CUDA EP' if dev != 'cpu' else 'CPU EP'}", "batch": 1, "p50_ms": p50, "p95_ms": p95, "fps": 1000 / p50, "notes": "640x640 photos, includes resize + decode of boxes"})
    for dev in (a.device, "cpu"):
        p = get_predictor()
        p.load(dev)
        frames = [Frame(i, i / 10, cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for i, im in enumerate(ims[:60])]
        p.predict(frames[:3])
        p.reset()
        lat = []
        for f in frames:
            t = time.perf_counter()
            p.predict([f])
            lat.append((time.perf_counter() - t) * 1000)
        p50, p95 = pct(lat)
        out.append({"stage": "ident predictor per frame (detect + read + vote)", "hardware": hw if dev != "cpu" else "CPU (shared box, 32 threads)", "runtime": f"onnxruntime {'CUDA EP' if dev != 'cpu' else 'CPU EP'}",
                    "batch": 1, "p50_ms": p50, "p95_ms": p95, "fps": 1000 / p50, "notes": "frames = real-UZ test photos (one plate each)"})
    d = RUNS / "ident" / a.exp
    d.mkdir(parents=True, exist_ok=True)
    (d / "speed_results.json").write_text(json.dumps(out, indent=1))
    for o in out:
        print(f"{o['stage'][:70]:70s} {o['runtime']:22s} p50={o['p50_ms']:.1f} ms p95={o['p95_ms']:.1f} ms")


if __name__ == "__main__":
    main()
