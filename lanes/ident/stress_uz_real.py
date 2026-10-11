"""Resolution / blur / JPEG stress test on the REAL Uzbek test crops (the real plates are degraded artificially, so these results are
tagged synthetic-degradation and kept apart from the plain real numbers). Shows how far the reader degrades towards CCTV-like crops.

  python -m lanes.ident.stress_uz_real --exp final --ocr-dir $SEEN_ROOT/artifacts/ident/plate_ocr
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from lanes.ident import evalkit as ek
from lanes.ident.data.pack import load_test_pack
from lanes.ident.ocr_onnx import PlateOcrOnnx, default_providers
from lanes.ident.reader import PlateReader
from seen.paths import RUNS


def degrade(img: np.ndarray, kind: str, v: float) -> np.ndarray:
    if kind == "width":
        h, w = img.shape[:2]
        return cv2.resize(img, (int(v), max(4, int(h * v / w))), interpolation=cv2.INTER_AREA)
    if kind == "blur":
        return cv2.GaussianBlur(img, (0, 0), v)
    if kind == "jpeg":
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(v)])
        return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if kind == "noise":
        rng = np.random.default_rng(0)
        return np.clip(img.astype(np.float32) + rng.normal(0, v, img.shape), 0, 255).astype(np.uint8)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--ocr-dir", required=True)
    ap.add_argument("--mode", default="auto")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    imgs, gts, _, _ = load_test_pack("test_uz_real")
    ocr = PlateOcrOnnx(Path(a.ocr_dir) / "model.onnx", Path(a.ocr_dir) / "plate_config.yaml", providers=default_providers(a.device))
    rd = PlateReader(ocr, mode=a.mode, orient=False)
    comps = []
    plan = [("width", [300, 160, 100, 70, 50, 35]), ("blur", [1.0, 2.0, 3.0]), ("jpeg", [30, 10]), ("noise", [8, 16])]
    for kind, vals in plan:
        for v in vals:
            res = rd.read([degrade(im, kind, v) for im in imgs])
            em = ek.ocr_metrics([r.text for r in res], gts)
            native_w = int(np.median([im.shape[1] for im in imgs]))
            comps.append(ek.component(f"ident.ocr.{a.exp}.stress_uz_real.{kind}{v}.{a.mode}", f"Real-UZ crops degraded: {kind}={v}", "synthetic",
                                      "real Uzbek test crops with artificial degradation (resize to width / gaussian blur sigma / JPEG quality / gaussian noise sigma)",
                                      "Roboflow Uzbek_Plates valid originals + synthetic degradation", "held-out cars", len(gts), a.ocr_dir.rstrip("/").split("/")[-1], em,
                                      notes=f"median native crop width {native_w}px; decoding={a.mode}; labels verified by Claude (AI), pending human spot-check"))
            e = em[0]
            print(f"{kind:6s} {v:6}  EM={e['value']:.3f} [{e['ci_low']:.3f},{e['ci_high']:.3f}]", flush=True)
    ek.write_results(RUNS / "ident" / a.exp / "component_results_stress.json", comps)


if __name__ == "__main__":
    main()
