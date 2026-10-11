"""End-to-end (detector + OCR) evaluation on the real Uzbek test images: a GT plate counts as correct only if a detection with IoU >= 0.5
exists AND its text equals the verified label. Reports exact match over all GT plates (so detection misses count as errors) plus the
detection-conditional OCR accuracy. Baseline = fast-alpr as shipped (YOLOv9-t-384 detector + pretrained OCR, no orientation search).

  python -m lanes.ident.eval_e2e --exp baseline_e2e --det fastalpr --ocr-model-dir <dir> --ocr-model cct_s_v2_global --mode argmax --no-orient
  python -m lanes.ident.eval_e2e --exp ours_e2e --det onnx:<det dir> --ocr-dir <artifact dir> --mode auto --orient
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from eval.stats import iou_matrix
from lanes.ident import evalkit as ek
from lanes.ident.data import det_sets as ds
from lanes.ident.eval_det import make_detector
from lanes.ident.eval_ocr import resolve_model
from lanes.ident.ocr_onnx import PlateOcrOnnx, default_providers
from lanes.ident.reader import PlateReader
from seen.paths import RUNS, fast_dir

FAST = fast_dir("ident")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--det", default="fastalpr")
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--ocr-dir", default=None, help="artifact-style dir with model.onnx + plate_config.yaml")
    ap.add_argument("--ocr-model-dir", default=None)
    ap.add_argument("--ocr-model", default=None)
    ap.add_argument("--mode", default="argmax", choices=["argmax", "constrained", "auto"])
    ap.add_argument("--orient", action="store_true")
    ap.add_argument("--pad", type=float, default=0.04)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    labels = {r["id"]: r for r in map(json.loads, (Path(__file__).parent / "labels" / "uz_real_test_labels.jsonl").read_text().splitlines())}
    recs = [r for r in ds.load_uz_real("test") if labels.get(r["id"], {}).get("readable")]
    det = make_detector(a.det, a.device, a.det_conf)
    if a.ocr_dir:
        ocr = PlateOcrOnnx(Path(a.ocr_dir) / "model.onnx", Path(a.ocr_dir) / "plate_config.yaml", providers=default_providers(a.device))
    else:
        onnx, cfg = resolve_model(Path(a.ocr_model_dir), a.ocr_model)
        ocr = PlateOcrOnnx(onnx, cfg, providers=default_providers(a.device))
    rd = PlateReader(ocr, mode=a.mode, orient=a.orient)
    n = len(recs)
    det_ok = e2e_ok = 0
    rows = []
    for r in recs:
        gt_text = labels[r["id"]]["text"]
        im = cv2.cvtColor(cv2.imread(r["path"]), cv2.COLOR_BGR2RGB)
        H, W = im.shape[:2]
        dets = det.detect([cv2.cvtColor(im, cv2.COLOR_RGB2BGR)])[0]
        dets = [d for d in dets if d[0] >= a.det_conf]
        pred_text, hit = "", False
        if dets:
            boxes = np.array([d[1:] for d in dets])
            ious = iou_matrix(boxes, r["gt"])[:, 0]
            j = int(np.argmax(ious))
            if ious[j] >= 0.5:
                hit = True
                x1, y1, x2, y2 = boxes[j]
                bw, bh = x2 - x1, y2 - y1
                crop = im[int(max(0, y1 - a.pad * bh)):int(min(H, y2 + a.pad * bh)), int(max(0, x1 - a.pad * bw)):int(min(W, x2 + a.pad * bw))]
                if crop.size:
                    pred_text = rd.read([crop])[0].text
        det_ok += hit
        e2e_ok += int(hit and ek.norm(pred_text) == ek.norm(gt_text))
        rows.append({"id": r["id"], "gt": gt_text, "pred": pred_text, "detected": int(hit)})
    m = [ek.prop_metric("e2e_exact_match(all GT plates)", e2e_ok, n), ek.prop_metric("detected@IoU0.5", det_ok, n)]
    if det_ok:
        m.append(ek.prop_metric("exact_match|detected", e2e_ok, det_ok))
    comp = ek.component(f"ident.e2e.{a.exp}.uz_real{a.tag}", "End-to-end plate read: detector + OCR on real Uzbek photos", "real",
                        "Uzbek vehicle-inspection station photos (stretched 640x640)", "Roboflow Uzbek_Plates valid originals", "held-out cars", n,
                        f"det={a.det}; ocr={a.ocr_dir or a.ocr_model}; mode={a.mode}; orient={a.orient}", m,
                        notes=f"labels verified by Claude (AI), pending human spot-check; {len(labels) - n} unreadable/partial crops excluded; det conf>={a.det_conf}")
    out = RUNS / "ident" / a.exp
    ek.write_results(out / "component_results.json", [comp])
    ek.write_preds_csv(out / f"preds_e2e{a.tag}.csv", rows)
    print(f"e2e exact match {e2e_ok}/{n} = {e2e_ok / n:.3f}; detected {det_ok}/{n} = {det_ok / n:.3f}")


if __name__ == "__main__":
    main()
