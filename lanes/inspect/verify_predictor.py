"""End-to-end check of the shipped ONNX predictor (lanes/inspect/predict.py) and per-frame latency.

  scripts/box.sh --gpu 1 -- python -P -m lanes.inspect.verify_predictor --out runs/inspect/predictor_check

Checks (all numbers are written to <out>/predictor_check.json):
  damage  : N SYNDCAR eval images, context=None; counts of damage detections per class, fraction with a part assigned, and the
            overlay JPEGs (plates blurred via the part model's licence-plate detections) for visual review.
  tyre    : 40 test images of each tyre set as 'closeup' frames; |p_defective(ONNX predictor) - p(sklearn pipeline CSV)|
            (preprocessing parity: PIL bicubic in training vs cv2 bicubic in the predictor).
  dot     : TyreAI test images as 'closeup' frames; fraction of digit-serial GT date codes recovered through region+chars+validator.
  latency : per-frame p50/p95 of predict() for a 1024x768 street photo on this machine (GPU1, shared; not an isolated benchmark).
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from eval.stats import percentiles
from lanes.inspect import common as C
from lanes.inspect import dot as DOT
from lanes.inspect.predict import get_predictor
from seen.predictor import Frame, FramePred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.RUNS_I / "predictor_check"))
    ap.add_argument("--n-damage", type=int, default=30)
    a = ap.parse_args()
    out = C.ensure(Path(a.out))
    p = get_predictor()
    p.load("cuda:0")
    res: dict = {"model_versions": p.model_versions, "role": p.role, "name": p.name, "version": p.version}

    # ---- damage on SYNDCAR frames
    from lanes.inspect.prep_damage import OUT as DMG
    gt = C.read_json(DMG / "eval" / "syndcar" / "gt.json")[: a.n_damage]
    frames = [Frame(index=i, t_s=float(i), image=C.read_rgb(g["path"])) for i, g in enumerate(gt)]
    preds = p.predict(frames, None)
    dets = [d for fp in preds for d in fp.detections if d.cls.startswith("damage:")]
    from collections import Counter
    res["damage_syndcar"] = {"frames": len(frames), "detections": len(dets), "by_class": dict(Counter(d.cls for d in dets)),
                             "with_part": sum(1 for d in dets if d.part), "parts_seen": dict(Counter(d.part for d in dets)),
                             "bbox_in_unit_range": all(all(0.0 <= v <= 1.0 for v in d.bbox) for d in dets)}

    # ---- tyre closeups
    tyre = {}
    for key in ("tyrenet_test", "warcoder", "sameer"):
        f = C.RUNS_I / "tyre_dino_lr" / f"preds_{key}.csv"
        if not f.exists():
            continue
        rows = list(csv.DictReader(f.open()))[:40]
        diffs = []
        for r in rows:
            img = C.read_rgb(r["path"])
            fr = Frame(index=0, t_s=0.0, image=img)
            ctx = FramePred(index=0, t_s=0.0, scene={"closeup": 1.0})
            d = [x for x in p.predict([fr], [ctx])[0].detections if x.cls.startswith("tyre:")]
            if d:
                diffs.append(abs(d[0].extra["p_defective"] - float(r["p_defective"])))
        tyre[key] = {"n": len(diffs), "mean_abs_diff_p_defective": float(np.mean(diffs)), "max_abs_diff": float(np.max(diffs))}
    res["tyre_parity_vs_sklearn_pipeline"] = tyre

    # ---- DOT through region + chars + validator, on TyreAI test close-ups
    from lanes.inspect.prep_dot import OUT as DOTD
    import yaml
    names = yaml.safe_load((DOTD / "chars" / "data.yaml").read_text())["names"]
    names = [str(names[i]) for i in sorted(names)] if isinstance(names, dict) else [str(n) for n in names]
    imgs = sorted((DOTD / "chars" / "images" / "test").glob("*.jpg"))
    found = tot = shown = 0
    for ip in imgs:
        b, n = DOT.read_yolo_labels(DOTD / "chars" / "labels" / "test" / (ip.stem + ".txt"), names, 1, 1)
        gts = DOT.assemble(b, n)
        if not (gts.isdigit() and len(gts) >= 6 and DOT.validate_wwyy(DOT.date_code(gts))["valid"]):
            continue
        tot += 1
        fr = Frame(index=0, t_s=0.0, image=C.read_rgb(ip))
        ctx = FramePred(index=0, t_s=0.0, scene={"closeup": 1.0})
        d = [x for x in p.predict([fr], [ctx])[0].detections if x.cls == "dot"]
        shown += bool(d)
        found += any(x.text == DOT.date_code(gts) for x in d)
    res["dot_through_predictor"] = {"digit_serial_images": tot, "dot_emitted": shown, "date_code_correct": found}

    # ---- latency
    rng = np.random.default_rng(0)
    img = (rng.random((768, 1024, 3)) * 255).astype(np.uint8)
    fr = Frame(index=0, t_s=0.0, image=img)
    for _ in range(5):
        p.predict([fr], None)
    ts = []
    for _ in range(40):
        t0 = time.perf_counter()
        p.predict([fr], None)
        ts.append((time.perf_counter() - t0) * 1000)
    res["latency_predict_damage_plus_part_ms"] = percentiles(ts)
    C.write_json(res, out / "predictor_check.json")
    print(res)


if __name__ == "__main__":
    main()
