"""Plate-detection evaluation: recall / precision @IoU0.5 (Wilson CIs) + AP50 (bootstrap CI), per subset, + latency.

  python -m lanes.ident.eval_det --exp baseline_det --detector fastalpr --sets uz_real_test rodosol_test ccpd_db ...
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from lanes.ident import evalkit as ek
from lanes.ident.data import det_sets as ds
from seen.paths import RUNS

SETS = {
    "uz_real_test": ("real", "Uzbek vehicle-inspection station photos (stretched 640x640)", "Roboflow Uzbek_Plates valid originals", "held-out cars", lambda n: ds.load_uz_real("test")),
    "rodosol_test": ("real", "fixed toll cameras (BR/Mercosur), day+night, rain", "RodoSol-ALPR (cars only, Kaggle mirror)", "block-split test", lambda n: ds.load_staged("rodosol_test")[:n]),
    "ccpd_db": ("real", "CCPD-DB: very bright/dark/uneven illumination (Chinese plates)", "CCPD2019 ccpd_db", "subset sample", lambda n: ds.load_ccpd("db", n, 0, "eval")),
    "ccpd_weather": ("real", "CCPD-Weather: rain/snow/fog (Chinese plates)", "CCPD2019 ccpd_weather", "subset sample", lambda n: ds.load_ccpd("weather", n, 0, "eval")),
    "ccpd_challenge": ("real", "CCPD-Challenge: hardest cases (Chinese plates)", "CCPD2019 ccpd_challenge", "subset sample", lambda n: ds.load_ccpd("challenge", n, 0, "eval")),
}


def make_detector(name: str, device: str, conf: float):
    if name == "fastalpr":
        from lanes.ident.detectors import FastAlprDetector
        return FastAlprDetector(conf=conf, device=device)
    if name.startswith("onnx:"):
        from lanes.ident.predict import OnnxPlateDetector
        return OnnxPlateDetector(name[5:], device=device, conf=conf)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--detector", default="fastalpr")
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--conf", type=float, default=0.05, help="detector floor for AP; reported recall/precision use --thr")
    ap.add_argument("--thr", type=float, default=0.25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    det = make_detector(a.detector, a.device, a.conf)
    comps = []
    for sname in a.sets:
        kind, domain, dataset, split, loader = SETS[sname]
        recs = loader(a.n)
        if not recs:
            print("skip", sname)
            continue
        gts, preds, lat = [], [], []
        for r in recs:
            im = cv2.imread(r["path"], cv2.IMREAD_COLOR)
            t0 = time.perf_counter()
            p = det.detect([im])[0]
            lat.append((time.perf_counter() - t0) * 1000)
            gts.append(r["gt"])
            preds.append([(s, np.array([x1, y1, x2, y2])) for s, x1, y1, x2, y2 in p])
        imgs = ek.det_images(gts, preds)
        m = ek.det_metrics(imgs, a.thr, 0.5)
        m += ek.det_metrics(imgs, a.thr, 0.75)[:1]
        lat = np.array(lat[5:] or lat)
        m.append(ek.metric("latency_p50_ms_per_image", float(np.percentile(lat, 50)), unit="ms"))
        m.append(ek.metric("latency_p95_ms_per_image", float(np.percentile(lat, 95)), unit="ms"))
        comps.append(ek.component(f"ident.det.{a.exp}.{sname}{a.tag}", "Plate detection recall/precision @IoU0.5, AP50", kind, domain, dataset, split, len(recs), a.detector, m,
                                  notes=f"{sum(len(g) for g in gts)} GT plates; recall/precision at conf>={a.thr}; AP over conf>={a.conf}; latency = batch-1 per image incl. pre/post-processing on {a.device}"))
        rec = [x for x in m if x["name"].startswith("recall@IoU0.5")][0]
        print(f"{sname:16s} n={len(recs)}  recall={rec['value']:.3f} [{rec['ci_low']:.3f},{rec['ci_high']:.3f}]  " + "  ".join(f"{x['name']}={x['value']:.3f}" for x in m if x['name'].startswith(('precision', 'AP'))) + f"  p50={np.percentile(lat, 50):.1f}ms", flush=True)
    ek.write_results(RUNS / "ident" / a.exp / "component_results.json", comps)


if __name__ == "__main__":
    main()
