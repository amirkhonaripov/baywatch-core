"""Face-blur recall on the WIDER FACE validation set (3,226 images, valid faces only).

    python -m lanes.flow.run_detect --model rfdetr-base --dataset wider_val       # person boxes for the head-crop pass / fallback
    python -m lanes.flow.eval_blur --person-model rfdetr-base --exp blur_wider

Modes compared (all produce the regions that ``seen.blur.blur_faces`` would blur):
  heuristic     old baseline: upper 30% of every detected person box
  yunet_full    YuNet on the whole frame, +25% padding
  yunet_crops   + YuNet on an upscaled head-and-shoulders crop of every detected person
  final         yunet_crops + the heuristic box for every person where no face was found  (what the pipeline ships)
Metrics (per face-height bucket, 95% CI = bootstrap over images):
  covered80      share of GT faces whose box is >= 80% covered by the blur regions   <- the privacy-relevant number
  recall_iou50   share of GT faces matched by a detected face box with IoU >= 0.5 (detector recall; not defined for 'heuristic')
  blur_area      mean blurred area as % of the frame (cost of over-blurring)
"""
from __future__ import annotations

import argparse
import pickle
import time

import cv2
import numpy as np

from eval.stats import metric
from lanes.flow.common import CLASSES, component, save_results
from lanes.flow.data_extra import prepare_wider, wider_items
from lanes.flow.run_detect import cache_path
from lanes.flow.tracker import iou_batch
from seen import blur
from seen.predictor import Det

BUCKETS = [("h<16", 0, 16), ("h16-31", 16, 32), ("h32-63", 32, 64), ("h>=64", 64, 1e9), ("all", 0, 1e9)]


def person_dets(dets: np.ndarray, w: int, h: int, conf: float = 0.35) -> list[Det]:
    out = []
    for x1, y1, x2, y2, s, c in dets:
        if int(c) == 0 and s >= conf:
            out.append(Det("person", float(s), (x1 / w, y1 / h, x2 / w, y2 / h)))
    return out


def regions_all(img_rgb: np.ndarray, pdets: list[Det]) -> dict:
    """All four modes with the expensive YuNet passes computed once: {mode: (blur boxes px (n,4), raw face boxes px (m,4) | None)}."""
    h, w = img_rgb.shape[:2]
    out = {}
    out["heuristic"] = (np.array([[d.bbox[0] * w, d.bbox[1] * h, d.bbox[2] * w, d.bbox[1] * h + 0.3 * (d.bbox[3] - d.bbox[1]) * h] for d in pdets]).reshape(-1, 4), None)

    def pad(raw):
        if not len(raw):
            return raw
        return np.stack([raw[:, 0] - blur.PAD * (raw[:, 2] - raw[:, 0]), raw[:, 1] - blur.PAD * (raw[:, 3] - raw[:, 1]),
                         raw[:, 2] + blur.PAD * (raw[:, 2] - raw[:, 0]), raw[:, 3] + blur.PAD * (raw[:, 3] - raw[:, 1])], 1)

    ff, _ = blur.detect_faces(img_rgb, [])
    raw_full = np.array([f[:4] for f in ff]).reshape(-1, 4)
    out["yunet_full"] = (pad(raw_full), raw_full)
    fc, info = blur.detect_faces(img_rgb, pdets)
    raw_crops = np.array([f[:4] for f in fc]).reshape(-1, 4)
    out["yunet_crops"] = (pad(raw_crops), raw_crops)
    fb = [pad(raw_crops)] if len(raw_crops) else []
    fb += [np.array([[d.bbox[0] * w, d.bbox[1] * h, d.bbox[2] * w, d.bbox[1] * h + blur.HEURISTIC_FRAC * (d.bbox[3] - d.bbox[1]) * h]]) for i, d in enumerate(pdets)
           if i in info["persons_without_face"]]
    out["final"] = (np.concatenate(fb, 0) if fb else np.zeros((0, 4)), raw_crops)
    return out


def covered_fraction(gt: np.ndarray, boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        mask[max(int(y1), 0):min(int(np.ceil(y2)), h), max(int(x1), 0):min(int(np.ceil(x2)), w)] = True
    out = []
    for x1, y1, x2, y2 in gt:
        a, b, c, d = max(int(x1), 0), max(int(y1), 0), min(int(np.ceil(x2)), w), min(int(np.ceil(y2)), h)
        out.append(mask[b:d, a:c].mean() if c > a and d > b else 1.0)
    return np.array(out)


MODES = ["heuristic", "yunet_full", "yunet_crops", "final"]
_PCACHE: dict = {}


def _work(it):
    cv2.setNumThreads(1)
    img = cv2.cvtColor(cv2.imread(it.path), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    pd = person_dets(_PCACHE.get(it.id, np.zeros((0, 6))), w, h)
    gh = it.gt[:, 3] - it.gt[:, 1]
    res, tm = {}, {}
    import time as _t

    t0 = _t.perf_counter()
    reg = regions_all(img, pd)
    dt = _t.perf_counter() - t0
    for m, (bx, raw) in reg.items():
        bx = bx[np.isfinite(bx).all(1)] if len(bx) else bx
        cov = covered_fraction(it.gt, bx, w, h)
        hit = (iou_batch(it.gt, raw).max(1) >= 0.5) if raw is not None and len(raw) else np.zeros(len(it.gt), dtype=bool)
        mask = np.zeros((h, w), dtype=bool)
        for x1, y1, x2, y2 in bx:
            mask[max(int(y1), 0):min(int(np.ceil(y2)), h), max(int(x1), 0):min(int(np.ceil(x2)), w)] = True
        res[m] = {"h": gh, "cov": cov, "hit": hit, "area": mask.mean(), "n_boxes": len(bx), "has_raw": raw is not None}
    return res, dt


def run(person_model: str, exp: str, limit: int | None, n_boot: int = 500, workers: int = 6, stride: int = 1):
    prepare_wider()
    items = wider_items(limit)[::stride]
    global _PCACHE
    _PCACHE = pickle.load(open(cache_path("wider_val", person_model), "rb"))["dets"]
    modes = MODES
    from multiprocessing import get_context

    with get_context("fork").Pool(workers) as pool:
        outs = []
        for k, r in enumerate(pool.imap(_work, items, chunksize=4)):
            outs.append(r)
            if k % 200 == 0:
                print(k, len(items), flush=True)
    rec = {m: [o[0][m] for o in outs] for m in modes}
    total_s = float(np.sum([o[1] for o in outs]))
    t_mode = {m: total_s / len(items) * len(items) for m in modes}  # all modes share one pass: report the per-image total below
    results = []
    rng = np.random.default_rng(0)
    nI = len(items)
    boot_idx = rng.integers(0, nI, (max(n_boot // 2, 100), nI))
    for m in modes:
        R = rec[m]
        ms = []
        for bname, lo, hi in BUCKETS:
            sel = [(r["h"] >= lo) & (r["h"] < hi) for r in R]
            den = np.array([x.sum() for x in sel], dtype=float)
            n_faces = int(den.sum())
            for name, key, thr in (("covered80", "cov", 0.8), ("recall_iou50", "hit", None)):
                if key == "hit" and not R[0]["has_raw"]:
                    continue
                num = np.array([((r[key][x] >= thr) if thr is not None else r[key][x]).sum() for r, x in zip(R, sel)], dtype=float)
                if den.sum() == 0:
                    continue
                base = num.sum() / den.sum()
                dd = den[boot_idx].sum(1)
                bs = num[boot_idx].sum(1)[dd > 0] / dd[dd > 0]
                ms.append(metric(f"{name}[{bname}]", float(base), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), "fraction"))
            ms.append(metric(f"n_faces[{bname}]", n_faces, unit="count"))
        ms.append(metric("blur_area_pct", 100 * float(np.mean([r["area"] for r in R])), unit="% of frame"))
        ms.append(metric("blur_boxes_per_image", float(np.mean([r["n_boxes"] for r in R])), unit="count"))
        ms.append(metric("ms_per_image_cpu_all_modes_total", 1000 * total_s / len(items), unit="ms"))
        results.append(component(f"flow.blur.{m}", f"Face blur coverage ({m})", "WIDER FACE val", len(items), ms, split="val", model=f"{blur.backend()}:{m}",
                                 domain="web photos (crowds, small faces)",
                                 notes=f"{int(sum(len(r['h']) for r in R))} valid faces in {len(items)} images; person boxes from {person_model} (conf>=0.35); "
                                       f"YuNet score>={blur.SCORE_THR}, pad {blur.PAD:.0%}, full-frame scales {blur.FULL_FRAME_SCALES}; bootstrap over images"))
        print(m, {x["name"]: round(x["value"], 3) for x in ms if x["name"].startswith(("covered80", "recall")) and "all" in x["name"]})
    save_results(exp, results)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--person-model", default="rfdetr-base")
    ap.add_argument("--exp", default="blur_wider")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    run(a.person_model, a.exp, a.limit, workers=a.workers, stride=a.stride)
