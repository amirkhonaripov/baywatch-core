"""Detection / segmentation evaluation with image-level bootstrap CIs (box and mask AP50, recall at fixed FP/image).

Records (one per image)::

    {"id": str, "w": int, "h": int,
     "gt":   [{"cls": str, "box": [x1,y1,x2,y2], "poly": [[[x,y],...], ...] | None}, ...],   # poly = list of polygons
     "pred": [{"cls": str, "score": float, "box": [...], "poly": [[[x,y],...], ...] | None}, ...]}

Matching is greedy by score at IoU >= thr (VOC/COCO style, one IoU). Matching depends only on the image, so it is
done once; the bootstrap resamples *images* and re-assembles the PR curve from the stored per-image (score, tp).
Mask IoU is computed on masks rasterised at <= ``RAST`` px on the long side.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

from eval.stats import iou_matrix, metric

RAST = 384


def _raster(polys, h: int, w: int, s: float) -> np.ndarray:
    m = np.zeros((max(1, round(h * s)), max(1, round(w * s))), np.uint8)
    for poly in polys:
        pts = np.round(np.asarray(poly, dtype=np.float64) * s).astype(np.int32).reshape(-1, 1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def _mask_iou(pred_polys: list, gt_polys: list, h: int, w: int) -> np.ndarray:
    s = min(1.0, RAST / max(h, w))
    pm = [_raster(p, h, w, s) for p in pred_polys]
    gm = [_raster(g, h, w, s) for g in gt_polys]
    if not pm or not gm:
        return np.zeros((len(pm), len(gm)))
    P = np.stack([m.ravel() for m in pm]).astype(np.float32)
    G = np.stack([m.ravel() for m in gm]).astype(np.float32)
    inter = P @ G.T
    union = P.sum(1)[:, None] + G.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def _greedy(ious: np.ndarray, scores: np.ndarray, thr: float) -> np.ndarray:
    """tp flags aligned with the input order of predictions."""
    n_p, n_g = ious.shape
    tp = np.zeros(n_p, bool)
    if n_p == 0 or n_g == 0:
        return tp
    used = np.zeros(n_g, bool)
    for i in np.argsort(-scores, kind="stable"):
        cand = np.where(~used, ious[i], -1.0)
        j = int(np.argmax(cand))
        if cand[j] >= thr:
            used[j] = True
            tp[i] = True
    return tp


class Evaluator:
    """Build once per (records, kind, thr); then ask for AP / recall@FPPI with bootstrap CIs."""

    def __init__(self, records: Sequence[dict], classes: Sequence[str], kind: str = "box", thr: float = 0.5,
                 groups: Sequence[int] | None = None):
        """``groups``: optional cluster id per record (near-duplicate clusters); the bootstrap then resamples whole clusters."""
        assert kind in ("box", "mask")
        self.members: list[np.ndarray] | None = None
        if groups is not None:
            g = np.asarray(groups)
            self.members = [np.where(g == u)[0] for u in np.unique(g)]
        self.kind, self.thr, self.classes = kind, thr, list(classes)
        self.n_img = len(records)
        # per image: n_gt per class, (scores, tp) per class; agnostic as key "*"
        self.per_img: list[dict] = []
        for r in records:
            gts, prs = r["gt"], r["pred"]
            entry: dict = {"ngt": {}, "sc": {}, "tp": {}}
            groups = {c: ([g for g in gts if g["cls"] == c], [p for p in prs if p["cls"] == c]) for c in self.classes}
            groups["*"] = (gts, prs)
            for c, (g, p) in groups.items():
                entry["ngt"][c] = len(g)
                if not p:
                    entry["sc"][c] = np.zeros(0)
                    entry["tp"][c] = np.zeros(0, bool)
                    continue
                sc = np.array([x["score"] for x in p], dtype=float)
                if not g:
                    tp = np.zeros(len(p), bool)
                elif kind == "box":
                    tp = _greedy(iou_matrix(np.array([x["box"] for x in p]), np.array([x["box"] for x in g])), sc, thr)
                else:
                    gp = [x["poly"] for x in g]
                    pp = [x["poly"] or [] for x in p]  # a prediction without a mask cannot overlap anything
                    if any(q is None for q in gp):
                        raise ValueError("mask evaluation needs polygon GT for every instance")
                    tp = _greedy(_mask_iou([q for q in pp], gp, r["h"], r["w"]), sc, thr)
                entry["sc"][c], entry["tp"][c] = sc, tp
            self.per_img.append(entry)

    # ------------------------------------------------------------------ curves
    def _gather(self, idx: np.ndarray, c: str):
        n_gt = sum(self.per_img[i]["ngt"][c] for i in idx)
        sc = np.concatenate([self.per_img[i]["sc"][c] for i in idx]) if len(idx) else np.zeros(0)
        tp = np.concatenate([self.per_img[i]["tp"][c] for i in idx]) if len(idx) else np.zeros(0, bool)
        return n_gt, sc, tp

    @staticmethod
    def _ap(n_gt: int, sc: np.ndarray, tp: np.ndarray) -> float:
        if n_gt == 0:
            return float("nan")
        if len(sc) == 0:
            return 0.0
        o = np.argsort(-sc, kind="stable")
        t = tp[o]
        ctp, cfp = np.cumsum(t), np.cumsum(~t)
        rec = ctp / n_gt
        prec = ctp / np.maximum(ctp + cfp, 1e-12)
        mrec = np.concatenate([[0.0], rec, [1.0]])
        mpre = np.maximum.accumulate(np.concatenate([[1.0], prec, [0.0]])[::-1])[::-1]
        i = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))

    @staticmethod
    def _recall_at_fppi(n_gt: int, sc: np.ndarray, tp: np.ndarray, n_img: int, fppi: float) -> float:
        """Highest recall reachable with <= fppi false positives per image (threshold sweep)."""
        if n_gt == 0:
            return float("nan")
        if len(sc) == 0:
            return 0.0
        o = np.argsort(-sc, kind="stable")
        t = tp[o]
        ctp, cfp = np.cumsum(t), np.cumsum(~t)
        ok = cfp / n_img <= fppi
        return float(ctp[ok].max() / n_gt) if ok.any() else 0.0

    # ------------------------------------------------------------------ metrics with CIs
    def _boot(self, fn, n_boot: int = 500, seed: int = 0):
        full = fn(np.arange(self.n_img))
        if not math.isfinite(full):
            return float("nan"), float("nan"), float("nan")
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(n_boot):
            if self.members is not None:  # cluster bootstrap: near-duplicate images are not independent samples
                pick = rng.integers(0, len(self.members), len(self.members))
                v = fn(np.concatenate([self.members[i] for i in pick]))
            else:
                v = fn(rng.integers(0, self.n_img, self.n_img))
            if math.isfinite(v):
                vals.append(v)
        if not vals:
            return full, float("nan"), float("nan")
        return full, float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))

    def ap(self, c: str, **kw):
        return self._boot(lambda idx: self._ap(*self._gather(idx, c)), **kw)

    def map(self, **kw):
        cls = [c for c in self.classes if sum(e["ngt"][c] for e in self.per_img) > 0]

        def fn(idx):
            v = [self._ap(*self._gather(idx, c)) for c in cls]
            v = [x for x in v if math.isfinite(x)]
            return float(np.mean(v)) if v else float("nan")
        return self._boot(fn, **kw)

    def recall_at_fppi(self, c: str, fppi: float, **kw):
        def fn(idx):
            n_gt, sc, tp = self._gather(idx, c)
            return self._recall_at_fppi(n_gt, sc, tp, len(idx), fppi)
        return self._boot(fn, **kw)

    def n_gt(self, c: str) -> int:
        return sum(e["ngt"][c] for e in self.per_img)

    def summary(self, prefix: str, fppis: Sequence[float] = (0.1, 0.5, 1.0), n_boot: int = 300) -> list[dict]:
        """Metric dicts: per-class AP, mAP over classes with GT, class-agnostic AP and recall@FPPI."""
        out = []
        p, lo, hi = self.map(n_boot=n_boot)
        if math.isfinite(p):
            out.append(metric(f"{prefix}mAP50_{self.kind}", p, lo, hi))
        for c in self.classes:
            if self.n_gt(c) == 0:
                continue
            p, lo, hi = self.ap(c, n_boot=n_boot)
            out.append(metric(f"{prefix}AP50_{self.kind}[{c}]", p, lo, hi))
        p, lo, hi = self.ap("*", n_boot=n_boot)
        if math.isfinite(p):
            out.append(metric(f"{prefix}AP50_{self.kind}[any_damage]", p, lo, hi))
        for f in fppis:
            p, lo, hi = self.recall_at_fppi("*", f, n_boot=n_boot)
            if math.isfinite(p):
                out.append(metric(f"{prefix}recall_{self.kind}[any_damage]@{f:g}FP/img", p, lo, hi))
        for c in self.classes:
            if self.n_gt(c) == 0:
                continue
            p, lo, hi = self.recall_at_fppi(c, 0.5, n_boot=n_boot)
            if math.isfinite(p):
                out.append(metric(f"{prefix}recall_{self.kind}[{c}]@0.5FP/img", p, lo, hi))
        return out
