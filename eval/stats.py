"""Shared statistics for every lane (Lane G owns this file; lanes may import it, never the reverse).

Everything here is plain numpy so it runs on the Mac and on the box.

    from eval.stats import wilson_ci, prop_metric, bootstrap_ci, ece, detection_ap, confusion

Conventions
- All intervals are two-sided 95 % unless ``conf`` is given.
- Proportions (accuracy, recall, exact-match, ...) use the Wilson score interval: ``prop_metric("recall", k, n)``.
- Anything that is not a simple proportion (mAP, MAE, F1, means, medians) uses a percentile bootstrap over the
  independent unit (image, clip, video): ``bootstrap_ci(units, stat_fn)``. Resample *images*, not boxes, and
  *videos*, not frames, otherwise the interval is too narrow.
- ``metric(...)`` / ``prop_metric(...)`` return dicts that match ``contracts/metrics.schema.json#/$defs/Metric``.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

Z95 = 1.959963984540054


def _z(conf: float) -> float:
    if abs(conf - 0.95) < 1e-12:
        return Z95
    from statistics import NormalDist

    return NormalDist().inv_cdf(0.5 + conf / 2)


# ----------------------------------------------------------------------------------------------- proportions
def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float, float]:
    """Return (p_hat, low, high) for k successes out of n (Wilson score interval). n == 0 -> (nan, 0, 1)."""
    if n <= 0:
        return float("nan"), 0.0, 1.0
    if not 0 <= k <= n:
        raise ValueError(f"k={k} not in [0, n={n}]")
    z = _z(conf)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return p, lo, hi


def metric(name: str, value: float, ci_low: float | None = None, ci_high: float | None = None,
           unit: str | None = None) -> dict[str, Any]:
    """A ``Metric`` dict (contracts/metrics.schema.json). NaN/inf values are refused: report n=0 instead."""
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"metric {name!r} has non-finite value {value!r}")
    m: dict[str, Any] = {"name": name, "value": float(value)}
    if ci_low is not None and math.isfinite(ci_low):
        m["ci_low"] = float(ci_low)
    if ci_high is not None and math.isfinite(ci_high):
        m["ci_high"] = float(ci_high)
    if unit:
        m["unit"] = unit
    return m


def prop_metric(name: str, k: int, n: int, unit: str = "fraction", conf: float = 0.95) -> dict[str, Any]:
    """Proportion metric with a Wilson interval, e.g. ``prop_metric("exact_match", 812, 1068)``."""
    p, lo, hi = wilson_ci(int(k), int(n), conf)
    if n == 0:
        raise ValueError(f"metric {name!r}: n == 0")
    return metric(name, p, lo, hi, unit)


# ------------------------------------------------------------------------------------------------- bootstrap
def bootstrap_ci(units: Sequence[Any] | np.ndarray, stat_fn: Callable[[Any], float] = np.mean, *,
                 n_boot: int = 2000, conf: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap: resample ``units`` with replacement, apply ``stat_fn``.

    ``units`` is a 1-D array of per-unit values (``stat_fn`` = mean/median, e.g. absolute errors for MAE) or a list
    of arbitrary per-unit records (``stat_fn`` receives a list, e.g. per-image detections for mAP).
    Returns (point estimate on the full sample, low, high).
    """
    n = len(units)
    if n == 0:
        raise ValueError("bootstrap_ci: no units")
    rng = np.random.default_rng(seed)
    is_arr = isinstance(units, np.ndarray) or (n and isinstance(units[0], (int, float, np.number)))
    arr = np.asarray(units) if is_arr else None
    point = float(stat_fn(arr if is_arr else list(units)))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = arr[idx] if is_arr else [units[i] for i in idx]
        stats[b] = stat_fn(sample)
    stats = stats[np.isfinite(stats)]
    if len(stats) == 0:
        return point, float("nan"), float("nan")
    a = (1 - conf) / 2
    return point, float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))


def boot_metric(name: str, units, stat_fn=np.mean, unit: str | None = None, **kw) -> dict[str, Any]:
    p, lo, hi = bootstrap_ci(units, stat_fn, **kw)
    return metric(name, p, lo, hi, unit)


def mae_metric(errors: Sequence[float], name: str = "mae", unit: str | None = None, **kw) -> dict[str, Any]:
    return boot_metric(name, np.abs(np.asarray(errors, dtype=float)), np.mean, unit, **kw)


# --------------------------------------------------------------------------------------------- calibration
def ece(confidences: Sequence[float], correct: Sequence[bool | int], n_bins: int = 15) -> float:
    """Expected calibration error with equal-width bins (top-label confidence vs accuracy)."""
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    if len(c) == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(c, bins[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(c[m].mean() - y[m].mean())
    return float(total)


def reliability_bins(confidences, correct, n_bins: int = 10) -> list[dict[str, float]]:
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(c, bins[1:-1], right=True), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out.append({"lo": float(bins[b]), "hi": float(bins[b + 1]), "n": int(m.sum()),
                        "conf": float(c[m].mean()), "acc": float(y[m].mean())})
    return out


def accuracy_coverage(confidences, correct, points: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.5)) -> list[dict]:
    """Accuracy when only the most confident ``coverage`` fraction is answered (the rest -> 'unknown')."""
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    order = np.argsort(-c, kind="stable")
    out = []
    for cov in points:
        k = max(1, int(round(cov * len(c))))
        sel = y[order[:k]]
        p, lo, hi = wilson_ci(int(sel.sum()), k)
        out.append({"coverage": float(cov), "n": k, "threshold": float(c[order[k - 1]]),
                    "accuracy": p, "ci_low": lo, "ci_high": hi})
    return out


# ----------------------------------------------------------------------------------------------- confusion
def confusion(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] | None = None) -> dict:
    """``{"labels": [...], "matrix": [[...]]}``; rows = truth, columns = prediction (schema ``confusion``)."""
    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    ix = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in ix and p in ix:
            m[ix[t], ix[p]] += 1
    return {"labels": labels, "matrix": m.tolist()}


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else float("nan")


# ----------------------------------------------------------------------------------------------- detection
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (N,4) and b (M,4) in xyxy."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def _match_image(pred_boxes, pred_scores, gt_boxes, iou_thr):
    """Greedy (score-ordered) matching for one image and one class -> (scores, is_tp)."""
    order = np.argsort(-np.asarray(pred_scores, dtype=float), kind="stable")
    pb = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)[order]
    ps = np.asarray(pred_scores, dtype=float)[order]
    gb = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    ious = iou_matrix(pb, gb)
    used = np.zeros(len(gb), dtype=bool)
    tp = np.zeros(len(pb), dtype=bool)
    for i in range(len(pb)):
        if len(gb) == 0:
            break
        cand = np.where(~used, ious[i], -1)
        j = int(np.argmax(cand))
        if cand[j] >= iou_thr:
            used[j] = True
            tp[i] = True
    return ps, tp


def average_precision(images: Sequence[dict], cls: str, iou_thr: float = 0.5) -> float:
    """All-point interpolated AP (VOC2010+/COCO-style at a single IoU) for one class.

    ``images``: list of {"gt": [(cls, x1,y1,x2,y2), ...], "pred": [(cls, score, x1,y1,x2,y2), ...]}; boxes in any
    consistent unit. Returns NaN if the class has no ground truth in ``images``.
    """
    scores, tps, n_gt = [], [], 0
    for im in images:
        g = [b[1:5] for b in im.get("gt", []) if b[0] == cls]
        p = [b for b in im.get("pred", []) if b[0] == cls]
        n_gt += len(g)
        if p:
            s, t = _match_image([b[2:6] for b in p], [b[1] for b in p], g, iou_thr)
            scores.append(s)
            tps.append(t)
    if n_gt == 0:
        return float("nan")
    if not scores:
        return 0.0
    s = np.concatenate(scores)
    t = np.concatenate(tps)
    order = np.argsort(-s, kind="stable")
    t = t[order]
    ctp = np.cumsum(t)
    cfp = np.cumsum(~t)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def detection_map(images: Sequence[dict], classes: Sequence[str], iou_thr: float = 0.5) -> float:
    aps = [average_precision(images, c, iou_thr) for c in classes]
    aps = [a for a in aps if math.isfinite(a)]
    return float(np.mean(aps)) if aps else float("nan")


def detection_ap(images: Sequence[dict], classes: Sequence[str], iou_thr: float = 0.5, *, n_boot: int = 1000,
                 seed: int = 0, name: str | None = None) -> dict[str, Any]:
    """mAP@iou with an image-level bootstrap CI, as a Metric dict."""
    fn = lambda ims: detection_map(ims, classes, iou_thr)  # noqa: E731
    p, lo, hi = bootstrap_ci(list(images), fn, n_boot=n_boot, seed=seed)
    return metric(name or f"mAP@{iou_thr:g}", p, lo, hi, "fraction")


def detection_recall(images: Sequence[dict], cls: str, iou_thr: float = 0.5, score_thr: float = 0.0) -> tuple[int, int]:
    """(matched gt, total gt) at a score threshold, for a Wilson interval on recall."""
    k = n = 0
    for im in images:
        g = [b[1:5] for b in im.get("gt", []) if b[0] == cls]
        p = [b for b in im.get("pred", []) if b[0] == cls and b[1] >= score_thr]
        n += len(g)
        if g and p:
            _, t = _match_image([b[2:6] for b in p], [b[1] for b in p], g, iou_thr)
            k += int(t.sum())
    return k, n


def percentiles(values_ms: Sequence[float]) -> dict[str, float]:
    v = np.asarray(values_ms, dtype=float)
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
            "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
