"""Binary per-state metrics with CIs (bootstrap over independent units = videos/clips/sources)."""
from __future__ import annotations

import numpy as np

from eval.stats import bootstrap_ci, confusion, ece, metric, prop_metric


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    # average ranks for ties
    sp = p[order]
    i = 0
    r = np.arange(1, len(p) + 1, dtype=float)
    while i < len(p):
        j = i
        while j + 1 < len(p) and sp[j + 1] == sp[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2
        i = j + 1
    ranks[order] = r
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def f1(y, p, thr=0.5) -> float:
    yh = np.asarray(p) >= thr
    y = np.asarray(y).astype(bool)
    tp, fp, fn = int((yh & y).sum()), int((yh & ~y).sum()), int((~yh & y).sum())
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else float("nan")


def binary_ece(y, p, n_bins: int = 10) -> float:
    p = np.asarray(p, float)
    y = np.asarray(y).astype(int)
    conf = np.maximum(p, 1 - p)
    correct = (p >= 0.5).astype(int) == y
    return ece(conf, correct, n_bins)


def state_metrics(y, p, groups, thr: float = 0.5, n_boot: int = 1000) -> tuple[list[dict], dict, dict]:
    """Returns (metrics, confusion, counts). Bootstrap resamples groups (videos/clips), not frames."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    groups = np.asarray(groups)
    ug = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in ug}
    units = [by[g] for g in ug]

    def cat(us):
        idx = np.concatenate(us) if us else np.array([], int)
        return y[idx], p[idx]

    out = []
    a = auroc(y, p)
    if np.isfinite(a):
        pt, lo, hi = bootstrap_ci(units, lambda us: auroc(*cat(us)), n_boot=n_boot)
        out.append(metric("auroc", a, lo, hi, "fraction"))
    fv = f1(y, p, thr)
    if np.isfinite(fv):
        pt, lo, hi = bootstrap_ci(units, lambda us: f1(*cat(us), thr), n_boot=n_boot)
        out.append(metric(f"f1@{thr:g}", fv, lo, hi, "fraction"))
    yh = (p >= thr).astype(int)
    out.append(prop_metric(f"accuracy@{thr:g}", int((yh == y).sum()), len(y)))
    if y.sum() > 0:
        out.append(prop_metric(f"recall@{thr:g}", int(((yh == 1) & (y == 1)).sum()), int(y.sum())))
    if yh.sum() > 0:
        out.append(prop_metric(f"precision@{thr:g}", int(((yh == 1) & (y == 1)).sum()), int(yh.sum())))
    e = binary_ece(y, p)
    if np.isfinite(e):
        pt, lo, hi = bootstrap_ci(units, lambda us: binary_ece(*cat(us)), n_boot=n_boot)
        out.append(metric("ece", e, lo, hi, "fraction"))
    conf = confusion(["pos" if v else "neg" for v in y], ["pos" if v else "neg" for v in yh], ["neg", "pos"])
    counts = {"n": int(len(y)), "n_pos": int(y.sum()), "n_groups": int(len(ug))}
    return out, conf, counts
