"""HOTA / IDF1 / MOTA for one sequence set, following the TrackEval definitions (Luiten et al. 2021; Ristani et al. 2016).

Inputs per sequence: ``gt`` and ``pr`` = dict frame -> (ids int array (n,), boxes xyxy (n,4)). Frames present in
either dict are evaluated (a frame missing from one side means "no objects there").
Results are *pooled* over the sequences passed in (sum the counts, then divide), which is how the benchmarks aggregate.
Bootstrap CIs resample whole sequences (``bootstrap``).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from lanes.flow.tracker import iou_batch

ALPHAS = np.arange(0.05, 0.99, 0.05)
EPS = np.finfo(float).eps


def seq_counts(gt: dict, pr: dict) -> dict:
    """Sufficient statistics for one sequence (additive over sequences for DetA/IDF1/MOTA; AssA needs a per-sequence pass)."""
    frames = sorted(set(gt) | set(pr))
    gt_ids = sorted({int(i) for f in gt.values() for i in f[0]})
    pr_ids = sorted({int(i) for f in pr.values() for i in f[0]})
    gmap = {g: k for k, g in enumerate(gt_ids)}
    pmap = {p: k for k, p in enumerate(pr_ids)}
    G, T = len(gt_ids), len(pr_ids)
    potential = np.zeros((G, T))  # HOTA global alignment numerator
    potential_idf1 = np.zeros((G, T))  # IDF1: frames with IoU >= 0.5
    gt_cnt = np.zeros(G)
    pr_cnt = np.zeros(T)
    per_frame = []
    n_gt = n_pr = 0
    for f in frames:
        gi, gb = gt.get(f, (np.zeros(0, int), np.zeros((0, 4))))
        pi, pb = pr.get(f, (np.zeros(0, int), np.zeros((0, 4))))
        gk = np.array([gmap[int(i)] for i in gi], dtype=int)
        pk = np.array([pmap[int(i)] for i in pi], dtype=int)
        sim = iou_batch(gb, pb) if len(gi) and len(pi) else np.zeros((len(gi), len(pi)))
        n_gt += len(gi)
        n_pr += len(pi)
        if len(gk):
            gt_cnt[gk] += 1
        if len(pk):
            pr_cnt[pk] += 1
        if len(gk) and len(pk):
            den = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
            simiou = np.where(den > EPS, sim / np.maximum(den, EPS), 0.0)
            potential[gk[:, None], pk[None, :]] += simiou
            potential_idf1[gk[:, None], pk[None, :]] += sim >= 0.5
        per_frame.append((gk, pk, sim))
    return {"G": G, "T": T, "potential": potential, "potential_idf1": potential_idf1, "gt_cnt": gt_cnt, "pr_cnt": pr_cnt,
            "per_frame": per_frame, "n_gt": n_gt, "n_pr": n_pr, "gt_ids": gt_ids, "pr_ids": pr_ids}


def hota_from_counts(c: dict) -> dict:
    """Additive HOTA statistics of one sequence (per-alpha arrays)."""
    G, T = c["G"], c["T"]
    nA = len(ALPHAS)
    tp = np.zeros(nA)
    loc_sum = np.zeros(nA)
    matches = [np.zeros((G, T)) for _ in range(nA)]
    if G and T:
        galign = c["potential"] / np.maximum(c["gt_cnt"][:, None] + c["pr_cnt"][None, :] - c["potential"], EPS)
    for gk, pk, sim in c["per_frame"]:
        if not len(gk) or not len(pk):
            continue
        score = galign[gk[:, None], pk[None, :]] * sim
        r, cc = linear_sum_assignment(-score)
        s = sim[r, cc]
        for a, al in enumerate(ALPHAS):
            m = s >= al - EPS
            tp[a] += m.sum()
            loc_sum[a] += s[m].sum()
            matches[a][gk[r[m]], pk[cc[m]]] += 1
    ass_tp = np.zeros(nA)  # sum over TPs of the association score A(c) (AssA = ass_tp / tp when pooled)
    for a in range(nA):
        mc = matches[a]
        if G and T:
            ass_a = mc / np.maximum(1, c["gt_cnt"][:, None] + c["pr_cnt"][None, :] - mc)
            ass_tp[a] = (mc * ass_a).sum()
    return {"tp": tp, "ass_tp": ass_tp, "loc_sum": loc_sum, "n_gt": c["n_gt"], "n_pr": c["n_pr"]}


def idf1_from_counts(c: dict) -> dict:
    G, T = c["G"], c["T"]
    if G == 0 or T == 0:
        return {"idtp": 0.0, "idfp": float(c["pr_cnt"].sum()), "idfn": float(c["gt_cnt"].sum())}
    fp_mat = np.zeros((G + T, G + T))
    fn_mat = np.zeros((G + T, G + T))
    fp_mat[G:, :T] = 1e10
    fn_mat[:G, T:] = 1e10
    for g in range(G):
        fn_mat[g, :T] = c["gt_cnt"][g]
        fn_mat[g, T + g] = c["gt_cnt"][g]
    for t in range(T):
        fp_mat[:G, t] = c["pr_cnt"][t]
        fp_mat[t + G, t] = c["pr_cnt"][t]
    fn_mat[:G, :T] -= c["potential_idf1"]
    fp_mat[:G, :T] -= c["potential_idf1"]
    r, cc = linear_sum_assignment(fn_mat + fp_mat)
    idfn = float(fn_mat[r, cc].sum())
    idfp = float(fp_mat[r, cc].sum())
    return {"idtp": float(c["gt_cnt"].sum() - idfn), "idfp": idfp, "idfn": idfn}


def clear_from_counts(c: dict, thr: float = 0.5) -> dict:
    """MOTA statistics with TrackEval's CLEAR matching (prefers keeping the previous match; counts ID switches)."""
    fn = fp = idsw = 0
    prev: dict[int, int] = {}
    for gk, pk, sim in c["per_frame"]:
        gt_n, pr_n = len(gk), len(pk)
        if gt_n == 0:
            fp += pr_n
            prev = {}
            continue
        if pr_n == 0:
            fn += gt_n
            prev = {}
            continue
        prev_pr = np.array([[prev.get(int(g), -1) == int(p) for p in pk] for g in gk])
        score = np.where(sim >= thr - EPS, 1000.0 * prev_pr + sim + 1.0, 0.0)
        r, cc = linear_sum_assignment(-score)
        ok = score[r, cc] > 0
        r, cc = r[ok], cc[ok]
        cur = {}
        for i, j in zip(r, cc):
            g, p = int(gk[i]), int(pk[j])
            if g in prev and prev[g] != p:
                idsw += 1
            cur[g] = p
        fn += gt_n - len(r)
        fp += pr_n - len(r)
        for g, p in prev.items():
            cur.setdefault(g, p)
        prev = cur
    return {"fn": fn, "fp": fp, "idsw": idsw, "n_gt": c["n_gt"]}


def seq_stats(gt: dict, pr: dict) -> dict:
    """All additive sufficient statistics of one sequence (pool with ``pool``)."""
    c = seq_counts(gt, pr)
    return {"hota": hota_from_counts(c), "idf1": idf1_from_counts(c), "clear": clear_from_counts(c)}


def pool(stats: list[dict]) -> dict:
    """Benchmark-style pooling: sums of TP/FP/FN, TP-weighted AssA, IDF1 from summed IDTP/IDFP/IDFN."""
    tp = sum(s["hota"]["tp"] for s in stats)
    ass_tp = sum(s["hota"]["ass_tp"] for s in stats)
    loc = sum(s["hota"]["loc_sum"] for s in stats)
    n_gt = sum(s["hota"]["n_gt"] for s in stats)
    n_pr = sum(s["hota"]["n_pr"] for s in stats)
    det_a = tp / np.maximum(1, tp + (n_gt - tp) + (n_pr - tp))
    ass_a = ass_tp / np.maximum(1, tp)
    hota = np.sqrt(det_a * ass_a)
    idtp = sum(s["idf1"]["idtp"] for s in stats)
    idfp = sum(s["idf1"]["idfp"] for s in stats)
    idfn = sum(s["idf1"]["idfn"] for s in stats)
    fn = sum(s["clear"]["fn"] for s in stats)
    fp = sum(s["clear"]["fp"] for s in stats)
    idsw = sum(s["clear"]["idsw"] for s in stats)
    ng = sum(s["clear"]["n_gt"] for s in stats)
    return {"HOTA": float(hota.mean()), "DetA": float(det_a.mean()), "AssA": float(ass_a.mean()), "LocA": float(np.mean(loc / np.maximum(1e-12, tp))),
            "IDF1": float(idtp / max(idtp + 0.5 * idfp + 0.5 * idfn, 1)), "IDP": float(idtp / max(idtp + idfp, 1)), "IDR": float(idtp / max(idtp + idfn, 1)),
            "MOTA": float(1 - (fn + fp + idsw) / max(ng, 1)), "IDSW": int(idsw), "IDSW_per_1k_gt": float(1000 * idsw / max(ng, 1)), "FN": int(fn), "FP": int(fp), "n_gt": int(ng)}


def all_metrics(c: dict) -> dict:
    """Single-sequence convenience wrapper (used by tests)."""
    return pool([{"hota": hota_from_counts(c), "idf1": idf1_from_counts(c), "clear": clear_from_counts(c)}])


def bootstrap(per_seq: dict[str, dict], keys=("HOTA", "IDF1", "MOTA", "DetA", "AssA"), n_boot: int = 300, seed: int = 0) -> dict[str, tuple[float, float, float]]:
    """``per_seq``: {sequence: seq_stats(...)}. Percentile bootstrap over whole sequences."""
    names = sorted(per_seq)
    rng = np.random.default_rng(seed)
    base = pool([per_seq[n] for n in names])
    vals = {k: [] for k in keys}
    for _ in range(n_boot):
        m = pool([per_seq[names[i]] for i in rng.integers(0, len(names), len(names))])
        for k in keys:
            vals[k].append(m[k])
    return {k: (base[k], float(np.quantile(vals[k], 0.025)), float(np.quantile(vals[k], 0.975))) for k in keys}
