"""Slot occupancy from detector boxes + slot polygons (seen.zones) on PKLot and CNRPark-EXT.

    python -m lanes.flow.eval_occupancy --model rfdetr-large --exp occupancy_rfdetr-large

Protocol
  * Detector: zero-shot COCO detector (vehicle classes), run by run_detect on the 'pklot_occ' / 'pklot_cal' / 'cnrext_occ' /
    'cnrext_cal' item lists (full frames; PKLot 1280x720, CNR-EXT 1000x750).
  * A slot is "occupied" if the best vehicle box with score >= c covers >= tau of the slot polygon while >= 20% of that box lies inside
    it (``seen.zones.slot_occupancy_score``). (c, tau) is chosen to maximise F1 on a *calibration* set drawn from other days.
  * Same-dataset: calibrate on PKLot calibration days, test on PKLot test days, per weather (sunny/cloudy/rainy).
    Cross-dataset: calibrate on PKLot only, test on CNR-EXT (sunny/overcast/rainy) and the reverse (CNR-EXT -> PKLot).
  * Metrics: accuracy, precision/recall/F1 for 'occupied'. 95% CI = bootstrap over images (slots in one frame are not independent).
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict

import numpy as np

from eval.stats import metric
from lanes.flow.common import VEHICLE_IDX, component, save_results
from lanes.flow.run_detect import cache_path, get_items
from seen.zones import slot_occupancy_score

CONFS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
TAUS = tuple(np.round(np.arange(0.05, 0.9, 0.05), 2))


def slot_scores(item, dets: np.ndarray, conf: float) -> np.ndarray:
    """Overlap score for every slot of an item (vectorised pre-filter, exact polygon clipping for candidates)."""
    v = dets[np.isin(dets[:, 5].astype(int), VEHICLE_IDX) & (dets[:, 4] >= conf)][:, :4]
    out = np.zeros(len(item.extra["slots"]))
    if len(v) == 0:
        return out
    for k, s in enumerate(item.extra["slots"]):
        poly = np.asarray(s["poly"])
        x1, y1 = poly.min(0)
        x2, y2 = poly.max(0)
        cand = v[(v[:, 2] > x1) & (v[:, 0] < x2) & (v[:, 3] > y1) & (v[:, 1] < y2)]
        if len(cand):
            out[k] = slot_occupancy_score([tuple(b) for b in cand], poly)
    return out


def collect(dataset: str, model: str):
    """-> list of dicts {cond, cluster, truth (S,), scores {conf: (S,)}}."""
    items = get_items(dataset)
    blob = pickle.load(open(cache_path(dataset, model), "rb"))
    rows = []
    for it in items:
        d = blob["dets"].get(it.id)
        if d is None:
            continue
        rows.append({"id": it.id, "cond": it.cond, "cluster": it.cluster, "lot": it.extra.get("lot"),
                     "truth": np.array([s["occ"] for s in it.extra["slots"]], dtype=int),
                     "scores": {c: slot_scores(it, d, c) for c in CONFS}})
    return rows


def counts(rows, conf, tau):
    tp = fp = fn = tn = 0
    for r in rows:
        p = r["scores"][conf] >= tau
        t = r["truth"] == 1
        tp += int((p & t).sum())
        fp += int((p & ~t).sum())
        fn += int((~p & t).sum())
        tn += int((~p & ~t).sum())
    return tp, fp, fn, tn


def f1_of(c):
    tp, fp, fn, tn = c
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def calibrate(rows):
    best = (-1.0, None, None)
    for c in CONFS:
        for t in TAUS:
            f = f1_of(counts(rows, c, t))
            if f > best[0]:
                best = (f, c, float(t))
    return best


def boot(rows, conf, tau, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    per = [counts([r], conf, tau) for r in rows]
    per = np.array(per, dtype=float)

    def stats(sel):
        tp, fp, fn, tn = per[sel].sum(0)
        n = tp + fp + fn + tn
        return {"accuracy": (tp + tn) / n, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "F1": 2 * tp / max(2 * tp + fp + fn, 1)}

    base = stats(np.arange(len(rows)))
    bs = defaultdict(list)
    for _ in range(n_boot):
        s = stats(rng.integers(0, len(rows), len(rows)))
        for k, v in s.items():
            bs[k].append(v)
    return {k: (base[k], float(np.quantile(bs[k], 0.025)), float(np.quantile(bs[k], 0.975))) for k in base}, per.sum(0)


def run(model: str, exp: str, n_boot: int = 500):
    data = {k: collect(k, model) for k in ("pklot_cal", "pklot_occ", "cnrext_cal", "cnrext_occ")}
    results, table = [], []
    cal = {"pklot": calibrate(data["pklot_cal"]), "cnrext": calibrate(data["cnrext_cal"])}
    print("calibration (F1, conf, tau):", cal)
    protocols = [("pklot", "pklot_occ", "same-dataset (calibrated on PKLot calibration days)"),
                 ("pklot", "cnrext_occ", "cross-dataset: calibrated on PKLot only, tested on CNR-EXT"),
                 ("cnrext", "cnrext_occ", "same-dataset (calibrated on CNR-EXT calibration days)"),
                 ("cnrext", "pklot_occ", "cross-dataset: calibrated on CNR-EXT only, tested on PKLot")]
    for cal_name, test_name, desc in protocols:
        f, c, t = cal[cal_name]
        rows_all = data[test_name]
        conds = sorted({r["cond"] for r in rows_all})
        for cond in conds + ["all"]:
            rows = [r for r in rows_all if cond == "all" or r["cond"] == cond]
            if not rows:
                continue
            m, cnt = boot(rows, c, t, n_boot)
            n_slots = int(cnt.sum())
            occ_rate = (cnt[0] + cnt[2]) / max(cnt.sum(), 1)
            ms = [metric(k, v[0], v[1], v[2], "fraction") for k, v in m.items()]
            ms.append(metric("majority_class_accuracy", max(occ_rate, 1 - occ_rate), unit="fraction"))
            ms.append(metric("occupied_rate", occ_rate, unit="fraction"))
            rid = f"flow.occ.{cal_name}_to_{test_name.split('_')[0]}.{model}.{cond}"
            results.append(component(rid, f"Slot occupancy ({cond})", test_name.split("_")[0], len(rows), ms, split=cond, model=model,
                                     domain="parking camera",
                                     notes=f"{desc}; conf>={c}, tau={t} (calibration F1 {f:.3f}); {n_slots} slots in {len(rows)} images; bootstrap over images"))
            table.append((cal_name, test_name.split("_")[0], cond, len(rows), n_slots, m["accuracy"], m["F1"], m["precision"], m["recall"], occ_rate))
    p = save_results(exp, results)
    md = "| calibrated on | tested on | weather | images | slots | accuracy | F1 | precision | recall | occupied rate |\n|---|---|---|---|---|---|---|---|---|---|\n"
    for a, b, cnd, ni, ns, acc, f1, pr, rc, oc in table:
        md += f"| {a} | {b} | {cnd} | {ni} | {ns} | {acc[0]:.3f} [{acc[1]:.3f},{acc[2]:.3f}] | {f1[0]:.3f} [{f1[1]:.3f},{f1[2]:.3f}] | {pr[0]:.3f} | {rc[0]:.3f} | {oc:.2f} |\n"
    (p.parent / "table.md").write_text(md)
    (p.parent / "calibration.json").write_text(json.dumps({k: {"F1": v[0], "conf": v[1], "tau": v[2]} for k, v in cal.items()}))
    print(md)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args()
    run(a.model, a.exp, a.n_boot)
