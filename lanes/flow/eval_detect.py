"""AP50/AP75 per condition from cached detections (see run_detect.py) -> ComponentResult JSON + markdown table.

    python -m lanes.flow.eval_detect --dataset detrac_test_s5 --models rfdetr-base,rfdetr-large,yolo11x,yolo26x --exp zeroshot_detrac

Protocol (class-agnostic "vehicle", as in the DETRAC benchmark): every predicted car/motorcycle/bus/truck box counts as a
vehicle detection; predictions whose area lies >50% inside an ignored region are dropped (DETRAC); AP is all-point
interpolated (VOC2010+, same function as eval.stats.average_precision). 95% CIs are percentile bootstrap over
independent units: sequences for DETRAC (40 test sequences), images for NDISPark.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict

import numpy as np

from eval.stats import metric
from lanes.flow import data as D
from lanes.flow.common import APAccumulator, VEHICLE_IDX, component, save_results
from lanes.flow.run_detect import cache_path, get_items


def filter_ignored(d: np.ndarray, ignore: np.ndarray, thr: float = 0.5) -> np.ndarray:
    if len(d) == 0 or len(ignore) == 0:
        return d
    keep = np.ones(len(d), dtype=bool)
    area = np.maximum((d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1]), 1e-9)
    for r in ignore:
        iw = np.clip(np.minimum(d[:, 2], r[2]) - np.maximum(d[:, 0], r[0]), 0, None)
        ih = np.clip(np.minimum(d[:, 3], r[3]) - np.maximum(d[:, 1], r[1]), 0, None)
        keep &= (iw * ih / area) <= thr
    return d[keep]


def vehicle_preds(d: np.ndarray, target: str = "vehicle") -> np.ndarray:
    m = np.isin(d[:, 5].astype(int), VEHICLE_IDX if target == "vehicle" else (0,))
    return d[m][:, :5]


def target_gt(it, target: str) -> np.ndarray:
    if target == "person" and "gt_person" in it.extra:
        return np.asarray(it.extra["gt_person"], dtype=np.float32).reshape(-1, 4)
    return it.gt


def diff_ci(fa, ca, fb, cb, n_boot: int = 500, seed: int = 1):
    """Bootstrap CI for fa(ca) - fb(cb) with independent resampling of the two cluster sets."""
    rng = np.random.default_rng(seed)
    base = fa(ca) - fb(cb)
    vals = []
    for _ in range(n_boot):
        a = [ca[i] for i in rng.integers(0, len(ca), len(ca))]
        b = [cb[i] for i in rng.integers(0, len(cb), len(cb))]
        v = fa(a) - fb(b)
        if np.isfinite(v):
            vals.append(v)
    return float(base), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def evaluate(dataset: str, model: str, op_conf: float = 0.4, n_boot: int = 500, target: str = "vehicle"):
    items = get_items(dataset)
    blob = pickle.load(open(cache_path(dataset, model), "rb"))
    dets, meta = blob["dets"], blob["meta"]
    img_cluster = dataset.startswith(("ndispark", "cocoval", "miotcd"))
    accs: dict[str, APAccumulator] = defaultdict(APAccumulator)
    n_img: dict[str, int] = defaultdict(int)
    for it in items:
        if it.id not in dets:
            continue
        p = filter_ignored(vehicle_preds(dets[it.id], target), it.ignore)
        cl = it.id if img_cluster else it.cluster
        for g in (it.cond, "all"):
            accs[g].add(cl, p, target_gt(it, target))
            n_img[g] += 1
    return accs, n_img, meta


def results_for(dataset: str, model: str, accs, n_img, meta, notes: str = "", n_boot: int = 500, op_conf: float = 0.4, target: str = "vehicle") -> list[dict]:
    out = []
    for g, acc in sorted(accs.items()):
        ms = acc.metrics(n_boot=n_boot) + acc.op_metrics(op_conf, n_boot=n_boot)
        out.append(component(f"flow.det.{dataset}.{model}.{g}" + ("" if target == "vehicle" else f".{target}"), f"{target.capitalize()} detection ({g})", dataset, n_img[g], ms, split=g, model=model,
                             domain={"detrac": "fixed CCTV", "ndispark": "parking camera", "miotcd": "traffic cameras", "cocoval": "web photos"}.get(dataset.split("_")[0], "fixed CCTV"),
                             notes=f"{len(acc.by_cluster)} bootstrap units, {acc.n_gt} GT boxes; {notes}".strip("; ")))
    # day -> night / clear -> rain gaps
    gaps = []
    if dataset.startswith("detrac"):
        day = [c for g in ("sunny", "cloudy") for c in accs[g].by_cluster] if "sunny" in accs else []
        pairs = [("night", ["night"]), ("rainy", ["rainy"])]
        for nm, gs in pairs:
            if gs[0] in accs and day:
                for t, tn in ((0.5, "AP50"), (0.75, "AP75")):
                    a = accs["all"]
                    fa = lambda cl, t=t: a.ap(cl, t)  # noqa: E731
                    cb = [c for g in gs for c in accs[g].by_cluster]
                    v, lo, hi = diff_ci(fa, day, fa, cb, n_boot)
                    gaps.append(metric(f"{tn}_drop_clear_to_{nm}", v, lo, hi, "fraction"))
    elif dataset == "ndispark_pair":
        pass
    if gaps:
        out.append(component(f"flow.det.{dataset}.{model}.gaps", "Condition gap (clear=sunny+cloudy minus condition)", dataset,
                             n_img["all"], gaps, split="gap", model=model, domain="fixed CCTV",
                             notes="positive = AP lost in the harder condition; bootstrap over sequences, groups resampled independently"))
    return out


def table(rows: list[dict], conds: list[str]) -> str:
    hdr = "| model | " + " | ".join(f"{c} AP50 / AP75" for c in conds) + " |\n|---|" + "---|" * len(conds) + "\n"
    s = hdr
    for r in rows:
        s += f"| {r['model']} | " + " | ".join(r.get(c, "-") for c in conds) + " |\n"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--op-conf", type=float, default=0.4)
    ap.add_argument("--target", default="vehicle", choices=["vehicle", "person"])
    a = ap.parse_args()
    allres, trows = [], []
    for m in a.models.split(","):
        accs, n_img, meta = evaluate(a.dataset, m, a.op_conf, a.n_boot, a.target)
        res = results_for(a.dataset, m, accs, n_img, meta, f"inference {meta['ms_per_img_infer']:.1f} ms/img (batch {meta['batch']}, {meta['env'].get('gpu', 'cpu')})",
                          a.n_boot, a.op_conf, a.target)
        allres += res
        row = {"model": m}
        for r in res:
            if r["split"] in ("gap",):
                continue
            ms = {x["name"]: x for x in r["metrics"]}
            row[r["split"]] = f"{ms['AP50']['value']:.3f} [{ms['AP50'].get('ci_low', float('nan')):.2f},{ms['AP50'].get('ci_high', float('nan')):.2f}] / {ms['AP75']['value']:.3f}"
        trows.append(row)
        print(json.dumps({r["id"]: {x["name"]: round(x["value"], 4) for x in r["metrics"]} for r in res}, indent=0)[:3000])
    p = save_results(a.exp, allres)
    conds = sorted({r["split"] for r in allres if r["split"] != "gap"})
    md = table(trows, conds)
    (p.parent / "table.md").write_text(md)
    print(md)
    print("wrote", p)


if __name__ == "__main__":
    main()
