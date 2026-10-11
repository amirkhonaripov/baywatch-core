"""Tread depth in mm from TyreCV stereo pairs (lane D, task 3c; the cut-able item).

Data: Kaggle maxfinkenbeiner/tyrecv-two-views-are-all-you-need (MIT label): 508 stereo pairs (Raspberry Pi 5 + IMX219 stereo
camera, close-range, gauge-measured depth 1.5-8 mm in 0.5 mm steps). Images = the un-augmented copy of each pair
(lanes.inspect.fetch_tyrecv); exact mm from the original files' names.

Method: frozen DINOv2 ViT-B/14 features (@448) of the left and right view -> ridge regression; variants: left only, mean of both
views, concat(left, right). Grouped 5-fold CV where the group is the stereo pair (uuid). Physical-tyre identity is NOT in the
metadata: if several pairs show the same tyre the CV is optimistic (stated in the notes).
Baselines: predict the training-fold median; per-bucket accuracy (nearest 1 mm bucket) and within-1-mm rate.
External reference (not reproducible here, data unreleased): HF Space UnCodeSpecial/tire-tread-depth reports MAE ~0.35 mm on
730 single-view close-up photos.

    scripts/box.sh --gpu 1 -- python -P -m lanes.inspect.tread --exp tread_dino_ridge
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from eval.stats import bootstrap_ci, metric, prop_metric
from lanes.inspect import common as C
from lanes.inspect import tyre_features as TF
from seen.paths import RAW


def read_pairs() -> list[dict]:
    root = RAW / "inspect_tyrecv"
    labels = json.loads((root / "labels_mm.json").read_text())
    pairs = {}
    for p in sorted(root.rglob("*.png")):
        stem = p.stem  # <uuid>_<bucket>_<L|R>
        uuid, bucket, side = stem.rsplit("_", 2)[0], stem.rsplit("_", 2)[1], stem.rsplit("_", 2)[2]
        pairs.setdefault(uuid, {})[side] = str(p)
        pairs[uuid]["bucket"] = int(bucket)
    out = []
    for u, d in pairs.items():
        if u in labels and "L" in d and "R" in d:
            out.append({"id": u, "L": d["L"], "R": d["R"], "mm": float(labels[u]), "bucket": d["bucket"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="tread_dino_ridge")
    ap.add_argument("--size", type=int, default=448)
    a = ap.parse_args()
    out = C.ensure(C.RUNS_I / a.exp)
    rows = read_pairs()
    print("pairs", len(rows), flush=True)
    # stage to NVMe once (small PNGs -> JPEG <= 896 px)
    recs = []
    for r in rows:
        for side in "LR":
            recs.append({"id": f"tyrecv:{r['id']}_{side}", "path": r[side]})
    staged = {r["id"]: r["fast"] for r in C.stage_images(recs, C.FAST_I / "tyre" / "tyrecv", 896)}
    keep = [r for r in rows if f"tyrecv:{r['id']}_L" in staged and f"tyrecv:{r['id']}_R" in staged]
    paths = [staged[f"tyrecv:{r['id']}_{s}"] for r in keep for s in "LR"]
    feats = TF.extract(paths, size=a.size, cache_name=f"tyrecv_{len(keep)}").reshape(len(keep), 2, -1)
    y = np.array([r["mm"] for r in keep])
    groups = np.array([r["id"] for r in keep])
    variants = {"left_only": feats[:, 0], "mean_of_views": feats.mean(1), "concat_left_right": feats.reshape(len(keep), -1)}
    results, preds_out = [], {}
    for vname, X in variants.items():
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            best = None
            for al in [10.0, 100.0, 1000.0, 3000.0, 10000.0]:  # alpha picked by inner CV on the TRAIN fold only
                inner = np.zeros(len(tr))
                for itr, ite in GroupKFold(n_splits=4).split(X[tr], y[tr], groups[tr]):
                    m = Ridge(alpha=al).fit((X[tr][itr] - mu) / sd, y[tr][itr])
                    inner[ite] = m.predict((X[tr][ite] - mu) / sd)
                mae = np.abs(inner - y[tr]).mean()
                if best is None or mae < best[0]:
                    best = (mae, al)
            m = Ridge(alpha=best[1]).fit((X[tr] - mu) / sd, y[tr])
            pred[te] = np.clip(m.predict((X[te] - mu) / sd), 1.0, 9.0)
        preds_out[vname] = pred
    # baseline: training-fold median
    base = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(feats[:, 0], y, groups):
        base[te] = np.median(y[tr])
    for vname, pred in {**preds_out, "baseline_train_median": base}.items():
        err = np.abs(pred - y)
        m = [metric("mae_mm", *bootstrap_ci(err, np.mean, n_boot=1000), unit="mm"),
             metric("rmse_mm", float(np.sqrt(np.mean((pred - y) ** 2))), unit="mm"),
             prop_metric("within_1mm_rate", int((err <= 1.0).sum()), len(y)),
             prop_metric("within_0.5mm_rate", int((err <= 0.5).sum()), len(y)),
             prop_metric("bucket_accuracy_nearest_mm", int((np.round(pred) == np.round(y)).sum()), len(y)),
             metric("pearson_r", float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 0 else 0.0)]
        results.append({
            "id": f"inspect.tread_depth.{a.exp}.{vname}", "name": f"Tread depth (mm) from a stereo pair, {vname}", "lane": "inspect", "data_kind": "real",
            "domain": "lab-style close-range stereo pair (Raspberry Pi 5 + IMX219 stereo camera); NOT a phone photo and NOT fixed CCTV",
            "dataset": "Kaggle maxfinkenbeiner/tyrecv-two-views-are-all-you-need (MIT label)", "split": "grouped 5-fold CV (group = stereo pair)", "n": len(y),
            "model": "DINOv2 ViT-B/14 @448 frozen features + ridge (alpha by inner CV)" if vname != "baseline_train_median" else "predict training-fold median",
            "metrics": m,
            "notes": "Label range 1.5-8 mm in 0.5 mm steps (few extreme depths). Group = stereo pair uuid; physical tyre identity unknown, so CV may be optimistic. "
                     "Single-camera phone accuracy is NOT measured. External reference (unreproducible): HF Space reports about 0.35 mm MAE on 730 single-view photos."})
    C.write_json(results, out / "component_results_tread.json")
    C.write_json({"id": [r["id"] for r in keep], "mm": y.tolist(), **{k: v.tolist() for k, v in preds_out.items()}, "baseline": base.tolist()}, out / "preds_tread.json")
    for r in results:
        print(r["id"], [(m["name"], round(m["value"], 3)) for m in r["metrics"][:3]])


if __name__ == "__main__":
    main()
