"""Tyre condition classifier (lane D, task 3a): good vs defective, trained on TyreNet, tested cross-dataset.

Data
  TyreNet (Mendeley 32b5vfj6tc, CC BY 4.0): Good / Defective folders. Re-split 70 / 10 / 20 (train / val / in-source test) by
    pHash cluster. The in-source test is therefore optimistic (same workshops / phones); the CROSS-dataset numbers are the ones to quote.
  Kaggle warcoder/tyre-quality-classification (CC BY 4.0, 1,856 files): cross-dataset test AFTER removing every image that has a
    pHash near-duplicate in TyreNet (the overlap rate is itself a reported finding).
  Kaggle sameersambhare1/tyre-condition-classification-dataset (CC BY-SA 4.0): NEW + SERVICEABLE -> good, UNUSABLE -> defective
    (our mapping; a 3-way view of the score by original class is also reported).
Model: frozen DINOv2 ViT-B/14 features @448 + L2-regularised logistic regression; C selected on TyreNet val AUROC only.

Run (GPU1, inference):  scripts/box.sh --gpu 1 -- python -P -m lanes.inspect.tyre_cls --exp tyre_dino_lr
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from eval.stats import accuracy_coverage, bootstrap_ci, confusion, ece, metric, prop_metric
from lanes.inspect import common as C
from lanes.inspect import tyre_features as TF
from seen.paths import RAW

LABELS = ["good", "defective"]


def read_tyrenet() -> list[dict]:
    root = RAW / "inspect_tyrenet" / "extracted" / "Tyre dataset"
    out = []
    for p in C.list_images(root):
        lab = 1 if p.parent.name.lower().startswith("def") else 0
        out.append({"path": str(p), "y": lab, "src": "tyrenet", "orig": p.parent.name})
    return out


def read_warcoder() -> list[dict]:
    root = RAW / "inspect_tyre_warcoder"
    out = []
    for p in C.list_images(root):
        par = p.parent.name.lower()
        if "def" in par:
            lab = 1
        elif "good" in par:
            lab = 0
        else:
            continue
        out.append({"path": str(p), "y": lab, "src": "warcoder", "orig": p.parent.name})
    return out


def read_sameer() -> list[dict]:
    root = RAW / "inspect_tyre_sameer"
    out = []
    for p in C.list_images(root):
        par = p.parent.name.upper()
        if par not in ("NEW", "SERVICEABLE", "UNUSABLE"):
            continue
        out.append({"path": str(p), "y": 1 if par == "UNUSABLE" else 0, "src": "sameer", "orig": par})
    return out


def binary_metrics(prefix: str, y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> tuple[list[dict], dict]:
    pred = (p >= thr).astype(int)
    n = len(y)
    m = [prop_metric(f"{prefix}accuracy", int((pred == y).sum()), n)]
    pos, neg = y == 1, y == 0
    if pos.any() and neg.any():
        m.append(prop_metric(f"{prefix}recall[defective]", int((pred[pos] == 1).sum()), int(pos.sum())))
        m.append(prop_metric(f"{prefix}recall[good]", int((pred[neg] == 0).sum()), int(neg.sum())))
        if (pred == 1).any():
            m.append(prop_metric(f"{prefix}precision[defective]", int(((pred == 1) & pos).sum()), int((pred == 1).sum())))
        units = list(zip(y.tolist(), p.tolist()))
        pt, lo, hi = bootstrap_ci(units, lambda u: roc_auc_score([a for a, _ in u], [b for _, b in u]) if len({a for a, _ in u}) == 2 else float("nan"),
                                  n_boot=500)
        m.append(metric(f"{prefix}auroc", pt, lo, hi))

        def f1(u):
            a = np.array([x for x, _ in u]); b = np.array([int(z >= thr) for _, z in u])
            tp, fp, fn = ((a == 1) & (b == 1)).sum(), ((a == 0) & (b == 1)).sum(), ((a == 1) & (b == 0)).sum()
            return 2 * tp / max(2 * tp + fp + fn, 1)
        pt, lo, hi = bootstrap_ci(units, f1, n_boot=500)
        m.append(metric(f"{prefix}f1[defective]", pt, lo, hi))
        m.append(metric(f"{prefix}balanced_accuracy", 0.5 * ((pred[pos] == 1).mean() + (pred[neg] == 0).mean())))
    conf = np.maximum(p, 1 - p)
    m.append(metric(f"{prefix}ece", ece(conf, (pred == y).astype(int))))
    cm = confusion([LABELS[i] for i in y], [LABELS[i] for i in pred], LABELS)
    return m, cm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="tyre_dino_lr")
    ap.add_argument("--size", type=int, default=448)
    a = ap.parse_args()
    out = C.ensure(C.RUNS_I / a.exp)

    tn, wc, sm = read_tyrenet(), read_warcoder(), read_sameer()
    print("raw: tyrenet", len(tn), "warcoder", len(wc), "sameer", len(sm), flush=True)
    # ONE sequential staging pass HDD -> NVMe (<= 1024 px long side); everything below reads the NVMe copies
    staged = {}
    for name, rows in [("tyrenet", tn), ("warcoder", wc), ("sameer", sm)]:
        for i, r in enumerate(rows):
            r["id"] = f"{name}:{Path(r['path']).parent.name}/{Path(r['path']).name}"
        rows[:] = C.stage_images(rows, C.FAST_I / "tyre" / name, 1024)
        for r in rows:
            r["raw"], r["path"] = r["path"], r["fast"]
    print("staged", len(tn), len(wc), len(sm), flush=True)

    # --- TyreNet split by near-duplicate cluster
    cl = C.group_near_dups([r["path"] for r in tn])
    for r, c in zip(tn, cl):
        u = C.seeded_fraction(f"tyrenet-{c}")
        r["split"] = "test" if u < 0.20 else "val" if u < 0.30 else "train"
    # --- cross-dataset: remove near-duplicates of ANY TyreNet image (all splits) from the test side
    tn_paths = [r["path"] for r in tn]
    overlap = {}
    cross = {}
    for name, rows in [("warcoder", wc), ("sameer", sm)]:
        tw = C.near_dup_matches([r["path"] for r in rows], tn_paths) if rows else {}
        overlap[name] = {"n": len(rows), "near_dup_of_tyrenet": len(tw),
                         "examples": [{"a": k, "b": v[0], "dist": v[1]} for k, v in list(tw.items())[:5]]}
        cross[name] = [r for r in rows if r["path"] not in tw]
    # duplicates inside a cross set (burst photos) are kept: they are test items, not training
    print(json.dumps(overlap, indent=1)[:1500], flush=True)

    allrows = tn + cross["warcoder"] + cross["sameer"]
    feats = TF.extract([r["path"] for r in allrows], size=a.size, cache_name=f"tyre_all_{len(allrows)}")
    F = {id(r): feats[i] for i, r in enumerate(allrows)}
    np.savez(out / "features.npz", f=feats, path=np.array([r["path"] for r in allrows]), y=np.array([r["y"] for r in allrows]),
             src=np.array([r["src"] for r in allrows]), orig=np.array([r["orig"] for r in allrows]),
             split=np.array([r.get("split", "cross") for r in allrows]))  # reused by lanes.inspect.tyre_cls3

    def XY(rows):
        return np.stack([F[id(r)] for r in rows]), np.array([r["y"] for r in rows])

    Xtr, ytr = XY([r for r in tn if r["split"] == "train"])
    Xva, yva = XY([r for r in tn if r["split"] == "val"])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    best = None
    for Cc in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        clf = LogisticRegression(C=Cc, max_iter=3000, class_weight="balanced").fit((Xtr - mu) / sd, ytr)
        auc = roc_auc_score(yva, clf.predict_proba((Xva - mu) / sd)[:, 1])
        print("C", Cc, "val auroc", round(auc, 4), flush=True)
        if best is None or auc > best[0]:
            best = (auc, Cc)
    clf = LogisticRegression(C=best[1], max_iter=3000, class_weight="balanced").fit(
        (np.vstack([Xtr, Xva]) - mu) / sd, np.concatenate([ytr, yva]))  # refit on train+val with the chosen C
    np.savez(out / "head.npz", mu=mu, sd=sd, w=clf.coef_[0], b=clf.intercept_[0], C=best[1], size=a.size, backbone=TF.BACKBONE)
    (out / "train_files.txt").write_text("\n".join(r["path"] for r in tn if r["split"] in ("train", "val")))  # staged NVMe paths

    results = []
    sets = [("tyrenet_test", [r for r in tn if r["split"] == "test"], "TyreNet held-out 20 % (in-source)", "Mendeley TyreNet (CC BY 4.0)"),
            ("warcoder", cross["warcoder"], "Kaggle warcoder tyre-quality, minus TyreNet near-duplicates (cross-dataset)", "Kaggle warcoder/tyre-quality-classification (CC BY 4.0)"),
            ("sameer", cross["sameer"], "Kaggle sameersambhare1 tyre-condition (NEW+SERVICEABLE=good, UNUSABLE=defective) (cross-dataset)", "Kaggle sameersambhare1/tyre-condition-classification-dataset (CC BY-SA 4.0)")]
    for key, rows, disp, ds in sets:
        if not rows:
            continue
        X, y = XY(rows)
        p = clf.predict_proba((X - mu) / sd)[:, 1]
        m, cm = binary_metrics("", y, p)
        ac = accuracy_coverage(np.maximum(p, 1 - p), ((p >= 0.5).astype(int) == y).astype(int))
        extra = ""
        if key == "sameer":
            by = {}
            for r, pp in zip(rows, p):
                by.setdefault(r["orig"], []).append(pp)
            extra = " Mean P(defective) by original class: " + ", ".join(f"{k}={np.mean(v):.2f} (n={len(v)})" for k, v in sorted(by.items()))
        results.append({
            "id": f"inspect.tyre_condition.{a.exp}.{key}", "name": f"Tyre condition good/defective on {disp}", "lane": "inspect", "data_kind": "real",
            "domain": "phone photo close-ups of tyres (service stations / workshops); not fixed camera", "dataset": ds, "split": key, "n": len(rows),
            "model": f"DINOv2 ViT-B/14 @{a.size} frozen + logistic regression (C={best[1]}, chosen on TyreNet val AUROC={best[0]:.3f})",
            "metrics": m, "confusion": cm,
            "notes": ("Threshold 0.5 on P(defective). " + (f"{overlap.get(key, {}).get('near_dup_of_tyrenet', 0)} of {overlap.get(key, {}).get('n', 0)} source images were pHash near-duplicates of TyreNet and were removed. " if key in overlap else "In-source: same sources as training, optimistic. ")
                      + "accuracy_coverage: " + json.dumps(ac) + extra)})
        with open(out / f"preds_{key}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "orig_class", "p_defective"])
            for r, pp in zip(rows, p):
                w.writerow([r["path"], LABELS[r["y"]], r["orig"], f"{pp:.4f}"])
        print(key, len(rows), [(x["name"], round(x["value"], 3)) for x in m if x["name"] in ("accuracy", "auroc", "f1[defective]")], flush=True)
    C.write_json({"overlap": overlap, "val_auroc": best[0], "C": best[1],
                  "n_train": int(len(ytr)), "n_val": int(len(yva))}, out / "overlap_and_selection.json")
    C.write_json(results, out / "component_results_tyre_cond.json")


if __name__ == "__main__":
    main()
