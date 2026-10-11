"""Experimental 3-way tyre condition head: good / worn / defective (lane D, task 3a extension).

Label sources: the sameersambhare1 README defines NEW = "no visible wear", SERVICEABLE = "moderate wear but safe to use",
UNUSABLE = "severe wear or damage"; we map NEW -> good, SERVICEABLE -> worn, UNUSABLE -> defective (OUR mapping of the README text).
TyreNet / warcoder only have good / defective (no 'worn'), so 'worn' labels come from ONE small source (547 smartphone photos from
Indian tyre workshops). Therefore:
  * numbers are 5-fold cross-validation INSIDE sameer, folds grouped by pHash cluster (near-duplicate photos of one tyre stay
    together), i.e. in-source and optimistic about other workshops;
  * variant (b) adds the TyreNet good/defective images to every training fold to see whether more data helps.
Reuses the frozen DINOv2 features written by lanes.inspect.tyre_cls (runs/inspect/tyre_dino_lr/features.npz).

    scripts/box.sh -- python -P -m lanes.inspect.tyre_cls3 --exp tyre_dino_lr3
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from eval.stats import bootstrap_ci, confusion, metric, prop_metric
from lanes.inspect import common as C

LABELS = ["good", "worn", "defective"]
MAP3 = {"NEW": 0, "SERVICEABLE": 1, "UNUSABLE": 2}


def macro_f1(y, p, k=3):
    f = []
    for c in range(k):
        tp, fp, fn = ((y == c) & (p == c)).sum(), ((y != c) & (p == c)).sum(), ((y == c) & (p != c)).sum()
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def fit_predict(Xtr, ytr, Xte, Cc):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LogisticRegression(C=Cc, max_iter=4000, class_weight="balanced").fit((Xtr - mu) / sd, ytr)
    return clf.predict_proba((Xte - mu) / sd), (mu, sd, clf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="tyre_dino_lr3")
    ap.add_argument("--from-exp", default="tyre_dino_lr")
    ap.add_argument("--C", type=float, default=0.01)
    a = ap.parse_args()
    out = C.ensure(C.RUNS_I / a.exp)
    z = np.load(C.RUNS_I / a.from_exp / "features.npz", allow_pickle=False)
    F, path, y2, src, orig, split = z["f"], z["path"], z["y"], z["src"], z["orig"], z["split"]
    sm = src == "sameer"
    tn = (src == "tyrenet") & np.isin(split, ["train", "val"])
    y3_sm = np.array([MAP3[o] for o in orig[sm]])
    Xs, ps = F[sm], path[sm]
    cl = np.array(C.group_near_dups([str(p) for p in ps]))
    print("sameer", len(Xs), "clusters", len(set(cl)), "class counts", np.bincount(y3_sm, minlength=3).tolist(), flush=True)
    Xt, yt = F[tn], np.where(y2[tn] == 1, 2, 0)

    results, preds = [], {}
    for variant in ["sameer_only", "tyrenet_plus_sameer"]:
        P = np.zeros((len(Xs), 3))
        for tr, te in GroupKFold(n_splits=5).split(Xs, y3_sm, cl):
            Xtr, ytr = Xs[tr], y3_sm[tr]
            if variant == "tyrenet_plus_sameer":
                Xtr, ytr = np.vstack([Xt, Xtr]), np.concatenate([yt, ytr])
            P[te], _ = fit_predict(Xtr, ytr, Xs[te], a.C)
        pred = P.argmax(1)
        preds[variant] = P
        acc_k = int((pred == y3_sm).sum())
        units = list(zip(y3_sm.tolist(), pred.tolist()))
        mf = bootstrap_ci(units, lambda u: macro_f1(np.array([x for x, _ in u]), np.array([z_ for _, z_ in u])), n_boot=500)
        m = [prop_metric("accuracy_3class", acc_k, len(y3_sm)), metric("macro_f1", *mf)]
        for c, nm in enumerate(LABELS):
            m.append(prop_metric(f"recall[{nm}]", int(((y3_sm == c) & (pred == c)).sum()), int((y3_sm == c).sum())))
        maj = np.bincount(y3_sm).max() / len(y3_sm)
        m.append(metric("majority_class_baseline_accuracy", float(maj)))
        results.append({
            "id": f"inspect.tyre_condition3.{a.exp}.{variant}", "name": f"Tyre condition good/worn/defective, 5-fold CV inside sameer ({variant})", "lane": "inspect",
            "data_kind": "real", "domain": "smartphone photos from Indian tyre workshops (one source); not fixed camera",
            "dataset": "Kaggle sameersambhare1/tyre-condition-classification-dataset (CC BY-SA 4.0)" + (" + TyreNet good/defective in training (CC BY 4.0)" if variant != "sameer_only" else ""),
            "split": "5-fold CV grouped by pHash cluster", "n": len(y3_sm), "model": f"DINOv2 ViT-B/14 @448 frozen + logistic regression (C={a.C})",
            "metrics": m, "confusion": confusion([LABELS[i] for i in y3_sm], [LABELS[i] for i in pred], LABELS),
            "notes": "Mapping NEW->good, SERVICEABLE->worn, UNUSABLE->defective is ours (from the dataset README). 'worn' labels exist in this ONE source only, "
                     "so this is an in-source CV and says nothing about other workshops, tyre types or lighting."})
        print(variant, [(x["name"], round(x["value"], 3)) for x in m[:3]], flush=True)

    # final 3-way head: TyreNet(train+val) good/defective + ALL sameer, for the shipped model if the CV supports it
    Xall, yall = np.vstack([Xt, Xs]), np.concatenate([yt, y3_sm])
    mu, sd = Xall.mean(0), Xall.std(0) + 1e-6
    clf = LogisticRegression(C=a.C, max_iter=4000, class_weight="balanced").fit((Xall - mu) / sd, yall)
    z0 = np.load(C.RUNS_I / a.from_exp / "head.npz")
    np.savez(out / "head3.npz", mu=mu, sd=sd, w=clf.coef_, b=clf.intercept_, C=a.C, size=int(z0["size"]), backbone=str(z0["backbone"]))
    (out / "train_files.txt").write_text("\n".join(list(path[tn]) + list(ps)))
    # cross-dataset check of the FINAL 3-way head on sets it never saw: warcoder (good/defective) and the TyreNet in-source test.
    # Binary view: P(defective) = probability of class 'defective'; 'worn' counts as not defective.
    from lanes.inspect.tyre_cls import binary_metrics
    for key, mask, ds in [("warcoder", src == "warcoder", "Kaggle warcoder tyre-quality (CC BY 4.0), TyreNet near-duplicates removed"),
                          ("tyrenet_test", (src == "tyrenet") & (split == "test"), "TyreNet held-out 20 % (in-source)")]:
        Xe, ye = F[mask], y2[mask]
        P = clf.predict_proba((Xe - mu) / sd)
        m, cm = binary_metrics("", ye, P[:, 2])
        three = np.bincount(P.argmax(1), minlength=3).tolist()
        results.append({
            "id": f"inspect.tyre_condition3_final_head.{a.exp}.{key}", "name": f"Final good/worn/defective head viewed as good-vs-defective on {key}", "lane": "inspect",
            "data_kind": "real", "domain": "phone photo close-ups of tyres", "dataset": ds, "split": key, "n": int(mask.sum()),
            "model": f"DINOv2 ViT-B/14 @448 frozen + 3-class logistic regression trained on TyreNet(train+val) + all of sameer (C={a.C})",
            "metrics": m, "confusion": cm,
            "notes": f"Predicted 3-way counts [good, worn, defective] = {three}. worn counts as not defective. {'Cross-dataset: warcoder never used in training.' if key == 'warcoder' else 'In-source.'}"})
        print("final head on", key, [(x["name"], round(x["value"], 3)) for x in m if x["name"] in ("accuracy", "auroc", "recall[defective]", "recall[good]")], three, flush=True)
    C.write_json(results, out / "component_results_tyre_cond3.json")
    np.savez(out / "cv_probs.npz", **{k: v for k, v in preds.items()}, y=y3_sm)


if __name__ == "__main__":
    main()
