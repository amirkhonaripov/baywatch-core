"""Train the bay-state head on frozen crop embeddings + detector geometry; evaluate on held-out handheld/YouTube data.

  python -m lanes.ops.train --exp heads_v1 --backbone dinov2l [--teacher data/prep/ops/vlm_train.parquet]

Writes runs/ops/<exp>/{head.npz, metrics.json, component_results.json, train_files.txt, test_preds.parquet}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval.manifest import filter_train, split_by_source
from lanes.ops.featurize import FEAT, dets_by_frame
from lanes.ops.prep import fpath
from lanes.ops.geometry import geom_features
from lanes.ops.heads import HEAD_STATES, HeadModel, fit_head, fit_platt
from lanes.ops.metrics import auroc, state_metrics
from seen.paths import PREP, RAW, RUNS

OPS = PREP / "ops"
TEACHER_W = 0.5
E_DIM = 0  # weight of a teacher (VLM) soft label relative to a human/derived label


def load_features(backbone: str, table: pd.DataFrame) -> np.ndarray:
    ids = pd.read_parquet(FEAT / f"emb_{backbone}_ids.parquet")["frame_id"].tolist()
    E = np.load(FEAT / f"emb_{backbone}.npy").astype(np.float32)
    pos = {f: i for i, f in enumerate(ids)}
    E = E[[pos[f] for f in table["frame_id"]]]
    dets = dets_by_frame()
    parts = dets_by_frame(FEAT / "parts.parquet") if (FEAT / "parts.parquet").exists() else {}
    G = np.stack([geom_features(dets.get(f, []), None, parts.get(f)) for f in table["frame_id"]])
    return np.concatenate([E, G], 1)


def labels(table: pd.DataFrame, teacher: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Y (soft), weight, source flag (1 = human/derived, 2 = teacher, 0 = none) for HEAD_STATES."""
    n, s = len(table), len(HEAD_STATES)
    Y = np.zeros((n, s), np.float32)
    W = np.zeros((n, s), np.float32)
    F = np.zeros((n, s), np.int8)
    T = None
    if teacher is not None:
        T = teacher.set_index("item_id").reindex(table["frame_id"])
    for j, st in enumerate(HEAD_STATES):
        y = table[st].to_numpy(float) if st in table else np.full(n, np.nan)
        m = ~np.isnan(y)
        Y[m, j], W[m, j], F[m, j] = y[m], 1.0, 1
        if T is not None and f"p_{st}" in T:
            t = T[f"p_{st}"].to_numpy(float)
            mt = (~m) & ~np.isnan(t)
            # fixed-camera workshop clips have no other label and are closest to the bay domain: full weight
            wt = np.where(table["dataset"].to_numpy() == "domain", 1.0, TEACHER_W)
            Y[mt, j], W[mt, j], F[mt, j] = t[mt], wt[mt], 2
    return Y, W, F


def onset_errors(te: pd.DataFrame, state: str, on_steps) -> tuple[np.ndarray, int]:
    from lanes.ops.prep import _segments

    coin = json.loads((RAW / "ops_howto" / "COIN.json").read_text())["database"]
    errs, miss = [], 0
    if "video" not in te.columns:
        return np.asarray([]), 0
    for vid, g in te[te["dataset"].isin(["crosstask", "coin"])].groupby("video"):
        g = g.sort_values("t_s")
        ds = g["dataset"].iloc[0]
        tid = str(g["task_id"].iloc[0]) if "task_id" in g.columns else ""
        segs = _segments(ds, tid, vid, coin)
        ons = [e for name, _, e in segs if name in on_steps]
        if not ons or g[state].notna().sum() == 0:
            continue
        true_on = min(ons)
        p = g[f"p_{state}"].rolling(5, center=True, min_periods=1).median().to_numpy()
        t = g["t_s"].to_numpy()
        hit = np.where(p >= 0.5)[0]
        if len(hit) == 0:
            miss += 1
            continue
        errs.append(float(t[hit[0]] - true_on))
    return np.asarray(errs), miss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--backbone", default="dinov2l")
    ap.add_argument("--teacher", default="")
    ap.add_argument("--no-geom", action="store_true")
    a = ap.parse_args()
    out = RUNS / "ops" / a.exp
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(OPS / "table.parquet")
    X = load_features(a.backbone, df)
    global E_DIM
    E_DIM = X.shape[1] - 24
    if a.no_geom:
        X = X[:, :-24]
    teacher = pd.read_parquet(a.teacher) if a.teacher else None
    Y, W, F = labels(df, teacher)

    # train / val (by source, from the train split) / test
    tr_mask = (df["split"] == "train").to_numpy().copy()
    from eval.manifest import components as manifest_components

    # every held-out manifest (incl. the jury set: 7 Commons lift photos are jury items) -> path/id/pHash exclusion
    comps = manifest_components()
    dropped_set: set[str] = set()
    dropped = []
    if comps:
        _, dropped = filter_train([str(fpath(p)) for p in df.loc[tr_mask, "path"]], components=comps)
        dropped_set = {d["path"] for d in dropped}
        tr_mask &= ~df["path"].map(lambda p: str(fpath(p)) in dropped_set).to_numpy()
    (out / "dropped_by_manifests.json").write_text(json.dumps(dropped, indent=1, default=str))
    srcs = sorted(df.loc[tr_mask, "source"].unique())
    vs = split_by_source([dict(id=s, path=s, source=s) for s in srcs], test_frac=0.15, seed="seen-ops-val")
    val_src = {r["source"] for r in vs if r["split"] == "test"}
    va_mask = tr_mask & df["source"].isin(val_src).to_numpy()
    fit_mask = tr_mask & ~va_mask
    te_mask = (df["split"] == "test").to_numpy()
    (out / "train_files.txt").write_text("\n".join(str(fpath(p)) for p in df.loc[tr_mask, "path"]) + "\n")
    print("rows fit/val/test", fit_mask.sum(), va_mask.sum(), te_mask.sum(), "dropped by manifests", len(dropped_set), flush=True)

    # weight decay selection on val (mean AUROC over states with both classes in val, human/derived labels only)
    best = None
    for wd in (1e-4, 1e-3, 1e-2, 5e-2):
        mean, std, Wm, b = fit_head(X[fit_mask], Y[fit_mask], W[fit_mask], wd=wd)
        z = ((X[va_mask] - mean) / std) @ Wm + b
        aucs = []
        for j in range(len(HEAD_STATES)):
            m = F[va_mask, j] == 1
            if m.sum() > 20:
                v = auroc(Y[va_mask, j][m].round(), z[m, j])
                if np.isfinite(v):
                    aucs.append(v)
        score = float(np.mean(aucs)) if aucs else -1
        print("wd", wd, "val mean AUROC", round(score, 4), flush=True)
        if best is None or score > best[0]:
            best = (score, wd)
    wd = best[1]
    # final fit on fit split, Platt on val split (human/derived + teacher labels)
    mean, std, Wm, b = fit_head(X[fit_mask], Y[fit_mask], W[fit_mask], wd=wd)
    zv = ((X[va_mask] - mean) / std) @ Wm + b
    pa, pb = np.ones(len(HEAD_STATES)), np.zeros(len(HEAD_STATES))
    for j in range(len(HEAD_STATES)):
        m = W[va_mask, j] > 0
        yy = np.where(m, Y[va_mask, j], np.nan)
        pa[j], pb[j] = fit_platt(zv[:, j], yy, W[va_mask, j])
    model = HeadModel(mean, std, Wm, b, pa, pb, HEAD_STATES)
    model.save(out / "head.npz")

    # test predictions
    P = model.predict_proba(X[te_mask])
    te = df.loc[te_mask].reset_index(drop=True)
    from lanes.ops.geometry import GEOM_NAMES
    from lanes.ops.rules import apply_rules

    car_conf = X[te_mask][:, E_DIM + GEOM_NAMES.index("car_conf")] if not a.no_geom else np.ones(len(te))
    ruled = apply_rules({st: P[:, j] for j, st in enumerate(HEAD_STATES)}, car_conf)  # same rules as the predictor
    for j, st in enumerate(HEAD_STATES):  # handheld metrics use the learned head as-is ...
        te[f"p_{st}"] = P[:, j]
    for st, v in ruled.items():  # ... the deployed post-processing is stored alongside for reference
        te[f"pr_{st}"] = v
    keep_cols = ["frame_id", "dataset", "source", "video", "t_s", "path", "task_id"] + [c for c in te.columns if c.startswith(("p_", "pr_"))] + \
                [s for s in HEAD_STATES if s in te.columns]
    te[[c for c in keep_cols if c in te.columns]].to_parquet(out / "test_preds.parquet")

    results = []
    model_name = f"{a.backbone}+geom linear head ({a.exp})"
    # (1) work: K700 val service vs car-context classes
    for ds, name, split_desc in [("k700", "work: Kinetics-700 val service classes vs car-context classes", "K700 validation (held out by clip)"),
                                 ("crosstask", "work proxy: inside vs outside annotated steps (CrossTask)", "test videos (split by video)"),
                                 ("coin", "work proxy: inside vs outside annotated steps (COIN)", "test videos (split by video)")]:
        d = te[(te["dataset"] == ds) & te["work"].notna()]
        if len(d) and d["work"].nunique() == 2:
            ms, conf, cnt = state_metrics(d["work"], d["p_work"], d["source"])
            results.append(dict(id=f"ops_work_{ds}_{a.exp}", name=name, lane="ops", data_kind="real", domain="handheld/YouTube",
                                dataset=ds, split=split_desc, n=cnt["n"], model=model_name, metrics=ms, confusion=conf,
                                notes=f"frames={cnt['n']} positives={cnt['n_pos']} videos/clips={cnt['n_groups']}; CI bootstrap over videos"))
    # (2) raised / wheel_off from derived step spans
    for st in ("raised", "wheel_off"):
        d = te[te["dataset"].isin(["crosstask", "coin"]) & te[st].notna()]
        if len(d) and d[st].nunique() == 2:
            ms, conf, cnt = state_metrics(d[st], d[f"p_{st}"], d["source"])
            results.append(dict(id=f"ops_{st}_howto_{a.exp}", name=f"{st}: derived step-span labels (CrossTask+COIN)", lane="ops",
                                data_kind="real", domain="handheld/YouTube", dataset="crosstask+coin", split="test videos (split by video)",
                                n=cnt["n"], model=model_name, metrics=ms, confusion=conf,
                                notes=f"labels derived from step timestamps (jack up..jack down / withdraw..put wheel); frames={cnt['n']} "
                                      f"positives={cnt['n_pos']} videos={cnt['n_groups']}"))
    # (3) hood_open: Roboflow in-project test and cross-project
    for ds, split_desc in [("rf_car_hood_open", "Roboflow test split (same project)"),
                           ("rf_car_condition", "all images of another uploader's project (cross-project)"),
                           ("rf_car_open_hood", "Roboflow test split (classification project)")]:
        d = te[(te["dataset"] == ds) & te["hood_open"].notna()]
        if len(d) and d["hood_open"].nunique() == 2:
            ms, conf, cnt = state_metrics(d["hood_open"], d["p_hood_open"], d["source"])
            results.append(dict(id=f"ops_hood_{ds}_{a.exp}", name=f"hood_open: {ds}", lane="ops", data_kind="real",
                                domain="phone/web photos", dataset=ds, split=split_desc, n=cnt["n"], model=model_name,
                                metrics=ms, confusion=conf, notes=f"positives={cnt['n_pos']}"))
    # (4) step-onset error on derived spans: predicted onset = first time a 5-s median-filtered probability >= 0.5
    for st, on_steps in (("raised", ("jack up", "raise jack", "jack up the car", "raise the jack up")),
                         ("wheel_off", ("withdraw wheel", "remove the tire"))):
        errs, miss = onset_errors(te, st, on_steps)
        if len(errs) >= 3:
            from eval.stats import mae_metric, prop_metric

            ms = [mae_metric(errs, "onset_mae", "s"),
                  {"name": "onset_median_abs_error", "value": float(np.median(np.abs(errs))), "unit": "s"},
                  prop_metric("onset_within_10s", int((np.abs(errs) <= 10).sum()), len(errs))]
            results.append(dict(id=f"ops_{st}_onset_{a.exp}", name=f"{st}: step-onset error on CrossTask/COIN test videos",
                                lane="ops", data_kind="real", domain="handheld/YouTube", dataset="crosstask+coin",
                                split="test videos (split by video)", n=len(errs), model=model_name, metrics=ms,
                                notes=f"onset = end of the '{on_steps[0]}'-type step; prediction = first frame with 5-s median "
                                      f"filtered p>=0.5 (stand-in for Lane E HMM); videos with no predicted onset: {miss} "
                                      "(counted as misses, excluded from MAE)"))
    (out / "component_results.json").write_text(json.dumps(results, indent=1))
    (out / "metrics.json").write_text(json.dumps(dict(wd=wd, val_mean_auroc=best[0], backbone=a.backbone,
                                                      used_parts=(FEAT / "parts.parquet").exists() and not a.no_geom,
                                                      teacher=a.teacher, n_fit=int(fit_mask.sum()), n_val=int(va_mask.sum()),
                                                      n_test=int(te_mask.sum()), platt_a=pa.tolist(), platt_b=pb.tolist()), indent=1))
    for r in results:
        print(r["name"], r["n"], {m["name"]: round(m["value"], 3) for m in r["metrics"]})
    print("TRAIN_DONE", out)


if __name__ == "__main__":
    main()
