"""Real-domain test: public fixed-camera workshop clips (CCTV / timelapse), hand-labelled bay ROIs. Never trained on.

  python -m lanes.ops.eval_cctv detect                    # RF-DETR context for the 420 frames -> FAST/ops/cctv/dets.parquet
  python -m lanes.ops.eval_cctv vlm_items                 # items parquet for lanes.ops.vlm
  python -m lanes.ops.eval_cctv predict --exp <exp>       # run lanes.ops.predict per bay sequence -> runs/ops/<exp>/cctv_preds.csv
  python -m lanes.ops.eval_cctv metrics --exp <exp> [--vlm runs/ops/vlm/cctv_vlm.parquet]
Bootstrap unit = (clip, bay ROI) sequence, so the CIs reflect that frames of one bay are strongly correlated.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from lanes.ops.metrics import state_metrics
from seen.paths import FAST, RUNS
from seen.predictor import Det, Frame, FramePred

HERE = Path(__file__).resolve().parent / "labels"
CF = FAST / "ops" / "cctv"
STATES = ["car_present", "raised", "wheel_off", "hood_open", "person_at_car", "person_under_car", "work"]


def load_items() -> pd.DataFrame:
    it = pd.read_csv(HERE / "cctv_items.csv")
    lab = pd.read_csv(HERE / "cctv_labels.csv")
    df = it.merge(lab, on="item_id")
    df = df[df["exclude"] == 0].copy()
    df["group"] = df["clip"] + "|" + df["roi_id"]
    return df.reset_index(drop=True)


def detect() -> None:
    from seen.baseline import PretrainedDetector

    idx = pd.read_csv(CF / "index.csv")
    det = PretrainedDetector(threshold=0.3)
    det.load("cuda:0")
    rows = []
    for r in idx.itertuples():
        img = np.asarray(Image.open(CF / "frames" / r.frame).convert("RGB"))
        p = det.predict([Frame(0, 0.0, img)])[0]
        for d in p.detections:
            rows.append((r.frame, d.cls, d.confidence, *d.bbox))
    pd.DataFrame(rows, columns=["frame", "cls", "conf", "x1", "y1", "x2", "y2"]).to_parquet(CF / "dets.parquet")
    print("DETECT_DONE", len(rows))


def vlm_items() -> None:
    df = load_items()
    out = df[["item_id", "roi_x1", "roi_y1", "roi_x2", "roi_y2"]].copy()
    out["path"] = [str(CF / "frames" / f) for f in df["frame"]]
    out.to_parquet(CF / "vlm_items.parquet")
    print("items", len(out))


def predict(exp: str, fixed_camera: bool, tag: str) -> None:
    from lanes.ops.predict import get_predictor

    df = load_items()
    dets = pd.read_parquet(CF / "dets.parquet")
    by_frame: dict[str, list[Det]] = {}
    for r in dets.itertuples():
        by_frame.setdefault(r.frame, []).append(Det(cls=r.cls, confidence=float(r.conf), bbox=(r.x1, r.y1, r.x2, r.y2)))
    rows = []
    timing = []
    pred = None
    for g, gd in df.groupby("group", sort=False):
        gd = gd.sort_values("k")
        roi = tuple(gd.iloc[0][["roi_x1", "roi_y1", "roi_x2", "roi_y2"]].astype(float))
        pred = get_predictor(roi=roi, fixed_camera=fixed_camera, artifact_dir=RUNS / "ops" / exp / "artifact")
        if not hasattr(predict, "_loaded"):
            pred.load("cuda:0")
            predict._loaded = pred  # reuse heavy models across groups
        else:
            src = predict._loaded
            pred.card, pred.device, pred.states, pred.thresholds = src.card, src.device, src.states, src.thresholds
            pred._ort, pred.bb, pred.parts = src._ort, src.bb, src.parts
            if src._ort is None:
                pred._head = src._head
        frames = [Frame(index=i, t_s=float(r.t_s), image=np.asarray(Image.open(CF / "frames" / r.frame).convert("RGB")))
                  for i, r in enumerate(gd.itertuples())]
        ctx = [FramePred(index=i, t_s=f.t_s, detections=by_frame.get(r.frame, [])) for i, (f, r) in enumerate(zip(frames, gd.itertuples()))]
        out = pred.predict(frames, ctx)
        for r, o in zip(gd.itertuples(), out):
            rows.append({"item_id": r.item_id, "group": g, **{f"p_{k}": v for k, v in o.states.items()}})
            timing.append(dict(o.timings_ms))
    p = pd.DataFrame(rows)
    d = RUNS / "ops" / exp
    p.to_csv(d / f"cctv_preds{tag}.csv", index=False)
    t = pd.DataFrame(timing)
    tj = {f"{c.replace('ops_', '')}_ms": float(t[c].median()) for c in t.columns}
    tj["note"] = "median per-frame GPU time inside lanes.ops.predict (batch 16 frames of one bay sequence); detector excluded"
    if not tag:
        (d / "cctv_timing.json").write_text(json.dumps(tj, indent=1))
    print("PRED_DONE", len(p), tj)


def metrics(exp: str, vlm: str | None) -> None:
    df = load_items()
    d = RUNS / "ops" / exp
    results = []
    systems = [(f"student ({exp})", d / "cctv_preds.csv"), (f"student ({exp}), no fixed-camera lift rule", d / "cctv_preds_nolift.csv")]
    if vlm:
        systems.append(("Qwen3-VL-8B zero-shot", Path(vlm)))
    summary = {}
    for name, path in systems:
        if not Path(path).exists():
            continue
        p = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
        m = df.merge(p.drop(columns=[c for c in ("group",) if c in p.columns]), on="item_id", how="inner")
        for st in STATES:
            y = pd.to_numeric(m[st], errors="coerce")
            pp = m.get(f"p_{st}")
            if pp is None:
                continue
            ok = y.notna() & pp.notna()
            if ok.sum() < 10 or y[ok].nunique() < 2:
                continue
            ms, conf, cnt = state_metrics(y[ok].astype(int).to_numpy(), pp[ok].to_numpy(), m.loc[ok, "group"].to_numpy())
            results.append(dict(id=f"ops_cctv_{st}_{'vlm' if 'Qwen' in name else ('nolift_' if 'no fixed' in name else '') + exp}",
                                name=f"{st} on public workshop CCTV/timelapse clips", lane="ops", data_kind="real",
                                domain="CCTV (fixed-camera public workshop clips)", dataset="6 public YouTube clips, hand-labelled bay ROIs",
                                split="test only (never trained on)", n=cnt["n"], model=name, metrics=ms, confusion=conf,
                                notes=f"positives={cnt['n_pos']}; bay sequences={cnt['n_groups']}; labels by one annotator (Claude) from frames; "
                                      "CI bootstrap over bay sequences"))
            summary.setdefault(name, {})[st] = {mm["name"]: round(mm["value"], 3) for mm in ms} | {"n": cnt["n"], "pos": cnt["n_pos"]}
    (d / "cctv_component_results.json").write_text(json.dumps(results, indent=1))
    (d / "cctv_summary.json").write_text(json.dumps(summary, indent=1))
    for name, s in summary.items():
        print("==", name)
        for st, v in s.items():
            print(f"  {st:18s}", v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["detect", "vlm_items", "predict", "metrics"])
    ap.add_argument("--exp", default="heads_v1")
    ap.add_argument("--vlm", default="")
    a = ap.parse_args()
    if a.cmd == "detect":
        detect()
    elif a.cmd == "vlm_items":
        vlm_items()
    elif a.cmd == "predict":
        predict(a.exp, True, "")
        predict(a.exp, False, "_nolift")
    else:
        metrics(a.exp, a.vlm or None)
