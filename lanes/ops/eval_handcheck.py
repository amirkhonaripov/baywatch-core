"""Hand-labelled handheld check set (96 test frames from K700 val / CrossTask-COIN test videos).

(a) student metrics per state on human labels in the handheld domain (gives the handheld side of the domain-drop
    comparison for states without derived labels: car_present, hood_open, person_at_car, person_under_car);
(b) teacher (Qwen3-VL) agreement with the human labels on the same frames (the teacher labels are used for training).
  python -m lanes.ops.eval_handcheck --exp heads_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval.stats import prop_metric
from lanes.ops.metrics import state_metrics
from seen.paths import RUNS

HERE = Path(__file__).resolve().parent / "labels"
STATES = ["car_present", "raised", "wheel_off", "hood_open", "person_at_car", "person_under_car", "work"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    a = ap.parse_args()
    lab = pd.read_csv(HERE / "handheld_check_labels.csv")
    lab = lab[lab["exclude"] == 0]
    d = RUNS / "ops" / a.exp
    stu = pd.read_parquet(d / "test_preds.parquet")
    vlm = pd.read_parquet(RUNS / "ops" / "vlm" / "handcheck_vlm.parquet")
    results, summary = [], {}
    for name, preds, key in (("student", stu.rename(columns={"frame_id": "item_id"}), "student"),
                             ("Qwen3-VL-8B-FP8 (teacher)", vlm, "teacher")):
        m = lab.merge(preds, on="item_id", how="inner", suffixes=("", "_pred"))
        groups = m["source"] if "source" in m else m["item_id"]
        for st in STATES:
            y = pd.to_numeric(m[st], errors="coerce")
            p = m.get(f"p_{st}")
            if p is None:
                continue
            ok = y.notna() & p.notna()
            if ok.sum() < 10 or y[ok].nunique() < 2:
                if ok.sum() >= 10:  # one class only: report agreement only
                    agree = int(((p[ok] >= 0.5).astype(int) == y[ok].astype(int)).sum())
                    summary.setdefault(key, {})[st] = {"agreement": round(agree / ok.sum(), 3), "n": int(ok.sum()),
                                                       "pos": int(y[ok].sum())}
                continue
            ms, conf, cnt = state_metrics(y[ok].astype(int).to_numpy(), p[ok].to_numpy(), np.asarray(groups[ok]))
            agree = int(((p[ok] >= 0.5).astype(int) == y[ok].astype(int)).sum())
            ms.append(prop_metric("agreement_with_human@0.5", agree, int(ok.sum())))
            results.append(dict(id=f"ops_handcheck_{st}_{key}_{a.exp}", name=f"{st} on hand-labelled handheld test frames ({key})",
                                lane="ops", data_kind="real", domain="handheld/YouTube",
                                dataset="96 hand-labelled frames from K700 val + CrossTask/COIN test videos",
                                split="test", n=cnt["n"], model=name, metrics=ms, confusion=conf,
                                notes=f"positives={cnt['n_pos']}; one annotator (Claude) labelled from the frames; "
                                      + ("teacher labels are used as training targets where no human/derived label exists"
                                         if key == "teacher" else "")))
            summary.setdefault(key, {})[st] = {mm["name"]: round(mm["value"], 3) for mm in ms} | {"n": cnt["n"], "pos": cnt["n_pos"]}
    (d / "handcheck_component_results.json").write_text(json.dumps(results, indent=1))
    (d / "handcheck_summary.json").write_text(json.dumps(summary, indent=1))
    for k, v in summary.items():
        print("==", k)
        for st, x in v.items():
            print(f"  {st:18s}", x)


if __name__ == "__main__":
    main()
