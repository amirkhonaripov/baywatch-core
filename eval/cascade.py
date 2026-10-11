"""Cascade analysis on the hand-labelled CCTV bay set: fast state heads on every frame, VLM only on uncertain frames.

For each state and escalation rate r, the r-fraction of frames whose fast probability is closest to 0.5 is answered by
Qwen3-VL (zero-shot), the rest by the fast heads (threshold 0.5). Reports accuracy with Wilson 95% CI and the
resulting VLM calls per camera-hour at 1 fps.

Usage: python -m eval.cascade [--fast runs/ops/heads_v1/cctv_preds.csv] [--vlm runs/ops/vlm/cctv_vlm.parquet]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval.stats import prop_metric
from seen.paths import RUNS

STATES = ["car_present", "raised", "wheel_off", "hood_open", "person_at_car", "person_under_car", "work"]
RATES = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
LABELS = Path(__file__).resolve().parents[1] / "lanes" / "ops" / "labels" / "cctv_labels.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", default=str(RUNS / "ops" / "heads_v1" / "cctv_preds.csv"))
    ap.add_argument("--vlm", default=str(RUNS / "ops" / "vlm" / "cctv_vlm.parquet"))
    ap.add_argument("--out", default=str(RUNS / "eval" / "cascade.json"))
    a = ap.parse_args()
    lab = pd.read_csv(LABELS)
    lab = lab[lab["exclude"].fillna(0).astype(int) == 0]
    fast = pd.read_csv(a.fast).set_index("item_id")
    vlm = pd.read_parquet(a.vlm).set_index("item_id")
    rows, results = [], []
    for s in STATES:
        d = lab[["item_id", s]].dropna()
        d = d[d["item_id"].isin(fast.index) & d["item_id"].isin(vlm.index)]
        y = d[s].astype(int).to_numpy()
        pf = fast.loc[d["item_id"], f"p_{s}"].to_numpy()
        pv = vlm.loc[d["item_id"], f"p_{s}"].to_numpy()
        order = np.argsort(np.abs(pf - 0.5))  # most uncertain first
        metrics = []
        for r in RATES:
            k = int(round(r * len(y)))
            esc = np.zeros(len(y), bool)
            esc[order[:k]] = True
            pred = np.where(esc, pv >= 0.5, pf >= 0.5).astype(int)
            correct = int((pred == y).sum())
            m = prop_metric(f"accuracy@escalate_{int(r*100)}pct", correct, len(y))
            metrics.append(m)
            rows.append({"state": s, "escalate": r, "n": len(y), "acc": m["value"], "ci_low": m["ci_low"], "ci_high": m["ci_high"],
                         "vlm_calls_per_cam_hour_1fps": round(3600 * r)})
        results.append({"id": f"ops_cascade_{s}", "name": f"Bay state '{s}': fast heads + Qwen3-VL on uncertain frames",
                        "lane": "ops", "data_kind": "real", "domain": "fixed CCTV (public workshop clips, hand-labelled)",
                        "dataset": "ops CCTV real-domain set (6 clips)", "split": "test", "n": int(len(y)),
                        "model": "DINOv2-L+geometry heads v0.1 / Qwen3-VL-8B zero-shot", "metrics": metrics,
                        "notes": "Escalation = frames whose fast probability is closest to 0.5. Single annotator (Claude). "
                                 "~12 independent bay sequences: CIs are optimistic about independence."})
    df = pd.DataFrame(rows)
    print(df.pivot(index="state", columns="escalate", values="acc").round(3).to_string())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1))
    df.to_csv(Path(a.out).with_suffix(".csv"), index=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
