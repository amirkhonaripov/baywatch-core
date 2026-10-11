"""Save CCTV failure cases (face-blurred ROI crops) for the notes: confident student errors per state.

  python -m lanes.ops.failures --exp heads_v1 --out <dir>   (then copy <dir> to docs/lanes/failures/ops/)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from lanes.ops.eval_cctv import CF, STATES, load_items
from seen.blur import blur_faces
from seen.paths import RUNS
from seen.predictor import Det


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-state", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_items().merge(pd.read_csv(RUNS / "ops" / a.exp / "cctv_preds.csv").drop(columns=["group"]), on="item_id")
    dets = pd.read_parquet(CF / "dets.parquet")
    rows = []
    used = set()
    for st in STATES:
        y = pd.to_numeric(df[st], errors="coerce")
        p = df[f"p_{st}"]
        err = (y.notna()) & (((y == 1) & (p < 0.5)) | ((y == 0) & (p >= 0.5)))
        cand = df[err].assign(margin=(p[err] - 0.5).abs()).sort_values("margin", ascending=False)
        cand = cand[~cand["frame"].isin(used)].drop_duplicates("group")
        for r in cand.head(a.per_state).itertuples():
            used.add(r.frame)
            img = np.asarray(Image.open(CF / "frames" / r.frame).convert("RGB"))
            d = [Det(cls=x.cls, confidence=float(x.conf), bbox=(x.x1, x.y1, x.x2, x.y2)) for x in dets[dets["frame"] == r.frame].itertuples()]
            img = blur_faces(img, d)
            h, w = img.shape[:2]
            x1, y1, x2, y2 = r.roi_x1, r.roi_y1, r.roi_x2, r.roi_y2
            crop = Image.fromarray(img).crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
            crop.thumbnail((640, 640))
            truth = int(getattr(r, st))
            fn = f"{st}_{r.item_id.replace('|', '_')}.jpg"
            crop.save(out / fn, quality=88)
            kind = "missed" if truth == 1 else "false alarm"
            rows.append(dict(id=f"ops_{st}_{r.item_id}", component=f"ops/{st}", file=fn, category=kind,
                             description=f"{st}: hand label {truth}, predicted p={getattr(r, f'p_{st}'):.2f} ({kind}); clip {r.clip}, bay {r.roi_id}"))
    (out / "failures.json").write_text(json.dumps(rows, indent=1))
    print("FAILURES", len(rows))


if __name__ == "__main__":
    main()
