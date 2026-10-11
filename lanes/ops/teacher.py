"""Qwen3-VL teacher labels for training frames that lack a human/derived label, and a hand-labelled handheld check set.

  python -m lanes.ops.teacher items        # -> FAST/ops/teacher_items.parquet (train sample) + handcheck_items.parquet (test)
  python -m lanes.ops.teacher sheets --out <dir>   # contact sheets of the hand-check sample (labelled by hand)
The hand labels live in lanes/ops/labels/handheld_check_labels.csv. They give (a) handheld metrics for states that have
no derived label and (b) the teacher's agreement with a human on the same frames (the teacher is only used for training).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lanes.ops.prep import FAST_OPS, TABLE, fpath

N_CHECK = 96


def _student_crop(sel: pd.DataFrame) -> pd.DataFrame:
    """The crop the student embeds (primary vehicle box + 25 %), so teacher and student look at the same region.
    NaN = no vehicle detected (teacher then sees the full frame)."""
    from PIL import Image

    from lanes.ops.featurize import dets_by_frame
    from lanes.ops.geometry import VEH, crop_box, primary_vehicle

    dets = dets_by_frame()
    rows = []
    for fid, p in zip(sel["frame_id"], sel["path"]):
        pv = primary_vehicle([(c, b) for k, c, b in dets.get(fid, []) if k in VEH])
        if pv is None:
            rows.append((np.nan,) * 4)
            continue
        with Image.open(fpath(p)) as im:
            w, h = im.size
        rows.append(crop_box(pv[1], 0.25, w / h))
    return pd.DataFrame(rows, columns=["roi_x1", "roi_y1", "roi_x2", "roi_y2"])


def items(seed: int = 0) -> None:
    df = pd.read_parquet(TABLE)
    rng = np.random.default_rng(seed)
    tr = df[df["split"] == "train"]
    k = tr[(tr["dataset"] == "k700") & (tr["j"] == 1)]
    h = tr[tr["dataset"].isin(["crosstask", "coin"])]
    h = h.sort_values(["video", "t_s"]).copy()
    h["_i"] = h.groupby("video").cumcount()
    h["_n"] = h.groupby("video")["_i"].transform("max") + 1
    keep_i = h.apply(lambda r: r["_i"] in set(np.linspace(0, r["_n"] - 1, min(6, r["_n"])).astype(int)), axis=1)
    h = h[keep_i]
    dom = tr[tr["dataset"] == "domain"]  # unlabelled fixed-camera workshop clips: all sampled frames
    sel = pd.concat([k, h, dom])
    out = pd.DataFrame({"item_id": sel["frame_id"].to_numpy(), "path": [str(fpath(p)) for p in sel["path"]]})
    out = pd.concat([out, _student_crop(sel)], axis=1)
    out.to_parquet(FAST_OPS / "teacher_items.parquet")
    # hand-check sample: handheld TEST frames, stratified over K700 classes and CrossTask/COIN tasks. Drawn once and
    # frozen (the hand labels refer to these frame ids); delete the file to redraw.
    if (FAST_OPS / "handcheck_items.parquet").exists():
        print("teacher items", len(out), "(hand-check set kept)")
        return
    te = df[(df["split"] == "test") & df["dataset"].isin(["k700", "crosstask", "coin"])].copy()
    te["stratum"] = np.where(te["dataset"] == "k700", te["k700_label"], te["task"])
    per = max(1, N_CHECK // te["stratum"].nunique())
    chk = te.sample(frac=1.0, random_state=seed).drop_duplicates("video")  # at most one frame per video/clip
    chk = chk.groupby("stratum").head(per).head(N_CHECK)
    c = pd.DataFrame({"item_id": chk["frame_id"].to_numpy(), "path": [str(fpath(p)) for p in chk["path"]],
                      "stratum": chk["stratum"].to_numpy(), "source": chk["source"].to_numpy()})
    c = pd.concat([c, _student_crop(chk)], axis=1)
    c.to_parquet(FAST_OPS / "handcheck_items.parquet")
    print("teacher items", len(out), "hand-check items", len(c), c["stratum"].value_counts().to_dict())


def sheets(out: Path, per: int = 12, cols: int = 4, tile: int = 400) -> None:
    import cv2

    c = pd.read_parquet(FAST_OPS / "handcheck_items.parquet")
    out.mkdir(parents=True, exist_ok=True)
    for s in range(0, len(c), per):
        tiles = []
        for i, r in enumerate(c.iloc[s:s + per].itertuples()):
            im = cv2.imread(r.path)
            H, W = im.shape[:2]
            if not np.isnan(r.roi_x1):  # the crop the student and the teacher look at
                cv2.rectangle(im, (int(r.roi_x1 * W), int(r.roi_y1 * H)), (int(r.roi_x2 * W) - 1, int(r.roi_y2 * H) - 1), (0, 255, 255), 2)
            im = cv2.resize(im, (tile, int(im.shape[0] * tile / im.shape[1])))[:tile]
            pad = np.zeros((tile, tile, 3), np.uint8)
            pad[:im.shape[0]] = im
            cv2.rectangle(pad, (0, 0), (tile, 20), (0, 0, 0), -1)
            cv2.putText(pad, f"{s + i}", (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(pad)
        rows = (len(tiles) + cols - 1) // cols
        canvas = np.zeros((rows * tile, cols * tile, 3), np.uint8)
        for i, t in enumerate(tiles):
            r_, c_ = divmod(i, cols)
            canvas[r_ * tile:(r_ + 1) * tile, c_ * tile:(c_ + 1) * tile] = t
        cv2.imwrite(str(out / f"hc_{s // per:03d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    c[["item_id", "stratum"]].to_csv(out / "index.csv")
    print("sheets", (len(c) + per - 1) // per)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["items", "sheets"])
    ap.add_argument("--out", default=str(FAST_OPS / "handcheck_sheets"))
    a = ap.parse_args()
    items() if a.cmd == "items" else sheets(Path(a.out))
