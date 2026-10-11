"""CCTV real-domain test set: expand bay ROIs into items and render crops for hand labelling.

  python -m lanes.ops.label_cctv items    # -> lanes/ops/labels/cctv_items.csv
  python -m lanes.ops.label_cctv sheets --out <dir>
Hand labels live in lanes/ops/labels/cctv_labels.csv (item_id + one column per state: 1 / 0 / ? ; exclude=1 for
title cards, black frames or views where the bay is not visible). These clips are NEVER used for training.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from seen.paths import PREP

HERE = Path(__file__).resolve().parent / "labels"
FR = PREP / "ops" / "cctv"
LABEL_STATES = ["car_present", "raised", "wheel_off", "hood_open", "person_at_car", "person_under_car", "work"]


def items() -> list[dict]:
    idx = {(r["clip"], int(r["k"])): r for r in csv.DictReader((FR / "index.csv").open())}
    out = []
    for r in csv.DictReader((HERE / "cctv_rois.csv").open()):
        for k in range(int(r["k_from"]), int(r["k_to"]) + 1, int(r["k_step"])):
            f = idx.get((r["clip"], k))
            if f is None:
                continue
            out.append(dict(item_id=f"{f['frame'][:-4]}|{r['roi_id']}", frame=f["frame"], clip=r["clip"], k=k,
                            t_s=f["t_s"], roi_id=r["roi_id"], roi_x1=r["x1"], roi_y1=r["y1"], roi_x2=r["x2"], roi_y2=r["y2"]))
    with (HERE / "cctv_items.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("items", len(out))
    return out


def crop(img: np.ndarray, roi, margin: float = 0.08) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = roi
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return img[int(max(0, y1 - dy) * h):int(min(1, y2 + dy) * h), int(max(0, x1 - dx) * w):int(min(1, x2 + dx) * w)]


def sheets(out: Path, per: int = 12, cols: int = 4, tile: int = 400) -> None:
    its = list(csv.DictReader((HERE / "cctv_items.csv").open()))
    out.mkdir(parents=True, exist_ok=True)
    for s in range(0, len(its), per):
        chunk = its[s:s + per]
        tiles = []
        for it in chunk:
            img = cv2.imread(str(FR / "frames" / it["frame"]))
            c = crop(img, [float(it[k]) for k in ("roi_x1", "roi_y1", "roi_x2", "roi_y2")])
            c = cv2.resize(c, (tile, int(c.shape[0] * tile / c.shape[1])))
            c = c[:tile]  # cap height
            pad = np.zeros((tile, tile, 3), np.uint8)
            pad[:c.shape[0]] = c
            cv2.rectangle(pad, (0, 0), (tile, 20), (0, 0, 0), -1)
            cv2.putText(pad, f"{s + len(tiles)}: {it['item_id']}", (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(pad)
        rows = (len(tiles) + cols - 1) // cols
        canvas = np.zeros((rows * tile, cols * tile, 3), np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            canvas[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = t
        cv2.imwrite(str(out / f"lab_{s // per:03d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("sheets", (len(its) + per - 1) // per)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["items", "sheets"])
    ap.add_argument("--out", default=str(FR / "label_sheets"))
    a = ap.parse_args()
    items() if a.cmd == "items" else sheets(Path(a.out))
