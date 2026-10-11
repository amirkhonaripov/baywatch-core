"""Contact sheets for hand labelling / visual verification (numbered tiles).

  python -m lanes.ops.sheets --index data/prep/ops/cctv/index.csv --frames data/prep/ops/cctv/frames \
      --out data/prep/ops/cctv/sheets --per-sheet 12 --cols 4 --tile 480
Optional --boxes <csv with frame,x1,y1,x2,y2> draws a rectangle (e.g. the bay ROI) on each tile.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def make_sheets(names: list[str], frames_dir: Path, out: Path, per: int, cols: int, tile: int,
                rois: dict[str, list[tuple[float, float, float, float]]] | None = None, prefix: str = "sheet") -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in range(0, len(names), per):
        chunk = names[s:s + per]
        rows = (len(chunk) + cols - 1) // cols
        tiles = []
        for n in chunk:
            img = cv2.imread(str(frames_dir / n))
            h, w = img.shape[:2]
            sc = tile / w
            img = cv2.resize(img, (tile, int(h * sc)))
            for (x1, y1, x2, y2) in (rois or {}).get(n, []):
                cv2.rectangle(img, (int(x1 * tile), int(y1 * h * sc)), (int(x2 * tile), int(y2 * h * sc)), (0, 255, 255), 2)
            cv2.rectangle(img, (0, 0), (len(n) * 9 + 8, 22), (0, 0, 0), -1)
            cv2.putText(img, n, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
        th = max(t.shape[0] for t in tiles)
        canvas = np.zeros((rows * th, cols * tile, 3), np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            canvas[r * th:r * th + t.shape[0], c * tile:(c + 1) * tile] = t
        p = out / f"{prefix}_{s // per:03d}.jpg"
        cv2.imwrite(str(p), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(p)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-sheet", type=int, default=12)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--tile", type=int, default=480)
    ap.add_argument("--filter", default="", help="only frames whose name contains this")
    ap.add_argument("--rois", default="", help="csv frame,x1,y1,x2,y2 (normalised)")
    ap.add_argument("--prefix", default="sheet")
    a = ap.parse_args()
    names = [r["frame"] for r in csv.DictReader(open(a.index)) if a.filter in r["frame"]]
    rois = {}
    if a.rois:
        for r in csv.DictReader(open(a.rois)):
            rois.setdefault(r["frame"], []).append(tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2")))
    ps = make_sheets(names, Path(a.frames), Path(a.out), a.per_sheet, a.cols, a.tile, rois, a.prefix)
    print(len(ps), "sheets")
