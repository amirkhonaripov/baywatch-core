"""Failure gallery for docs/lanes/failures/ident/: real-UZ test photos where the final predictor is wrong.

Privacy: faces are blurred with seen.blur.blur_faces and the plate itself is pixelated (real plates of private vehicles must not appear in
decks/UI); the caption JSON carries the verified label vs the prediction as text for the notes. Synthetic failures are saved unpixelated.

  python -m lanes.ident.make_failures --e2e-preds runs/ident/ours_e2e/preds_e2e.csv --out docs/lanes/failures/ident --n 12
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from lanes.ident.data import det_sets as ds
from seen.blur import blur_faces


def pixelate(img: np.ndarray, box, block: int = 6):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return img
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    img[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e-preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    recs = {r["id"]: r for r in ds.load_uz_real("test")}
    rows = list(csv.DictReader(open(a.e2e_preds)))
    bad = [r for r in rows if r["gt"] != r["pred"]]
    # one example per failure kind: missed detection, wrong text with detection
    missed = [r for r in bad if r["detected"] == "0"]
    wrong = [r for r in bad if r["detected"] == "1"]
    pick = missed[: a.n // 2] + wrong[: a.n - min(len(missed), a.n // 2)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = []
    for i, r in enumerate(pick):
        rec = recs[r["id"]]
        im = cv2.imread(rec["path"])
        for b in rec["gt"]:
            im = pixelate(im, b)
        im = blur_faces(im, [])
        fn = f"uz_real_{i:02d}_{'missed' if r['detected'] == '0' else 'misread'}.jpg"
        cv2.imwrite(str(out / fn), im)
        cap.append({"id": r["id"], "component": "ident.e2e", "file": fn, "category": "missed detection" if r["detected"] == "0" else "misread text",
                    "description": f"verified label {r['gt']} vs prediction '{r['pred']}' (plate pixelated here on purpose)"})
    (out / "failures.json").write_text(json.dumps(cap, indent=1))
    print(len(cap), "failure images")


if __name__ == "__main__":
    main()
