"""Failure gallery for the tyre and DOT components (damage failures: lanes/inspect/failures.py).

    python -P -m lanes.inspect.failures_other tyre --exp tyre_dino_lr --out docs/lanes/failures/inspect
    python -P -m lanes.inspect.failures_other dot  --exp dot_chars    --out docs/lanes/failures/inspect

Tyre close-ups and cropped sidewall text contain no faces; each picked image is still viewed by eye before it is kept.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from lanes.inspect import common as C
from lanes.inspect import dot as DOT


def _label(img: np.ndarray, lines: list[str]) -> np.ndarray:
    h, w = img.shape[:2]
    pad = 22 * len(lines) + 6
    out = np.full((h + pad, w, 3), 255, np.uint8)
    out[:h] = img
    for i, t in enumerate(lines):
        cv2.putText(out, t, (6, h + 18 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def tyre(a) -> list[dict]:
    items = []
    for key in ("warcoder", "sameer"):
        f = C.RUNS_I / a.exp / f"preds_{key}.csv"
        if not f.exists():
            continue
        rows = list(csv.DictReader(f.open()))
        for r in rows:
            r["p"] = float(r["p_defective"])
        fn = sorted([r for r in rows if r["label"] == "defective" and r["p"] < 0.5], key=lambda r: r["p"])[: a.n]
        fp = sorted([r for r in rows if r["label"] == "good" and r["p"] >= 0.5], key=lambda r: -r["p"])[: a.n]
        for cat, lst in [("defective_predicted_good", fn), ("good_predicted_defective", fp)]:
            for i, r in enumerate(lst):
                im = cv2.imread(r["path"])
                if im is None:
                    continue
                s = 520 / max(im.shape[:2])
                im = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)))
                im = _label(im, [f"{key}: label={r['label']} (orig {r['orig_class']})", f"P(defective)={r['p']:.2f}"])
                tag = f"tyre_{key}_{cat}_{i:02d}"
                cv2.imwrite(str(Path(a.out) / f"{tag}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 82])
                items.append({"id": tag, "component": f"inspect.tyre_condition.{a.exp}.{key}", "file": f"{tag}.jpg", "category": cat,
                              "description": f"{key} image labelled {r['label']} (source class {r['orig_class']}), model P(defective)={r['p']:.2f}"})
    return items


def dot(a) -> list[dict]:
    rows = C.read_json(C.RUNS_I / a.exp / "preds_dot_chars.json")
    from lanes.inspect.prep_dot import OUT
    items = []
    wrong = [r for r in rows if r["gt"] != r["pred"] and r["gt"]]
    # prefer date-code failures on digit serials, then others
    dc = [r for r in wrong if r["gt"].isdigit() and DOT.date_code(r["gt"]) != DOT.date_code(r["pred"])]
    other = [r for r in wrong if r not in dc]
    for cat, lst in [("date_code_wrong", dc[: a.n]), ("string_wrong", other[: a.n])]:
        for i, r in enumerate(lst):
            im = cv2.imread(str(OUT / "chars" / "images" / "test" / r["file"]))
            if im is None:
                continue
            valid = DOT.validate_wwyy(DOT.date_code(r["pred"]))
            im = _label(im, [f"GT:   {r['gt']}", f"pred: {r['pred']}", f"pred date code valid: {valid['valid']} ({valid.get('reason', '')})"])
            tag = f"dot_{cat}_{i:02d}"
            cv2.imwrite(str(Path(a.out) / f"{tag}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 82])
            items.append({"id": tag, "component": f"inspect.dot_chars.{a.exp}", "file": f"{tag}.jpg", "category": cat,
                          "description": f"GT {r['gt']!r} read as {r['pred']!r}"})
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["tyre", "dot"])
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    C.ensure(Path(a.out))
    items = tyre(a) if a.kind == "tyre" else dot(a)
    C.write_json(items, Path(a.out) / f"failures_{a.kind}_{a.exp}.json")
    print(len(items), "images ->", a.out)


if __name__ == "__main__":
    main()
