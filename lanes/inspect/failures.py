"""Failure gallery for lane D: pick the clearest failures from saved predictions, draw GT (green) and predictions (red),
blur licence plates (parts-model plate detections or GT plate polygons) and write JPEGs + failures.json to
docs/lanes/failures/inspect/ (G re-blurs faces; we also look at every picked image by eye and drop any with a visible face).

  python -P -m lanes.inspect.failures damage --preds runs/inspect/dmg_B/preds_syndcar.json --set syndcar --exp dmg_B --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from eval.stats import iou_matrix
from lanes.inspect import common as C


def _blur_boxes(img: np.ndarray, boxes) -> np.ndarray:
    for x1, y1, x2, y2 in boxes:
        a, b, c, d = max(int(x1), 0), max(int(y1), 0), min(int(x2), img.shape[1]), min(int(y2), img.shape[0])
        if c - a > 3 and d - b > 3:
            k = max(((c - a) // 3) | 1, 15)
            img[b:d, a:c] = cv2.GaussianBlur(img[b:d, a:c], (k, k), 0)
    return img


def damage_failures(preds: list[dict], set_name: str, conf: float = 0.25, n_each: int = 4):
    """-> list of (category, record, description) for missed GT, high-confidence unmatched predictions and wrong classes."""
    miss, fp, wrong = [], [], []
    for r in preds:
        gts = r["gt"]
        pr = [p for p in r["pred"] if p["score"] >= conf]
        if gts and pr:
            iou = iou_matrix(np.array([p["box"] for p in pr]), np.array([g["box"] for g in gts]))
        else:
            iou = np.zeros((len(pr), len(gts)))
        for j, g in enumerate(gts):
            cand = [i for i in range(len(pr)) if iou[i, j] >= 0.5]
            if not cand:
                a = (g["box"][2] - g["box"][0]) * (g["box"][3] - g["box"][1]) / (r["w"] * r["h"])
                miss.append((a, r, f"missed {g['cls']} (box area {a:.3%} of image)"))
            elif all(pr[i]["cls"] != g["cls"] for i in cand) and g["cls"] != "damage":
                wrong.append((max(pr[i]["score"] for i in cand), r, f"{g['cls']} predicted as {pr[cand[0]]['cls']}"))
        for i, p in enumerate(pr):
            if (len(gts) == 0 or iou[i].max() < 0.1):
                fp.append((p["score"], r, f"unmatched {p['cls']} prediction, score {p['score']:.2f} (may be unlabelled damage)"))
    out = []
    # misses: prefer large missed boxes (clear failures, not tiny scratches); keep image diversity
    for cat, lst, rev in [("missed_damage", miss, True), ("false_positive", fp, True), ("wrong_class", wrong, True)]:
        seen, k = set(), 0
        for _, r, desc in sorted(lst, key=lambda t: -t[0] if rev else t[0]):
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            out.append((cat, r, desc))
            k += 1
            if k >= n_each:
                break
    return out


def render(rec: dict, set_name: str, out_dir: Path, tag: str, plates: list | None = None, conf: float = 0.25) -> str:
    img = C.read_rgb(rec["path"])
    img = np.ascontiguousarray(img[:, :, ::-1])  # BGR for cv2 drawing
    if plates:
        img = _blur_boxes(img, plates)
    th = max(2, img.shape[1] // 400)
    for g in rec["gt"]:
        x1, y1, x2, y2 = [int(v) for v in g["box"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), th)
    for p in rec["pred"]:
        if p["score"] < conf:
            continue
        x1, y1, x2, y2 = [int(v) for v in p["box"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), th)
        cv2.putText(img, f"{p['cls']} {p['score']:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, img.shape[1] / 1600, (0, 0, 255), max(1, th // 2))
    s = 1000 / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    fn = f"{tag}.jpg"
    cv2.imwrite(str(out_dir / fn), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["damage"])
    ap.add_argument("--preds", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-each", type=int, default=3)
    ap.add_argument("--plates", default="", help="optional parts-preds json for the same set: unified 'license_plate' boxes are blurred")
    a = ap.parse_args()
    out = C.ensure(Path(a.out))
    preds = C.read_json(Path(a.preds))
    plates = {}
    if a.plates:
        for r in C.read_json(Path(a.plates)):
            plates[r["id"]] = [p["box"] for p in r["pred"] if p.get("unified") == "license_plate" and p["score"] >= 0.2]
    items = []
    for cat, rec, desc in damage_failures(preds, a.set, n_each=a.n_each):
        tag = f"{a.exp}_{a.set}_{cat}_{len(items):02d}"
        fn = render(rec, a.set, out, tag, plates.get(rec["id"]))
        items.append({"id": tag, "component": f"inspect.damage.{a.exp}.{a.set}", "file": fn, "description": desc, "category": cat})
    C.write_json(items, out / f"failures_{a.exp}_{a.set}.json")
    print(len(items), "failure images written to", out)


if __name__ == "__main__":
    main()
