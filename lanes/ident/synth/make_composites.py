"""Build the synthetic-composite detection set (and composite OCR crops): rendered Uzbek plates pasted onto real vehicle photos.

Sources and where the quad comes from:
  rodosol   RodoSol-ALPR TRAIN split (block split, see det_sets.py): 4 annotated corners -> exact homography
  ccpd_*    CCPD2019 db/weather/challenge/base, the 'train' half by file-name hash (the 'eval' half is what we measure on): 4 annotated vertices
  photos    Mendeley Central Asia (blurred seller plates) + Wikimedia Commons Uzbek car photos: quads = boxes of the fast-alpr detector
            (conf >= 0.35, axis-aligned; per-photo list in data/prep/ident/det/photo_boxes.jsonl)
Outputs under data/prep/ident/det/synth_comp/: images/*.jpg, labels/*.txt (YOLO, class 0 = plate), meta.jsonl;
and OCR crops (native resolution, 6 % pad) under data/prep/ident/ocr/synth_comp_train + .csv (plate_region = format label).
Everything produced here is SYNTHETIC (real photo, invented plate) and is reported as such.
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from lanes.ident import uz_format as uz
from lanes.ident.data import det_sets as ds
from lanes.ident.synth.composite import box_to_quad, order_quad, paste_plate
from seen.paths import RAW, fast_dir

FAST = fast_dir("ident")
OUT = FAST / "det" / "synth_comp"
OCR = FAST / "ocr"


def text_for(rng):
    r = rng.random()
    if r < 0.62:
        return uz.sample_text(rng, uz.UZ_PRIVATE), uz.UZ_PRIVATE
    if r < 0.86:
        return uz.sample_text(rng, uz.UZ_LEGAL), uz.UZ_LEGAL
    return uz.sample_text(rng, uz.UZ_PRIVATE if rng.random() < 0.5 else uz.UZ_LEGAL), uz.UZ_EV


def job(a):
    idx, rec, seed, kind = a
    rng = random.Random(seed)
    im = cv2.imread(rec["path"], cv2.IMREAD_COLOR)
    if im is None:
        return None
    H, W = im.shape[:2]
    scale = 1.0
    if max(H, W) > 1400:
        scale = 1400.0 / max(H, W)
        im = cv2.resize(im, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    quads = []
    if kind == "rodosol":
        quads = [rec["corners"] * scale]
    elif kind == "ccpd":
        v = rec["verts"] * scale  # BR, BL, TL, TR
        quads = [np.array([v[2], v[3], v[0], v[1]])]
    else:
        quads = [box_to_quad(np.array(b) * scale) for b in rec["boxes"]]
    labels, texts, formats = [], [], []
    cur = rgb
    for q in quads:
        t, f = text_for(rng)
        res = paste_plate(cur, q, t, f, rng, match=True, match_blur=(kind != "photos"))
        if res is None:
            continue
        cur, nq, box, info = res
        labels.append(box)
        texts.append(t)
        formats.append(info["format"])
    if not labels:
        return None
    name = f"{kind}_{idx:06d}"
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "images" / f"{name}.jpg"), cv2.cvtColor(cur, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, rng.randint(70, 95)])
    h, w = cur.shape[:2]
    with open(OUT / "labels" / f"{name}.txt", "w") as f:
        for b in labels:
            f.write(f"0 {(b[0] + b[2]) / 2 / w:.6f} {(b[1] + b[3]) / 2 / h:.6f} {(b[2] - b[0]) / w:.6f} {(b[3] - b[1]) / h:.6f}\n")
    # OCR crops (native res)
    crops = []
    for k, (b, t, fm) in enumerate(zip(labels, texts, formats)):
        bw, bh = b[2] - b[0], b[3] - b[1]
        x1, y1, x2, y2 = int(max(0, b[0] - 0.06 * bw)), int(max(0, b[1] - 0.06 * bh)), int(min(w, b[2] + 0.06 * bw)), int(min(h, b[3] + 0.06 * bh))
        if x2 - x1 < 24 or y2 - y1 < 8:
            continue
        crops.append({"img": cv2.resize(cur[y1:y2, x1:x2], (128, 64), interpolation=cv2.INTER_LINEAR), "text": t, "format": fm, "src": f"{name}_{k}"})
    return {"name": name, "kind": kind, "src": rec["path"], "texts": texts, "formats": formats, "crops": crops}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rodosol", type=int, default=3000)
    ap.add_argument("--ccpd", type=int, default=3500)
    ap.add_argument("--photos", action="store_true")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    jobs = []
    rng = np.random.default_rng(a.seed)
    if a.rodosol:
        recs = ds.load_rodosol("train")
        sel = rng.choice(len(recs), min(a.rodosol, len(recs)), replace=False)
        jobs += [(len(jobs) + i, recs[j], a.seed * 7919 + len(jobs) + i, "rodosol") for i, j in enumerate(sel)]
    if a.ccpd:
        per = {"db": 0.40, "weather": 0.25, "challenge": 0.12, "base": 0.23}
        for sub, fr in per.items():
            try:
                recs = ds.load_ccpd(sub, int(a.ccpd * fr), seed=a.seed, part="train")
            except Exception as e:  # subset not extracted yet
                print("skip ccpd", sub, e)
                continue
            jobs += [(len(jobs) + i, r, a.seed * 7919 + len(jobs) + i, "ccpd") for i, r in enumerate(recs)]
    if a.photos:
        pb = FAST / "det" / "photo_boxes.jsonl"
        if pb.exists():
            for ln in pb.read_text().splitlines():
                r = json.loads(ln)
                if r["boxes"]:
                    jobs.append((len(jobs), r, a.seed * 7919 + len(jobs), "photos"))
    print("jobs", len(jobs), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        res = [r for r in ex.map(job, jobs, chunksize=16) if r]
    cr = [c for r in res for c in r["crops"]]
    if cr:
        from lanes.ident.data.pack import save_train_pack
        save_train_pack("synth_comp_train", np.stack([c["img"] for c in cr]), [c["text"] for c in cr], [c["format"] for c in cr], [c["src"] for c in cr])
    for r in res:
        for c in r["crops"]:
            c.pop("img", None)
    (OUT / "meta.jsonl").write_text("".join(json.dumps(r) + "\n" for r in res))
    print("composites", len(res), "ocr crops", len(cr))


if __name__ == "__main__":
    main()
