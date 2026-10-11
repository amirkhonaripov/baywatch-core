"""Synthetic Uzbek plate OCR crops, written as packed arrays (see lanes/ident/data/pack.py).

  python -m lanes.ident.synth.gen_ocr --name synth_uz_train --n 80000 --seed 1000 --exclude-texts test_synth_uz.texts.txt
  python -m lanes.ident.synth.gen_ocr --name test_synth_uz --n 3000 --seed 7 --native      # held-out test, native resolution (pkl)

Training crops are the degraded native-resolution crop resized to the model input (128x64 RGB, cv2 linear). Test crops keep their native
resolution so evaluation goes through the same resize as deployment. Plate texts are INVENTED: random region (skewed to realistic codes),
random series letters/digits, never copied from real plates; the test split's texts are disjoint from the train split's texts.
plate_region = format label. Mix: private 62 %, legal 22 %, EV 10 % (half private / half legal layout), motorcycle 6 %.
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
from lanes.ident.data.pack import OCR, save_test_pack, save_train_pack
from lanes.ident.synth.augment import degrade
from lanes.ident.synth.render import render_plate

MIX = [(uz.UZ_PRIVATE, 0.62), (uz.UZ_LEGAL, 0.22), (uz.UZ_EV, 0.10), (uz.UZ_MOTO, 0.06)]


def make_one(args):
    seed, native, exclude, intensity = args
    rng = random.Random(seed)
    fmt = rng.choices([m for m, _ in MIX], weights=[w for _, w in MIX])[0]
    for _ in range(50):
        if fmt == uz.UZ_EV:
            layout = uz.UZ_PRIVATE if rng.random() < 0.5 else uz.UZ_LEGAL
            text = uz.sample_text(rng, layout)
        else:
            text = uz.sample_text(rng, fmt)
        if text not in exclude:
            break
    plate, f, style = render_plate(text, rng, fmt)
    crop, info = degrade(np.asarray(plate), seed * 7 + 1, intensity=intensity)
    if not native:
        crop = cv2.resize(crop, (128, 64), interpolation=cv2.INTER_LINEAR)
    return crop, text, f, {"scenario": info["scenario"], "out_w": info["out_w"], "ops": info["ops"], "font": style.font}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--native", action="store_true")
    ap.add_argument("--exclude-texts", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--intensity", type=float, default=1.0)
    a = ap.parse_args()
    exclude = set((OCR / a.exclude_texts).read_text().split()) if a.exclude_texts else set()
    jobs = [(a.seed * 1_000_003 + i, a.native, exclude, a.intensity) for i in range(a.n)]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(make_one, jobs, chunksize=64))
    imgs = [r[0] for r in res]
    texts = [r[1] for r in res]
    fmts = [r[2] for r in res]
    meta = [r[3] for r in res]
    srcs = [f"synth:{a.name}:{i}" for i in range(len(res))]
    if a.native:
        save_test_pack(a.name, imgs, texts, fmts, srcs)
    else:
        save_train_pack(a.name, np.stack(imgs), texts, fmts, srcs)
    (OCR / f"{a.name}.meta.jsonl").write_text("".join(json.dumps({"text": t, "format": f, **m}) + "\n" for t, f, m in zip(texts, fmts, meta)))
    (OCR / f"{a.name}.texts.txt").write_text("\n".join(sorted(set(texts))) + "\n")
    print(a.name, len(res))


if __name__ == "__main__":
    main()
