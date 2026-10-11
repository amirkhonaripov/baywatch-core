"""bf16 vs fp32 parity for the DINOv2-L crop embedding (Precision policy, docs/LANES.md).

  python -m lanes.ops.parity --exp heads_v1 --n 256
Embeds n random test crops in fp32 and bf16, reports embedding cosine and head-probability differences / decision flips.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image

from lanes.ops.backbones import load_backbone
from lanes.ops.featurize import FEAT, crop_for, dets_by_frame
from lanes.ops.heads import HeadModel
from lanes.ops.prep import TABLE, fpath
from seen.paths import RUNS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--n", type=int, default=256)
    a = ap.parse_args()
    df = pd.read_parquet(TABLE)
    df = df[df["split"] == "test"].sample(a.n, random_state=0)
    dets = dets_by_frame()
    head = HeadModel.load(RUNS / "ops" / a.exp / "head.npz")
    ids = pd.read_parquet(FEAT / "emb_dinov2l_ids.parquet")["frame_id"].tolist()
    E_all = np.load(FEAT / "emb_dinov2l.npy", mmap_mode="r")
    pos = {f: i for i, f in enumerate(ids)}
    from lanes.ops.geometry import geom_features

    parts = dets_by_frame(FEAT / "parts.parquet") if (FEAT / "parts.parquet").exists() else {}
    G = np.stack([geom_features(dets.get(f, []), None, parts.get(f)) for f in df["frame_id"]])
    embs = {}
    for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        bb = load_backbone("dinov2l", "cuda:0", dtype=dt)
        xs = torch.stack([bb.preprocess(crop_for(Image.open(fpath(p)).convert("RGB"), dets.get(f, []))[0])
                          for p, f in zip(df["path"], df["frame_id"])])
        with torch.inference_mode():
            embs[name] = np.concatenate([bb.forward(xs[i:i + 32].to("cuda:0")).float().cpu().numpy() for i in range(0, len(xs), 32)])
        del bb
        torch.cuda.empty_cache()
    e32, e16 = embs["fp32"], embs["bf16"]
    cos = (e32 * e16).sum(1) / np.linalg.norm(e32, axis=1) / np.linalg.norm(e16, axis=1)
    p32 = head.predict_proba(np.concatenate([e32, G], 1))
    p16 = head.predict_proba(np.concatenate([e16, G], 1))
    stored = np.asarray(E_all[[pos[f] for f in df["frame_id"]]], dtype=np.float32)
    ps = head.predict_proba(np.concatenate([stored, G], 1))
    res = dict(n=int(a.n), emb_cosine_min=float(cos.min()), emb_cosine_mean=float(cos.mean()),
               prob_absdiff_max=float(np.abs(p32 - p16).max()), prob_absdiff_mean=float(np.abs(p32 - p16).mean()),
               decision_flips_at_0p5=int(((p32 >= 0.5) != (p16 >= 0.5)).sum()), decisions=int(p32.size),
               stored_fp16_vs_fp32_prob_absdiff_max=float(np.abs(p32 - ps).max()),
               note="fp32 vs bf16 DINOv2-L on random test crops; same head; stored features are bf16 compute saved as fp16")
    (RUNS / "ops" / a.exp / "parity_bf16.json").write_text(json.dumps(res, indent=1))
    print("PARITY", json.dumps(res))


if __name__ == "__main__":
    main()
