"""Track-voting evaluation on SYNTHETIC tracks (no real Uzbek plate tracks exist in our data; declared synthetic).

A track = one invented plate (fixed style) rendered in K frames whose resolution grows from ~64 to ~240 px (car approaching) with an
independent degradation seed per frame. We compare: mean single-frame accuracy, best-confidence frame, majority vote over strings,
pooled per-slot voting (unconstrained), pooled + format-constrained voting, for the first k frames (k = 3, 5, K).

  python -m lanes.ident.eval_voting --exp ocr_main --model-dir <dir> --model model --tracks 400
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from lanes.ident import evalkit as ek
from lanes.ident import uz_format as uz
from lanes.ident.eval_ocr import resolve_model
from lanes.ident.ocr_onnx import PlateOcrOnnx, default_providers
from lanes.ident.reader import PlateReader
from lanes.ident.synth.augment import degrade
from lanes.ident.synth.render import render_plate
from lanes.ident.voting import Observation, vote
from seen.paths import RUNS


def make_tracks(n: int, k: int, seed: int):
    rng = random.Random(seed)
    tracks = []
    for t in range(n):
        fmt = rng.choices([uz.UZ_PRIVATE, uz.UZ_LEGAL, uz.UZ_EV], weights=[.65, .25, .10])[0]
        text = uz.sample_text(rng, uz.UZ_PRIVATE if fmt != uz.UZ_LEGAL else uz.UZ_LEGAL)
        plate, f, _ = render_plate(text, rng, fmt)
        arr = np.asarray(plate)
        frames = []
        for i in range(k):
            w = int(64 + (240 - 64) * i / max(k - 1, 1))
            img, _ = degrade(arr, seed * 100_003 + t * 101 + i, out_w=w)
            frames.append((img, w))
        tracks.append((text, frames))
    return tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tracks", type=int, default=400)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=99)
    a = ap.parse_args()
    onnx, cfg = resolve_model(Path(a.model_dir), a.model)
    ocr = PlateOcrOnnx(onnx, cfg, providers=default_providers(a.device))
    rd = PlateReader(ocr, mode="argmax", orient=False)       # per-frame reads are raw probabilities; constraint is applied by the voter
    tracks = make_tracks(a.tracks, a.frames, a.seed)
    obs_all = []
    for text, frames in tracks:
        res = rd.read([f for f, _ in frames])
        obs_all.append((text, [Observation(r.probs, r.text, r.conf, quality=w, frame=i) for i, (r, (_, w)) in enumerate(zip(res, frames))]))
    comps = []
    def add(name, preds, gts, extra=""):
        comps.append(ek.component(f"ident.vote.{a.exp}.{name}", f"Track voting [{name}]", "synthetic", "rendered Uzbek plates, K frames per track, rising resolution",
                                  "synthetic UZ tracks (ours)", "held-out seed", len(gts), a.model, ek.ocr_metrics(preds, gts),
                                  notes=f"{a.tracks} synthetic tracks x {a.frames} frames; {extra}"))
        em = [m for m in comps[-1]["metrics"] if m["name"] == "exact_match"][0]
        print(f"{name:34s} n={len(gts)} EM={em['value']:.3f} [{em['ci_low']:.3f},{em['ci_high']:.3f}]", flush=True)
    gts = [t for t, _ in obs_all]
    flat_p = [o.text for _, os_ in obs_all for o in os_]
    flat_g = [t for t, os_ in obs_all for _ in os_]
    add("single_frame_argmax(all frames)", flat_p, flat_g, "unit = frame, not track; frames are correlated within a track")
    for k in (3, 5, a.frames):
        sub = [(t, os_[:k]) for t, os_ in obs_all]
        add(f"first{k}_best_conf_frame", [vote(o, "best", constrained=False).text for _, o in sub], gts)
        add(f"first{k}_majority", [vote(o, "majority").text for _, o in sub], gts)
        add(f"first{k}_pooled", [vote(o, "pooled", constrained=False).text for _, o in sub], gts)
        add(f"first{k}_pooled+constrained", [vote(o, "pooled", constrained=True).text for _, o in sub], gts)
    ek.write_results(RUNS / "ident" / a.exp / "component_results_voting.json", comps)


if __name__ == "__main__":
    main()
