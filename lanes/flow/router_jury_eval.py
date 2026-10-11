"""External check of the scene router on Lane G's jury-style set (eval/manifests/jury.jsonl, 96 Wikimedia Commons photos with scene labels).

    python -m lanes.flow.router_jury_eval --images <dir with jury_001.jpg ...> --probe lanes/flow/assets/router_probe.npz --exp router_jury [--device mps]
The jury items were removed from the router training pool (pHash near-duplicate audit, see flow_notes.md); this script only reads them.
Strict accuracy: predicted scene == label.scene. Lenient accuracy: predicted scene in label.acceptable_scenes (Lane G's own tolerance).
'abstained' = the router answered "unknown" because the top probability was below the calibrated threshold; shown separately.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from eval.stats import confusion, prop_metric
from lanes.flow.common import component, save_results
from lanes.flow.router_model import PROMPT_SETS, Probe, SigLIP2Encoder, zero_shot_probs
from seen.vocab import SCENES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--model", default="google/siglip2-so400m-patch14-384")
    ap.add_argument("--exp", default="router_jury")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--blend", type=float, default=0.5)  # overridden by the probe card
    a = ap.parse_args()
    man = [json.loads(l) for l in open(Path(__file__).resolve().parents[2] / "eval" / "manifests" / "jury.jsonl") if l.strip()]
    imgs, rows = [], []
    for r in man:
        p = Path(a.images) / Path(r["path"]).name
        im = cv2.imread(str(p))
        if im is None:
            continue
        imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        rows.append(r)
    enc = SigLIP2Encoder(a.model, a.device)
    emb = enc.encode_images(imgs, 16)
    probe = Probe.load(Path(a.probe))
    card = json.loads(Path(a.probe).with_suffix(".json").read_text())
    text = enc.class_text_embeddings(PROMPT_SETS[card.get("prompt_set", "v2")])
    scale, bias = enc.logit_params
    tau = card.get("unknown_threshold", 0.45)
    a.blend = card.get("blend_probe_weight", a.blend)
    heads = {"zeroshot": zero_shot_probs(emb, text, scale, bias), "probe": probe.probs(emb)}
    heads["blend"] = a.blend * heads["probe"] + (1 - a.blend) * heads["zeroshot"]
    y = [r["label"]["scene"] for r in rows]
    acc = [set(r["label"].get("acceptable_scenes", [r["label"]["scene"]])) | {r["label"]["scene"]} for r in rows]
    res, report = [], {"n": len(rows), "tau": tau}
    for h, p in heads.items():
        pred = np.array(SCENES)[p.argmax(1)]
        ans = p.max(1) >= tau
        final = np.where(ans, pred, "unknown")
        strict = int(sum(f == t for f, t in zip(final, y)))
        lenient = int(sum(f in ac for f, ac in zip(final, acc)))
        raw_strict = int(sum(f == t for f, t in zip(pred, y)))
        ms = [prop_metric("accuracy_strict", strict, len(y)), prop_metric("accuracy_lenient", lenient, len(y)), prop_metric("accuracy_strict_no_rejection", raw_strict, len(y)),
              prop_metric("answered_fraction", int(ans.sum()), len(y))]
        for c in SCENES:
            m = [t == c for t in y]
            if any(m):
                ms.append(prop_metric(f"recall[{c}]", int(sum(f == c for f, mm in zip(final, m) if mm)), int(sum(m))))
        res.append(component(f"router.jury.{h}", f"Scene router on the jury-style set ({h})", "Lane G jury set (Wikimedia Commons)", len(y), ms, split="jury (held out)",
                             model=f"{a.model.split('/')[-1]}:{h}", domain="web photos", lane="router", confusion=confusion(y, list(final), list(SCENES)),
                             notes=f"unknown threshold {tau} (from leave-one-source-out on the training sources); lenient = prediction in label.acceptable_scenes"))
        report[h] = {"strict": strict / len(y), "lenient": lenient / len(y), "confusion": Counter(zip(y, final))}
        print(h, "strict %.3f lenient %.3f answered %.2f" % (strict / len(y), lenient / len(y), ans.mean()))
    save_results(a.exp, res)


if __name__ == "__main__":
    main()
