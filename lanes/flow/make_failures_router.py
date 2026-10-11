"""Router failure gallery from the held-out sources (train-only probe, blend, threshold from leave-one-source-out): the most confident wrong answers
per (true scene -> predicted scene) pair. Faces are blurred with seen.blur (YuNet) before saving.

    python -m lanes.flow.make_failures_router --out docs/lanes/failures/flow --max 10
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from lanes.flow.common import prep_dir
from lanes.flow.router_eval import (choose_blend, choose_tau, fit_probe, load_emb, load_manifest, loso_parts, logo_cv_choose_c, pick_holdout)
from lanes.flow.router_model import zero_shot_probs
from seen import blur
from seen.vocab import SCENES

MODEL = "google/siglip2-so400m-patch14-384"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=10)
    ap.add_argument("--prompts", default="v2")
    a = ap.parse_args()
    z = load_emb(MODEL)
    X, y, src, ids = z["emb"], z["labels"], z["sources"], z["ids"]
    text, scale, bias = z[f"text_{a.prompts}"], float(z["scale"]), float(z["bias"])
    held = pick_holdout(y, src)
    te = np.array([s in held for s in src])
    tr = ~te
    C, _ = logo_cv_choose_c(X[tr], y[tr], src[tr])
    probe = fit_probe(X[tr], y[tr], C)
    PP, PZ, il = loso_parts(X[tr], y[tr], src[tr], C, text, scale, bias)
    w, _ = choose_blend(PP, PZ, y[tr][il])
    tau, _, _ = choose_tau(w * PP + (1 - w) * PZ, y[tr][il])
    P = w * probe.probs(X[te]) + (1 - w) * zero_shot_probs(X[te], text, scale, bias)
    pred = np.array(SCENES)[P.argmax(1)]
    conf = P.max(1)
    final = np.where(conf >= tau, pred, "unknown")
    yt, idt = y[te], ids[te]
    paths = {r["id"]: r for r in load_manifest()}
    groups = defaultdict(list)
    for i in np.where(final != yt)[0]:
        groups[(yt[i], final[i])].append((-conf[i], i))
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta, n = [], 0
    for (t, p), lst in ranked:
        for _, i in sorted(lst)[:2]:
            if n >= a.max:
                break
            r = paths[idt[i]]
            im = cv2.cvtColor(cv2.imread(r["path"]), cv2.COLOR_BGR2RGB)
            im = blur.blur_faces(im, [])
            s = 640 / max(im.shape[:2])
            if s < 1:
                im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            txt = f"true {t} -> router said {p} (p={conf[i]:.2f})"
            cv2.putText(bgr, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(bgr, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            fn = f"router_{n:02d}_{t}_as_{p}.jpg"
            cv2.imwrite(str(out / fn), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            meta.append({"id": f"flow_router_{n:02d}", "component": "router.so400m.blend", "file": fn,
                         "description": f"held-out source {r['source']} ({r['dataset']}): true scene '{t}', router answered '{p}' with p={conf[i]:.2f}. Faces blurred with seen.blur.",
                         "category": f"router: {t} -> {p}"})
            n += 1
    (out / "failures_router.json").write_text(json.dumps(meta, indent=1))
    print(n, "images;", [(k, len(v)) for k, v in ranked[:8]])


if __name__ == "__main__":
    main()
