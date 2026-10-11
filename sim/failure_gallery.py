"""Pick failure examples of the synthetic end-to-end run for docs/lanes/failures/sim/ (renders only: cartoon
proxy people, no real faces or plates).

    python -m sim.failure_gallery RUN_DIR RENDER_DIR OUT_DIR [--n 10]

Categories: car present but no vehicle detected in the bay zone (missed car, split by lifted / on the floor) and
vehicle detected in an empty bay zone (false car). Writes the bay crop JPEGs and failures.json
[{id, component, file, description, category}].
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("render_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(Path(a.run_dir) / "frame_cue.jsonl")]
    cats = {
        "missed_car_lifted": [r for r in rows if r["gt"] and r["raised"] and not r["pred"]],
        "missed_car_floor": [r for r in rows if r["gt"] and not r["raised"] and not r["pred"]],
        "false_car_empty_bay": [r for r in rows if not r["gt"] and r["pred"]],
    }
    rng = np.random.default_rng(0)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = []
    per = max(1, a.n // len(cats))
    for cat, rs in cats.items():
        if not rs:
            continue
        for r in [rs[i] for i in sorted(rng.choice(len(rs), min(per, len(rs)), replace=False))]:
            ep_dir = Path(a.render_dir) / r["episode"]
            roi = json.loads((ep_dir / "rois.json").read_text())["rois"][r["bay"]]
            im = Image.open(ep_dir / "frames" / f"{r['frame']:06d}.jpg")
            W, H = im.size
            crop = im.crop((int(roi[0] * W), int(roi[1] * H), int(roi[2] * W), int(roi[3] * H)))
            name = f"{cat}_{r['episode']}_{r['bay']}_{r['frame']:06d}.jpg"
            crop.save(out / name, quality=85)
            conf = "none" if r["conf"] is None else f"{r['conf']:.2f}"
            items.append({"id": name[:-4], "component": "e2e.synthetic", "file": name, "category": cat,
                          "description": f"{r['episode']} {r['bay']} frame {r['frame']}: ground truth car_present={r['gt']} "
                                         f"(raised={r['raised']}); best in-zone vehicle confidence {conf}. Renders are "
                                         "stylised Kenney models; the detector is COCO-pretrained RF-DETR (default pipeline)."})
    (out / "failures.json").write_text(json.dumps(items, indent=1))
    print(f"{len(items)} examples -> {out}; counts:", {k: len(v) for k, v in cats.items()}, "of", len(rows), "frames")


if __name__ == "__main__":
    main()
