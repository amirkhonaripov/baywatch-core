"""Slot-occupancy failure gallery: worst images per (dataset, weather) for the untiled and the tiled detector, slots coloured by outcome.

    python -m lanes.flow.make_failures_occ --out <dir>
Colours: red = occupied slot predicted vacant (missed car), yellow = vacant slot predicted occupied, green = correct occupied, blue = correct vacant.
"""
import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

from lanes.flow.common import run_dir
from lanes.flow.eval_occupancy import slot_scores
from lanes.flow.run_detect import cache_path, get_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="rfdetr-large")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for ds, weather, which in (("pklot_occ", "rainy", "pklot"), ("pklot_occ", "sunny", "pklot"), ("cnrext_occ", "rainy", "cnrext")):
        items = [i for i in get_items(ds) if i.cond == weather]
        res = {}
        for m in (a.base, a.base + "+tile"):
            cal = json.loads((run_dir(f"occupancy_{m}") / "calibration.json").read_text())[which]
            dets = pickle.load(open(cache_path(ds, m), "rb"))["dets"]
            errs = []
            for it in items:
                sc = slot_scores(it, dets[it.id], cal["conf"])
                pred = sc >= cal["tau"]
                truth = np.array([s["occ"] for s in it.extra["slots"]]) == 1
                errs.append((int((pred != truth).sum()), pred, truth))
            res[m] = (cal, errs)
        order = np.argsort([-e[0] for e in res[a.base][2 - 1 + 0][1]]) if False else np.argsort([-e[0] for e in res[a.base][1]])
        for rank, idx in enumerate(order[:2]):
            it = items[idx]
            for m in (a.base, a.base + "+tile"):
                cal, errs = res[m]
                n_err, pred, truth = errs[idx]
                im = cv2.imread(it.path)
                for s, p, t in zip(it.extra["slots"], pred, truth):
                    poly = np.array(s["poly"], np.int32)
                    col = (0, 255, 0) if (p and t) else (255, 128, 0) if (not p and not t) else (0, 0, 255) if t else (0, 255, 255)
                    cv2.polylines(im, [poly], True, col, 3 if col in ((0, 0, 255), (0, 255, 255)) else 1)
                txt = f"{ds} {weather} {it.id.split('/')[-1]}  {m}: {n_err} of {len(truth)} slots wrong"
                cv2.putText(im, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(im, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                fn = f"occ_{which}_{weather}_{rank}_{m.replace('+', '_')}.jpg"
                cv2.imwrite(str(out / fn), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
                meta.append({"id": f"flow_occ_{which}_{weather}_{rank}_{m}", "component": f"flow.occ.{which}.{m}.{weather}", "file": fn,
                             "description": f"{m} on {it.id}: {n_err} of {len(truth)} slots wrong (conf>={cal['conf']}, overlap>={cal['tau']}). Red = occupied slot predicted vacant, yellow = vacant predicted occupied, "
                                            "green = correct occupied, blue = correct vacant.", "category": "occupancy: " + ("small cars missed without tiling" if "tile" not in m else "after tiling")})
    (out / "failures.json").write_text(json.dumps(meta, indent=1))
    print(len(meta), "images")


if __name__ == "__main__":
    main()
