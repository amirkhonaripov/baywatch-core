"""CarDD via the Hugging Face FiftyOne mirror harpreetsahota/CarDD (2,816 images = the official training portion).

Licence: CarDD research / non-commercial (CarDD_license.pdf shipped in the mirror; copyright stays with the Flickr /
Shutterstock photographers; no redistribution). Used for training and evaluation inside this research prototype only.
FiftyOne stores each instance mask as a zlib-compressed .npy of a bool array covering the bounding box
(`bounding_box` = [x, y, w, h] relative to the image), which we turn into polygons.
"""
from __future__ import annotations

import base64
import io
import json
import zlib

import cv2
import numpy as np

from lanes.inspect import common as C
from seen.paths import RAW

CARDD_MAP = {"scratch": "scratch", "dent": "dent", "crack": "crack", "lamp broken": "lamp_broken",
             "glass shatter": "glass_shatter", "tire flat": "tire_flat"}


def _mask_to_polys(det: dict, W: int, H: int) -> list[list[list[float]]]:
    raw = zlib.decompress(base64.b64decode(det["mask"]["$binary"]["base64"]))
    m = np.load(io.BytesIO(raw), allow_pickle=False).astype(np.uint8)  # bool array; allow_pickle=False for untrusted data
    x, y, w, h = det["bounding_box"]
    x0, y0 = int(round(x * W)), int(round(y * H))
    bw, bh = max(1, int(round(w * W))), max(1, int(round(h * H)))
    if m.shape != (bh, bw):
        m = cv2.resize(m, (bw, bh), interpolation=cv2.INTER_NEAREST)
    full = np.zeros((H, W), np.uint8)
    xs, ys = max(0, x0), max(0, y0)
    xe, ye = min(W, x0 + bw), min(H, y0 + bh)
    if xe <= xs or ye <= ys:
        return []
    full[ys:ye, xs:xe] = m[ys - y0: ye - y0, xs - x0: xe - x0]
    cnts, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < 4:
            continue
        c = cv2.approxPolyDP(c, 0.8, True).reshape(-1, 2)
        if len(c) >= 3:
            polys.append([[float(a), float(b)] for a, b in c])
    return polys


def read_cardd() -> list[dict]:
    root = RAW / "inspect_cardd"
    samples = json.loads((root / "samples.json").read_text())["samples"]
    out = []
    for s in samples:
        p = root / s["filepath"]
        W, H = s["metadata"]["width"], s["metadata"]["height"]
        inst = []
        for d in s["segmentations"]["detections"]:
            cls = CARDD_MAP.get(d["label"])
            if cls is None:
                continue
            polys = _mask_to_polys(d, W, H)
            if polys:
                inst.append({"cls": cls, "src_cls": d["label"], "poly": polys})
        out.append({"id": f"cardd:{p.name}", "path": str(p), "source": "cardd", "inst": inst})
    return out


if __name__ == "__main__":
    from collections import Counter
    r = read_cardd()
    print(len(r), Counter(i["cls"] for x in r for i in x["inst"]))
