"""SYNDCAR (Mendeley hzpj48krdt, CC BY 4.0): 245 photos of one Citroen Ami micro-car (two vehicle IDs) in parking lots with
digitally generated damage (our reading of the images; the Mendeley page does not describe the method).
Damage labels: YOLO boxes (4 classes). Part labels: YOLO polygons (28 classes) on the same images.
TEST ONLY - never used for training.
"""
from __future__ import annotations

import numpy as np

from lanes.inspect import common as C
from seen.paths import RAW

_CACHE: dict | None = None


def _one(ip, root):
    w, h = C.image_size(ip)
    dl, pl = root / "labels_damage" / (ip.stem + ".txt"), root / "labels_parts" / (ip.stem + ".txt")
    di = []
    if dl.exists():
        for line in dl.read_text().splitlines():
            t = line.split()
            if len(t) < 5:
                continue
            c, cx, cy, bw, bh = int(t[0]), *map(float, t[1:5])
            di.append({"cls": C.SYND_DAMAGE_MAP[C.SYND_DAMAGE[c]], "src_cls": C.SYND_DAMAGE[c], "poly": None,
                       "box": [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]})
    pi = []
    if pl.exists():
        for line in pl.read_text().splitlines():
            t = line.split()
            if len(t) < 7:
                continue
            nm = C.SYND_PARTS[int(t[0])]
            v = np.asarray(t[1:], dtype=np.float64).reshape(-1, 2)
            pi.append({"cls": C.SYND_PART_FAMILY[nm], "src_cls": nm, "poly": [[[float(x * w), float(y * h)] for x, y in v]]})
    base = {"id": f"syndcar:{ip.name}", "path": str(ip), "source": "syndcar", "orig_wh": (w, h)}
    return {**base, "inst": di}, {**base, "inst": pi}


def read_syndcar() -> dict[str, list[dict]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    from concurrent.futures import ThreadPoolExecutor
    root = RAW / "inspect_syndcar" / "extracted"
    ips = C.list_images(root / "images")
    with ThreadPoolExecutor(32) as ex:  # threaded: serial small-file reads crawl on the shared HDD
        res = list(ex.map(lambda ip: _one(ip, root), ips))
    _CACHE = {"damage": [r[0] for r in res], "parts": [r[1] for r in res]}
    return _CACHE


if __name__ == "__main__":
    from collections import Counter
    r = read_syndcar()
    print(len(r["damage"]), Counter(i["cls"] for x in r["damage"] for i in x["inst"]))
    print(Counter(i["cls"] for x in r["parts"] for i in x["inst"]))


def syndcar_parts_from_staged() -> list[dict]:
    """Parts records for SYNDCAR pointing at the staged (resized) JPEGs written by prep_damage; polygons scaled by
    staged_width / original_width. Images removed there as near-duplicates of training data are skipped."""
    from lanes.inspect.prep_damage import OUT as DMG
    gt = {g["id"]: g for g in C.read_json(DMG / "eval" / "syndcar" / "gt.json")}
    out = []
    for r in read_syndcar()["parts"]:
        g = gt.get(r["id"])
        if g is None:
            continue
        s = g["w"] / r["orig_wh"][0]
        inst = [{**i, "poly": [[[x * s, y * s] for x, y in poly] for poly in i["poly"]]} for i in r["inst"]]
        out.append({**r, "fast": g["path"], "wh": (g["w"], g["h"]), "scale": 1.0, "inst": inst})  # polys already staged-scale
    return out
