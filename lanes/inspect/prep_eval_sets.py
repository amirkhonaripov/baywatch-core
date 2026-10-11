"""Readers for the cross-source damage test sets (SYNDCAR, Kaggle Coco Car Damage, Roboflow jijo/car-damage-new1).

Each reader returns records {id, path, source, inst:[{cls, box:[x1,y1,x2,y2] | None, poly:[[[x,y]..]] | None}]} in
original pixel coordinates. These sets are NEVER used for training.
"""
from __future__ import annotations

import json
from pathlib import Path

from lanes.inspect import common as C
from seen.paths import RAW


def read_cococardamage() -> list[dict]:
    """Kaggle Coco Car Damage (CC0): one class 'damage' with polygons; 59 train + 11 val annotated images (the 8 'test'
    images ship without annotations). Class-agnostic evaluation only (cls = 'damage')."""
    root = RAW / "inspect_cococardamage"
    out = []
    for sub, f in [("train", "COCO_train_annos.json"), ("val", "COCO_val_annos.json")]:
        p = root / sub / f
        if not p.exists():
            continue
        j = json.loads(p.read_text())
        by_img: dict[int, list] = {}
        for a in j["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        for im in j["images"]:
            fp = root / sub / im["file_name"]
            if not fp.exists():
                continue
            inst = []
            for a in by_img.get(im["id"], []):
                polys = [[[seg[i], seg[i + 1]] for i in range(0, len(seg) - 1, 2)] for seg in a["segmentation"] if len(seg) >= 6]
                if polys:
                    inst.append({"cls": "damage", "poly": polys})
            out.append({"id": f"cococardamage:{sub}/{im['file_name']}", "path": str(fp), "source": "cococardamage", "inst": inst})
    return out


def read_jijo() -> list[dict]:
    """Roboflow jijo/car-damage-new (CC BY label, provenance unknown), 416x416 aspect-stretched. dent / scratch boxes."""
    root = RAW / "inspect_jijo_car_damage_new1" / "x"
    out = []
    for sub in ["test", "valid", "train"]:
        p = root / sub / "_annotations.coco.json"
        if not p.exists():
            continue
        j = json.loads(p.read_text())
        names = {c["id"]: c["name"] for c in j["categories"]}
        by_img: dict[int, list] = {}
        for a in j["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        for im in j["images"]:
            fp = root / sub / im["file_name"]
            if not fp.exists():
                continue
            inst = []
            for a in by_img.get(im["id"], []):
                nm = names[a["category_id"]]
                if nm not in ("dent", "scratch"):
                    continue
                x, y, w, h = a["bbox"]
                inst.append({"cls": nm, "box": [x, y, x + w, y + h], "poly": None})
            out.append({"id": f"jijo_roboflow:{sub}/{im['file_name']}", "path": str(fp), "source": "jijo_roboflow", "inst": inst})
    return out


def read_syndcar_damage() -> list[dict]:
    from lanes.inspect.prep_syndcar import read_syndcar
    return read_syndcar()["damage"]
