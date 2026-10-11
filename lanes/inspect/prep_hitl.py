"""Humans in the Loop 'Car Parts and Car Damages' (CC0), Supervisely format.

NOTE: the two top-level folders are labelled the wrong way round in the Kaggle release ('Car damages dataset' holds
the 998 PARTS images, 'Car parts dataset' holds the 814 DAMAGE images), so we decide per image from the class titles,
never from the folder names.
"""
from __future__ import annotations

import json
from pathlib import Path

from lanes.inspect import common as C
from seen.paths import RAW

HITL_PARTS = ["Windshield", "Back-windshield", "Front-window", "Back-window", "Front-door", "Back-door", "Front-wheel",
              "Back-wheel", "Front-bumper", "Back-bumper", "Headlight", "Tail-light", "Hood", "Trunk", "License-plate",
              "Mirror", "Roof", "Grille", "Rocker-panel", "Quarter-panel", "Fender"]


def _load_one(annf: Path):
    imgf = annf.parent.parent / "img" / annf.name[: -len(".json")]
    j = json.loads(annf.read_text())
    titles = {o["classTitle"] for o in j["objects"]}
    kind = "damage" if titles & set(C.HITL_DMG_MAP) else "parts" if titles & set(HITL_PARTS) else None
    if kind is None:
        return None
    inst = []
    for o in j["objects"]:
        if o.get("geometryType") != "polygon":
            continue
        ext = o["points"]["exterior"]
        if len(ext) < 3:
            continue
        t = o["classTitle"]
        cls = C.HITL_DMG_MAP.get(t) if kind == "damage" else t
        if cls is None:
            continue
        inst.append({"cls": cls, "src_cls": t, "poly": [[[float(x), float(y)] for x, y in ext]]})
    return kind, {"id": f"hitl:{imgf.name}", "path": str(imgf), "source": "hitl", "inst": inst}


def _scan() -> dict[str, list[dict]]:
    from concurrent.futures import ThreadPoolExecutor
    root = RAW / "inspect_hitl"
    out = {"damage": [], "parts": []}
    files = sorted(root.glob("*/*/ann/*.json"))
    with ThreadPoolExecutor(32) as ex:  # deep request queue: serial reads crawl on the saturated shared HDD
        for res in ex.map(_load_one, files):
            if res is not None:
                out[res[0]].append(res[1])
    return out


_CACHE: dict | None = None


def read_hitl_damage() -> list[dict]:
    global _CACHE
    _CACHE = _CACHE or _scan()
    return _CACHE["damage"]


def read_hitl_parts() -> list[dict]:
    global _CACHE
    _CACHE = _CACHE or _scan()
    return _CACHE["parts"]


if __name__ == "__main__":
    d, p = read_hitl_damage(), read_hitl_parts()
    from collections import Counter
    print("damage images", len(d), Counter(i["src_cls"] for r in d for i in r["inst"]))
    print("parts images", len(p), Counter(i["src_cls"] for r in p for i in r["inst"]))
