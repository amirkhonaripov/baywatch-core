"""Prepare car-parts segmentation data (lane D, task 2). Prepared data lives on the NVMe (FAST/inspect/parts).

Training pool : Ultralytics carparts-seg (CC BY 4.0, Roboflow origin; ALL its images pooled, because its own splits contain
                Roboflow augmentation copies of the same photo in several splits) + Humans in the Loop parts half (CC0).
                Re-split by near-duplicate cluster: 15 % in-source test, 5 % val, rest train.
Cross-source  : SYNDCAR (28 parts -> coarse families, ONE micro-car), CarDamagePanel (565 images, 21 panels, unified level).
                Near-duplicates of training images are removed from the TEST side.
Classes are mapped to the unified side-agnostic vocabulary lanes.inspect.common.PART_CLASSES (SYNDCAR at family level).

Requires lanes.inspect.prep_damage to have run first (SYNDCAR is staged there once).
Run: python -P -m lanes.inspect.prep_parts
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from lanes.inspect import common as C
from seen.paths import RAW

OUT = C.FAST_I / "parts"
POOL_IMG, POOL_LAB = OUT / "yolo" / "images" / "all", OUT / "yolo" / "labels" / "all"
TRAIN_MAX = 1024


def _read_yolo_seg_dir(img_dir: Path, lab_dir: Path, names: dict[int, str], mapping: dict, dataset: str, tag: str) -> list[dict]:
    """YOLO-seg labels -> records with NORMALISED polygons (no image is opened). Threaded: serial small-file reads crawl on the
    saturated shared HDD (~1 file/s) while 32 requests in flight give ~100 files/s."""
    from concurrent.futures import ThreadPoolExecutor

    def one(lp: Path):
        ip = img_dir / (lp.stem + ".jpg")  # carparts-seg and CarDamagePanel images are .jpg; fall back to other extensions
        if not ip.exists():
            ip = next((img_dir / (lp.stem + e) for e in (".jpeg", ".png") if (img_dir / (lp.stem + e)).exists()), None)
            if ip is None:
                return None
        inst = []
        for line in lp.read_text().splitlines():
            t = line.split()
            if len(t) < 7:
                continue
            nm = names[int(t[0])]
            cls = mapping.get(nm)
            if cls is None:
                continue
            inst.append({"cls": cls, "src_cls": nm, "polyn": np.asarray(t[1:], dtype=np.float64).reshape(-1, 2).tolist()})
        return {"id": f"{dataset}:{tag}/{ip.name}", "path": str(ip), "source": dataset, "inst": inst}

    with ThreadPoolExecutor(32) as ex:
        return [r for r in ex.map(one, sorted(lab_dir.glob("*.txt"))) if r is not None]


def read_carparts_seg() -> list[dict]:
    import yaml
    root = RAW / "inspect_carparts_seg" / "extracted"
    names = {int(k): v for k, v in yaml.safe_load((root / "carparts-seg.yaml").read_text())["names"].items()}
    out = []
    for split in ["train", "val", "test"]:
        out += _read_yolo_seg_dir(root / "images" / split, root / "labels" / split, names, C.CARPARTS_MAP, "carparts_seg", split)
    return out


def read_hitl_parts() -> list[dict]:
    from lanes.inspect.prep_hitl import read_hitl_parts as rd
    out = []
    for r in rd():
        inst = [{**i, "cls": C.HITL_PART_MAP[i["src_cls"].lower()]} for i in r["inst"] if i["src_cls"].lower() in C.HITL_PART_MAP]
        out.append({**r, "inst": inst})
    return out


def read_cardamagepanel() -> list[dict]:
    import yaml
    root = RAW / "inspect_cardamagepanel"
    names = {int(k): v for k, v in yaml.safe_load((root / "data.yaml").read_text())["names"].items()}
    out = []
    for split in ["train", "val", "test"]:
        out += _read_yolo_seg_dir(root / "images" / split, root / "labels" / split, names, C.HITL_PART_MAP, "cardamagepanel", split)
    return out


def finalize(recs: list[dict]) -> list[dict]:
    """After staging: make every instance carry `polyn` (normalised) and `poly` (ORIGINAL pixels, as scale_inst expects)."""
    for r in recs:
        w0, h0 = r["orig_wh"]
        for i in r["inst"]:
            if "polyn" not in i:  # HITL (Supervisely, pixel coordinates in original size)
                i["polyn"] = [[x / w0, y / h0] for x, y in i["poly"][0]]
            i["poly"] = [[[x * w0, y * h0] for x, y in i["polyn"]]]
    return recs


def write_labels(recs: list[dict]) -> None:
    C.ensure(POOL_LAB)
    idx = {c: i for i, c in enumerate(C.PART_CLASSES)}
    for r in recs:
        lines = []
        for ins in r["inst"]:
            a = np.clip(np.asarray(ins["polyn"], dtype=np.float64), 0, 1)
            if len(a) >= 3 and (a.max(0) - a.min(0)).min() > 1e-4:
                lines.append(f"{idx[ins['cls']]} " + " ".join(f"{v:.5f}" for v in a.ravel()))
        (POOL_LAB / (Path(r["fast"]).stem + ".txt")).write_text("\n".join(lines))


def write_eval_gt(name: str, recs: list[dict]) -> None:
    gt = [{"id": r["id"], "path": r["fast"], "w": r["wh"][0], "h": r["wh"][1], "gt": C.scale_inst(r["inst"], r["scale"]),
           "source": r.get("source", "")} for r in recs]
    C.write_json(gt, OUT / "eval" / name / "gt.json")


def main() -> None:
    info: dict = {"sets": {}}
    cs, hp, cp = read_carparts_seg(), read_hitl_parts(), read_cardamagepanel()
    print("raw: carparts_seg", len(cs), "hitl_parts", len(hp), "cardamagepanel", len(cp), flush=True)
    pool = [r for r in cs + hp if r["inst"]]
    pool = finalize(C.stage_images(pool, POOL_IMG, TRAIN_MAX)) if pool else []
    cp = finalize(C.stage_images([r for r in cp if r["inst"]], OUT / "eval" / "cardamagepanel" / "images", TRAIN_MAX))

    cl = C.group_near_dups([r["fast"] for r in pool])
    tr, va, te_cs, te_hp = [], [], [], []
    for r, c in zip(pool, cl):
        u = C.seeded_fraction(f"parts-{c}")
        if u < 0.15:
            (te_cs if r["source"] == "carparts_seg" else te_hp).append(r)
        elif u < 0.20:
            va.append(r)
        else:
            tr.append(r)
    in_test = {"carparts_test": te_cs, "hitl_parts_test": te_hp}

    from lanes.inspect.prep_syndcar import syndcar_parts_from_staged
    cross = {"cardamagepanel": cp, "syndcar_parts": syndcar_parts_from_staged()}
    keep = [r["fast"] for r in tr + va]
    for name, recs in list(cross.items()):
        tw = C.near_dup_matches([r["fast"] for r in recs], keep) if recs else {}
        info["sets"][name] = {"n_before": len(recs), "removed_as_train_near_dup": len(tw),
                              "removed_examples": [{"test": a, "train": b, "dist": d} for a, (b, d) in list(tw.items())[:5]]}
        cross[name] = [r for r in recs if r["fast"] not in tw]
    held = [r["fast"] for v in in_test.values() for r in v] + [r["fast"] for v in cross.values() for r in v]
    # train/val must not contain near-duplicates of the in-source tests (already clustered) - nothing more to drop

    write_labels(tr + va)
    (OUT / "trainP.txt").write_text("\n".join(r["fast"] for r in tr))
    (OUT / "valP.txt").write_text("\n".join(r["fast"] for r in va))
    (OUT / "data.yaml").write_text(f"path: {OUT}\ntrain: trainP.txt\nval: valP.txt\nnames:\n"
                                   + "".join(f"  {i}: {c}\n" for i, c in enumerate(C.PART_CLASSES)))
    (OUT / "train_files.txt").write_text("\n".join(r["fast"] for r in tr + va))
    for name, recs in {**in_test, **cross}.items():
        if recs:
            write_eval_gt(name, recs)
            info["sets"].setdefault(name, {})["n_after"] = len(recs)
    info["train"] = {"images": len(tr), "val": len(va), "by_source": dict(Counter(r["source"] for r in tr)),
                     "instances_by_class": dict(Counter(i["cls"] for r in tr for i in r["inst"]))}
    C.write_json(info, OUT / "splits.json")
    C.write_json(info, C.PREP_I / "parts_splits.json")
    print(json.dumps(info, indent=1)[:3000])


if __name__ == "__main__":
    main()
