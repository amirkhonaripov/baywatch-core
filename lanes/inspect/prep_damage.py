"""Prepare the exterior-damage datasets (lane D, task 1).  Prepared data lives on the NVMe (FAST/inspect/damage).

Flow: (1) ONE sequential pass over the raw HDD files -> resized JPEGs on the NVMe (training pool <= 1024 px, SYNDCAR/Coco <= 1536 px);
(2) hashing / near-duplicate removal / splits on the NVMe copies; (3) YOLO-seg labels + list files + eval gt.json.

Sources (7 classes = seen.vocab.DAMAGE_TYPES: dent, scratch, crack, glass_shatter, lamp_broken, tire_flat, other)
  VehiDE   (Kaggle copy of the official release, 13,945 images): train JSON 11,621 -> train / val(5 %); official validation
            2,324 -> in-source TEST `vehide_val`.
  HITL     (Humans in the Loop, CC0, 814 damage images): 80 / 10 / 10 train / val / test (`hitl_test`) by pHash cluster.
  CarDD    (HF FiftyOne mirror, 2,816 images): 20 % clustered hold-out `cardd_test`, 5 % val, rest train (model B only).
Cross-source TESTS (never used in training; near-duplicates of training images are removed from the TEST side):
  SYNDCAR (4 classes, boxes only; one micro-car, digitally generated damage), Kaggle Coco Car Damage (70 annotated images,
  class-agnostic), Roboflow jijo/car-damage-new (dent/scratch; 416x416 stretched, provenance unknown), and
  `cardd_trainpart` = the CarDD images that model B trains on, used ONLY as a cross-source test for model A.
Two training variants over ONE shared image pool:
  A = VehiDE + HITL  (no CarDD at all: clean cross-source numbers on CarDD / SYNDCAR / Coco / Jijo)
  B = A + CarDD train (the shipped model)
All dedup is by perceptual hash (eval.manifest pHash, Hamming <= 8, original and mirrored).

Outputs under FAST/inspect/damage/ :
    yolo/{images,labels}/all/*, trainA.txt valA.txt trainB.txt valB.txt, data_A.yaml data_B.yaml
    eval/<set>/gt.json (+ eval/<cross set>/images/*), splits.json, train_files_{A,B}.txt (staged paths, for the leakage audit)

Run:  python -P -m lanes.inspect.prep_damage
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from lanes.inspect import common as C

OUT = C.FAST_I / "damage"
POOL_IMG, POOL_LAB = OUT / "yolo" / "images" / "all", OUT / "yolo" / "labels" / "all"
TRAIN_MAX, EVAL_MAX = 1024, 1536


def read_vehide() -> tuple[list[dict], list[dict]]:
    """-> (train_records, official_val_records); record = {id, path, source, inst:[{cls, poly}]}"""
    from seen.paths import RAW
    root = RAW / "inspect_vehide"
    out = []
    for jf, sub, tag in [("0Train_via_annos.json", "image/image", "train"), ("0Val_via_annos.json", "validation/validation", "val")]:
        j = json.loads((root / jf).read_text())
        recs = []
        for name, v in j.items():
            p = root / sub / name  # existence is checked by the staging pass (threaded), not by serial stat() calls
            inst = []
            for r in v["regions"]:
                pts = list(zip(r["all_x"], r["all_y"]))
                if len(pts) >= 3 and r["class"] in C.VEHIDE_MAP:
                    inst.append({"cls": C.VEHIDE_MAP[r["class"]], "src_cls": r["class"], "poly": [[[float(a), float(b)] for a, b in pts]]})
            recs.append({"id": f"vehide:{tag}/{name}", "path": str(p), "inst": inst, "source": "vehide"})
        out.append(recs)
    return out[0], out[1]


def write_labels(recs: list[dict]) -> None:
    """YOLO-seg label file per staged pool image (polygons normalised by the ORIGINAL size, so scale-free)."""
    C.ensure(POOL_LAB)
    cls_idx = {c: i for i, c in enumerate(C.DAMAGE_CLASSES)}
    for r in recs:
        w0, h0 = r["orig_wh"]
        lines = []
        for ins in r["inst"]:
            for poly in ins["poly"]:
                a = np.asarray(poly, dtype=np.float64)
                a[:, 0] = np.clip(a[:, 0] / w0, 0, 1)
                a[:, 1] = np.clip(a[:, 1] / h0, 0, 1)
                if len(a) >= 3 and (a.max(0) - a.min(0)).min() > 1e-4:
                    lines.append(f"{cls_idx[ins['cls']]} " + " ".join(f"{v:.5f}" for v in a.ravel()))
        (POOL_LAB / (Path(r["fast"]).stem + ".txt")).write_text("\n".join(lines))


def write_eval_gt(name: str, recs: list[dict]) -> None:
    gt = [{"id": r["id"], "path": r["fast"], "w": r["wh"][0], "h": r["wh"][1], "gt": C.scale_inst(r["inst"], r["scale"]),
           "source": r.get("source", "")} for r in recs]
    C.write_json(gt, OUT / "eval" / name / "gt.json")


def _cluster_split(recs: list[dict], fracs: tuple[float, float], tag: str) -> tuple[list, list, list]:
    """-> (train, val, test): test fraction fracs[0], val fracs[1]; near-duplicate clusters never straddle."""
    cl = C.group_near_dups([r["fast"] for r in recs])
    tr, va, te = [], [], []
    for r, c in zip(recs, cl):
        u = C.seeded_fraction(f"{tag}-{c}")
        (te if u < fracs[0] else va if u < fracs[0] + fracs[1] else tr).append(r)
    return tr, va, te


def main() -> None:
    info: dict = {"dedup": "eval.manifest pHash, Hamming <= 8, original + mirrored; hashes taken on the staged NVMe copies", "sets": {}}
    from lanes.inspect.prep_cardd import read_cardd
    from lanes.inspect.prep_eval_sets import read_cococardamage, read_jijo, read_syndcar_damage
    from lanes.inspect.prep_hitl import read_hitl_damage

    vh_train, vh_test = read_vehide()
    hitl, cardd = read_hitl_damage(), read_cardd()
    synd = [r for r in read_syndcar_damage() if r["inst"]]
    coco, jijo = read_cococardamage(), [r for r in read_jijo() if r["inst"]]
    print("raw: vehide train", len(vh_train), "val", len(vh_test), "hitl", len(hitl), "cardd", len(cardd),
          "syndcar", len(synd), "coco", len(coco), "jijo", len(jijo), flush=True)

    # (1) single sequential staging pass (HDD -> NVMe)
    pool_ok = C.stage_images(vh_train + vh_test + hitl + cardd, POOL_IMG, TRAIN_MAX)
    ok_ids = {r["id"] for r in pool_ok}
    vh_train, vh_test = [r for r in vh_train if r["id"] in ok_ids], [r for r in vh_test if r["id"] in ok_ids]
    hitl, cardd = [r for r in hitl if r["id"] in ok_ids], [r for r in cardd if r["id"] in ok_ids]
    synd = C.stage_images(synd, OUT / "eval" / "syndcar" / "images", EVAL_MAX)
    coco = C.stage_images(coco, OUT / "eval" / "cococardamage" / "images", EVAL_MAX)
    jijo = C.stage_images(jijo, OUT / "eval" / "jijo_roboflow" / "images", TRAIN_MAX)
    print("staged", len(pool_ok), len(synd), len(coco), len(jijo), flush=True)

    # (2) splits + dedup on the NVMe copies
    vh_tr, vh_va, _ = _cluster_split(vh_train, (0.0, 0.05), "vehide")
    h_tr, h_va, h_te = _cluster_split(hitl, (0.10, 0.10), "hitl")
    c_tr, c_va, c_te = _cluster_split(cardd, (0.20, 0.05), "cardd")
    in_source = {"vehide_val": vh_test, "hitl_test": h_te, "cardd_test": c_te}

    cand = {"vh_tr": vh_tr, "vh_va": vh_va, "h_tr": h_tr, "h_va": h_va, "c_tr": c_tr, "c_va": c_va}
    held = [r["fast"] for v in in_source.values() for r in v]
    info["dropped_for_in_source_tests"] = {}
    for k, lst in cand.items():  # in-source tests stay intact: drop the TRAIN-side twins
        tw = C.near_dup_matches([r["fast"] for r in lst], held)
        info["dropped_for_in_source_tests"][k] = len(tw)
        cand[k] = [r for r in lst if r["fast"] not in tw]
    poolA = cand["vh_tr"] + cand["vh_va"] + cand["h_tr"] + cand["h_va"]
    poolB_extra = cand["c_tr"] + cand["c_va"]
    all_cand = [r["fast"] for r in poolA + poolB_extra]
    A_paths = [r["fast"] for r in poolA]

    cross = {"syndcar": synd, "cococardamage": coco, "jijo_roboflow": jijo}
    for name, recs in list(cross.items()):  # cross-source tests: drop the TEST-side twins
        tw = C.near_dup_matches([r["fast"] for r in recs], all_cand)
        info["sets"][name] = {"n_before": len(recs), "removed_as_train_near_dup": len(tw),
                              "removed_examples": [{"test": a, "train": b, "dist": d} for a, (b, d) in list(tw.items())[:5]]}
        cross[name] = [r for r in recs if r["fast"] not in tw]
    tw = C.near_dup_matches([r["fast"] for r in poolB_extra], A_paths)
    info["sets"]["cardd_trainpart"] = {"n_before": len(poolB_extra), "removed_as_A_pool_near_dup": len(tw),
                                       "note": "CarDD images model B trains on; cross-source test for model A only"}
    cardd_trainpart = [r for r in poolB_extra if r["fast"] not in tw]

    # (3) labels, lists, eval ground truth
    write_labels(poolA + poolB_extra)
    trA, vaA = [r["fast"] for r in cand["vh_tr"] + cand["h_tr"]], [r["fast"] for r in cand["vh_va"] + cand["h_va"]]
    trB, vaB = trA + [r["fast"] for r in cand["c_tr"]], vaA + [r["fast"] for r in cand["c_va"]]
    for tag, tr, va in [("A", trA, vaA), ("B", trB, vaB)]:
        (OUT / f"train{tag}.txt").write_text("\n".join(tr))
        (OUT / f"val{tag}.txt").write_text("\n".join(va))
        (OUT / f"data_{tag}.yaml").write_text(f"path: {OUT}\ntrain: train{tag}.txt\nval: val{tag}.txt\nnames:\n"
                                              + "".join(f"  {i}: {c}\n" for i, c in enumerate(C.DAMAGE_CLASSES)))
        (OUT / f"train_files_{tag}.txt").write_text("\n".join(tr + va))
    for name, recs in {**in_source, **cross, "cardd_trainpart": cardd_trainpart}.items():
        if recs:
            write_eval_gt(name, recs)
            info["sets"].setdefault(name, {})["n_after"] = len(recs)
    inst = lambda rs: dict(Counter(i["cls"] for r in rs for i in r["inst"]))  # noqa: E731
    info["train_A"] = {"images": len(trA), "val": len(vaA), "instances": inst(cand["vh_tr"] + cand["h_tr"])}
    info["train_B"] = {"images": len(trB), "val": len(vaB), "instances": inst(cand["vh_tr"] + cand["h_tr"] + cand["c_tr"])}
    info["in_source_tests"] = {k: {"images": len(v), "instances": inst(v)} for k, v in in_source.items()}
    C.write_json(info, OUT / "splits.json")
    C.write_json(info, C.PREP_I / "damage_splits.json")  # small copy on the HDD
    print(json.dumps(info, indent=1)[:4000])


if __name__ == "__main__":
    main()
