"""Evaluate a damage-seg model on the prepared eval sets -> ComponentResult JSON (+ per-image predictions).

    scripts/box.sh --gpu 1 -- python -P -m lanes.inspect.eval_damage --exp dmg_A --weights <best.pt> --sets syndcar,cococardamage,...

Per set: box AP50 / mask AP50 per class (mask only where GT has polygons), mAP over classes with GT, class-agnostic
("any damage") AP50 and recall at 0.1 / 0.5 / 1 false positives per image, all with image-level bootstrap 95 % CIs.
Predictions are restricted to the label space of each test set (e.g. SYNDCAR has no 'dent' GT, so dent predictions are
ignored there instead of being counted as false positives; this is stated in the result notes).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lanes.inspect import common as C
from lanes.inspect.evaldet import Evaluator
from lanes.inspect.prep_damage import OUT as DMG

SETS = {
    # name: (display name, dataset string, split, label space for predictions, class list for per-class metrics, domain notes)
    "vehide_val": ("VehiDE official validation split (in-source)", "VehiDE (Kaggle copy of official release)", "official val 2,324 (never trained on)",
                   ["dent", "scratch", "crack", "glass_shatter", "lamp_broken", "other"], "web photos, Vietnam"),
    "hitl_test": ("Humans in the Loop damage, held-out 10 % (in-source)", "Humans in the Loop Car Damages (CC0)", "10 % clustered hold-out",
                  ["dent", "scratch", "crack", "other"], "web/phone photos"),
    "cardd_test": ("CarDD held-out 20 % (in-source for model B, cross-source for model A)", "CarDD (HF FiftyOne mirror)", "20 % clustered hold-out",
                   ["dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"], "Flickr/Shutterstock web photos"),
    "cardd_trainpart": ("CarDD train portion (cross-source for model A only; model B trained on it)", "CarDD (HF FiftyOne mirror)", "the 80 % CarDD portion that model B trains on",
                        ["dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"], "Flickr/Shutterstock web photos"),
    "syndcar": ("SYNDCAR (cross-source)", "SYNDCAR (Mendeley hzpj48krdt, CC BY 4.0)", "all 245 (never trained on)",
                ["glass_shatter", "lamp_broken", "crack", "scratch"], "see notes"),
    "cococardamage": ("Kaggle Coco Car Damage (cross-source, class-agnostic)", "Kaggle lplenka/coco-car-damage-detection-dataset (CC0)", "70 annotated images (never trained on)",
                      None, "web photos"),
    "jijo_roboflow": ("Roboflow jijo/car-damage-new (cross-source, 416x416 stretched)", "Roboflow Universe jijo/car-damage-new v2 (CC BY label, provenance unknown)", "all (never trained on)",
                      ["dent", "scratch"], "web photos, aspect-stretched to 416x416"),
}
AGNOSTIC_ONLY = {"cococardamage"}


def load_gt(name: str) -> list[dict]:
    return C.read_json(DMG / "eval" / name / "gt.json")


def predict_set(model, name: str, imgsz: int, conf: float = 0.001) -> list[dict]:
    gt = load_gt(name)
    recs = []
    paths = [g["path"] for g in gt]
    names = model.names
    t_inf = []
    for g, res in zip(gt, C.predict_chunks(model, paths, 4, imgsz=imgsz, conf=conf, iou=0.6, max_det=100, device=0)):
        pred = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            cl = res.boxes.cls.cpu().numpy().astype(int)
            sc = res.boxes.conf.cpu().numpy()
            polys = res.masks.xy if res.masks is not None else [None] * len(cl)
            for b, c, s, p in zip(xyxy, cl, sc, polys):
                pred.append({"cls": names[int(c)], "score": float(s), "box": b.tolist(),
                             "poly": [p.tolist()] if p is not None and len(p) >= 3 else None})
        t_inf.append(sum(res.speed.values()))
        recs.append({"id": g["id"], "path": g["path"], "w": g["w"], "h": g["h"], "gt": g["gt"], "pred": pred})
    return recs, t_inf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--tag", default="")
    ap.add_argument("--model-name", default="yolo11m-seg")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)
    results = []
    out = C.ensure(C.RUNS_I / a.exp)
    for name in a.sets.split(","):
        if not (DMG / "eval" / name / "gt.json").exists():
            print("skip", name, "(no gt.json)")
            continue
        disp, ds, split, label_space, domain = SETS[name]
        recs, t_inf = predict_set(model, name, a.imgsz)
        agnostic = name in AGNOSTIC_ONLY
        if agnostic:
            for r in recs:
                for g in r["gt"]:
                    g["cls"] = "damage"
        space = set(label_space) if label_space else None
        eval_recs = []
        for r in recs:
            pr = [p for p in r["pred"] if space is None or p["cls"] in space]
            eval_recs.append({**r, "pred": pr})
        classes = [] if agnostic else [c for c in label_space if any(g["cls"] == c for r in recs for g in r["gt"])]
        metrics = []
        groups = C.group_near_dups([r["path"] for r in eval_recs])  # CIs resample near-duplicate clusters, not images
        has_poly = all(g.get("poly") for r in recs for g in r["gt"]) and any(r["gt"] for r in recs)
        metrics += Evaluator(eval_recs, classes, "box", 0.5, groups=groups).summary("")
        if has_poly:
            metrics += Evaluator(eval_recs, classes, "mask", 0.5, groups=groups).summary("")
        n_inst = sum(len(r["gt"]) for r in recs)
        metrics.append({"name": "n_gt_instances", "value": float(n_inst)})
        metrics.append({"name": "n_near_duplicate_clusters", "value": float(len(set(groups)))})
        metrics.append({"name": "n_images_without_gt", "value": float(sum(1 for r in recs if not r["gt"]))})
        results.append({
            "id": f"inspect.damage.{a.exp}.{name}", "name": f"Exterior damage ({a.model_name}) on {disp}", "lane": "inspect",
            "data_kind": "synthetic" if name == "syndcar" else "real",
            "domain": f"phone/web photo ({domain}); no fixed-camera damage data exists" + ("; DIGITALLY GENERATED damage on real photos of ONE vehicle" if name == "syndcar" else ""),
            "dataset": ds, "split": split, "n": len(recs), "model": f"{a.model_name} fine-tuned ({a.exp}) @ imgsz {a.imgsz}",
            "metrics": metrics,
            "notes": ("Predictions restricted to the test set's label space: " + ", ".join(sorted(space)) if space else "Class-agnostic: all predicted classes count as 'damage'.")
                     + ("; mask AP not computed (GT is boxes only)" if not has_poly else "") + a.tag,
        })
        C.write_json(recs, out / f"preds_{name}.json")
        print(name, "n=", len(recs), "latency ms (mean incl. pre/post):", float(np.mean(t_inf)), flush=True)
    C.write_json(results, out / "component_results_damage.json")
    print("wrote", out / "component_results_damage.json")


if __name__ == "__main__":
    main()
