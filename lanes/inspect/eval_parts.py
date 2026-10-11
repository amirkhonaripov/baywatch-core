"""Evaluate the parts-seg model and damage -> part assignment.

  python -P -m lanes.inspect.eval_parts parts  --exp parts_A --weights <best.pt>            # mask AP50 per part
  python -P -m lanes.inspect.eval_parts assign --exp assign_B --parts-weights <best.pt> --damage-preds runs/inspect/dmg_B/preds_syndcar.json

Part masks are evaluated at the unified part level on carparts_test / hitl_parts_test / cardamagepanel and at the coarse
FAMILY level on SYNDCAR (its labels are a different taxonomy for one micro-car, see common.PART_FAMILY).

Assignment on SYNDCAR (the only set with damage AND parts on the same images). SYNDCAR has no explicit damage->part labels,
so the reference part of a damage box is derived from the GROUND-TRUTH part polygons (max overlap, >= 10 % of the box inside
that part); the metric is therefore "agreement with the geometric reference", reported
 (a) oracle damage: GT damage box + PREDICTED parts, and
 (b) end-to-end: PREDICTED damage (conf >= thr, matched to GT damage boxes at IoU >= 0.5) + PREDICTED parts.
Each also on the 'clear' subset where the reference part covers >= 60 % of the box.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eval.stats import iou_matrix, prop_metric
from lanes.inspect import common as C
from lanes.inspect.assign import assign, box_to_poly
from lanes.inspect.evaldet import Evaluator
from lanes.inspect.prep_parts import OUT as PARTS

SETS = {
    "carparts_test": ("Ultralytics carparts-seg + HITL-parts pool, held-out 10 % (in-source)", "Ultralytics carparts-seg (CC BY 4.0)", "unified", "web photos"),
    "hitl_parts_test": ("Humans in the Loop parts, held-out (in-source)", "Humans in the Loop Car Parts (CC0)", "unified", "web/phone photos"),
    "cardamagepanel": ("CarDamagePanel 565 (cross-source)", "HF kadireks/cardamagepanel-originals (licence: other)", "unified", "photos of damaged cars"),
    "syndcar_parts": ("SYNDCAR parts, family level (cross-source, ONE micro-car)", "SYNDCAR (Mendeley hzpj48krdt, CC BY 4.0)", "family", "phone photos of one Citroen Ami in parking lots"),
}


def load_gt(name: str) -> list[dict]:
    return C.read_json(PARTS / "eval" / name / "gt.json")


def predict_parts(model, name: str, imgsz: int, family: bool, conf: float = 0.001):
    gt = load_gt(name)
    names = model.names
    recs = []
    for g, res in zip(gt, C.predict_chunks(model, [x["path"] for x in gt], 4, imgsz=imgsz, conf=conf, iou=0.6, max_det=100, device=0)):
        pred = []
        if res.boxes is not None and len(res.boxes):
            xyxy, cl, sc = res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int), res.boxes.conf.cpu().numpy()
            polys = res.masks.xy if res.masks is not None else [None] * len(cl)
            for b, c, s, p in zip(xyxy, cl, sc, polys):
                nm = names[int(c)]
                pred.append({"cls": C.PART_FAMILY[nm] if family else nm, "unified": nm, "score": float(s), "box": b.tolist(),
                             "poly": [p.tolist()] if p is not None and len(p) >= 3 else None})
        recs.append({"id": g["id"], "path": g["path"], "w": g["w"], "h": g["h"], "gt": g["gt"], "pred": pred})
    return recs


def run_parts(a) -> None:
    from ultralytics import YOLO
    model = YOLO(a.weights)
    out = C.ensure(C.RUNS_I / a.exp)
    results = []
    for name in a.sets.split(","):
        if not (PARTS / "eval" / name / "gt.json").exists():
            print("skip", name)
            continue
        disp, ds, level, domain = SETS[name]
        recs = predict_parts(model, name, a.imgsz, level == "family")
        classes = C.FAMILIES if level == "family" else C.PART_CLASSES
        classes = [c for c in classes if any(g["cls"] == c for r in recs for g in r["gt"])]
        recs = [r for r in recs if r["gt"]]
        groups = C.group_near_dups([r["path"] for r in recs])  # CIs resample near-duplicate clusters
        metrics = Evaluator(recs, classes, "mask", 0.5, groups=groups).summary("", fppis=(0.5,), n_boot=200)
        metrics += [m for m in Evaluator(recs, classes, "box", 0.5, groups=groups).summary("", fppis=(0.5,), n_boot=200) if m["name"].startswith("mAP50")]
        for m in metrics:
            m["name"] = m["name"].replace("any_damage", "any_part")  # class-agnostic AP over part instances
        metrics.append({"name": "n_near_duplicate_clusters", "value": float(len(set(groups)))})
        results.append({
            "id": f"inspect.parts.{a.exp}.{name}", "name": f"Car-part segmentation on {disp}", "lane": "inspect", "data_kind": "real",
            "domain": f"phone/web photo ({domain}); no fixed-camera data", "dataset": ds, "split": name, "n": len(recs),
            "model": f"{a.model_name} fine-tuned ({a.exp}) @ imgsz {a.imgsz}", "metrics": metrics,
            "notes": f"label level: {level}; images without GT instances removed. Mask AP50 per class with image-level bootstrap CIs. {a.tag}"})
        C.write_json(recs, out / f"preds_{name}.json")
        print(name, len(recs), flush=True)
    C.write_json(results, out / "component_results_parts.json")


def _gt_part_for_box(box, gt_parts: list[dict], h: int, w: int, min_frac: float = 0.10):
    """Reference part family of a GT damage box from GT part polygons -> (family | None, covered fraction)."""
    polys = [g["poly"] for g in gt_parts]
    labels = [g["cls"] for g in gt_parts]
    res = assign([box_to_poly(box)], polys, labels, None, h, w, min_frac)
    return res[0]


def run_assign(a) -> None:
    from ultralytics import YOLO
    model = YOLO(a.parts_weights)
    out = C.ensure(C.RUNS_I / a.exp)
    gt_d = {g["id"]: g for g in C.read_json(PARTS.parent / "damage" / "eval" / "syndcar" / "gt.json")}
    gt_p = {g["id"]: g for g in C.read_json(PARTS / "eval" / "syndcar_parts" / "gt.json")}
    dmg_pred = {r["id"]: r for r in C.read_json(Path(a.damage_preds))}
    ids = [i for i in gt_d if i in gt_p]
    rows_a, rows_b = [], []  # (correct, clear)
    n_b_matched = n_b_gt = 0
    unknown_a = 0
    for i, res in zip(ids, C.predict_chunks(model, [gt_p[i]["path"] for i in ids], 4, imgsz=a.imgsz, conf=a.part_conf, iou=0.6, max_det=100, device=0)):
        g, gp = gt_d[i], gt_p[i]
        h, w = g["h"], g["w"]
        if res.masks is not None and len(res.masks.xy):
            labels = [C.PART_FAMILY[res.names[int(c)]] for c in res.boxes.cls.cpu().numpy().astype(int)]
            ppolys = [[p.tolist()] for p in res.masks.xy]
            pscores = res.boxes.conf.cpu().numpy().tolist()
        else:
            labels, ppolys, pscores = [], [], []
        # reference parts for GT damage boxes
        refs = [_gt_part_for_box(x["box"], gp["gt"], h, w) for x in g["gt"]]
        # (a) oracle damage box
        pa = assign([box_to_poly(x["box"]) for x in g["gt"]], ppolys, labels, pscores, h, w)
        for (rp, rf), (pp, _) in zip(refs, pa):
            if rp is None:
                continue
            rows_a.append((pp == rp, rf >= 0.6))
            unknown_a += pp is None
        # (b) predicted damage matched to GT damage
        pr = [p for p in dmg_pred.get(i, {"pred": []})["pred"] if p["score"] >= a.damage_conf and p["cls"] in C.SYND_DAMAGE_MAP.values() and p["poly"]]
        n_b_gt += sum(1 for r in refs if r[0] is not None)
        if pr and g["gt"]:
            ious = iou_matrix(np.array([p["box"] for p in pr]), np.array([x["box"] for x in g["gt"]]))
            used = set()
            for pi in np.argsort([-p["score"] for p in pr]):
                j = int(np.argmax(np.where([k in used for k in range(ious.shape[1])], -1, ious[pi])))
                if ious[pi, j] >= 0.5 and j not in used:
                    used.add(j)
                    rp, rf = refs[j]
                    if rp is None:
                        continue
                    n_b_matched += 1
                    pp, _ = assign([pr[pi]["poly"]], ppolys, labels, pscores, h, w)[0]
                    rows_b.append((pp == rp, rf >= 0.6))
    def met(rows, prefix):
        k, n = sum(r[0] for r in rows), len(rows)
        kc = sum(r[0] for r in rows if r[1]); nc = sum(1 for r in rows if r[1])
        m = [prop_metric(f"{prefix}part_accuracy", k, n)]
        if nc:
            m.append(prop_metric(f"{prefix}part_accuracy_clear_subset", kc, nc))
        m.append({"name": f"{prefix}n_damage_instances", "value": float(n)})
        return m
    metrics = met(rows_a, "oracle_damage.") + met(rows_b, "end_to_end.")
    metrics.append(prop_metric("end_to_end.damage_matched_rate", n_b_matched, max(1, n_b_gt)))
    metrics.append(prop_metric("oracle_damage.pred_part_unknown_rate", unknown_a, max(1, len(rows_a))))
    res = [{
        "id": f"inspect.damage_to_part.{a.exp}", "name": "Damage assigned to the car part it lies on (SYNDCAR, family level)", "lane": "inspect",
        "data_kind": "synthetic", "domain": "phone photos of ONE micro-car (Citroen Ami) with digitally generated damage; no fixed-camera data",
        "dataset": "SYNDCAR (Mendeley hzpj48krdt, CC BY 4.0)", "split": "all 245 images (test only)", "n": len(ids),
        "model": f"parts: {a.parts_weights}; damage preds: {a.damage_preds}", "metrics": metrics,
        "notes": ("Reference part = GT part polygon with max overlap of the GT damage box (>=10%% of box). Accuracy = predicted part family == reference family. "
                  "(a) GT damage boxes + predicted parts; (b) predicted damage (conf>=%.2f) matched to GT at IoU>=0.5, then assigned. "
                  "Damage on SYNDCAR is synthetic; single vehicle model; see docs/lanes/inspect_notes.md." % a.damage_conf)}]
    C.write_json(res, out / "component_results_assign.json")
    print(res[0]["metrics"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parts")
    p.add_argument("--exp", required=True); p.add_argument("--weights", required=True)
    p.add_argument("--sets", default=",".join(SETS)); p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--model-name", default="yolo11m-seg"); p.add_argument("--tag", default="")
    q = sub.add_parser("assign")
    q.add_argument("--exp", required=True); q.add_argument("--parts-weights", required=True); q.add_argument("--damage-preds", required=True)
    q.add_argument("--imgsz", type=int, default=1024); q.add_argument("--part-conf", type=float, default=0.25)
    q.add_argument("--damage-conf", type=float, default=0.25)
    a = ap.parse_args()
    run_parts(a) if a.cmd == "parts" else run_assign(a)
