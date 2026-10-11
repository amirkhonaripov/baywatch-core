"""DOT date-code reading (lane D, task 3b): characters -> string -> WWYY with a week/year validity check.

The date code of a tyre DOT is the last four digits of the DOT string: WW = production week (01-53), YY = year (tyres since 2000).
Validity: 1 <= WW <= 53 and 2000 <= 2000+YY <= current year (+0).  A reading that fails the check is returned with
valid=False and never reported as a tyre age.

Evaluation sets: TyreAI `ocr-yphpv` character boxes (CC BY label), re-split by pHash cluster (lanes/inspect/prep_dot.py).
Ground-truth string = TyreAI character labels in reading order. GT date code = last 4 chars when they are digits and valid.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

import numpy as np

CHAR_MAP = {"slash": "/", "hyphen": "-"}
NOW_YEAR = dt.date.today().year


def to_char(name: str) -> str | None:
    if name == "Boundary":
        return None
    return CHAR_MAP.get(name, name)


def reading_order(centers: np.ndarray) -> np.ndarray:
    """Indices of characters in reading order: along the principal axis of the centres, oriented left->right (top->bottom for
    vertical text). Handles rotated text lines (sidewall text curves)."""
    n = len(centers)
    if n <= 1:
        return np.arange(n)
    c = centers - centers.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    ax = vt[0]
    if abs(ax[0]) >= abs(ax[1]):
        if ax[0] < 0:
            ax = -ax
    elif ax[1] < 0:
        ax = -ax
    return np.argsort(c @ ax)


def assemble(boxes_xyxy: np.ndarray, names: list[str]) -> str:
    keep = [i for i, n in enumerate(names) if to_char(n) is not None]
    if not keep:
        return ""
    b = boxes_xyxy[keep]
    centers = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
    order = reading_order(centers)
    return "".join(to_char(names[keep[i]]) for i in order)


def validate_wwyy(code: str | None, now_year: int = NOW_YEAR) -> dict:
    """{valid, week, year, age_years, reason}"""
    if not code or not re.fullmatch(r"\d{4}", code):
        return {"valid": False, "reason": "not 4 digits"}
    ww, yy = int(code[:2]), int(code[2:])
    if not 1 <= ww <= 53:
        return {"valid": False, "reason": f"week {ww} out of 1..53"}
    year = 2000 + yy
    if year > now_year:
        return {"valid": False, "reason": f"year {year} in the future"}
    return {"valid": True, "week": ww, "year": year, "age_years": round(now_year - year - (dt.date.today().isocalendar().week < ww) * 0.0, 1),
            "reason": ""}


def date_code(s: str) -> str | None:
    """Last four characters if they are all digits, else None."""
    t = s.strip()
    return t[-4:] if len(t) >= 4 and t[-4:].isdigit() else None


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def read_yolo_labels(label_path: Path, names: list[str], w: int = 1, h: int = 1):
    boxes, nm = [], []
    for ln in label_path.read_text().splitlines():
        t = ln.split()
        if len(t) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in t[1:5])
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        nm.append(names[int(t[0])])
    return np.array(boxes).reshape(-1, 4), nm


def region_main(a) -> None:
    """DOT-region detector on the cross-source test set (dot-code-6shg0): box AP50 and recall at fixed FP/image."""
    from lanes.inspect import common as C
    from lanes.inspect.evaldet import Evaluator
    from lanes.inspect.prep_dot import OUT
    from ultralytics import YOLO
    model = YOLO(a.weights)
    img_dir, lab_dir = OUT / "region" / "images" / "test", OUT / "region" / "labels" / "test"
    paths = sorted(img_dir.glob("*.*"))
    recs = []
    for p, res in zip(paths, C.predict_chunks(model, [str(p) for p in paths], 8, imgsz=a.imgsz, conf=0.001, iou=0.6, max_det=20, device=0)):
        h, w = res.orig_shape
        gb, _ = read_yolo_labels(lab_dir / (p.stem + ".txt"), ["dot_code"], w, h) if (lab_dir / (p.stem + ".txt")).exists() else (np.zeros((0, 4)), [])
        pr = []
        if res.boxes is not None and len(res.boxes):
            for b, sc in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                pr.append({"cls": "dot_code", "score": float(sc), "box": b.tolist(), "poly": None})
        recs.append({"id": p.name, "w": w, "h": h, "gt": [{"cls": "dot_code", "box": b.tolist(), "poly": None} for b in gb], "pred": pr})
    ev = Evaluator(recs, ["dot_code"], "box", 0.5)
    metrics = ev.summary("", fppis=(0.1, 0.5, 1.0), n_boot=500)
    res_json = [{"id": f"inspect.dot_region.{a.exp}", "name": "DOT-code region detection on an independent source", "lane": "inspect", "data_kind": "real",
                 "domain": "phone photos of tyre sidewalls (Roboflow uploader dot-code-6shg0, 512x512 stretched)", "dataset": "Roboflow subhams-workspace-l47ut/dot-code-6shg0 (CC BY label)",
                 "split": "all 97 images (cross-source; trained on tire-dot-finder + tyre-sidewall-text-label)", "n": len(recs),
                 "model": f"YOLO11s one-class detector ({a.exp}) @ {a.imgsz}", "metrics": metrics + [{"name": "n_gt_instances", "value": float(ev.n_gt("*"))}],
                 "notes": "Box IoU 0.5. Training sources are tiny (about 150 images): expect high variance; see the CIs."}]
    C.write_json(res_json, C.RUNS_I / a.exp / "component_results_dot_region.json")
    print([(m["name"], round(m["value"], 3)) for m in metrics])


def cluster_prop(name: str, flags, clusters, n_boot: int = 1000):
    """Proportion with a CLUSTER bootstrap CI (resample near-duplicate clusters, ratio estimator sum(k)/sum(n))."""
    from eval.stats import metric
    flags = np.asarray(flags, dtype=float)
    cl = np.asarray(clusters)
    ks = [flags[cl == u].sum() for u in np.unique(cl)]
    ns = [float((cl == u).sum()) for u in np.unique(cl)]
    ks, ns = np.array(ks), np.array(ns)
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(ks), len(ks))
        vals.append(ks[i].sum() / ns[i].sum())
    return metric(name, float(ks.sum() / ns.sum()), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))


def main() -> None:  # evaluation of a character-detection model on the held-out TyreAI split
    from eval.stats import metric, prop_metric, bootstrap_ci
    from lanes.inspect import common as C
    from lanes.inspect.prep_dot import OUT
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["chars", "region"])
    ap.add_argument("--exp", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    if a.kind == "region":
        return region_main(a)
    from ultralytics import YOLO
    model = YOLO(a.weights)
    names = yaml.safe_load((OUT / "chars" / "data.yaml").read_text())["names"]
    names = [str(names[i]) for i in sorted(names)] if isinstance(names, dict) else [str(n) for n in names]
    img_dir, lab_dir = OUT / "chars" / "images" / "test", OUT / "chars" / "labels" / "test"
    paths = sorted(img_dir.glob("*.jpg"))
    rows = []
    for p, res in zip(paths, C.predict_chunks(model, [str(p) for p in paths], 8, imgsz=a.imgsz, conf=a.conf, iou=0.5, agnostic_nms=True, max_det=60, device=0)):
        gb, gn = read_yolo_labels(lab_dir / (p.stem + ".txt"), names, res.orig_shape[1], res.orig_shape[0])
        gt = assemble(gb, gn)
        if res.boxes is not None and len(res.boxes):
            pb = res.boxes.xyxy.cpu().numpy()
            pn = [names[int(c)] for c in res.boxes.cls.cpu().numpy()]  # names from data.yaml by index, never from the weights
        else:
            pb, pn = np.zeros((0, 4)), []
        pred = assemble(pb, pn)
        rows.append({"file": p.name, "gt": gt, "pred": pred})
    n = len(rows)
    exact = np.array([r["gt"] == r["pred"] for r in rows])
    cer_num = sum(edit_distance(r["gt"], r["pred"]) for r in rows)
    cer_den = max(1, sum(len(r["gt"]) for r in rows))
    from lanes.inspect import common as C0
    clusters = C0.group_near_dups([str(p) for p in paths])  # TyreAI test images include near-duplicate copies: cluster bootstrap
    metrics = [cluster_prop("string_exact_match", exact, clusters),
               metric("char_error_rate", cer_num / cer_den, unit="fraction"),
               {"name": "n_near_duplicate_clusters", "value": float(len(set(clusters)))}]
    # date-code subset: GT ends in 4 digits forming a valid WWYY, and the string is digits only (a DOT-style serial)
    sub = [r for r in rows if r["gt"].isdigit() and len(r["gt"]) >= 6 and validate_wwyy(date_code(r["gt"]))["valid"]]
    if sub:
        ok = [date_code(r["pred"]) == date_code(r["gt"]) for r in sub]
        cl_of = dict(zip([p.name for p in paths], clusters))
        sub_cl = [cl_of[r["file"]] for r in sub]
        metrics.append(cluster_prop("date_code_exact_match[digit-serial subset]", ok, sub_cl))
        wrong = [r for r, o in zip(sub, ok) if not o]
        right = [r for r, o in zip(sub, ok) if o]
        flagged = sum(1 for r in wrong if not validate_wwyy(date_code(r["pred"]))["valid"])
        if wrong:
            metrics.append(prop_metric("validator_rejects_wrong_readings", flagged, len(wrong)))
        metrics.append(prop_metric("validator_accepts_correct_readings", sum(validate_wwyy(date_code(r["pred"]))["valid"] for r in right), max(1, len(right))))
        # accepted-reading precision (what the user would see: only valid-looking readings are shown)
        shown = [(date_code(r["pred"]) == date_code(r["gt"])) for r in sub if validate_wwyy(date_code(r["pred"]))["valid"]]
        if shown:
            metrics.append(prop_metric("date_code_precision_when_valid", int(sum(shown)), len(shown)))
        metrics.append({"name": "n_digit_serial_subset", "value": float(len(sub))})
    res_json = [{"id": f"inspect.dot_chars.{a.exp}", "name": "Tyre sidewall character reading (TyreAI) with week/year validity check", "lane": "inspect",
                 "data_kind": "real", "domain": "phone close-ups of embossed tyre text; rectified/cropped by the dataset (no scene search)",
                 "dataset": "Roboflow Universe tyreai/ocr-yphpv v3 (CC BY label)", "split": "15 % pHash-cluster hold-out of train+valid", "n": n,
                 "model": f"YOLO11s character detector ({a.exp}) @ {a.imgsz}", "metrics": metrics,
                 "notes": "GT string = TyreAI character boxes in reading order (principal-axis ordering, same function as for predictions). "
                          "Date-code subset = images whose GT string is all digits, >= 6 chars, ending in a valid WWYY. Not measured: the stage that finds "
                          "the DOT region in a full tyre photo end-to-end, night/rain, tilted close-ups beyond the dataset's own variation."}]
    out = C.ensure(C.RUNS_I / a.exp)
    C.write_json(res_json, out / "component_results_dot_chars.json")
    C.write_json(rows, out / "preds_dot_chars.json")
    print(metrics)


if __name__ == "__main__":
    main()
