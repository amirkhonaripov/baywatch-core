"""Metric helpers for the ident lane. Every reported number goes through here (Wilson CIs from eval.stats).

ComponentResult dicts follow contracts/metrics.schema.json#/$defs/ComponentResult.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from eval.stats import boot_metric, detection_ap, detection_recall, iou_matrix, metric, prop_metric, wilson_ci


def norm(t: str | None) -> str:
    return re.sub(r"[^0-9A-Z]", "", (t or "").upper())


def lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_acc(pred: str, gt: str) -> float:
    """1 - Levenshtein/len(gt), clipped to [0, 1]."""
    g = norm(gt)
    if not g:
        return 0.0
    return max(0.0, 1.0 - lev(norm(pred), g) / len(g))


def ocr_metrics(preds: Sequence[str], gts: Sequence[str], extra: dict | None = None) -> list[dict]:
    n = len(gts)
    ok = [norm(p) == norm(g) for p, g in zip(preds, gts)]
    ca = np.array([char_acc(p, g) for p, g in zip(preds, gts)])
    m = [prop_metric("exact_match", int(sum(ok)), n), boot_metric("char_accuracy", ca, np.mean, unit="fraction")]
    # off-by-<=1 character reads (useful to show how much constrained decoding / voting can still fix)
    near = sum(1 for p, g in zip(preds, gts) if lev(norm(p), norm(g)) <= 1)
    m.append(prop_metric("within_1_char", near, n))
    if extra:
        for k, v in extra.items():
            m.append(metric(k, v))
    return m


def component(id: str, name: str, data_kind: str, domain: str, dataset: str, split: str, n: int, model: str,
              metrics: list[dict], notes: str = "", predictions_csv: str | None = None, confusion: dict | None = None) -> dict:
    d = {"id": id, "name": name, "lane": "ident", "data_kind": data_kind, "domain": domain, "dataset": dataset,
         "split": split, "n": int(n), "model": model, "metrics": metrics}
    if notes:
        d["notes"] = notes
    if predictions_csv:
        d["predictions_csv"] = predictions_csv
    if confusion:
        d["confusion"] = confusion
    return d


def write_results(path: str | Path, comps: list[dict], merge: bool = True) -> None:
    """Write/merge ComponentResult list (merge by id)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cur: dict[str, dict] = {}
    if merge and p.exists():
        for c in json.loads(p.read_text()):
            cur[c["id"]] = c
    for c in comps:
        cur[c["id"]] = c
    p.write_text(json.dumps(list(cur.values()), indent=1))


def write_preds_csv(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------------------------------- detection
def det_images(gts: list[np.ndarray], preds: list[list[tuple[float, np.ndarray]]]) -> list[dict]:
    """Build eval.stats image records: gt boxes xyxy (pixels or normalised, consistent), preds (score, xyxy)."""
    ims = []
    for g, p in zip(gts, preds):
        ims.append({"gt": [("plate", *map(float, b)) for b in g], "pred": [("plate", float(s), *map(float, b)) for s, b in p]})
    return ims


def det_metrics(images: list[dict], score_thr: float = 0.25, iou: float = 0.5) -> list[dict]:
    k, n = detection_recall(images, "plate", iou, score_thr)
    # precision: predicted boxes >= thr that match a gt at IoU
    tp = fp = 0
    for im in images:
        g = np.array([b[1:5] for b in im["gt"]], dtype=float).reshape(-1, 4)
        p = [b for b in im["pred"] if b[1] >= score_thr]
        used = np.zeros(len(g), bool)
        for b in sorted(p, key=lambda x: -x[1]):
            if len(g):
                ious = iou_matrix(np.array([b[2:6]]), g)[0]
                ious[used] = -1
                j = int(np.argmax(ious))
                if ious[j] >= iou:
                    used[j] = True
                    tp += 1
                    continue
            fp += 1
    m = [prop_metric(f"recall@IoU{iou:g}(conf>={score_thr:g})", k, n)]
    if tp + fp:
        m.append(prop_metric(f"precision@IoU{iou:g}(conf>={score_thr:g})", tp, tp + fp))
    m.append(detection_ap(images, ["plate"], iou, n_boot=500, name=f"AP@IoU{iou:g}"))
    return m


def wilson(k: int, n: int):
    return wilson_ci(k, n)
