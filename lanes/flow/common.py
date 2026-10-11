"""Shared helpers for Lane B (flow): paths, detection AP engine with cluster bootstrap, ComponentResult writer.

Everything here is CPU/numpy so it runs on the Mac and the box. Statistics follow ``eval.stats`` (Wilson for
proportions, percentile bootstrap over the independent unit: sequence / camera / image, never over boxes).
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from eval.stats import iou_matrix, metric, prop_metric, wilson_ci  # noqa: F401  (re-exported)
from seen.paths import ARTIFACTS, DATA, PREP, RAW, RUNS, fast_dir

LANE = "flow"
# Detector label space (COCO subset we care about). Index = class id in cached detection arrays.
CLASSES = ("person", "car", "motorcycle", "bus", "truck")
VEHICLE_IDX = (1, 2, 3, 4)
PERSON_IDX = 0


def run_dir(exp: str) -> Path:
    p = RUNS / LANE / exp
    p.mkdir(parents=True, exist_ok=True)
    return p


def prep_dir(*parts: str) -> Path:
    p = PREP / LANE
    for x in parts:
        p = p / x
    p.mkdir(parents=True, exist_ok=True)
    return p


def fast_prep(*parts: str) -> Path:
    """Prepared image data lives on the NVMe (seen.paths.fast_dir, flow budget 12 GB); raw/runs/artifacts stay on the HDD."""
    p = fast_dir(LANE)
    for x in parts:
        p = p / x
    p.mkdir(parents=True, exist_ok=True)
    return p


def stage_items(items: list, sub: str) -> list:
    """One sequential pass: copy every item's image to the NVMe (``fast_prep(sub)``) in source-path order and repoint ``item.path``.
    Already-staged files are not copied again."""
    import shutil

    dst_root = fast_prep(sub)
    for it in sorted(items, key=lambda i: i.path):
        src = Path(it.path)
        if str(src).startswith(str(fast_dir(LANE))):
            continue
        # keep a short unique relative name: <parent>__<name>
        dst = dst_root / f"{src.parent.name}__{src.name}"
        if not dst.exists():
            shutil.copyfile(src, dst)
        it.path = str(dst)
    return items


def raw_dir(name: str) -> Path:
    return RAW / f"flow_{name}"


def artifact_dir(model: str) -> Path:
    p = ARTIFACTS / LANE / model
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------------------------- result writing
def component(id: str, name: str, dataset: str, n: int, metrics: list[dict], *, split: str = "test", model: str | None = None,
              domain: str = "fixed CCTV", data_kind: str = "real", notes: str | None = None, lane: str = LANE,
              confusion: dict | None = None, predictions_csv: str | None = None) -> dict[str, Any]:
    """One ``ComponentResult`` (contracts/metrics.schema.json). Metrics must be Metric dicts (eval.stats.metric)."""
    if data_kind not in ("real", "synthetic"):
        raise ValueError(data_kind)
    for m in metrics:
        if not math.isfinite(m["value"]):
            raise ValueError(f"non-finite metric {m['name']} in {id}")
    out: dict[str, Any] = {"id": id, "name": name, "lane": lane, "data_kind": data_kind, "domain": domain, "dataset": dataset,
                           "split": split, "n": int(n), "metrics": metrics}
    if model:
        out["model"] = model
    if notes:
        out["notes"] = notes
    if confusion is not None:
        out["confusion"] = confusion
    if predictions_csv:
        out["predictions_csv"] = predictions_csv
    return out


def save_results(exp: str, results: list[dict], name: str = "component_results.json") -> Path:
    p = run_dir(exp) / name
    p.write_text(json.dumps(results, indent=1))
    return p


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t0


# --------------------------------------------------------------------------------------------- AP engine
IOU_THRS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


def match_image(pred: np.ndarray, gt: np.ndarray, thrs: Sequence[float] = IOU_THRS) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Greedy score-ordered matching for one image, one class. ``pred`` (N,5)=[x1,y1,x2,y2,score], ``gt`` (M,4).

    Returns {thr: (scores sorted desc, is_tp)}. Each GT is matched at most once (best-IoU unmatched GT >= thr).
    """
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 5)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 4)
    order = np.argsort(-pred[:, 4], kind="stable")
    pred = pred[order]
    out: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    if len(pred) == 0:
        return {t: (np.zeros(0), np.zeros(0, dtype=bool)) for t in thrs}
    ious = iou_matrix(pred[:, :4], gt) if len(gt) else np.zeros((len(pred), 0))
    for t in thrs:
        tp = np.zeros(len(pred), dtype=bool)
        if len(gt):
            used = np.zeros(len(gt), dtype=bool)
            for i in range(len(pred)):
                cand = np.where(used, -1.0, ious[i])
                j = int(cand.argmax())
                if cand[j] >= t:
                    used[j] = True
                    tp[i] = True
        out[t] = (pred[:, 4], tp)
    return out


def ap_from_matches(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """All-point interpolated AP (VOC2010+ / identical to eval.stats.average_precision)."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    t = tp[order]
    ctp = np.cumsum(t)
    cfp = np.cumsum(~t)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


class APAccumulator:
    """Collect per-image matches tagged with a cluster id, then compute AP50/AP75/AP[.5:.95] + cluster bootstrap CI."""

    def __init__(self, thrs: Sequence[float] = IOU_THRS):
        self.thrs = tuple(thrs)
        self.by_cluster: dict[Any, dict[float, list]] = defaultdict(lambda: {t: [[], [], 0] for t in self.thrs})
        self.n_images = 0
        self.n_gt = 0

    def add(self, cluster: Any, pred: np.ndarray, gt: np.ndarray) -> None:
        m = match_image(pred, gt, self.thrs)
        bucket = self.by_cluster[cluster]
        for t in self.thrs:
            s, tp = m[t]
            bucket[t][0].append(s)
            bucket[t][1].append(tp)
            bucket[t][2] += len(gt)
        self.n_images += 1
        self.n_gt += len(gt)

    def _pool(self, clusters: Sequence[Any], t: float) -> float:
        ss, tps, n = [], [], 0
        for c in clusters:
            b = self.by_cluster[c][t]
            ss += b[0]
            tps += b[1]
            n += b[2]
        if n == 0:
            return float("nan")
        s = np.concatenate(ss) if ss else np.zeros(0)
        tp = np.concatenate(tps) if tps else np.zeros(0, dtype=bool)
        return ap_from_matches(s, tp, n)

    def ap(self, clusters: Sequence[Any] | None = None, t: float = 0.5) -> float:
        return self._pool(list(self.by_cluster) if clusters is None else clusters, t)

    def ap_coco(self, clusters: Sequence[Any] | None = None) -> float:
        cl = list(self.by_cluster) if clusters is None else clusters
        v = [self._pool(cl, t) for t in self.thrs]
        v = [x for x in v if math.isfinite(x)]
        return float(np.mean(v)) if v else float("nan")

    def counts_at(self, score_thr: float, t: float = 0.5, clusters: Sequence[Any] | None = None) -> tuple[int, int, int]:
        """(tp, fp, fn) pooled over clusters for predictions with score >= score_thr at IoU ``t``."""
        tp = fp = n = 0
        for c in (list(self.by_cluster) if clusters is None else clusters):
            b = self.by_cluster[c][t]
            for s, m in zip(b[0], b[1]):
                k = s >= score_thr
                tp += int((m & k).sum())
                fp += int((~m & k).sum())
            n += b[2]
        return tp, fp, n - tp

    def op_metrics(self, score_thr: float, n_boot: int = 500, seed: int = 0, t: float = 0.5) -> list[dict]:
        """precision / recall / F1 at a fixed score threshold with a cluster bootstrap CI."""
        clusters = list(self.by_cluster)
        rng = np.random.default_rng(seed)

        def prf(cl):
            tp, fp, fn = self.counts_at(score_thr, t, cl)
            p = tp / (tp + fp) if tp + fp else float("nan")
            r = tp / (tp + fn) if tp + fn else float("nan")
            f = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
            return p, r, f

        base = prf(clusters)
        boots = [prf([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))]) for _ in range(n_boot)] if len(clusters) >= 2 and n_boot else []
        out = []
        for k, nm in enumerate(("precision", "recall", "F1")):
            b = np.array([x[k] for x in boots if math.isfinite(x[k])])
            lo, hi = (float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))) if len(b) > 20 else (None, None)
            if math.isfinite(base[k]):
                out.append(metric(f"{nm}@conf{score_thr:g}", base[k], lo, hi, "fraction"))
        return out

    def metrics(self, prefix: str = "", n_boot: int = 500, seed: int = 0) -> list[dict]:
        """AP50 / AP75 / AP[.5:.95] with a percentile bootstrap over clusters (sequences / cameras)."""
        clusters = list(self.by_cluster)
        rng = np.random.default_rng(seed)
        fns: dict[str, Callable] = {
            "AP50": lambda cl: self._pool(cl, 0.5),
            "AP75": lambda cl: self._pool(cl, 0.75),
            "AP50:95": lambda cl: self.ap_coco(cl),
        }
        out = []
        boots: dict[str, list[float]] = {k: [] for k in fns}
        if len(clusters) >= 2 and n_boot:
            for _ in range(n_boot):
                idx = rng.integers(0, len(clusters), len(clusters))
                cl = [clusters[i] for i in idx]
                for k, f in fns.items():
                    boots[k].append(f(cl))
        for k, f in fns.items():
            v = f(clusters)
            b = np.array([x for x in boots[k] if math.isfinite(x)])
            lo, hi = (float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))) if len(b) > 20 else (None, None)
            out.append(metric(prefix + k, v, lo, hi, "fraction"))
        return out


# --------------------------------------------------------------------------------------------- misc
def load_json(p: str | Path):
    return json.loads(Path(p).read_text())


def write_lines(p: str | Path, lines: Sequence[str]) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text("\n".join(lines) + "\n")


def env_info() -> dict[str, str]:
    info = {"host": os.uname().nodename}
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return info
