"""Run a detector over a dataset and cache raw detections (score >= conf) so metrics can be recomputed without a GPU.

    scripts/box.sh --gpu 1 -- python -m lanes.flow.run_detect --model rfdetr-base --dataset detrac_test_s5
    datasets: detrac_test_s5 | detrac_val_s10 | ndispark_train | ndispark_validation | ndispark_test | cocoval_person | miotcd_eval

Output: runs/flow/det/<dataset>/<model>.pkl = {"dets": {item_id: float32 (N,6)}, "meta": {...timings...}}.
"""
from __future__ import annotations

import argparse
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from lanes.flow import data as D
from lanes.flow.common import env_info, run_dir, stage_items
from lanes.flow.detectors import get_detector


def get_items(name: str) -> list[D.Item]:
    if name.startswith("detrac_test_s"):
        return D.detrac_items("test", int(name.split("_s")[1]))
    if name.startswith("detrac_train_s"):
        return D.detrac_items("train", int(name.split("_s")[1]))
    if name == "ndispark_train":
        return D.ndispark_items("train")
    if name == "ndispark_validation":
        return D.ndispark_items("validation")
    if name == "ndispark_test":
        return D.ndispark_test_counts()
    if name == "cocoval_person":
        from lanes.flow.data_extra import cocoval_items

        return stage_items(cocoval_items(), "coco")
    if name in ("pklot_occ", "pklot_cal"):
        from lanes.flow.data_extra import pklot_items

        return stage_items(pklot_items(150, 0, "test") if name == "pklot_occ" else pklot_items(100, 1, "cal"), "pklot")
    if name in ("cnrext_occ", "cnrext_cal"):
        from lanes.flow.data_extra import cnrext_items

        return stage_items(cnrext_items(150, 0, "test") if name == "cnrext_occ" else cnrext_items(100, 1, "cal"), "cnrext")
    if name == "wider_val":
        from lanes.flow.data_extra import wider_items

        return stage_items(wider_items(), "wider")
    if name == "miotcd_eval":
        from lanes.flow.data_extra import miotcd_items

        return miotcd_items()
    raise ValueError(name)


def cache_path(dataset: str, model: str):
    safe = model.replace("/", "_").replace("@", "_at_").replace("#", "_r")
    return run_dir(f"det/{dataset}") / f"{safe}.pkl"


def run(model: str, dataset: str, conf: float = 0.05, batch: int = 8, limit: int | None = None, max_side: int | None = None,
        device: str = "cuda:0", det=None, skip_existing: bool = True) -> dict:
    """Run one model on one dataset. Pass ``det`` (a loaded detector) to reuse a model across datasets in one process."""
    p = cache_path(dataset, model)
    if skip_existing and p.exists() and not limit:
        print("exists, skipping", p, flush=True)
        return pickle.load(open(p, "rb"))["meta"]
    items = get_items(dataset)
    if limit:
        items = items[:limit]
    if det is None:
        det = get_detector(model)
        det.load(device)
    dets: dict[str, np.ndarray] = {}
    lat: list[float] = []
    ex = ThreadPoolExecutor(6)

    def ld(it):
        return D.load_rgb(it.path, max_side)

    det.infer([ld(items[0])], conf)  # warm-up
    t_all = time.perf_counter()
    batches = [items[i:i + batch] for i in range(0, len(items), batch)]
    fut = ex.submit(lambda b: [ld(it) for it in b], batches[0]) if batches else None
    for bi, b in enumerate(batches):
        imgs = fut.result()
        if bi + 1 < len(batches):
            fut = ex.submit(lambda bb: [ld(it) for it in bb], batches[bi + 1])
        t0 = time.perf_counter()
        out = det.infer(imgs, conf)
        lat.append((time.perf_counter() - t0) * 1000 / len(b))
        for it, d, im in zip(b, out, imgs):
            h0, w0 = it.size[1] or im.shape[0], it.size[0] or im.shape[1]
            if max_side and im.shape[:2] != (h0, w0):  # map back to original resolution
                d = d.copy()
                d[:, [0, 2]] *= w0 / im.shape[1]
                d[:, [1, 3]] *= h0 / im.shape[0]
            dets[it.id] = d.astype(np.float32)
        if bi % 50 == 0:
            print(f"{model} {dataset} {bi * batch}/{len(items)}", flush=True)
    meta = {"model": model, "dataset": dataset, "conf": conf, "n": len(items), "wall_s": time.perf_counter() - t_all,
            "ms_per_img_infer": float(np.mean(lat)), "batch": batch, "env": env_info(), "max_side": max_side}
    with open(p, "wb") as f:
        pickle.dump({"dets": dets, "meta": meta}, f)
    print("saved", p, meta, flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "--models", dest="models", required=True, help="comma-separated detector specs")
    ap.add_argument("--dataset", "--datasets", dest="datasets", required=True, help="comma-separated dataset names")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-side", type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for m in a.models.split(","):
        det = None
        for d in a.datasets.split(","):
            if not a.force and not a.limit and cache_path(d, m).exists():
                print("exists, skipping", cache_path(d, m), flush=True)
                continue
            if det is None:
                det = get_detector(m)
                det.load(a.device)
            run(m, d, a.conf, a.batch, a.limit, a.max_side, a.device, det=det, skip_existing=not a.force)
        del det


if __name__ == "__main__":
    main()
