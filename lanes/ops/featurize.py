"""GPU feature extraction for the ops frame table (run on GPU1).

  python -m lanes.ops.featurize detect   # RF-DETR (COCO) vehicles+persons -> FAST/ops/feat/dets.parquet
  python -m lanes.ops.featurize parts    # trained part detector (wheel/hood) -> FAST/ops/feat/parts.parquet
  python -m lanes.ops.featurize embed --backbone dinov2l|siglip2   -> FAST/ops/feat/emb_<backbone>.npy
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from lanes.ops.backbones import BACKBONES, load_backbone
from lanes.ops.geometry import VEH, crop_box, primary_vehicle
from lanes.ops.prep import FAST_OPS, TABLE, fpath
from seen.paths import RUNS

FEAT = FAST_OPS / "feat"


def _table() -> pd.DataFrame:
    return pd.read_parquet(TABLE)


def detect(batch: int = 16, threshold: float = 0.3) -> None:
    from rfdetr import RFDETRBase
    from rfdetr.assets.coco_classes import COCO_CLASSES

    keep = {"car": "car", "truck": "truck", "bus": "bus", "motorcycle": "motorcycle", "person": "person"}
    df = _table()
    FEAT.mkdir(parents=True, exist_ok=True)
    out_p = FEAT / "dets.parquet"
    done = set()
    old = None
    if out_p.exists():
        old = pd.read_parquet(out_p)
        done = set(pd.read_parquet(FEAT / "dets_done.parquet")["frame_id"])
    todo = df[~df["frame_id"].isin(done)]
    print("detect todo", len(todo), "of", len(df), flush=True)
    model = RFDETRBase()
    rows, ids = [], []
    t0 = time.time()
    paths = todo["path"].tolist()
    fids = todo["frame_id"].tolist()
    for s in range(0, len(paths), batch):
        imgs = [np.asarray(Image.open(fpath(p)).convert("RGB")) for p in paths[s:s + batch]]
        res = model.predict(imgs, threshold=threshold)
        if not isinstance(res, list):
            res = [res]
        for fid, img, det in zip(fids[s:s + batch], imgs, res):
            h, w = img.shape[:2]
            for (x1, y1, x2, y2), c, k in zip(det.xyxy, det.confidence, det.class_id):
                name = keep.get(COCO_CLASSES.get(int(k), ""))
                if name:
                    rows.append((fid, name, float(c), x1 / w, y1 / h, x2 / w, y2 / h))
            ids.append(fid)
        if (s // batch) % 200 == 0:
            print(s, len(paths), f"{time.time() - t0:.0f}s", flush=True)
    new = pd.DataFrame(rows, columns=["frame_id", "cls", "conf", "x1", "y1", "x2", "y2"])
    allr = pd.concat([old, new]) if old is not None else new
    allr.to_parquet(out_p)
    pd.DataFrame({"frame_id": sorted(done | set(ids))}).to_parquet(FEAT / "dets_done.parquet")
    print("DETECT_DONE", len(allr), "boxes", f"{time.time() - t0:.0f}s")


def dets_by_frame(path=None) -> dict[str, list[tuple[str, float, tuple]]]:
    d = pd.read_parquet(path or FEAT / "dets.parquet")
    out: dict[str, list] = {}
    for r in d.itertuples(index=False):
        out.setdefault(r.frame_id, []).append((r.cls, float(r.conf), (r.x1, r.y1, r.x2, r.y2)))
    return out


def crop_for(img: Image.Image, dets: list, roi=None) -> tuple[Image.Image, tuple]:
    w, h = img.size
    pv = primary_vehicle([(c, b) for k, c, b in dets if k in VEH], roi)
    if pv is None:
        box = roi if roi is not None else (0.0, 0.0, 1.0, 1.0)
        box = crop_box(box, 0.0, w / h) if roi is not None else box
    else:
        box = crop_box(pv[1], 0.25, w / h)
    x1, y1, x2, y2 = box
    return img.crop((int(x1 * w), int(y1 * h), max(int(x2 * w), int(x1 * w) + 2), max(int(y2 * h), int(y1 * h) + 2))), box


class _DS(torch.utils.data.Dataset):
    def __init__(self, paths, dets, fids, prep):
        self.paths, self.dets, self.fids, self.prep = paths, dets, fids, prep

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(fpath(self.paths[i])).convert("RGB")
        crop, _ = crop_for(img, self.dets.get(self.fids[i], []))
        return self.prep(crop)


def embed(backbone: str, batch: int = 64, workers: int = 8) -> None:
    df = _table()
    dets = dets_by_frame()
    bb = load_backbone(backbone, "cuda:0")
    ds = _DS(df["path"].tolist(), dets, df["frame_id"].tolist(), bb.preprocess)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=True)
    feats = []
    t0 = time.time()
    with torch.inference_mode():
        for i, x in enumerate(dl):
            feats.append(bb.forward(x.to("cuda:0", non_blocking=True)).float().cpu().numpy().astype(np.float16))
            if i % 100 == 0:
                print(i * batch, len(ds), f"{time.time() - t0:.0f}s", flush=True)
    F = np.concatenate(feats)
    np.save(FEAT / f"emb_{backbone}.npy", F)
    df[["frame_id"]].to_parquet(FEAT / f"emb_{backbone}_ids.parquet")
    print("EMBED_DONE", backbone, F.shape, f"{time.time() - t0:.0f}s")


def parts(batch: int = 32, conf: float = 0.2) -> None:
    """Run the trained part detectors (wheel from carparts, hood_open/closed from car-hood-open) on full frames."""
    from ultralytics import YOLO

    df = _table()
    specs = [("parts", RUNS / "ops" / "parts_yolo" / "weights" / "best.pt", {"wheel"}),
             ("hood", RUNS / "ops" / "hood_yolo" / "weights" / "best.pt", {"hood_open", "hood_closed"})]
    rows = []
    t0 = time.time()
    for tag, w, keep in specs:
        m = YOLO(str(w))
        names = m.names
        paths = [str(fpath(p)) for p in df["path"]]
        fids = df["frame_id"].tolist()
        for s in range(0, len(paths), batch):
            res = m.predict(paths[s:s + batch], conf=conf, imgsz=640, verbose=False, half=True)
            for fid, r in zip(fids[s:s + batch], res):
                h, w_ = r.orig_shape
                for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                    n = names[int(k)]
                    if n in keep:
                        rows.append((fid, n, float(c), x1 / w_, y1 / h, x2 / w_, y2 / h))
            if (s // batch) % 300 == 0:
                print(tag, s, len(paths), f"{time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows, columns=["frame_id", "cls", "conf", "x1", "y1", "x2", "y2"]).to_parquet(FEAT / "parts.parquet")
    print("PARTS_DONE", len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["detect", "embed", "parts"])
    ap.add_argument("--backbone", default="dinov2l", choices=list(BACKBONES))
    a = ap.parse_args()
    if a.cmd == "detect":
        detect()
    elif a.cmd == "embed":
        embed(a.backbone)
    else:
        parts()
