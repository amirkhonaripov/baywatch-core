"""Dataset adapters for the flow lane: UA-DETRAC (HF mirror), NDISPark (Zenodo), MIO-TCD, COCO val (person check).

All adapters return ``Item`` lists with pixel-space xyxy ground truth. Prepared (extracted) frames live in
``PREP/flow/<dataset>/``; raw archives in ``RAW/flow_<dataset>/``.

Run as a script to prepare data on the box::

    python -m lanes.flow.data detrac --split test --stride 5
    python -m lanes.flow.data detrac --split train --stride 10
    python -m lanes.flow.data ndispark
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lanes.flow.common import PREP, RAW, fast_prep, prep_dir, raw_dir


@dataclass
class Item:
    id: str  # "<dataset>:<relpath>" (matches eval/manifests ids)
    path: str  # absolute path of the image
    cluster: str  # independent unit for bootstrap: sequence / camera
    cond: str  # condition label (weather, day/night)
    gt: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))  # (N,4) xyxy px
    gt_cls: list[str] = field(default_factory=list)  # per-GT class name when the dataset has it
    ignore: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))  # (M,4) ignored regions
    size: tuple[int, int] = (0, 0)  # (w, h)
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------------- UA-DETRAC
DETRAC_ZIPS = {"test": "ua_detrac_test_set.zip", "train": "ua_detrac_training_set.zip"}


def detrac_meta() -> dict:
    return json.loads((raw_dir("detrac") / "sequence_meta.json").read_text())


def detrac_zip(split: str) -> Path:
    return raw_dir("detrac") / "hf" / DETRAC_ZIPS[split]


def _frame_no(name: str) -> int:
    return int(re.search(r"img(\d+)", name).group(1))


def prepare_detrac(split: str, stride: int, offset: int = 1) -> Path:
    """One sequential pass over the HF zip: write frames ``offset, offset+stride, ...`` of every sequence to the NVMe + ``index_s<stride>.json`` with GT.

    Frames come from the HF mirror ``abhineet123/ua_detrac`` (960x540, 25 fps originally). GT: every 'vehicle' box (+target_id for
    tracking) and 'ignored' regions; weather label per sequence from the original DETRAC XML. ``annotations.zip`` members are never read.
    """
    meta = detrac_meta()
    out = fast_prep("detrac", split)
    idx_path = out / f"index_s{stride}.json"
    # 32 MB read buffer: forward seeks over skipped frames stay inside the buffer, so the HDD sees large sequential reads
    z = zipfile.ZipFile(open(detrac_zip(split), "rb", buffering=32 * 1024 * 1024))
    index: dict = {}
    cur = None  # (seq, dir, frames dict)

    def finish():
        if cur is None:
            return
        seq, _, frames = cur
        index[seq] = {"weather": meta[seq]["weather"], "camera_state": meta[seq]["attr"], "frames": {str(fn): d for fn, d in sorted(frames.items())}}
        print(split, seq, meta[seq]["weather"], len(frames), flush=True)

    for info in z.infolist():
        n = info.filename
        if info.is_dir() or not n.startswith("detrac_"):
            continue
        sd, name = n.split("/", 1)
        if cur is None or cur[1] != sd:
            finish()
            cur = (sd.split("_", 2)[2], sd, {})
            (out / cur[0]).mkdir(parents=True, exist_ok=True)
        seq, _, frames = cur
        if name == "annotations.csv":
            for r in csv.DictReader(io.TextIOWrapper(io.BytesIO(z.read(n)), "utf-8")):
                fn = _frame_no(r["filename"])
                if (fn - offset) % stride:
                    continue
                d = frames.setdefault(fn, {"file": r["filename"], "w": int(r["width"]), "h": int(r["height"]), "gt": [], "ignore": []})
                box = [float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"])]
                if r["class"] == "ignored":
                    d["ignore"].append(box)
                else:
                    tid = int(float(r["target_id"])) if r.get("target_id") not in (None, "") else -1
                    d["gt"].append(box + [tid])
        elif name.endswith(".jpg"):
            fn = _frame_no(name)
            if (fn - offset) % stride == 0:
                frames.setdefault(fn, {"file": name, "w": 960, "h": 540, "gt": [], "ignore": []})
                dst = out / seq / name
                if not dst.exists():
                    dst.write_bytes(z.read(n))
    finish()
    idx_path.write_text(json.dumps(index))
    return idx_path


def detrac_items(split: str = "test", stride: int = 5, seqs: list[str] | None = None) -> list[Item]:
    root = fast_prep("detrac", split)
    idx = json.loads((root / f"index_s{stride}.json").read_text())
    items = []
    for seq, s in idx.items():
        if seqs and seq not in seqs:
            continue
        for fn, d in s["frames"].items():
            g = np.asarray(d["gt"], dtype=np.float32).reshape(-1, 5)
            items.append(Item(id=f"detrac:{split}/{seq}/{d['file']}", path=str(root / seq / d["file"]),
                              cluster=seq, cond=s["weather"], gt=g[:, :4], ignore=np.asarray(d["ignore"], dtype=np.float32).reshape(-1, 4),
                              size=(d["w"], d["h"]), extra={"frame": int(fn), "track_id": g[:, 4].astype(int).tolist()}))
    return items


# --------------------------------------------------------------------------------------------- NDISPark
def prepare_ndispark() -> Path:
    """Extract the Zenodo zip on the HDD (raw), then copy the 147 MB to the NVMe in one pass (``fast/ndispark``)."""
    import shutil

    out = raw_dir("ndispark") / "ex"
    if not out.exists():
        zipfile.ZipFile(raw_dir("ndispark") / "ndis_park.zip").extractall(out)
    fast = fast_prep("ndispark")
    if not (fast / "train").exists():
        shutil.copytree(out, fast, dirs_exist_ok=True)
    return fast


def _nd_root() -> Path:
    f = fast_prep("ndispark")
    return f if (f / "train").exists() else raw_dir("ndispark") / "ex"


def ndispark_items(split: str) -> list[Item]:
    """'train' (day, boxes), 'validation' (night, boxes). 'test' (night) has counts only: see ndispark_test_counts()."""
    root = _nd_root() / split
    js = json.loads((root / ("train_coco_annotations.json" if split == "train" else "val_coco_annotations.json")).read_text())
    boxes: dict[int, list] = {}
    for a in js["annotations"]:
        if a["category_id"] != 3:  # 'car' is the only populated category
            continue
        x, y, w, h = a["bbox"]
        boxes.setdefault(a["image_id"], []).append([x, y, x + w, y + h])
    cond = "day" if split == "train" else "night"
    items = []
    for im in js["images"]:
        cam = Path(im["file_name"]).name.split("_")[0]
        items.append(Item(id=f"ndispark:{split}/{im['file_name']}", path=str(root / "imgs" / Path(im["file_name"]).name), cluster=f"cam{cam}",
                          cond=cond, gt=np.asarray(boxes.get(im["id"], []), dtype=np.float32).reshape(-1, 4),
                          size=(im["width"], im["height"])))
    return items


def ndispark_test_counts() -> list[Item]:
    root = _nd_root() / "test"
    items = []
    for r in csv.DictReader(open(root / "ground_truth_test_counting.csv")):
        if r["numVehicles"] in ("", None):
            continue
        name = r["imgName"] + ".jpg"
        items.append(Item(id=f"ndispark:test/{name}", path=str(root / "imgs" / name), cluster=f"cam{name.split('_')[0]}", cond="night",
                          extra={"count": int(float(r["numVehicles"]))}))
    return items


def mean_luminance(path: str) -> float:
    import cv2

    im = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_4)
    return float(im.mean())


# --------------------------------------------------------------------------------------------- image loading
def load_rgb(path: str, max_side: int | None = None) -> np.ndarray:
    import cv2

    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if max_side and max(im.shape[:2]) > max_side:
        s = max_side / max(im.shape[:2])
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["detrac", "ndispark"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--stride", type=int, default=5)
    a = ap.parse_args()
    if a.what == "detrac":
        print(prepare_detrac(a.split, a.stride))
    else:
        print(prepare_ndispark())


if __name__ == "__main__":
    main()
