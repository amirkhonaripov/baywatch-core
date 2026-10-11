"""Build the RF-DETR fine-tuning set (Roboflow-style COCO folders) from UA-DETRAC train + NDISPark train (+ a COCO person/vehicle subset).

    python -m lanes.flow.prep_finetune --detrac-stride 10 --coco-n 2500

Output ``PREP/flow/ft/{train,valid}/`` with ``_annotations.coco.json`` and ``PREP/flow/ft/classes.json`` (our class order = CLASSES).
Held out and NEVER used here: UA-DETRAC test (40 official sequences), NDISPark validation + test (night), COCO images used for the person-AP check
(the first 2000 of the seeded shuffle in ``data_extra.cocoval_items``), everything in ``eval/manifests``.
Choices: DETRAC ignored regions are painted grey in the training frames (unlabelled far vehicles would otherwise teach "background");
classes: car+van -> car, bus -> bus, others -> truck. Validation = 6 whole train sequences (2 night, 1 rainy, 2 sunny, 1 cloudy) at stride 25.
The COCO subset (non-crowd person/car/motorcycle/bus/truck boxes) keeps the person class alive: without positives the person logit collapses.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import cv2
import numpy as np

from lanes.flow import data as D
from lanes.flow.common import CLASSES, PREP, fast_prep, prep_dir, raw_dir, write_lines

VAL_SEQS = {"night": 2, "rainy": 1, "sunny": 2, "cloudy": 1}
TYPE_MAP = {"car": "car", "van": "car", "bus": "bus", "others": "truck"}


class CocoWriter:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.images, self.anns, self.n = [], [], 0
        self.sources: list[str] = []

    def add(self, img: np.ndarray, boxes: list[tuple[float, float, float, float, str]], name: str, src: str):
        self.n += 1
        h, w = img.shape[:2]
        cv2.imwrite(str(self.root / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        self.images.append({"id": self.n, "file_name": name, "width": w, "height": h})
        for x1, y1, x2, y2, c in boxes:
            x1, y1, x2, y2 = (float(v) for v in (max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            self.anns.append({"id": len(self.anns) + 1, "image_id": self.n, "category_id": CLASSES.index(c), "bbox": [x1, y1, x2 - x1, y2 - y1],
                              "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
        self.sources.append(src)

    def save(self):
        cats = [{"id": i, "name": c, "supercategory": "object"} for i, c in enumerate(CLASSES)]
        (self.root / "_annotations.coco.json").write_text(json.dumps({"images": self.images, "annotations": self.anns, "categories": cats}))


def parse_xml(p: Path) -> tuple[list[list[float]], dict[int, list]]:
    root = ET.parse(p).getroot()
    ign: list[list[float]] = []
    ig = root.find("ignored_region")
    if ig is not None:
        for b in ig.findall("box"):
            l, t, w, h = (float(b.get(k)) for k in ("left", "top", "width", "height"))
            ign.append([l, t, l + w, t + h])
    frames: dict[int, list] = {}
    for fr in root.findall("frame"):
        n = int(fr.get("num"))
        for tg in fr.iter("target"):
            b = tg.find("box")
            a = tg.find("attribute")
            l, t, w, h = (float(b.get(k)) for k in ("left", "top", "width", "height"))
            frames.setdefault(n, []).append((l, t, l + w, t + h, TYPE_MAP.get(a.get("vehicle_type"), "car")))
    return ign, frames


def pick_val_seqs(meta: dict) -> list[str]:
    rng = random.Random(0)
    out = []
    for w, k in VAL_SEQS.items():
        c = sorted(s for s, v in meta.items() if v["split"] == "train" and v["weather"] == w)
        rng.shuffle(c)
        out += c[:k]
    return out


def build_detrac(train_w: CocoWriter, val_w: CocoWriter, stride: int, val_stride: int = 25):
    meta = D.detrac_meta()
    val_seqs = set(pick_val_seqs(meta))
    z = zipfile.ZipFile(D.detrac_zip("train"))
    names = set(z.namelist())
    xml_dir = raw_dir("detrac") / "xml"
    seqdirs = sorted({n.split("/")[0] for n in names if n.startswith("detrac_")})
    for sd in seqdirs:
        seq = sd.split("_", 2)[2]
        ign, frames = parse_xml(xml_dir / f"{seq}.xml")
        is_val = seq in val_seqs
        st = val_stride if is_val else stride
        w = val_w if is_val else train_w
        for fn in sorted(frames):
            if (fn - 1) % st:
                continue
            fname = f"image{fn:06d}.jpg"  # the HF train zip names frames image000001.jpg (the test zip uses img00001.jpg)
            if f"{sd}/{fname}" not in names:
                continue
            im = cv2.imdecode(np.frombuffer(z.read(f"{sd}/{fname}"), np.uint8), cv2.IMREAD_COLOR)
            for x1, y1, x2, y2 in ign:
                im[int(max(y1, 0)):int(y2), int(max(x1, 0)):int(x2)] = 114
            w.add(im, frames[fn], f"detrac_{seq}_{fname}", f"detrac_train/{seq}/{fname}")
        print("detrac", seq, "val" if is_val else "train", train_w.n, val_w.n, flush=True)
    return sorted(val_seqs)


def build_ndispark(train_w: CocoWriter, repeat: int = 4, max_side: int = 1600):
    for it in D.ndispark_items("train"):
        im = cv2.imread(it.path)
        s = min(1.0, max_side / max(im.shape[:2]))
        if s < 1:
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        boxes = [(*(b * s), "car") for b in it.gt]
        for r in range(repeat):
            train_w.add(im, boxes, f"ndispark_{Path(it.path).stem}_r{r}.jpg", f"ndispark_train/{Path(it.path).name}")


def build_coco(train_w: CocoWriter, n: int, skip: int = 2000):
    from lanes.flow.router_data import coco_annotations, fetch_coco_images

    root = raw_dir("coco")
    js = coco_annotations()
    imgs = {i["id"]: i for i in js["images"]}
    names = {1: "person", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
    by: dict[int, list] = {}
    for a in js["annotations"]:
        if a["category_id"] in names and not a["iscrowd"]:
            x, y, w, h = a["bbox"]
            by.setdefault(a["image_id"], []).append((x, y, x + w, y + h, names[a["category_id"]]))
    ids = sorted(imgs)
    random.Random(0).shuffle(ids)  # same shuffle as data_extra.cocoval_items: [:2000] is the held-out person-AP set
    pool = [i for i in ids[skip:] if i in by]
    fetch_coco_images([imgs[i]["file_name"] for i in pool[:n]])
    for iid in pool[:n]:
        im = cv2.imread(str(root / "val2017" / imgs[iid]["file_name"]))
        train_w.add(im, by[iid], f"coco_{imgs[iid]['file_name']}", f"coco_val2017/{imgs[iid]['file_name']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detrac-stride", type=int, default=10)
    ap.add_argument("--ndispark-repeat", type=int, default=4)
    ap.add_argument("--coco-n", type=int, default=2500)
    ap.add_argument("--name", default="ft")
    a = ap.parse_args()
    root = fast_prep(a.name)  # NVMe (flow budget 12 GB)
    for sub in ("train", "valid"):
        if (root / sub).exists():
            shutil.rmtree(root / sub)
    tw, vw = CocoWriter(root / "train"), CocoWriter(root / "valid")
    val_seqs = build_detrac(tw, vw, a.detrac_stride)
    n_detrac = tw.n
    if a.ndispark_repeat:
        build_ndispark(tw, a.ndispark_repeat)
    n_nd = tw.n - n_detrac
    if a.coco_n:
        build_coco(tw, a.coco_n)
    tw.save()
    vw.save()
    (root / "classes.json").write_text(json.dumps(list(CLASSES)))
    write_lines(root / "train_files.txt", sorted(set(tw.sources)))
    write_lines(root / "valid_files.txt", sorted(set(vw.sources)))
    summary = {"train_images": tw.n, "train_boxes": len(tw.anns), "valid_images": vw.n, "valid_boxes": len(vw.anns), "detrac_train_images": n_detrac,
              "ndispark_images(with repeats)": n_nd, "coco_images": tw.n - n_detrac - n_nd, "val_sequences": val_seqs}
    (root / "summary.json").write_text(json.dumps(summary, indent=1))
    print(summary)


if __name__ == "__main__":
    main()
