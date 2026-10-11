"""Vehicle + person detector wrappers with one output format.

``infer(images_rgb, conf)`` -> list of float32 arrays (N, 6) = [x1, y1, x2, y2, score, class_idx] in *pixels* of the
input image; class_idx indexes ``common.CLASSES`` = (person, car, motorcycle, bus, truck).

Registry names
  rfdetr-{nano,small,medium,base,large}   RF-DETR COCO weights (Apache-2.0 for these sizes)
  yolo11x / yolo26x ...                   Ultralytics COCO weights (AGPL-3.0: evaluated, not the shipping default)
  rfdetr-<size>@<checkpoint.pth>[#classes] fine-tuned RF-DETR (class names stored beside the checkpoint)
After inference vehicle classes (car/motorcycle/bus/truck) get a class-agnostic NMS so one object is not reported twice
as "car" and "truck"; persons are untouched. This is applied to every model equally.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from lanes.flow.common import CLASSES, VEHICLE_IDX, artifact_dir

RF_WEIGHTS = {"nano": "rf-detr-nano.pth", "small": "rf-detr-small.pth", "medium": "rf-detr-medium.pth", "base": "rf-detr-base.pth",
              "large": "rf-detr-large-2026.pth"}  # current Large (patch 16); the older rf-detr-large.pth (patch 14) does not match RFDETRLarge
RFDETR_COCO = {1: 0, 3: 1, 4: 2, 6: 3, 8: 4}  # COCO-91 category id -> our class idx
YOLO_COCO = {0: 0, 2: 1, 3: 2, 5: 3, 7: 4}  # COCO-80 index -> our class idx


def vehicle_nms(dets: np.ndarray, iou: float = 0.7) -> np.ndarray:
    """Class-agnostic NMS over the vehicle classes only."""
    if len(dets) == 0:
        return dets
    import torch
    from torchvision.ops import nms

    isveh = np.isin(dets[:, 5].astype(int), VEHICLE_IDX)
    v, p = dets[isveh], dets[~isveh]
    if len(v):
        keep = nms(torch.from_numpy(v[:, :4]).float(), torch.from_numpy(v[:, 4]).float(), iou).numpy()
        v = v[keep]
    out = np.concatenate([v, p], 0) if len(p) else v
    return out[np.argsort(-out[:, 4], kind="stable")] if len(out) else out


class Detector2D:
    name = "base"

    def load(self, device: str = "cuda:0") -> None:
        raise NotImplementedError

    def raw_infer(self, images: list[np.ndarray], conf: float) -> list[np.ndarray]:
        raise NotImplementedError

    def infer(self, images: list[np.ndarray], conf: float = 0.05, nms_iou: float = 0.7) -> list[np.ndarray]:
        return [vehicle_nms(d, nms_iou) for d in self.raw_infer(images, conf)]


class RFDETR(Detector2D):
    def __init__(self, size: str = "base", checkpoint: str | None = None, class_map: dict[int, int] | None = None,
                 resolution: int | None = None, precision: str = "fp32"):
        self.size, self.checkpoint, self.resolution, self.precision = size, checkpoint, resolution, precision
        self.class_map = class_map or RFDETR_COCO
        self.name = f"rfdetr-{size}" + (f"@{Path(checkpoint).parent.name}" if checkpoint else "")
        self.model = None

    def load(self, device: str = "cuda:0") -> None:
        import rfdetr

        cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium,
               "base": rfdetr.RFDETRBase, "large": rfdetr.RFDETRLarge}[self.size]
        kw = {"device": device}
        if self.checkpoint:
            kw["pretrain_weights"] = self.checkpoint
            meta = Path(self.checkpoint).with_name("classes.json")
            if meta.exists():  # fine-tuned: category ids follow classes.json (list of our class names, in training order)
                names = json.loads(meta.read_text())
                self.class_map = {i: CLASSES.index(n) for i, n in enumerate(names) if n in CLASSES}
                kw["num_classes"] = len(names)
        else:
            kw["pretrain_weights"] = str(artifact_dir("pretrained") / RF_WEIGHTS[self.size])
        if self.resolution:
            kw["resolution"] = self.resolution
        self.model = cls(**kw)
        try:
            self.model.optimize_for_inference()
        except Exception:  # noqa: BLE001  (optional speed-up)
            pass

    def raw_infer(self, images, conf):
        import torch

        if self.precision == "fp32":
            res = self.model.predict(images, threshold=conf, include_source_image=False)
        else:
            dt = {"bf16": torch.bfloat16, "fp16": torch.float16}[self.precision]
            with torch.autocast("cuda", dtype=dt):
                res = self.model.predict(images, threshold=conf, include_source_image=False)
        if not isinstance(res, list):
            res = [res]
        out = []
        for d in res:
            rows = []
            for (x1, y1, x2, y2), s, k in zip(d.xyxy, d.confidence, d.class_id):
                c = self.class_map.get(int(k))
                if c is not None:
                    rows.append((x1, y1, x2, y2, s, c))
            out.append(np.asarray(rows, dtype=np.float32).reshape(-1, 6))
        return out


class UltralyticsYOLO(Detector2D):
    def __init__(self, weights: str = "yolo11x.pt", imgsz: int = 960):
        self.weights, self.imgsz = weights, imgsz
        self.name = Path(weights).stem
        self.model = None
        self.device = "cuda:0"

    def load(self, device: str = "cuda:0") -> None:
        from ultralytics import YOLO

        wdir = artifact_dir("pretrained")
        w = self.weights if os.path.exists(self.weights) else str(wdir / self.weights)
        if not os.path.exists(w):  # download once into artifacts/flow/pretrained (never into code/ or cwd)
            cwd = os.getcwd()
            os.chdir(wdir)
            try:
                YOLO(self.weights)
            finally:
                os.chdir(cwd)
        self.model = YOLO(w)
        self.device = device

    def raw_infer(self, images, conf):
        bgr = [np.ascontiguousarray(im[:, :, ::-1]) for im in images]
        res = self.model.predict(bgr, conf=conf, iou=0.7, imgsz=self.imgsz, max_det=300, device=self.device, verbose=False,
                                 classes=list(YOLO_COCO), half=True)
        out = []
        for r in res:
            b = r.boxes
            xyxy, s, k = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)
            rows = [(*xyxy[i], s[i], YOLO_COCO[k[i]]) for i in range(len(s)) if k[i] in YOLO_COCO]
            out.append(np.asarray(rows, dtype=np.float32).reshape(-1, 6))
        return out


class Tiled(Detector2D):
    """Full-frame pass + overlapping square tiles along the long side (for wide frames with small objects, e.g. parking lots), merged by vehicle NMS.

    Tiles are ``min(h, w)`` wide with 25% overlap; frames that are not wider than 1.3x their height get the full-frame pass only.
    """

    def __init__(self, base: Detector2D, overlap: float = 0.25):
        self.base, self.overlap = base, overlap
        self.name = base.name + "+tile"

    def load(self, device: str = "cuda:0") -> None:
        self.base.load(device)

    def raw_infer(self, images, conf):
        out = []
        for im in images:
            h, w = im.shape[:2]
            crops, offs = [im], [(0, 0)]
            if w > 1.3 * h:
                t = h
                n = int(np.ceil((w - t) / (t * (1 - self.overlap)))) + 1
                for x0 in np.linspace(0, w - t, n).astype(int):
                    crops.append(im[:, x0:x0 + t])
                    offs.append((int(x0), 0))
            elif h > 1.3 * w:
                t = w
                n = int(np.ceil((h - t) / (t * (1 - self.overlap)))) + 1
                for y0 in np.linspace(0, h - t, n).astype(int):
                    crops.append(im[y0:y0 + t])
                    offs.append((0, int(y0)))
            res = self.base.raw_infer(crops, conf)
            merged = []
            for d, (ox, oy) in zip(res, offs):
                d = d.copy()
                d[:, [0, 2]] += ox
                d[:, [1, 3]] += oy
                merged.append(d)
            out.append(np.concatenate(merged, 0) if merged else np.zeros((0, 6), np.float32))
        return out


def get_detector(spec: str) -> Detector2D:
    if spec.endswith("+tile"):
        return Tiled(get_detector(spec[:-5]))
    return _get_detector(spec)


def _get_detector(spec: str) -> Detector2D:
    if spec == "rfdetr-large-ft":  # our fine-tuned RF-DETR Large (runs/flow/ft_large)
        from lanes.flow.common import run_dir

        d = run_dir("ft_large")
        ck = d / "checkpoint_best_total.pth"
        return RFDETR("large", str(ck if ck.exists() else d / "checkpoint_best_ema.pth"))
    for prec in ("bf16", "fp16"):
        if spec.endswith(":" + prec):
            d = _get_detector(spec[: -len(prec) - 1])
            d.precision = prec
            d.name += ":" + prec
            return d
    """'rfdetr-base' | 'rfdetr-large' | 'yolo11x' | 'yolo26x' | 'rfdetr-base@/path/checkpoint_best_total.pth'."""
    if spec.startswith("rfdetr"):
        head, _, ckpt = spec.partition("@")
        size = head.split("-", 1)[1] if "-" in head else "base"
        res = None
        if "#" in ckpt:
            ckpt, r = ckpt.split("#")
            res = int(r)
        return RFDETR(size, ckpt or None, resolution=res)
    if spec.startswith("yolo"):
        w = spec if spec.endswith(".pt") else spec + ".pt"
        return UltralyticsYOLO(w)
    raise ValueError(f"unknown detector {spec!r}")
