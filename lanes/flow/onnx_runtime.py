"""ONNX Runtime backends: RF-DETR detector (exported with ``export_onnx.py``) behind the ``Detector2D`` interface.

Pre/post-processing mirrors rfdetr: square resize to the export resolution (bilinear), /255, ImageNet mean/std, NCHW; outputs
``dets`` (1,Q,4 cxcywh in 0..1) and ``labels`` (1,Q,C logits); score = sigmoid(logit); the Q*C scores are flattened and the top-300
kept (rfdetr's PostProcess), then mapped to pixels of the *original* image.
"""
from __future__ import annotations

import numpy as np

from lanes.flow.detectors import RFDETR_COCO, Detector2D, vehicle_nms

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def providers(device: str = "cuda:0"):
    idx = int(device.split(":")[1]) if ":" in device else 0
    if device.startswith("cpu"):
        return ["CPUExecutionProvider"]
    return [("CUDAExecutionProvider", {"device_id": idx}), "CPUExecutionProvider"]


class OnnxRFDETR(Detector2D):
    def __init__(self, path, resolution: int | None = None, class_map: dict[int, int] | None = None, num_select: int = 300):
        self.path, self.resolution, self.num_select = str(path), resolution, num_select
        self.class_map = class_map or RFDETR_COCO
        self.name = "rfdetr-onnx"
        self.sess = None

    def load(self, device: str = "cuda:0") -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(self.path, so, providers=providers(device))
        inp = self.sess.get_inputs()[0]
        self.in_name = inp.name
        if self.resolution is None:
            self.resolution = int(inp.shape[-1]) if isinstance(inp.shape[-1], int) else 560
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.active = self.sess.get_providers()

    def preprocess(self, im: np.ndarray) -> np.ndarray:
        import cv2

        r = cv2.resize(im, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        return ((r - MEAN) / STD).transpose(2, 0, 1)[None]

    def run_raw(self, x: np.ndarray):
        return self.sess.run(self.out_names, {self.in_name: x})

    def postprocess(self, boxes: np.ndarray, logits: np.ndarray, w: int, h: int, conf: float) -> np.ndarray:
        q, c = logits.shape
        prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
        flat = prob.ravel()
        k = min(self.num_select, flat.size)
        top = np.argpartition(-flat, k - 1)[:k]
        top = top[np.argsort(-flat[top], kind="stable")]
        qi, ci = top // c, top % c
        keep = flat[top] >= conf
        qi, ci, sc = qi[keep], ci[keep], flat[top][keep]
        b = boxes[qi]
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1) * np.array([w, h, w, h], dtype=np.float32)
        rows = [(*xyxy[i], sc[i], self.class_map[int(ci[i])]) for i in range(len(sc)) if int(ci[i]) in self.class_map]
        return np.asarray(rows, dtype=np.float32).reshape(-1, 6)

    def raw_infer(self, images, conf):
        out = []
        for im in images:
            h, w = im.shape[:2]
            dets, labels = self.run_raw(self.preprocess(im))[:2]
            out.append(self.postprocess(dets[0], labels[0], w, h, conf))
        return out
