"""Day-1 baseline predictors built only from pretrained public models (no training by us).

They exist so the end-to-end skeleton works before the lanes deliver. Lane predictors replace them in
seen/pipeline.py::default_predictors.
"""
from __future__ import annotations

import time

import numpy as np

from seen.predictor import Det, Frame, FramePred, empty_preds

COCO_KEEP = {"car": "car", "truck": "truck", "bus": "bus", "motorcycle": "motorcycle", "person": "person"}


class PretrainedDetector:
    """RF-DETR (COCO-pretrained) vehicles + persons."""

    name = "rfdetr-coco"
    version = "pretrained"
    role = "vehicle"

    def __init__(self, size: str = "base", threshold: float = 0.4):
        self.size, self.threshold, self.model = size, threshold, None

    def load(self, device: str = "cuda:0") -> None:
        from rfdetr import RFDETRBase, RFDETRLarge
        from rfdetr.assets.coco_classes import COCO_CLASSES

        self.model = (RFDETRLarge if self.size == "large" else RFDETRBase)()
        self.names = COCO_CLASSES

    def predict(self, frames: list[Frame], context: list[FramePred] | None = None) -> list[FramePred]:
        out = empty_preds(frames)
        for f, p in zip(frames, out):
            t0 = time.perf_counter()
            det = self.model.predict(f.image, threshold=self.threshold)
            h, w = f.image.shape[:2]
            for (x1, y1, x2, y2), c, k in zip(det.xyxy, det.confidence, det.class_id):
                name = COCO_KEEP.get(self.names.get(int(k), ""))
                if name:
                    p.detections.append(Det(cls=name, confidence=float(c), bbox=(x1 / w, y1 / h, x2 / w, y2 / h)))
            p.timings_ms["vehicle_detector"] = (time.perf_counter() - t0) * 1000
        return out


class PretrainedALPR:
    """fast-alpr pretrained plate detector + global OCR (not Uzbek-tuned)."""

    name = "fast-alpr-global"
    version = "pretrained"
    role = "plate"

    def load(self, device: str = "cuda:0") -> None:
        from fast_alpr import ALPR

        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") else ["CPUExecutionProvider"]
        # Explicit providers: the library default tries TensorRT first (noisy, and FP16 TRT failed parity in our tests).
        self.alpr = ALPR(detector_providers=prov, ocr_providers=prov)

    def predict(self, frames: list[Frame], context: list[FramePred] | None = None) -> list[FramePred]:
        out = empty_preds(frames)
        for f, p in zip(frames, out):
            t0 = time.perf_counter()
            bgr = np.ascontiguousarray(f.image[:, :, ::-1])
            h, w = f.image.shape[:2]
            for r in self.alpr.predict(bgr):
                bb = r.detection.bounding_box
                text = r.ocr.text if r.ocr else None
                conf = float(np.mean(r.ocr.confidence)) if (r.ocr and r.ocr.confidence is not None) else float(r.detection.confidence)
                p.detections.append(Det(cls="plate", confidence=conf, bbox=(bb.x1 / w, bb.y1 / h, bb.x2 / w, bb.y2 / h),
                                        text=text, extra={"format": "unknown"}))
            p.timings_ms["plate"] = (time.perf_counter() - t0) * 1000
        return out
