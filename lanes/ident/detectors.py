"""Plate detectors with a common interface: ``detect(images_bgr) -> list[list[(score, x1, y1, x2, y2)]]`` in pixels."""
from __future__ import annotations

import time

import numpy as np

from lanes.ident.ocr_onnx import default_providers


class FastAlprDetector:
    """open-image-models / fast-alpr plate detector (YOLOv9-t end2end ONNX, MIT)."""

    def __init__(self, name: str = "yolo-v9-t-384-license-plate-end2end", conf: float = 0.05, device: str = "cuda:0"):
        from open_image_models import LicensePlateDetector

        self.name = name
        self.det = LicensePlateDetector(detection_model=name, conf_thresh=conf, providers=default_providers(device))

    def detect(self, images_bgr: list[np.ndarray]):
        out = []
        for im in images_bgr:
            res = self.det.predict(im)
            out.append([(float(r.confidence), r.bounding_box.x1, r.bounding_box.y1, r.bounding_box.x2, r.bounding_box.y2) for r in res])
        return out
