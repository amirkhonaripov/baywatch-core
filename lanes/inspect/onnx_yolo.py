"""ONNX Runtime wrapper for Ultralytics YOLO11-style segmentation / detection exports (no Ultralytics import at inference).

Export format expected (``YOLO(...).export(format="onnx", imgsz=S, dynamic=False, simplify=True, opset=17)``):
  seg : output0 [1, 4 + nc + 32, A]  (cx, cy, w, h, class scores..., 32 mask coefficients),  output1 [1, 32, S/4, S/4]
  det : output0 [1, 4 + nc, A]
Post-processing is re-implemented here (letterbox, class-wise NMS, mask = sigmoid(coef @ proto) cropped to the box) and
checked against the Ultralytics predictor in lanes/inspect/export.py (parity report in the model card).
"""
from __future__ import annotations

import cv2
import numpy as np


def letterbox(img: np.ndarray, size: int, color: int = 114) -> tuple[np.ndarray, float, int, int]:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), color, np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, left, top


def nms_classwise(boxes: np.ndarray, scores: np.ndarray, cls: np.ndarray, iou: float, max_det: int) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0, int)
    off = cls.astype(np.float32)[:, None] * 8192.0  # class offset trick => class-wise NMS in one call
    b = boxes + off
    xywh = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
    keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), 0.0, iou)
    keep = np.array(keep, int).reshape(-1)
    return keep[:max_det]


class YoloOnnx:
    def __init__(self, path: str, names: list[str], imgsz: int, providers: list[str] | None = None, task: str = "seg"):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path), so, providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.names, self.imgsz, self.task = names, imgsz, task
        self.input_name = self.sess.get_inputs()[0].name
        self.nc = len(names)

    # ------------------------------------------------------------------------------------------------ core
    def raw(self, img_rgb: np.ndarray):
        lb, r, left, top = letterbox(img_rgb, self.imgsz)
        x = lb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        outs = self.sess.run(None, {self.input_name: x})
        return outs, r, left, top

    def decode(self, outs, r: float, left: int, top: int, orig_hw: tuple[int, int], conf: float, iou: float, max_det: int,
               mask_canvas: int | None = 384):
        """-> list of {cls:int, name, score, box:[x1,y1,x2,y2] (orig px), mask: bool array on a canvas whose long side is
        ``mask_canvas`` (or None)}; canvas size is returned as the second value (ch, cw)."""
        H, W = orig_hw
        p = outs[0][0].T  # [A, C]
        boxes_xywh, rest = p[:, :4], p[:, 4:]
        sc_all = rest[:, : self.nc]
        cls = sc_all.argmax(1)
        score = sc_all[np.arange(len(p)), cls]
        m = score >= conf
        boxes_xywh, score, cls = boxes_xywh[m], score[m], cls[m]
        coef = rest[m][:, self.nc: self.nc + 32] if self.task == "seg" else None
        xyxy = np.stack([boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2, boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2,
                         boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2, boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2], 1)
        keep = nms_classwise(xyxy, score, cls, iou, max_det)
        xyxy, score, cls = xyxy[keep], score[keep], cls[keep]
        if coef is not None:
            coef = coef[keep]
        # map to original image
        ob = xyxy.copy()
        ob[:, [0, 2]] = (ob[:, [0, 2]] - left) / r
        ob[:, [1, 3]] = (ob[:, [1, 3]] - top) / r
        ob[:, [0, 2]] = ob[:, [0, 2]].clip(0, W)
        ob[:, [1, 3]] = ob[:, [1, 3]].clip(0, H)
        masks = [None] * len(keep)
        ch = cw = 0
        if coef is not None and len(keep) and mask_canvas:
            proto = outs[1][0]  # [32, mh, mw]
            _, mh, mw = proto.shape
            s = mask_canvas / max(H, W)
            ch, cw = max(1, round(H * s)), max(1, round(W * s))
            logits = (coef @ proto.reshape(32, -1)).reshape(-1, mh, mw)
            prob = 1.0 / (1.0 + np.exp(-logits))
            gain = mw / self.imgsz
            # valid (non-padded) region of the letterboxed image in proto coordinates
            x0, y0 = left * gain, top * gain
            x1, y1 = mw - left * gain, mh - top * gain
            for k in range(len(keep)):
                bx = xyxy[k] * gain  # box in proto coords
                pm = prob[k].copy()
                yy, xx = np.mgrid[0:mh, 0:mw]
                pm[~((xx + 0.5 >= bx[0]) & (xx + 0.5 <= bx[2]) & (yy + 0.5 >= bx[1]) & (yy + 0.5 <= bx[3]))] = 0.0
                crop = pm[int(round(y0)):int(round(y1)), int(round(x0)):int(round(x1))]
                if crop.size == 0:
                    masks[k] = np.zeros((ch, cw), bool)
                    continue
                masks[k] = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_LINEAR) > 0.5
        dets = [{"cls": int(c), "name": self.names[int(c)], "score": float(s_), "box": b.tolist(), "mask": mk}
                for c, s_, b, mk in zip(cls, score, ob, masks)]
        return dets, (ch, cw)

    def __call__(self, img_rgb: np.ndarray, conf: float = 0.25, iou: float = 0.6, max_det: int = 100, mask_canvas: int | None = 384):
        outs, r, left, top = self.raw(img_rgb)
        return self.decode(outs, r, left, top, img_rgb.shape[:2], conf, iou, max_det, mask_canvas)


def mask_to_polygon(mask: np.ndarray, scale_x: float, scale_y: float, eps: float = 1.0) -> list[list[float]]:
    """Largest external contour of a bool mask as a polygon in original-image pixels (for evidence overlays / eval)."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float64)
    return [[float(x * scale_x), float(y * scale_y)] for x, y in c]
