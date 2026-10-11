"""Face blur applied to every evidence frame before it is stored or shown.

Lane B: faces come from YuNet (OpenCV Zoo ``face_detection_yunet_2023mar.onnx``, MIT) run (1) on the full frame and (2) on an
upscaled head-and-shoulders crop of every detected person (CCTV faces are small); a person box with no face found falls back to
the heuristic "upper 30% of the person box". ``blur_faces`` keeps its signature ``(image, dets) -> image``.

Model lookup: ``$SEEN_YUNET`` -> ``lanes/flow/assets/`` -> ``$ARTIFACTS/flow/yunet/`` -> download once from the OpenCV Zoo. If the
model cannot be loaded the heuristic alone is used (never a silent no-op: ``backend()`` reports which one is active).
Measured recall on WIDER FACE val: see ``docs/lanes/flow_notes.md``.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from seen.predictor import Det
from seen.vocab import PERSON

YUNET_NAME = "face_detection_yunet_2023mar.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/" + YUNET_NAME
SCORE_THR = 0.5  # low on purpose: for privacy a false blur is cheaper than a missed face
NMS_THR = 0.3
PAD = 0.25  # relative padding added around every face box (hair/forehead/jitter)
HEURISTIC_FRAC = 0.30  # fallback: upper part of a person box
FULL_FRAME_SCALES = (1.0, 2.0)  # YuNet input scales for the full-frame pass; (1.0, 2.0) finds smaller faces at ~3x the cost (see flow_notes.md)

Box = tuple[float, float, float, float]
_state: dict = {"det": None, "tried": False, "path": None}


def _model_path() -> Path | None:
    cands = []
    if os.environ.get("SEEN_YUNET"):
        cands.append(Path(os.environ["SEEN_YUNET"]))
    cands.append(Path(__file__).resolve().parents[1] / "lanes" / "flow" / "assets" / YUNET_NAME)
    try:
        from seen.paths import ARTIFACTS

        cands.append(ARTIFACTS / "flow" / "yunet" / YUNET_NAME)
    except Exception:  # noqa: BLE001
        pass
    for c in cands:
        if c.exists() and c.stat().st_size > 100_000:
            return c
    try:  # one-off download (232 KB)
        dst = cands[-1] if cands else Path.home() / ".cache" / "seen" / YUNET_NAME
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(YUNET_URL, dst)
        return dst if dst.stat().st_size > 100_000 else None
    except Exception:  # noqa: BLE001
        return None


def _detector():
    if _state["det"] is None and not _state["tried"]:
        _state["tried"] = True
        p = _model_path()
        if p is not None and hasattr(cv2, "FaceDetectorYN"):
            try:
                _state["det"] = cv2.FaceDetectorYN.create(str(p), "", (320, 320), SCORE_THR, NMS_THR, 5000)
                _state["path"] = str(p)
            except Exception:  # noqa: BLE001
                _state["det"] = None
    return _state["det"]


def backend() -> str:
    return "yunet" if _detector() is not None else "heuristic"


def _yunet(bgr: np.ndarray, scale: float = 1.0) -> list[tuple[float, float, float, float, float]]:
    """Face boxes (x1, y1, x2, y2, score) in the pixel frame of ``bgr``; runs YuNet at ``scale`` x the input size."""
    det = _detector()
    if det is None or bgr.size == 0:
        return []
    h, w = bgr.shape[:2]
    if scale != 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    ih, iw = bgr.shape[:2]
    det.setInputSize((iw, ih))
    _, faces = det.detect(bgr)
    out = []
    if faces is not None:
        for f in faces:
            x, y, fw, fh, s = f[0], f[1], f[2], f[3], f[-1]
            if not np.all(np.isfinite([x, y, fw, fh, s])) or fw <= 0 or fh <= 0:
                continue
            out.append((x / scale, y / scale, (x + fw) / scale, (y + fh) / scale, float(s)))
    return out


def _nms(boxes: list[tuple], thr: float = 0.5) -> list[tuple]:
    boxes = sorted(boxes, key=lambda b: -b[4])
    keep: list[tuple] = []
    for b in boxes:
        ok = True
        for k in keep:
            iw = min(b[2], k[2]) - max(b[0], k[0])
            ih = min(b[3], k[3]) - max(b[1], k[1])
            if iw > 0 and ih > 0:
                inter = iw * ih
                u = (b[2] - b[0]) * (b[3] - b[1]) + (k[2] - k[0]) * (k[3] - k[1]) - inter
                if inter / max(u, 1e-9) > thr:
                    ok = False
                    break
        if ok:
            keep.append(b)
    return keep


def detect_faces(image: np.ndarray, dets: list[Det] | None = None) -> tuple[list[tuple[float, float, float, float, float]], dict]:
    """Pixel-space face boxes (x1,y1,x2,y2,score) from a full-frame pass + per-person head-crop passes.

    ``image`` is RGB uint8. Returns (boxes, info) with info = {"persons_without_face": [person det indices]}.
    """
    h, w = image.shape[:2]
    bgr = np.ascontiguousarray(image[:, :, ::-1])
    boxes = []
    for sc in FULL_FRAME_SCALES:
        boxes += _yunet(bgr, sc if max(h, w) >= 640 or sc >= 2.0 else 2.0)
    info: dict = {"persons_without_face": []}
    for i, d in enumerate(dets or []):
        if d.cls != PERSON:
            continue
        x1, y1, x2, y2 = d.bbox
        px1, py1, px2, py2 = x1 * w, y1 * h, x2 * w, y2 * h
        bw, bh = px2 - px1, py2 - py1
        if bw < 4 or bh < 8:
            info["persons_without_face"].append(i)
            continue
        # head-and-shoulders region; upscale so a ~15 px face becomes ~60 px for YuNet
        cx1, cx2 = int(max(px1 - 0.1 * bw, 0)), int(min(px2 + 0.1 * bw, w))
        cy1, cy2 = int(max(py1 - 0.05 * bh, 0)), int(min(py1 + 0.45 * bh, h))
        crop = bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        sc = float(np.clip(256.0 / max(crop.shape[1], 1), 1.0, 4.0))
        found = [(a + cx1, b + cy1, c + cx1, e + cy1, s) for a, b, c, e, s in _yunet(crop, sc)]
        if not found:
            info["persons_without_face"].append(i)
        boxes += found
    return _nms(boxes), info


def face_boxes(image: np.ndarray, dets: list[Det]) -> list[Box]:
    """Normalised [x1,y1,x2,y2] regions to blur: YuNet faces (padded) + heuristic head box for persons with no face found."""
    h, w = image.shape[:2]
    out: list[Box] = []
    faces, info = detect_faces(image, dets) if backend() == "yunet" else ([], {"persons_without_face": [i for i, d in enumerate(dets) if d.cls == PERSON]})
    for x1, y1, x2, y2, _ in faces:
        pw, ph = (x2 - x1) * PAD, (y2 - y1) * PAD
        out.append((max(x1 - pw, 0) / w, max(y1 - ph, 0) / h, min(x2 + pw, w) / w, min(y2 + ph, h) / h))
    for i in info["persons_without_face"]:
        x1, y1, x2, y2 = dets[i].bbox
        out.append((x1, y1, x2, y1 + HEURISTIC_FRAC * (y2 - y1)))
    return out


def _pixelate(roi: np.ndarray) -> np.ndarray:
    """Mosaic of ~6 blocks across the region + a soft blur: not recoverable by deconvolution, unlike a plain Gaussian."""
    h, w = roi.shape[:2]
    n = 6
    small = cv2.resize(roi, (n, max(int(round(n * h / w)), 1)), interpolation=cv2.INTER_AREA)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    k = max((w // 4) | 1, 3)
    return cv2.GaussianBlur(big, (k, k), 0)


def blur_faces(image: np.ndarray, dets: list[Det]) -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]
    for x1, y1, x2, y2 in face_boxes(image, dets):
        a, b = max(int(x1 * w), 0), max(int(y1 * h), 0)
        c, d = min(int(np.ceil(x2 * w)), w), min(int(np.ceil(y2 * h)), h)
        if c - a > 2 and d - b > 2:
            img[b:d, a:c] = _pixelate(img[b:d, a:c])
    return img
