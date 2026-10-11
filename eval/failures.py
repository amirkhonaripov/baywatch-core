"""Failure gallery: collect lanes' failure images + jury failures, face-blur them again, write the gallery.

    python -m eval.failures [--out <output root>]

Inputs  docs/lanes/failures/<lane>/*.{jpg,png}  (+ optional failures.json [{id, component, file, description, category}])
Output  docs/lanes/eval_gallery/<lane>/<file>.jpg   (<= 640 px, faces blurred by three independent passes)
        docs/lanes/eval_failures.json              list[FailureCase]; image_uri relative to docs/ and web/public/

Blur passes (any one is enough to hide a face; we run all that are available):
  1. seen.blur.blur_faces on person boxes, when detections are supplied (jury failures)
  2. OpenCV YuNet face detector (opencv_zoo, MIT) if its ONNX file is present or downloadable
  3. OpenCV Haar frontal + profile cascades (bundled with opencv-python 4.x; absent in OpenCV 5, then skipped)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MAX_W = 640
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _yunet_path() -> Path | None:
    try:
        from seen.paths import PREP

        p = PREP / "eval" / "models" / "face_detection_yunet_2023mar.onnx"
    except Exception:
        return None
    if p.exists():
        return p
    try:
        from eval.jury.select import http_get

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(http_get(YUNET_URL, timeout=60))
        return p
    except Exception:
        return None


_YUNET = {"det": None, "tried": False}


def _yunet_faces(bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    import cv2

    if not _YUNET["tried"]:
        _YUNET["tried"] = True
        p = _yunet_path()
        if p is not None and hasattr(cv2, "FaceDetectorYN"):
            try:
                _YUNET["det"] = cv2.FaceDetectorYN.create(str(p), "", (320, 320), 0.6, 0.3, 5000)
            except Exception:
                _YUNET["det"] = None
    det = _YUNET["det"]
    if det is None:
        return []
    h, w = bgr.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    if faces is None:
        return []
    return [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in faces]


def _haar_faces(bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    import cv2

    if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):  # OpenCV 5 dropped the Haar API
        return []
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = []
    for name in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml"):
        c = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / name))
        if c.empty():
            continue
        for (x, y, w, h) in c.detectMultiScale(g, 1.1, 5, minSize=(12, 12)):
            out.append((int(x), int(y), int(w), int(h)))
    return out


def safe_blur(rgb: np.ndarray, dets=None) -> np.ndarray:
    """Face blur for anything we publish: seen.blur on person boxes + YuNet + Haar; returns a new RGB image."""
    import cv2

    from seen import blur

    img = blur.blur_faces(rgb, dets or [])
    bgr = np.ascontiguousarray(img[:, :, ::-1])
    h, w = bgr.shape[:2]
    for (x, y, fw, fh) in _yunet_faces(bgr) + _haar_faces(bgr):
        px, py = int(0.25 * fw), int(0.25 * fh)
        a, b, c, d = max(0, x - px), max(0, y - py), min(w, x + fw + px), min(h, y + fh + py)
        if c - a > 2 and d - b > 2:
            k = max(((c - a) // 2) | 1, 15)
            bgr[b:d, a:c] = cv2.GaussianBlur(bgr[b:d, a:c], (k, k), 0)
    return bgr[:, :, ::-1].copy()


def save_gallery_image(rgb: np.ndarray, dst: Path, dets=None) -> Path:
    import cv2

    img = safe_blur(rgb, dets)
    h, w = img.shape[:2]
    if w > MAX_W:
        img = cv2.resize(img, (MAX_W, int(h * MAX_W / w)), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst.with_suffix(".jpg")), img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    return dst.with_suffix(".jpg")


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:80]


def collect(roots: list[Path], out: Path) -> list[dict]:
    import cv2

    cases: list[dict] = []
    lanes: dict[str, Path] = {}
    for r in roots:
        base = r / "docs" / "lanes" / "failures"
        if base.is_dir():
            for d in sorted(p for p in base.iterdir() if p.is_dir()):
                lanes[d.name] = d  # later roots override
    gal = out / "docs" / "lanes" / "eval_gallery"
    for lane, d in sorted(lanes.items()):
        meta = {}
        mj = d / "failures.json"
        if mj.exists():
            try:
                for m in json.loads(mj.read_text()):
                    meta[m.get("file") or m.get("image") or m.get("id")] = m
            except Exception:
                pass
        for img_path in sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS):
            m = meta.get(img_path.name, {})
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            dst = save_gallery_image(im[:, :, ::-1], gal / lane / _slug(img_path.stem))
            cases.append({"id": m.get("id") or f"{lane}:{img_path.stem}",
                          "component": m.get("component") or lane,
                          "image_uri": str(dst.relative_to(out / "docs")),
                          "description": m.get("description") or img_path.stem.replace("_", " "),
                          **({"category": m["category"]} if m.get("category") else {})})
    d = out / "docs" / "lanes"
    d.mkdir(parents=True, exist_ok=True)
    (d / "eval_failures.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False))
    return cases


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--out", default=str(REPO))
    a = ap.parse_args(argv)
    roots = [REPO] + [Path(r) for r in a.root]
    cases = collect(roots, Path(a.out))
    print(f"{len(cases)} failure cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
