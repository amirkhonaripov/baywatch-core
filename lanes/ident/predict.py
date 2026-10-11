"""Ident lane predictor: plate detection + Uzbek-adapted OCR + track voting, as a ``seen.predictor.Predictor``.

    from lanes.ident.predict import get_predictor
    p = get_predictor(); p.load("cuda:0"); preds = p.predict(frames, context)

Emits ``Det(cls="plate", confidence, bbox(normalised xyxy), track_id, text, extra={"format", "format_valid", ...})`` with
``extra["format"]`` in {uz_private, uz_legal, uz_ev, uz_moto, foreign, unknown}.
Models are loaded with onnxruntime (CUDA EP, CPU fallback) from ``artifacts/ident/plate_det`` and ``artifacts/ident/plate_ocr``
(``model.onnx`` + ``model_card.json`` [+ ``plate_config.yaml`` for the OCR]).

Plate tracking inside the predictor: plates are associated across consecutive calls by IoU (greedy) so per-track voting works on
video; if ``context`` carries vehicle detections with track ids, a plate inherits ``extra["vehicle_track_id"]``. ``reset()`` clears
the tracks (call between videos/cameras).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from lanes.ident import uz_format as uz
from lanes.ident.ocr_onnx import PlateOcrOnnx, default_providers
from lanes.ident.reader import PlateReader
from lanes.ident.voting import Observation, TrackVoter
from seen import vocab
from seen.paths import ARTIFACTS
from seen.predictor import Det, Frame, FramePred, Predictor

DET_DIR = ARTIFACTS / "ident" / "plate_det"
OCR_DIR = ARTIFACTS / "ident" / "plate_ocr"


# ------------------------------------------------------------------------------------------------------ detector
class OnnxPlateDetector:
    """RF-DETR / YOLO-style single-class plate detector exported to ONNX (see model_card.json: arch, input_size)."""

    def __init__(self, model_dir: str | Path, device: str = "cuda:0", conf: float = 0.25):
        import onnxruntime as ort

        d = Path(model_dir)
        self.card = json.loads((d / "model_card.json").read_text())
        self.arch = self.card.get("arch", "rfdetr")
        self.size = self.card["input_size"]
        self.conf = conf
        self.sess = ort.InferenceSession(str(d / "model.onnx"), providers=default_providers(device))
        self.inp = self.sess.get_inputs()[0].name
        self.outs = [o.name for o in self.sess.get_outputs()]
        self.mean = np.array(self.card.get("mean", [0.485, 0.456, 0.406]), np.float32)
        self.std = np.array(self.card.get("std", [0.229, 0.224, 0.225]), np.float32)

    def _prep(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        sh, sw = (self.size, self.size) if isinstance(self.size, int) else self.size
        if self.arch == "yolo":  # letterbox
            r = min(sh / h, sw / w)
            nh, nw = round(h * r), round(w * r)
            img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((sh, sw, 3), 114, np.uint8)
            top, left = (sh - nh) // 2, (sw - nw) // 2
            canvas[top:top + nh, left:left + nw] = img
            x = canvas[..., ::-1].astype(np.float32) / 255.0
            meta = (r, left, top)
        else:  # RF-DETR: plain resize, ImageNet normalisation
            img = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_LINEAR)
            x = (img[..., ::-1].astype(np.float32) / 255.0 - self.mean) / self.std
            meta = (sw / w, sh / h)
        return np.ascontiguousarray(x.transpose(2, 0, 1))[None], meta

    def detect(self, images_bgr: list[np.ndarray]) -> list[list[tuple[float, float, float, float, float]]]:
        out = []
        for im in images_bgr:
            h, w = im.shape[:2]
            x, meta = self._prep(im)
            res = self.sess.run(self.outs, {self.inp: x})
            dets: list[tuple[float, float, float, float, float]] = []
            if self.arch == "yolo":
                p = res[0][0]  # (5, N) or (N, 5)
                p = p if p.shape[0] < p.shape[1] else p.T
                cx, cy, bw, bh, sc = p[0], p[1], p[2], p[3], p[4]
                keep = sc >= self.conf
                r, left, top = meta
                boxes = np.stack([(cx - bw / 2 - left) / r, (cy - bh / 2 - top) / r, (cx + bw / 2 - left) / r, (cy + bh / 2 - top) / r], 1)[keep]
                scs = sc[keep]
                if len(scs):
                    idx = cv2.dnn.NMSBoxes([[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes], scs.tolist(), self.conf, 0.5)
                    for i in np.array(idx).reshape(-1):
                        b = boxes[i]
                        dets.append((float(scs[i]), *(float(np.clip(v, 0, m)) for v, m in zip(b, (w, h, w, h)))))
            else:
                names = dict(zip(self.outs, res))
                boxes = names.get("dets", res[0])[0]          # (Q, 4) cxcywh normalised
                logits = names.get("labels", res[1])[0]       # (Q, C)
                cls = self.card.get("class_index", 0)
                sc = 1 / (1 + np.exp(-logits[:, cls]))
                for q in np.where(sc >= self.conf)[0]:
                    cx, cy, bw, bh = boxes[q]
                    dets.append((float(sc[q]), float(np.clip((cx - bw / 2) * w, 0, w)), float(np.clip((cy - bh / 2) * h, 0, h)),
                                 float(np.clip((cx + bw / 2) * w, 0, w)), float(np.clip((cy + bh / 2) * h, 0, h))))
            out.append(sorted(dets, reverse=True))
        return out


# ------------------------------------------------------------------------------------------------------- tracker
def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


class _PlateTracks:
    def __init__(self, iou_thr: float = 0.25, max_age: int = 15):
        self.iou_thr, self.max_age = iou_thr, max_age
        self.tracks: dict[int, dict] = {}
        self.next_id = 1

    def update(self, boxes: list[tuple], frame_idx: int) -> list[int]:
        ids: list[int | None] = [None] * len(boxes)
        used = set()
        pairs = sorted(((_iou(b, t["box"]), i, tid) for i, b in enumerate(boxes) for tid, t in self.tracks.items()), reverse=True)
        for s, i, tid in pairs:
            if s < self.iou_thr:
                break
            if ids[i] is None and tid not in used:
                ids[i] = tid
                used.add(tid)
        for i, b in enumerate(boxes):
            if ids[i] is None:
                ids[i] = self.next_id
                self.next_id += 1
            self.tracks[ids[i]] = {"box": b, "last": frame_idx}
        for tid in [t for t, v in self.tracks.items() if frame_idx - v["last"] > self.max_age]:
            del self.tracks[tid]
        return ids  # type: ignore[return-value]


# ------------------------------------------------------------------------------------------------- the predictor
def _iou(a, b) -> float:
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


class IdentPredictor:
    name = "ident-plate"
    role = "plate"          # pipeline maps roles to the model/model_version shown on event evidence (docs/LANES.md)
    version = "0.1.0"       # replaced by the artifact model-card versions in load()

    def __init__(self, det_dir: str | Path = DET_DIR, ocr_dir: str | Path = OCR_DIR, det_conf: float = 0.3, mode: str = "auto",
                 orient: bool = True, vote: bool = True, vote_method: str = "pooled", pad: float = 0.04, max_plates: int = 8,
                 use_global: bool = True, min_read_conf: float = 0.6):
        self.det_dir, self.ocr_dir = Path(det_dir), Path(ocr_dir)
        self.det_conf, self.mode, self.orient, self.vote, self.vote_method = det_conf, mode, orient, vote, vote_method
        self.pad, self.max_plates = pad, max_plates
        self.use_global, self.min_read_conf = use_global, min_read_conf
        self.det: OnnxPlateDetector | None = None
        self.reader: PlateReader | None = None
        self._tracks = _PlateTracks()
        self._voter = TrackVoter(method=vote_method)
        self._ocr_regions: list[str] | None = None

    def load(self, device: str = "cuda:0") -> None:
        self.det = OnnxPlateDetector(self.det_dir, device=device, conf=self.det_conf)
        card = json.loads((self.ocr_dir / "model_card.json").read_text())
        try:
            dv, ov = json.loads((self.det_dir / "model_card.json").read_text()).get("version", "0"), card.get("version", "0")
            self.version = ov if dv == ov else f"det {dv} / ocr {ov}"
        except Exception:  # noqa: BLE001
            pass
        ocr = PlateOcrOnnx(self.ocr_dir / "model.onnx", self.ocr_dir / "plate_config.yaml", providers=default_providers(device))
        self._ocr_regions = ocr.regions
        self.reader = PlateReader(ocr, mode=self.mode, orient=self.orient, margin=float(card.get("auto_margin", 6.0)))
        self.global_det = None
        if self.use_global:
            try:  # worldwide plate detector (open-image-models, MIT): catches foreign plates our UZ/CIS-trained detector misses
                from open_image_models import LicensePlateDetector

                self.global_det = LicensePlateDetector(detection_model="yolo-v9-t-384-license-plate-end2end", conf_thresh=0.4,
                                                       providers=default_providers(device))
            except Exception as e:  # noqa: BLE001
                print(f"[ident] global plate detector unavailable: {e}")

    def reset(self) -> None:
        self._tracks = _PlateTracks()
        self._voter = TrackVoter(method=self.vote_method)

    def _crop(self, rgb: np.ndarray, box) -> np.ndarray | None:
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        x1, y1, x2, y2 = (int(max(0, x1 - self.pad * bw)), int(max(0, y1 - self.pad * bh)), int(min(w, x2 + self.pad * bw)), int(min(h, y2 + self.pad * bh)))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return rgb[y1:y2, x1:x2]

    def _green_region(self, crop_upright: np.ndarray) -> bool:
        """Visual EV cue: strongly green left field (region box)."""
        h, w = crop_upright.shape[:2]
        left = crop_upright[int(h * 0.12):int(h * 0.88), int(w * 0.02):int(w * 0.20)]
        if left.size == 0:
            return False
        hsv = cv2.cvtColor(left, cv2.COLOR_RGB2HSV)
        green = (hsv[..., 0] > 45) & (hsv[..., 0] < 90) & (hsv[..., 1] > 90) & (hsv[..., 2] > 50)
        return float(green.mean()) > 0.45

    def predict(self, frames: list[Frame], context: list[FramePred] | None = None) -> list[FramePred]:
        assert self.det is not None and self.reader is not None, "call load() first"
        out: list[FramePred] = []
        for fi, fr in enumerate(frames):
            t0 = time.perf_counter()
            rgb = fr.image
            H, W = rgb.shape[:2]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            raw = list(self.det.detect([bgr])[0][: self.max_plates])
            if self.global_det is not None:
                for g in self.global_det.predict(bgr):
                    b = g.bounding_box
                    gb = (float(g.confidence), float(b.x1), float(b.y1), float(b.x2), float(b.y2))
                    if all(_iou(gb[1:], tuple(d[1:5])) < 0.3 for d in raw):
                        raw.append(gb + ("global",))
            t1 = time.perf_counter()
            crops, keep = [], []
            for d in raw:
                c = self._crop(rgb, d[1:5])
                if c is not None:
                    crops.append(c)
                    keep.append(d)
            readings = self.reader.read(crops) if crops else []
            t2 = time.perf_counter()
            ids = self._tracks.update([d[1:5] for d in keep], fr.index)
            veh = [] if not context else [dd for dd in context[fi].detections if dd.cls in vocab.VEHICLE_CLASSES and dd.track_id is not None]
            fp = FramePred(index=fr.index, t_s=fr.t_s)
            for d, c, r, tid in zip(keep, crops, readings, ids):
                text, conf, fmt, valid, nv = r.text, r.conf, r.format, r.valid, 1
                if self.vote:
                    q = math.sqrt(max((d[3] - d[1]) * (d[4] - d[2]), 1.0))
                    self._voter.add(tid, Observation(r.probs, r.text, r.conf, quality=q, frame=fr.index))
                    v = self._voter.result(tid)
                    if v is not None:
                        text, conf, fmt, valid, nv = v.text, v.conf, v.format, v.valid, v.n_used
                up = np.ascontiguousarray(np.rot90(c, r.rot // 90))
                if valid and fmt in (uz.UZ_PRIVATE, uz.UZ_LEGAL) and self._green_region(up):
                    fmt = uz.UZ_EV
                extra = {"format": fmt, "format_valid": bool(valid), "ocr_conf": round(float(conf), 4), "rot": r.rot, "n_votes": nv,
                         "free_text": r.free_text}
                # Never guess: a reading counts only if confident and a valid UZ format (or a confident foreign read).
                pw, ph = float(d[3] - d[1]), float(d[4] - d[2])
                ratio = pw / max(ph, 1.0)
                if r.rot in (90, 270):  # the reader rotated the crop upright: measure the upright plate
                    ratio = 1.0 / max(ratio, 1e-6)
                # Layout must match the plate's shape: UZ single-row plates are wide (~4.6:1), moto plates are square-ish.
                shape_ok = not ((fmt == uz.UZ_MOTO and ratio > 1.9) or (fmt in (uz.UZ_PRIVATE, uz.UZ_LEGAL, uz.UZ_EV) and ratio < 1.2))
                from_global = len(d) > 5 and d[5] == "global"  # found only by the worldwide detector (likely a foreign plate)
                need = 0.9 if from_global else self.min_read_conf
                readable = float(conf) >= need and shape_ok and (bool(valid) or fmt == uz.FOREIGN)
                extra["detector"] = "global" if from_global else "uz"
                extra["box_ratio"] = round(ratio, 2)
                extra["readable"] = readable
                if not readable:
                    extra["unread_text"], text = text, None
                # visual format head, if the model has one (classes: uz_private/uz_legal/uz_ev/uz_moto/foreign)
                if r.region_probs is not None and self._ocr_regions:
                    j = int(np.argmax(r.region_probs))
                    extra["head_format"] = self._ocr_regions[j]
                    extra["head_format_prob"] = round(float(r.region_probs[j]), 4)
                    if self._ocr_regions[j] == "foreign" and float(r.region_probs[j]) > 0.8:
                        extra["format"], extra["format_valid"] = uz.FOREIGN, False
                    elif self._ocr_regions[j] == uz.UZ_EV and valid and float(r.region_probs[j]) > 0.6:
                        extra["format"] = uz.UZ_EV
                if veh:
                    cx, cy = (d[1] + d[3]) / 2, (d[2] + d[4]) / 2
                    for vd in veh:
                        x1, y1, x2, y2 = vd.bbox
                        if x1 * W <= cx <= x2 * W and y1 * H <= cy <= y2 * H:
                            extra["vehicle_track_id"] = vd.track_id
                            break
                fp.detections.append(Det(cls=vocab.PLATE, confidence=float(d[0]), bbox=(d[1] / W, d[2] / H, d[3] / W, d[4] / H), track_id=int(tid),
                                         text=text, extra=extra))
            if any(d.text for d in fp.detections):  # a readable plate exists: drop unreadable duplicates from the worldwide detector
                fp.detections = [d for d in fp.detections if d.text or d.extra.get("detector") != "global"]
            fp.timings_ms = {"ident_det": (t1 - t0) * 1000, "ident_ocr": (t2 - t1) * 1000}
            out.append(fp)
        return out


def get_predictor(**kw) -> Predictor:
    """Loads from artifacts/ident/{plate_det,plate_ocr}; call ``.load(device)`` (CUDA EP first, CPU fallback)."""
    return IdentPredictor(**kw)
