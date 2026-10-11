"""Lane D predictor: light visual inspection (exterior damage + part, tyre condition, DOT date code).

    from lanes.inspect.predict import get_predictor
    p = get_predictor(); p.load("cuda:0"); preds = p.predict(frames, context)

Emits seen.predictor.Det items (bboxes normalised 0..1 on the FULL frame):
  cls="damage:<dent|scratch|crack|glass_shatter|lamp_broken|tire_flat|other>", part=<car part name>|None,
      extra={"polygon": [[x,y],...] normalised, "part_overlap": float, "area_frac": float}
  cls="part:<name>" only if emit_parts=True (needs a vocab addition from the integrator)
  cls="tyre:<good|defective>"   (default; 'worn' only with emit_worn=True: experimental, its labels come from ONE small source)
  cls="dot", text="WWYY" only when the reading passes the week/year check, extra={"raw", "week", "year"}; only on close-up frames
      that are framed on the code (character model run on the whole frame). Full-photo DOT localisation is NOT supported.
Vehicle crops come from ``context`` (largest car/truck/bus detections); with ``context=None`` the whole frame is used. If a
context is given and no vehicle is in it, damage/parts are skipped (no vehicle -> no inspection); tyre/DOT still run on
close-up frames (context scene "closeup" >= 0.5) and on `wheel` detections that are at least ``min_wheel_px`` large.

Every model is loaded from artifacts/inspect/<name>/{model.onnx, model_card.json}. A missing artifact raises FileNotFoundError:
this predictor never fabricates output.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from lanes.inspect import dot as DOT
from lanes.inspect.assign import box_to_poly  # noqa: F401  (re-export for callers)
from lanes.inspect.onnx_yolo import YoloOnnx, mask_to_polygon
from seen.paths import ARTIFACTS
from seen.predictor import Det, Frame, FramePred, empty_preds
from seen.vocab import VEHICLE_CLASSES, WHEEL

ART = ARTIFACTS / "inspect"
PART_FAMILY_FALLBACK = None


def _card(name: str) -> dict:
    p = ART / name / "model_card.json"
    if not p.exists():
        raise FileNotFoundError(f"missing artifact {p} (run lanes.inspect.export first)")
    return json.loads(p.read_text())


def _onnx(name: str) -> Path:
    p = ART / name / "model.onnx"
    if not p.exists():
        raise FileNotFoundError(f"missing artifact {p}")
    return p


def _crop(img: np.ndarray, box, margin: float = 0.05) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    mw, mh = (x2 - x1) * margin, (y2 - y1) * margin
    a, b = max(0, int((x1 - mw) * w)), max(0, int((y1 - mh) * h))
    c, d = min(w, int((x2 + mw) * w) + 1), min(h, int((y2 + mh) * h) + 1)
    return img[b:d, a:c], (a, b)


class InspectPredictor:
    name = "seen-inspect"          # damage + part, tyre condition, DOT date code
    role = "inspection"            # docs/LANES.md "Model naming in events"
    version = "0.1.0"              # bumped with the artifact set; per-model versions in self.model_versions after load()

    def __init__(self, damage_conf: float = 0.25, part_conf: float = 0.30, tyre_threshold: float = 0.5, emit_parts: bool = False,
                 run_damage: bool = True, run_tyre: bool = True, run_dot: bool = True, min_wheel_px: int = 160, max_vehicles: int = 2,
                 emit_worn: bool = False, use_dot_region: bool = False,
                 artifacts: str | Path | None = None):
        global ART
        if artifacts:
            ART = Path(artifacts)
        self.damage_conf, self.part_conf, self.tyre_threshold = damage_conf, part_conf, tyre_threshold
        self.emit_parts, self.run_damage, self.run_tyre, self.run_dot = emit_parts, run_damage, run_tyre, run_dot
        self.min_wheel_px, self.max_vehicles, self.emit_worn = min_wheel_px, max_vehicles, emit_worn
        self.use_dot_region = use_dot_region  # the region detector FAILED on an independent set (AP50 ~ 0): opt-in only
        self.loaded = False

    # ------------------------------------------------------------------------------------------------ loading
    def _required(self) -> list[str]:
        r = []
        if self.run_damage:
            r += ["damage_seg", "parts_seg"]
        if self.run_tyre:
            r += ["tyre_cond"]
        if self.run_dot:
            r += (["dot_region"] if self.use_dot_region else []) + ["dot_chars"]
        return r

    def load(self, device: str = "cuda:0") -> None:
        for n in self._required():  # fail fast, before importing onnxruntime
            _card(n)
            _onnx(n)
        import onnxruntime as ort
        prov = ["CPUExecutionProvider"]
        if device.startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            idx = int(device.split(":")[1]) if ":" in device else 0
            prov = [("CUDAExecutionProvider", {"device_id": idx}), "CPUExecutionProvider"]
        self.providers = prov
        self.models: dict = {}
        if self.run_damage:
            c = _card("damage_seg")
            self.models["damage"] = YoloOnnx(_onnx("damage_seg"), c["classes"], c["input_size"], prov, "seg")
            c = _card("parts_seg")
            self.models["parts"] = YoloOnnx(_onnx("parts_seg"), c["classes"], c["input_size"], prov, "seg")
        if self.run_tyre:
            c = _card("tyre_cond")
            import onnxruntime as ort2
            so = ort2.SessionOptions(); so.log_severity_level = 3
            self.models["tyre"] = ort2.InferenceSession(str(_onnx("tyre_cond")), so, providers=prov)
            self.tyre_size = int(c["input_size"])
            self.tyre_labels = c["classes"]  # ["good", "defective"]
            self.tyre_labels3 = c.get("classes3") or ["good", "worn", "defective"]
        if self.run_dot:
            if self.use_dot_region:
                c = _card("dot_region")
                self.models["dot_region"] = YoloOnnx(_onnx("dot_region"), c["classes"], c["input_size"], prov, "det")
            c = _card("dot_chars")
            self.models["dot_chars"] = YoloOnnx(_onnx("dot_chars"), c["classes"], c["input_size"], prov, "det")
        self.model_versions = {n: _card(n).get("version", "?") for n in self._required()}
        self.loaded = True

    # ------------------------------------------------------------------------------------------------ helpers
    def _vehicles(self, fp: FramePred | None):
        if fp is None:
            return None
        v = [d for d in fp.detections if d.cls in VEHICLE_CLASSES and d.confidence >= 0.3]
        v.sort(key=lambda d: -(d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
        return v[: self.max_vehicles]

    def _tyre_prob(self, crop: np.ndarray) -> tuple[float, np.ndarray | None]:
        from lanes.inspect.tyre_features import MEAN, STD
        import cv2
        s = self.tyre_size
        h, w = crop.shape[:2]
        r = s / min(h, w)
        im = cv2.resize(crop, (max(s, round(w * r)), max(s, round(h * r))), interpolation=cv2.INTER_CUBIC)
        t, l = (im.shape[0] - s) // 2, (im.shape[1] - s) // 2
        im = im[t:t + s, l:l + s].astype(np.float32) / 255.0
        x = ((im - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        outs = self.models["tyre"].run(None, {"input": x})
        return float(outs[0].reshape(-1)[0]), (outs[1].reshape(-1) if len(outs) > 1 else None)

    def _read_dot(self, crop: np.ndarray):
        """DOT reading. Default: the frame IS the code region (close-up framed on the code), run the character model directly.
        With use_dot_region=True a region detector localises it first (not validated cross-source)."""
        if not self.use_dot_region:
            det = [{"box": [0, 0, crop.shape[1], crop.shape[0]], "score": 1.0}]
        else:
            det, _ = self.models["dot_region"](crop, conf=0.4, iou=0.5, max_det=3, mask_canvas=None)
        out = []
        for d in det:
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            px, py = int(0.05 * (x2 - x1)), int(0.05 * (y2 - y1))
            sub = crop[max(0, y1 - py): y2 + py, max(0, x1 - px): x2 + px]
            if sub.size == 0:
                continue
            chars, _ = self.models["dot_chars"](sub, conf=0.25, iou=0.5, max_det=60, mask_canvas=None)
            if not chars:
                continue
            boxes = np.array([c["box"] for c in chars])
            s = DOT.assemble(boxes, [c["name"] for c in chars])
            code = DOT.date_code(s)
            v = DOT.validate_wwyy(code)
            out.append((d, s, code, v, float(np.mean([c["score"] for c in chars]))))
        return out

    # ------------------------------------------------------------------------------------------------ main
    def predict(self, frames: list[Frame], context: list[FramePred] | None = None) -> list[FramePred]:
        if not self.loaded:
            raise RuntimeError("call load() first")
        preds = empty_preds(frames)
        for k, fr in enumerate(frames):
            ctx = context[k] if context else None
            img = fr.image
            H, W = img.shape[:2]
            t = {}
            veh = self._vehicles(ctx)
            regions = [(0.0, 0.0, 1.0, 1.0)] if veh is None else [tuple(v.bbox) for v in veh]
            sc = ctx.scene if ctx is not None else {}
            closeup = bool(sc) and (sc.get("closeup", 0.0) >= 0.5 or max(sc, key=sc.get) == "closeup")
            # ------------------------------------------------------------------ damage + part
            if self.run_damage:
                t0 = time.perf_counter()
                for box in regions:
                    crop, (ox, oy) = _crop(img, box)
                    if crop.size == 0:
                        continue
                    ch, cw = crop.shape[:2]
                    dets, (mh, mw) = self.models["damage"](crop, conf=self.damage_conf, iou=0.6, max_det=50, mask_canvas=384)
                    pdets, _ = self.models["parts"](crop, conf=self.part_conf, iou=0.6, max_det=40, mask_canvas=384)
                    pm = [p["mask"] for p in pdets]
                    for d in dets:
                        part, frac = None, 0.0
                        if d["mask"] is not None and pm:
                            dm = d["mask"]
                            inter = np.array([float((dm & m).sum()) for m in pm])
                            sc = np.array([p["score"] for p in pdets])
                            j = int(np.argmax(inter + 1e-3 * sc))
                            frac = inter[j] / max(float(dm.sum()), 1.0)
                            part = pdets[j]["name"] if frac >= 0.10 else None
                        x1, y1, x2, y2 = d["box"]
                        bbox = ((ox + x1) / W, (oy + y1) / H, (ox + x2) / W, (oy + y2) / H)
                        poly = []
                        if d["mask"] is not None:
                            poly = mask_to_polygon(d["mask"], cw / mw, ch / mh, eps=1.0)
                            poly = [[(ox + x) / W, (oy + y) / H] for x, y in poly][:60]
                        preds[k].detections.append(Det(
                            cls=f"damage:{d['name']}", confidence=d["score"], bbox=tuple(float(v) for v in bbox), part=part,
                            extra={"polygon": poly, "part_overlap": round(float(frac), 3),
                                   "area_frac": round(float(d["mask"].sum()) / (mh * mw), 5) if d["mask"] is not None else None}))
                    if self.emit_parts:
                        for p in pdets:
                            x1, y1, x2, y2 = p["box"]
                            preds[k].detections.append(Det(cls=f"part:{p['name']}", confidence=p["score"],
                                                           bbox=((ox + x1) / W, (oy + y1) / H, (ox + x2) / W, (oy + y2) / H)))
                t["inspect.damage_ms"] = (time.perf_counter() - t0) * 1000
            # ------------------------------------------------------------------ tyre + DOT
            tyre_regions = []
            if closeup:
                tyre_regions.append(((0.0, 0.0, 1.0, 1.0), "closeup"))
            if ctx is not None:
                for d in ctx.detections:
                    if d.cls == WHEEL and d.confidence >= 0.3:
                        bw, bh = (d.bbox[2] - d.bbox[0]) * W, (d.bbox[3] - d.bbox[1]) * H
                        if min(bw, bh) >= self.min_wheel_px:
                            tyre_regions.append((tuple(d.bbox), "wheel_crop"))
            if (self.run_tyre or self.run_dot) and tyre_regions:
                t0 = time.perf_counter()
                for box, kind in tyre_regions:
                    crop, (ox, oy) = _crop(img, box, 0.08 if kind == "wheel_crop" else 0.0)
                    if crop.size == 0:
                        continue
                    warn = {"warning": "wheel crop from a whole-vehicle photo; condition model validated only on tyre close-ups"} if kind == "wheel_crop" else {}
                    if self.run_tyre:
                        p, p3 = self._tyre_prob(crop)
                        if self.emit_worn and p3 is not None:  # experimental 3-way head (worn labels from ONE source)
                            j = int(np.argmax(p3))
                            lab, conf = self.tyre_labels3[j], float(p3[j])
                        else:
                            lab = self.tyre_labels[1] if p >= self.tyre_threshold else self.tyre_labels[0]
                            conf = max(p, 1 - p)
                        preds[k].detections.append(Det(cls=f"tyre:{lab}", confidence=conf, bbox=tuple(float(v) for v in box),
                                                       extra={"p_defective": round(p, 4), "source": kind, **warn}))
                    if self.run_dot and (kind == "closeup" or self.use_dot_region):
                        for d, s, code, v, conf in self._read_dot(crop):
                            if v["valid"]:
                                x1, y1, x2, y2 = d["box"]
                                preds[k].detections.append(Det(
                                    cls="dot", confidence=min(conf, d["score"]), text=code,
                                    bbox=((ox + x1) / W, (oy + y1) / H, (ox + x2) / W, (oy + y2) / H),
                                    extra={"raw": s, "week": v["week"], "year": v["year"], "source": kind, **warn}))
                t["inspect.tyre_dot_ms"] = (time.perf_counter() - t0) * 1000
            preds[k].timings_ms.update(t)
        return preds


def get_predictor(**kw) -> InspectPredictor:
    return InspectPredictor(**kw)
