"""Lane C bay-state predictor: per-frame probabilities for seen.vocab.BAY_STATES.

Design (frozen embeddings + detector rules + linear heads; no video model):
  * context (merged detector output) gives vehicle/person boxes; the "primary" vehicle is the one inside the bay ROI
    (largest-overlap) or, without an ROI, the largest one.
  * car_present      = confidence of the primary vehicle (detector rule; 0 if none).
  * person_at_car    = learned head on [crop embedding, geometry]; person_under_car = learned head x raised gate.
  * raised, wheel_off, hood_open, work, washing = linear head on a frozen DINOv2-L crop embedding (car box + 25 %
    margin) concatenated with detector geometry and part-detector cues (wheel count, hood open/closed boxes).
  * Fixed-camera lift rule (multi-frame input only): if the car's bottom edge rises > 25 % of its height above its
    lowest position in the window while staying in place, `raised` is pushed up (noisy-OR with the head).
  * idle = car_present x (1 - work).
Probabilities are per frame and Platt-calibrated on a held-out handheld split; temporal smoothing is Lane E's HMM.

    from lanes.ops.predict import get_predictor
    p = get_predictor(roi=(0.3, 0.1, 0.7, 0.8)); p.load("cuda:0"); preds = p.predict(frames, context)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from lanes.ops.geometry import GEOM_NAMES, VEH, crop_box, geom_features, primary_vehicle
from lanes.ops.rules import apply_rules
from seen.paths import ARTIFACTS
from seen.predictor import Frame, FramePred, empty_preds

ART = ARTIFACTS / "ops" / "states"


class OpsStatePredictor:
    name = "ops-bay-states"
    role = "states"
    version = "0.1"  # replaced by the model card version on load()

    def __init__(self, artifact_dir: str | Path | None = None, roi=None, use_parts: bool = True, fixed_camera: bool = True,
                 batch: int = 16):
        self.dir = Path(artifact_dir) if artifact_dir else ART
        self.roi = tuple(roi) if roi is not None else None
        self.use_parts = use_parts
        self.fixed_camera = fixed_camera
        self.batch = batch
        self.card: dict = {}

    # ------------------------------------------------------------------------------------------------ loading
    def load(self, device: str = "cuda:0") -> None:
        import torch

        self.card = json.loads((self.dir / "model_card.json").read_text())
        self.name = self.card.get("name", self.name)
        self.version = str(self.card.get("version", self.version))
        self.device = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
        self.states = self.card["head_states"]
        self.thresholds = self.card.get("thresholds", {})
        # linear head (ONNX if onnxruntime is present, else the npz)
        self._ort = None
        try:
            import onnxruntime as ort

            prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device != "cpu" else ["CPUExecutionProvider"]
            self._ort = ort.InferenceSession(str(self.dir / "head.onnx"), providers=prov)
        except Exception:  # noqa: BLE001
            from lanes.ops.heads import HeadModel

            self._head = HeadModel.load(self.dir / "head.npz")
        # backbone
        from lanes.ops.backbones import load_backbone

        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        local = self.dir / "backbone_hf"
        self.bb = load_backbone(self.card["backbone"], self.device, dtype=dtype, path=str(local) if local.exists() else None)
        # part detectors (wheel / hood); optional
        self.parts = []
        if self.use_parts:
            from ultralytics import YOLO

            for key, keep in (("parts_detector", {"wheel"}), ("hood_detector", {"hood_open", "hood_closed"})):
                w = self.dir / self.card.get(key, "")
                if self.card.get(key) and w.exists():
                    self.parts.append((YOLO(str(w)), keep))

    # --------------------------------------------------------------------------------------------- inference
    def _part_dets(self, images: list[np.ndarray]) -> list[list[tuple[str, float, tuple]]]:
        out = [[] for _ in images]
        for model, keep in self.parts:
            res = model.predict([im[:, :, ::-1] for im in images], conf=0.2, imgsz=640, verbose=False,
                                device=0 if self.device.startswith("cuda") else "cpu", half=self.device != "cpu")
            for i, r in enumerate(res):
                h, w = r.orig_shape
                for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                    n = model.names[int(k)]
                    if n in keep:
                        out[i].append((n, float(c), (x1 / w, y1 / h, x2 / w, y2 / h)))
        return out

    def _crop(self, image: np.ndarray, dets):
        from PIL import Image

        h, w = image.shape[:2]
        pv = primary_vehicle([(c, b) for k, c, b in dets if k in VEH], self.roi)
        if pv is None:
            box = crop_box(self.roi, 0.0, w / h) if self.roi is not None else (0.0, 0.0, 1.0, 1.0)
        else:
            box = crop_box(pv[1], 0.25, w / h)
        x1, y1, x2, y2 = box
        im = Image.fromarray(image).crop((int(x1 * w), int(y1 * h), max(int(x2 * w), int(x1 * w) + 2), max(int(y2 * h), int(y1 * h) + 2)))
        return im, pv

    def _head_probs(self, X: np.ndarray) -> np.ndarray:
        if self._ort is not None:
            return self._ort.run(None, {"features": X.astype(np.float32)})[0]
        return self._head.predict_proba(X)

    def predict(self, frames: list[Frame], context: list[FramePred] | None = None) -> list[FramePred]:
        import torch

        out = empty_preds(frames)
        if not frames:
            return out
        ctx = context or empty_preds(frames)
        dets_all = [[(d.cls, float(d.confidence), tuple(d.bbox)) for d in c.detections] for c in ctx]
        for s in range(0, len(frames), self.batch):
            fr = frames[s:s + self.batch]
            dets = dets_all[s:s + self.batch]
            t0 = time.perf_counter()
            parts = self._part_dets([f.image for f in fr]) if self.parts else [None] * len(fr)
            t1 = time.perf_counter()
            crops, pvs = zip(*[self._crop(f.image, d) for f, d in zip(fr, dets)])
            x = torch.stack([self.bb.preprocess(c) for c in crops]).to(self.device)
            with torch.inference_mode():
                emb = self.bb.forward(x).float().cpu().numpy()
            t2 = time.perf_counter()
            G = np.stack([geom_features(d, self.roi, p) for d, p in zip(dets, parts)])
            P = self._head_probs(np.concatenate([emb, G], 1))
            t3 = time.perf_counter()
            ruled = apply_rules({k: P[:, j] for j, k in enumerate(self.states)},
                                np.array([pv[0] if pv is not None else 0.0 for pv in pvs]))
            for i, f in enumerate(fr):
                st = {k: float(v[i]) for k, v in ruled.items()}
                o = out[s + i]
                o.states = st
                n = len(fr)
                o.timings_ms.update({"ops_parts": (t1 - t0) * 1000 / n, "ops_embed": (t2 - t1) * 1000 / n,
                                     "ops_head": (t3 - t2) * 1000 / n})
        if self.fixed_camera and len(frames) > 2:
            self._lift_rule(out, dets_all)
        return out

    def _lift_rule(self, out: list[FramePred], dets_all) -> None:
        """Fixed camera: car bottom edge well above its own lowest position, same x-extent -> raised."""
        boxes = []
        for d in dets_all:
            pv = primary_vehicle([(c, b) for k, c, b in d if k in VEH], self.roi)
            boxes.append(pv[1] if pv else None)
        valid = [b for b in boxes if b is not None]
        if len(valid) < 3:
            return
        for i, b in enumerate(boxes):
            if b is None:
                continue
            same = [v for v in valid if abs((v[0] + v[2]) / 2 - (b[0] + b[2]) / 2) < 0.1 * (b[2] - b[0])
                    and abs((v[2] - v[0]) - (b[2] - b[0])) < 0.15 * (b[2] - b[0])]
            if len(same) < 3:
                continue
            ground = max(v[3] for v in same)
            lift = (ground - b[3]) / max(b[3] - b[1], 1e-6)
            rule = float(np.clip((lift - 0.15) / 0.2, 0.0, 1.0))
            p = out[i].states.get("raised", 0.0)
            out[i].states["raised"] = 1 - (1 - p) * (1 - 0.9 * rule)


def get_predictor(**kw) -> OpsStatePredictor:
    return OpsStatePredictor(**kw)
