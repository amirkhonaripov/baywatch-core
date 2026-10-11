"""Lane B predictors: vehicle + person detector (optional tracker) and the scene router.

    from lanes.flow.predict import get_predictor, get_router
    det = get_predictor(model="rfdetr-base", tracker="ocsort")   # Predictor: fills FramePred.detections (+ Det.track_id)
    rt = get_router()                                            # Predictor: fills FramePred.scene

``model`` is a registry name (``lanes.flow.detectors.get_detector``: rfdetr-base|medium|large, yolo11x, yolo26x, ...) or the name of a
folder under ``$ARTIFACTS/flow/`` that holds a fine-tuned checkpoint described by ``model_card.json`` ({"weights": "...", "size": "base",
"classes": [...]}). ``runtime="onnx"`` uses ``model.onnx`` from the same folder with ONNX Runtime (CUDA EP, CPU fallback).
The tracker keeps state across ``predict`` calls (a stream); call ``reset()`` to start a new video.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from lanes.flow.common import CLASSES, artifact_dir
from seen.paths import ARTIFACTS
from seen.predictor import Det, Frame, FramePred, empty_preds

CLASS_NAMES = CLASSES  # person, car, motorcycle, bus, truck (all in seen.vocab)


PRETTY = {"rfdetr-nano": "RF-DETR Nano", "rfdetr-small": "RF-DETR Small", "rfdetr-medium": "RF-DETR Medium", "rfdetr-base": "RF-DETR Base",
          "rfdetr-large": "RF-DETR Large", "yolo11x": "YOLO11x", "yolo26x": "YOLO26x"}


class FlowDetector:
    """Vehicle + person detector with optional tracking. Implements ``seen.predictor.Predictor``."""

    role = "vehicle"

    def __init__(self, model: str = "rfdetr-base", tracker: str | None = None, conf: float = 0.35, runtime: str = "torch",
                 track_fps: float | None = None, backend=None):
        self.model_spec, self.tracker_name, self.conf, self.runtime, self.track_fps = model, tracker, conf, runtime, track_fps
        self.name = PRETTY.get(model, model) + (" (COCO, zero-shot)" if model in PRETTY else "")
        self.version = f"{model}-pretrained/{runtime}" + (f"+{tracker}" if tracker else "")
        self._backend = backend  # injectable for tests (anything with .infer(images, conf) -> list[(N,6)])
        self._tracker = None

    def _resolve(self):
        from lanes.flow.detectors import get_detector

        card = ARTIFACTS / "flow" / self.model_spec / "model_card.json"
        if card.exists():
            mc = json.loads(card.read_text())
            w = str(card.parent / mc["weights"])
            if w.endswith(".onnx") and (card.parent / "checkpoint.pth").exists():  # torch runtime needs the .pth
                w = str(card.parent / "checkpoint.pth")
            spec = f"rfdetr-{mc.get('size', 'base')}@{w}" + (f"#{mc['resolution']}" if mc.get("resolution") else "")
            self.name = mc.get("name", self.model_spec)
            self.version = f"{mc.get('name', self.model_spec)}-v{mc.get('version', '1')}/{self.runtime}" + ("+" + self.tracker_name if self.tracker_name else "")
            return get_detector(spec)
        return get_detector(self.model_spec)

    def load(self, device: str = "cuda:0") -> None:
        if self._backend is None:
            if self.runtime == "onnx":
                from lanes.flow.onnx_runtime import OnnxRFDETR

                self._backend = OnnxRFDETR(ARTIFACTS / "flow" / self.model_spec / "model.onnx")
                card = ARTIFACTS / "flow" / self.model_spec / "model_card.json"
                if card.exists():  # name/version from the model card, not the "pretrained" default
                    mc = json.loads(card.read_text())
                    self.name = mc.get("name", self.model_spec)
                    self.version = f"{self.name}-v{mc.get('version', '1')}/onnx" + ("+" + self.tracker_name if self.tracker_name else "")
            else:
                self._backend = self._resolve()
            self._backend.load(device)

    def reset(self) -> None:
        if self._tracker is not None:
            self._tracker.reset()

    def predict(self, frames: list[Frame], context: list[FramePred] | None = None) -> list[FramePred]:
        out = empty_preds(frames)
        if not frames:
            return out
        t0 = time.perf_counter()
        raw = self._backend.infer([f.image for f in frames], min(self.conf, 0.3))
        dt = (time.perf_counter() - t0) * 1000 / len(frames)
        for f, p, d in zip(frames, out, raw):
            h, w = f.image.shape[:2]
            for x1, y1, x2, y2, s, c in d:
                if s >= self.conf:
                    p.detections.append(Det(cls=CLASS_NAMES[int(c)], confidence=float(s),
                                            bbox=(float(np.clip(x1 / w, 0, 1)), float(np.clip(y1 / h, 0, 1)),
                                                  float(np.clip(x2 / w, 0, 1)), float(np.clip(y2 / h, 0, 1)))))
            p.timings_ms["vehicle_detector"] = dt
        if self.tracker_name:
            from lanes.flow.tracker import make_tracker, track_predictions

            if self._tracker is None:
                fps = self.track_fps
                if fps is None and len(frames) > 1:
                    d = np.diff([f.t_s for f in frames])
                    fps = 1.0 / max(float(np.median(d[d > 0])), 1e-3) if (d > 0).any() else 5.0
                self._tracker = make_tracker(self.tracker_name, fps=fps or 5.0, det_thresh=self.conf)
            else:  # keep state across calls: do not reset
                pass
            self._track_stream(out)
        return out

    def _track_stream(self, preds: list[FramePred]) -> None:
        from lanes.flow.tracker import iou_batch

        S = 1000.0
        for p in preds:
            idx = [i for i, d in enumerate(p.detections) if d.cls in ("car", "truck", "bus", "motorcycle")]
            arr = np.array([[*(np.asarray(p.detections[i].bbox) * S), p.detections[i].confidence] for i in idx]).reshape(-1, 5)
            res = self._tracker.update(arr, p.t_s)
            if len(res) and len(idx):
                iou = iou_batch(arr[:, :4], res[:, :4])
                for k, i in enumerate(idx):
                    j = int(iou[k].argmax())
                    if iou[k, j] > 0.9:
                        p.detections[i].track_id = int(res[j, 4])


def get_predictor(model: str | None = None, tracker: str | None = None, **kw) -> FlowDetector:
    """Default = the shipped fine-tuned detector ``artifacts/flow/rfdetr-large-flow`` when present, else zero-shot RF-DETR Large."""
    if model is None:
        model = "rfdetr-large-flow" if (ARTIFACTS / "flow" / "rfdetr-large-flow" / "model_card.json").exists() else "rfdetr-large"
    return FlowDetector(model=model, tracker=tracker, **kw)


def get_router(mode: str = "blend", encoder: str | None = None, probe_path: str | None = None, **kw):
    """Scene router. mode: 'blend' (default: probe/zero-shot mix with the leave-one-source-out weight), 'probe', 'zeroshot' (no training)."""
    from lanes.flow.router_model import DEFAULT_MODEL, SceneRouter

    here = Path(__file__).resolve().parent / "assets"
    art = ARTIFACTS / "flow" / "router" / "probe.npz"
    probe = Path(probe_path) if probe_path else (art if art.exists() else here / "router_probe.npz")  # artifacts first, repo copy as fallback
    card = probe.with_suffix(".json")
    meta = json.loads(card.read_text()) if card.exists() else {}
    if encoder is None:
        encoder = meta.get("encoder")
    kw.setdefault("blend", meta.get("blend_probe_weight", 0.5))
    kw.setdefault("prompt_set", meta.get("prompt_set", "v2"))
    return SceneRouter(encoder_name=encoder or DEFAULT_MODEL, probe_path=str(probe), mode=mode, **kw)
