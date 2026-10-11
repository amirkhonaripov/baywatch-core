"""ONNX plate-OCR runner exposing full per-slot probabilities (needed for constrained decoding and track voting).

Works with any fast-plate-ocr export (pretrained hub models and ours): ``model.onnx`` + ``plate_config.yaml``.
Input: uint8 RGB crops of any size (resized to the config's size, no letter-boxing unless the config says so).
Output per crop: ``probs`` (slots, vocab) float32 softmax rows and, when the model has a region/format head,
``region_probs`` (n_regions,).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml

_INTERP = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA,
           "lanczos4": cv2.INTER_LANCZOS4}


def default_providers(device: str = "cuda:0") -> list:
    """CUDA EP first (when requested and available), CPU fallback. TensorRT libs are not installed on the box."""
    avail = ort.get_available_providers()
    if device.startswith("cuda") and hasattr(ort, "preload_dlls"):
        try:  # CUDA 13 / cuDNN 9 shipped as pip wheels (nvidia/cu13/lib, nvidia/cudnn/lib) are not on LD_LIBRARY_PATH
            ort.preload_dlls()
        except Exception:  # noqa: BLE001
            pass
    out: list = []
    if device.startswith("cuda") and "CUDAExecutionProvider" in avail:
        idx = int(device.split(":")[1]) if ":" in device else 0
        out.append(("CUDAExecutionProvider", {"device_id": idx}))
    out.append("CPUExecutionProvider")
    return out


class PlateOcrOnnx:
    def __init__(self, model_path: str | Path, config_path: str | Path, providers: list | None = None,
                 sess_options: ort.SessionOptions | None = None):
        self.cfg = yaml.safe_load(Path(config_path).read_text())
        self.alphabet: str = self.cfg["alphabet"]
        self.slots: int = int(self.cfg["max_plate_slots"])
        self.pad: str = self.cfg["pad_char"]
        self.h, self.w = int(self.cfg["img_height"]), int(self.cfg["img_width"])
        self.rgb = self.cfg.get("image_color_mode", "grayscale") == "rgb"
        self.keep_ar = bool(self.cfg.get("keep_aspect_ratio", False))
        self.interp = _INTERP[self.cfg.get("interpolation", "linear")]
        self.pad_color = self.cfg.get("padding_color", (114, 114, 114))
        self.regions: list[str] | None = self.cfg.get("plate_regions")
        so = sess_options or ort.SessionOptions()
        self.sess = ort.InferenceSession(str(model_path), providers=providers or default_providers("cpu"), sess_options=so)
        outs = self.sess.get_outputs()
        names = [o.name for o in outs]
        # keras->ONNX (torch backend) exports name the outputs softmax_N: identify by shape (plate = rank 3, format head = rank 2)
        self.plate_out = "plate" if "plate" in names else next((o.name for o in outs if len(o.shape) == 3), names[0])
        self.region_out = "region" if "region" in names else next((o.name for o in outs if len(o.shape) == 2), None)
        self.in_name = self.sess.get_inputs()[0].name
        b = self.sess.get_inputs()[0].shape[0]
        self.static_b: int | None = b if isinstance(b, int) else None   # keras->ONNX (torch backend) exports have a fixed batch

    # ------------------------------------------------------------------------------------------------ preprocess
    def _prep(self, img: np.ndarray) -> np.ndarray:
        """img: HxWx3 uint8 RGB (or HxW gray)."""
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        if not self.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if self.keep_ar:
            oh, ow = img.shape[:2]
            r = min(self.h / oh, self.w / ow)
            nw, nh = round(ow * r), round(oh * r)
            img = cv2.resize(img, (nw, nh), interpolation=self.interp)
            dw, dh = (self.w - nw) / 2, (self.h - nh) / 2
            t, b, l, rr = round(dh - 0.1), round(dh + 0.1), round(dw - 0.1), round(dw + 0.1)
            pc = self.pad_color
            val = tuple(int(c) for c in pc) if self.rgb else int(pc[0] if isinstance(pc, (list, tuple)) else pc)
            img = cv2.copyMakeBorder(img, t, b, l, rr, cv2.BORDER_CONSTANT, value=val)
        else:
            img = cv2.resize(img, (self.w, self.h), interpolation=self.interp)
        if img.ndim == 2:
            img = img[..., None]
        return img.astype(np.uint8)

    # ----------------------------------------------------------------------------------------------------- run
    def probs(self, crops: list[np.ndarray], batch: int = 128) -> tuple[np.ndarray, np.ndarray | None]:
        """Return (probs[N, slots, vocab], region_probs[N, R] | None). Chunks by the model's static batch size when it has one."""
        if not crops:
            return np.zeros((0, self.slots, len(self.alphabet)), np.float32), None
        bs = self.static_b or batch
        pp, rr = [], []
        names = [self.plate_out] + ([self.region_out] if self.region_out else [])
        for i in range(0, len(crops), bs):
            chunk = crops[i:i + bs]
            n = len(chunk)
            x = np.stack([self._prep(c) for c in chunk])
            if self.static_b and n < bs:  # pad the last chunk by repeating the last crop
                x = np.concatenate([x, np.repeat(x[-1:], bs - n, 0)])
            out = self.sess.run(names, {self.in_name: x})
            pp.append(out[0][:n].reshape(n, self.slots, len(self.alphabet)))
            if self.region_out:
                rr.append(out[1][:n])
        return np.concatenate(pp).astype(np.float32), (np.concatenate(rr).astype(np.float32) if rr else None)

    def decode(self, probs: np.ndarray) -> list[tuple[str, float]]:
        """Unconstrained arg-max decode -> [(text without trailing pad, mean char prob)]."""
        idx = probs.argmax(-1)
        mx = probs.max(-1)
        res = []
        for ids, m in zip(idx, mx):
            s = "".join(self.alphabet[i] for i in ids).rstrip(self.pad)
            n = max(len(s), 1)
            res.append((s, float(m[:n].mean())))
        return res

    def read(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        p, _ = self.probs(crops)
        return self.decode(p)
