"""Cascade state predictor: fast heads on every frame, Qwen3-VL on uncertain frames + a periodic check.

Measured on the hand-labelled CCTV set (eval/cascade.py): escalating ~30% of frames lifts car_present 0.92->0.96,
hood_open 0.82->0.95, wheel_off 0.78->0.86 accuracy; 'work' and 'person_at_car' need most frames escalated (0.62->0.78
at 50%). For 'raised' and 'person_under_car' the fast heads are as good or better, so they are never overridden.

The VLM is pluggable: ``LocalQwen`` loads Qwen3-VL in-process (box); on Modal the serving layer passes a client whose
``answer(images)`` calls a separate GPU class. Each frame's states keep a provenance flag in FramePred.timings_ms
("vlm_frame": 1.0) and the per-call latency, so results can show which frames the VLM decided.
"""
from __future__ import annotations

import io
import time
from typing import Protocol

import numpy as np
from PIL import Image

from seen.predictor import Frame, FramePred
from seen.vocab import VEHICLE_CLASSES

VLM_STATES = ("car_present", "hood_open", "wheel_off", "person_at_car", "work", "washing")  # VLM better on CCTV
FAST_ONLY = ("raised", "person_under_car")  # fast heads as good or better on CCTV


class VLMClient(Protocol):
    name: str
    version: str

    def answer(self, images: list[np.ndarray]) -> list[dict[str, float]]: ...


def crop_main_vehicle(img: np.ndarray, pred: FramePred | None, roi=None, margin: float = 0.15) -> np.ndarray:
    h, w = img.shape[:2]
    box = None
    if pred is not None:
        vs = [d for d in pred.detections if d.cls in VEHICLE_CLASSES]
        if vs:
            box = max(vs, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])).bbox
    if box is None and roi is not None:
        box = roi
    if box is None:
        return img
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    a, b = int(max(0, x1 - dx) * w), int(max(0, y1 - dy) * h)
    c, d = int(min(1, x2 + dx) * w), int(min(1, y2 + dy) * h)
    return img[b:d, a:c] if (c - a) > 16 and (d - b) > 16 else img


class LocalQwen:
    """In-process Qwen3-VL (transformers, bf16). Same prompt and yes/no-probability parsing as lanes/ops/vlm.py."""

    name = "qwen3-vl-8b-zeroshot"
    version = "Qwen3-VL-8B-Instruct"

    def __init__(self, model: str = "Qwen/Qwen3-VL-8B-Instruct", device: str = "cuda:0"):
        self.model_id, self.device = model, device
        self.model = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.proc = AutoProcessor.from_pretrained(self.model_id, min_pixels=256 * 256, max_pixels=640 * 640)
        self.proc.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, torch_dtype=torch.bfloat16,
                                                                 device_map=self.device).eval()
        tok = self.proc.tokenizer
        self.yes_ids = {i for t in ("yes", " yes", "Yes", " Yes") for i in tok.encode(t, add_special_tokens=False)[:1]}
        self.no_ids = {i for t in ("no", " no", "No", " No") for i in tok.encode(t, add_special_tokens=False)[:1]}

    def answer(self, images: list[np.ndarray]) -> list[dict[str, float]]:
        import torch

        from lanes.ops.vlm import PROMPT, _hf_parse

        if self.model is None:
            self.load()
        ims = []
        for im in images:
            p = Image.fromarray(im)
            s = 448 / min(p.size)
            if s > 1:
                p = p.resize((int(p.size[0] * s), int(p.size[1] * s)), Image.Resampling.BICUBIC)
            ims.append(p)
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        texts = [self.proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)] * len(ims)
        enc = self.proc(text=texts, images=ims, return_tensors="pt", padding=True).to(self.device)
        with torch.inference_mode():
            g = self.model.generate(**enc, max_new_tokens=64, do_sample=False, return_dict_in_generate=True,
                                    output_scores=True)
        seqs = g.sequences[:, enc["input_ids"].shape[1]:]
        probs = [torch.softmax(sc.float(), -1).cpu() for sc in g.scores]
        tok = self.proc.tokenizer
        out = []
        for b in range(len(ims)):
            ids = seqs[b].tolist()
            toks = [tok.decode([i]) for i in ids]
            sps = [{"yes": float(sum(probs[t][b, i] for i in self.yes_ids)), "no": float(sum(probs[t][b, i] for i in self.no_ids))}
                   for t in range(len(ids))]
            out.append(_hf_parse(toks, sps, tok.decode(ids, skip_special_tokens=True)))
        return out


def jpeg_bytes(img: np.ndarray, q: int = 90) -> bytes:
    b = io.BytesIO()
    Image.fromarray(img).save(b, format="JPEG", quality=q)
    return b.getvalue()


class CascadePredictor:
    """Fast heads on every frame; VLM on (a) frames where any state's fast probability is within ``band`` of 0.5 and
    (b) one frame every ``periodic_s`` seconds (work/idle drifts slowly). At most ``max_calls`` VLM frames per input,
    keeping the most uncertain. band=0.03 escalates ~30% of frames on the CCTV test set."""

    role = "states"
    name = "ops-cascade"

    def __init__(self, fast=None, vlm: VLMClient | None = None, band: float = 0.03, periodic_s: float = 15.0,
                 max_calls: int = 8, batch: int = 8, roi=None):
        from lanes.ops.predict import get_predictor

        self.fast = fast or get_predictor(roi=roi)
        self.vlm = vlm
        self.band, self.periodic_s, self.max_calls, self.batch, self.roi = band, periodic_s, max_calls, batch, roi

    @property
    def version(self) -> str:
        v = getattr(self.vlm, "version", "none") if self.vlm is not None else "none"
        return f"{self.fast.name}@{self.fast.version}+{v}"

    def load(self, device: str = "cuda:0") -> None:
        self.fast.load(device)
        if self.vlm is not None and hasattr(self.vlm, "load") and getattr(self.vlm, "model", 1) is None:
            self.vlm.load()

    def select(self, preds: list[FramePred]) -> list[int]:
        """Indices (positions in ``preds``) to send to the VLM."""
        margins = []
        for p in preds:
            ms = [abs(p.states[s] - 0.5) for s in VLM_STATES if s in p.states]
            margins.append(min(ms) if ms else 0.5)
        pick = {i for i, m in enumerate(margins) if m < self.band}
        last = -1e9
        for i, p in enumerate(preds):  # periodic check
            if p.t_s - last >= self.periodic_s:
                pick.add(i)
                last = p.t_s
        ranked = sorted(pick, key=lambda i: margins[i])
        return sorted(ranked[: self.max_calls])

    def predict(self, frames: list[Frame], context: list[FramePred] | None = None) -> list[FramePred]:
        preds = self.fast.predict(frames, context=context)
        if self.vlm is None or not frames:
            return preds
        idx = self.select(preds)
        if not idx:
            return preds
        ctx = {c.index: c for c in (context or [])}
        crops = [crop_main_vehicle(frames[i].image, ctx.get(frames[i].index), self.roi) for i in idx]
        t0 = time.perf_counter()
        answers: list[dict[str, float]] = []
        for s in range(0, len(crops), self.batch):
            answers += self.vlm.answer(crops[s:s + self.batch])
        dt = (time.perf_counter() - t0) * 1000 / len(idx)
        for i, ans in zip(idx, answers):
            for s in VLM_STATES:
                if s in ans and ans[s] == ans[s]:  # not NaN
                    preds[i].states[s] = float(ans[s])
            if "work" in ans:
                preds[i].states["idle"] = 1.0 - float(ans["work"])
            preds[i].timings_ms["vlm_frame"] = 1.0
            preds[i].timings_ms["vlm_ms"] = dt
        return preds


def get_predictor(vlm: str | VLMClient | None = "local", **kw) -> CascadePredictor:
    client = LocalQwen() if vlm == "local" else (None if vlm in (None, "none") else vlm)
    return CascadePredictor(vlm=client, **kw)
