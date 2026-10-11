"""Frozen image backbones for crop embeddings (no Kinetics video pretraining: DINOv2 and SigLIP2 are image models)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

BACKBONES = {
    "dinov2l": dict(hf="facebook/dinov2-large", size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    "siglip2": dict(hf="google/siglip2-so400m-patch14-384", size=384, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}


@dataclass
class Backbone:
    name: str
    model: torch.nn.Module
    size: int
    preprocess: Callable  # PIL -> CHW float tensor
    forward: Callable  # BCHW -> B x D
    dim: int


def make_preprocess(size: int, mean, std):
    m = np.asarray(mean, np.float32)[:, None, None]
    s = np.asarray(std, np.float32)[:, None, None]

    def prep(img):
        from PIL import Image

        a = np.asarray(img.convert("RGB").resize((size, size), Image.Resampling.BICUBIC), np.float32) / 255.0
        return torch.from_numpy((a.transpose(2, 0, 1) - m) / s)

    return prep


def load_backbone(name: str, device: str = "cuda:0", dtype=torch.bfloat16, path: str | None = None) -> Backbone:
    """``path``: local copy of the HF weights (artifacts/ops/states/backbone_hf) instead of the hub id."""
    cfg = dict(BACKBONES[name])
    if path:
        cfg["hf"] = str(path)
    prep = make_preprocess(cfg["size"], cfg["mean"], cfg["std"])
    if name == "dinov2l":
        from transformers import AutoModel

        m = AutoModel.from_pretrained(cfg["hf"], torch_dtype=dtype).to(device).eval()

        def fwd(x):
            o = m(pixel_values=x.to(dtype)).last_hidden_state
            return torch.cat([o[:, 0], o[:, 1:].mean(1)], dim=1)  # CLS + mean patch token

        dim = 2 * m.config.hidden_size
    else:
        from transformers import AutoModel

        full = AutoModel.from_pretrained(cfg["hf"], torch_dtype=dtype)
        m = full.vision_model.to(device).eval()
        del full

        def fwd(x):
            return m(pixel_values=x.to(dtype)).pooler_output

        dim = m.config.hidden_size
    return Backbone(name, m, cfg["size"], prep, fwd, dim)
