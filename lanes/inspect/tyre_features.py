"""Frozen-backbone feature extraction for the tyre condition classifier (cached per file).

Backbone: DINOv2 ViT-B/14 (timm `vit_base_patch14_dinov2.lvd142m`, Apache-2.0 weights from the Hugging Face hub).
Features = [CLS token, mean of patch tokens] (1536-d). Preprocess: shorter side -> `size`, centre crop `size` x `size`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lanes.inspect import common as C

MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
BACKBONE = "vit_base_patch14_dinov2.lvd142m"


def load_image(path: str | Path, size: int, flip: bool = False) -> np.ndarray:
    with Image.open(path) as im:
        im.draft("RGB", (size * 2, size * 2))  # fast JPEG decode at reduced scale for 12 MP phone photos
        im = im.convert("RGB")
        w, h = im.size
        s = size / min(w, h)
        im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.Resampling.BICUBIC)
        w, h = im.size
        l, t = (w - size) // 2, (h - size) // 2
        im = im.crop((l, t, l + size, t + size))
    a = np.asarray(im, np.float32) / 255.0
    if flip:
        a = a[:, ::-1]
    return ((a - MEAN) / STD).transpose(2, 0, 1).copy()


class _DS(torch.utils.data.Dataset):
    def __init__(self, paths, size):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            return torch.from_numpy(load_image(self.paths[i], self.size)), i
        except Exception:  # noqa: BLE001
            return torch.zeros(3, self.size, self.size), -1


def make_backbone(device: str = "cuda:0", size: int = 448):
    import timm
    m = timm.create_model(BACKBONE, pretrained=True, num_classes=0, img_size=size)
    return m.eval().to(device)


@torch.no_grad()
def embed(m, x: torch.Tensor) -> torch.Tensor:
    tok = m.forward_features(x)  # [B, 1+N, D]
    return torch.cat([tok[:, 0], tok[:, 1:].mean(1)], dim=1)


def extract(paths: list[str], size: int = 448, device: str = "cuda:0", bs: int = 32, workers: int = 6, cache_name: str = "") -> np.ndarray:
    """Features for ``paths`` (N x 1536, float32); cached in PREP/inspect/tyre/feats/<cache_name or hash>.npz."""
    key = hashlib.sha1(("|".join(paths) + f"|{size}|{BACKBONE}").encode()).hexdigest()[:16]
    cf = C.ensure(C.FAST_I / "tyre" / "feats") / f"{cache_name or key}_{size}_{key}.npz"
    if cf.exists():
        return np.load(cf)["f"]
    m = make_backbone(device, size)
    dl = torch.utils.data.DataLoader(_DS(paths, size), batch_size=bs, num_workers=workers, shuffle=False)
    out = np.zeros((len(paths), 1536), np.float32)
    bad = 0
    for x, idx in dl:
        f = embed(m, x.to(device)).float().cpu().numpy()
        for j, i in enumerate(idx.tolist()):
            if i >= 0:
                out[i] = f[j]
            else:
                bad += 1
    if bad:
        print("warning: unreadable images:", bad)
    np.savez(cf, f=out)
    return out
