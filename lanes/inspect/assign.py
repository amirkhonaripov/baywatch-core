"""Damage -> car part assignment (shared by the predictor and the evaluation).

Rule: for each damage instance take the part instance whose mask overlaps the damage mask the most (intersection area);
the damage is assigned that part's class if the intersection covers >= ``min_frac`` of the damage area, else None
('unknown part'). Masks are rasterised at <= ``RAST`` px on the long side from polygons (list of polygons per instance).
"""
from __future__ import annotations

import numpy as np

from lanes.inspect.evaldet import _raster

RAST = 384


def _masks(polys_list, h: int, w: int) -> list[np.ndarray]:
    s = min(1.0, RAST / max(h, w))
    return [_raster(p, h, w, s) for p in polys_list]


def box_to_poly(box) -> list[list[list[float]]]:
    x1, y1, x2, y2 = box
    return [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]


def assign(damage_polys: list, part_polys: list, part_labels: list[str], part_scores: list[float] | None, h: int, w: int,
           min_frac: float = 0.10) -> list[tuple[str | None, float]]:
    """-> per damage instance (part_label | None, overlap fraction of the damage area covered by that part)."""
    if not damage_polys:
        return []
    dm = _masks(damage_polys, h, w)
    if not part_polys:
        return [(None, 0.0) for _ in dm]
    pm = _masks(part_polys, h, w)
    P = np.stack([m.ravel() for m in pm]).astype(np.float32)
    out = []
    for d in dm:
        da = float(d.sum())
        if da <= 0:
            out.append((None, 0.0))
            continue
        inter = P @ d.ravel().astype(np.float32)  # intersection area (raster px) with every part instance
        key = inter + (1e-3 * np.asarray(part_scores, dtype=np.float32) if part_scores is not None else 0.0)  # ties -> more confident part
        j = int(np.argmax(key))
        frac = float(inter[j] / da)
        out.append((part_labels[j] if frac >= min_frac else None, frac))
    return out
