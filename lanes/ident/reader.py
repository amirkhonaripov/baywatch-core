"""Crop -> Reading: OCR + (optional) orientation search + format-constrained decoding.

``PlateReader.read(crops)`` returns one ``Reading`` per crop:
  text        canonical text (no separators, upper case)
  conf        geometric-mean probability of the chosen characters (text slots only)
  format      uz_private / uz_legal / uz_moto / foreign / unknown (uz_ev is added from the format head or the green-field cue)
  valid       text matches a valid Uzbek layout (region 01-99)
  rot         rotation applied to the crop in degrees counter-clockwise (0/90/180/270)
  probs       (slots, vocab) per-slot probabilities of the chosen rotation (for track voting)
Decoding modes (``mode``):
  argmax        plain arg-max per slot (what fast-plate-ocr does)
  constrained   best valid Uzbek layout under the per-slot probabilities (always forced)
  auto          constrained, but fall back to arg-max (format 'foreign'/'unknown') when the constrained reading is much less
                likely than the free reading: score_free - score_constrained > ``margin`` nats.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from lanes.ident import uz_format as uz
from lanes.ident.ocr_onnx import PlateOcrOnnx


@dataclass
class Reading:
    text: str
    conf: float
    format: str = uz.UNKNOWN
    valid: bool = False
    rot: int = 0
    probs: np.ndarray | None = None
    region_probs: np.ndarray | None = None
    free_text: str = ""
    extra: dict = field(default_factory=dict)


def _rot(img: np.ndarray, k: int) -> np.ndarray:
    """k * 90 degrees counter-clockwise."""
    return np.ascontiguousarray(np.rot90(img, k))


def free_decode(ocr: PlateOcrOnnx, p: np.ndarray) -> tuple[str, float, float]:
    idx = p.argmax(-1)
    s = "".join(ocr.alphabet[i] for i in idx).rstrip(ocr.pad)
    n = max(len(s), 1)
    lp = np.log(np.clip(p.max(-1), 1e-9, 1.0))
    return s, float(math.exp(lp[:n].mean())), float(lp.sum())


class PlateReader:
    def __init__(self, ocr: PlateOcrOnnx, mode: str = "auto", orient: bool = True, margin: float = 6.0,
                 min_conf_fallback: float = 0.0):
        assert mode in ("argmax", "constrained", "auto")
        self.ocr, self.mode, self.orient, self.margin = ocr, mode, orient, margin

    def _decode(self, p: np.ndarray) -> tuple[str, float, str, bool, float]:
        """-> (text, conf, layout/format name, constrained_used, score)"""
        free, fconf, fscore = free_decode(self.ocr, p)
        if self.mode == "argmax":
            pr = uz.parse(free)
            return free, fconf, pr.format, False, fscore
        best = uz.constrained_decode(p, self.ocr.alphabet)
        if best is None:
            pr = uz.parse(free)
            return free, fconf, pr.format, False, fscore
        text, score, layout = best
        n = len(uz.LAYOUTS[layout])
        # geometric-mean probability over the text slots of the chosen layout
        lp = np.log(np.clip(p, 1e-9, 1.0))
        idx = [self.ocr.alphabet.index(c) for c in text]
        conf = float(math.exp(np.mean([lp[i, j] for i, j in enumerate(idx)])))
        if self.mode == "auto" and (fscore - score) > self.margin:
            pr = uz.parse(free)
            return free, fconf, uz.FOREIGN if not pr.valid else pr.format, False, fscore
        return text, conf, uz.LAYOUT_TO_FORMAT[layout], True, score

    def read(self, crops: list[np.ndarray]) -> list[Reading]:
        if not crops:
            return []
        hyps: list[tuple[int, int]] = []  # (crop index, k)
        imgs: list[np.ndarray] = []
        for i, c in enumerate(crops):
            ks = [0]
            if self.orient:
                h, w = c.shape[:2]
                ks = [0, 2] if w >= h * 1.15 else [1, 3] if h >= w * 1.15 else [0, 1, 2, 3]
            for k in ks:
                hyps.append((i, k))
                imgs.append(_rot(c, k))
        probs, rprobs = self.ocr.probs(imgs)
        best: dict[int, tuple[float, Reading]] = {}
        for (i, k), p, j in zip(hyps, probs, range(len(imgs))):
            text, conf, fmt, cons, score = self._decode(p)
            valid = uz.parse(text).valid
            # orientation score: the (constrained) log-likelihood of the reading; prefer valid layouts
            sel = score + (3.0 if valid else 0.0)
            r = Reading(text=text, conf=conf, format=fmt, valid=valid, rot=(k * 90) % 360, probs=p,
                        region_probs=None if rprobs is None else rprobs[j], free_text=free_decode(self.ocr, p)[0])
            if i not in best or sel > best[i][0]:
                best[i] = (sel, r)
        return [best[i][1] for i in range(len(crops))]
