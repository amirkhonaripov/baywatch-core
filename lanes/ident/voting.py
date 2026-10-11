"""Track voting: combine the per-frame plate reads of one track into one reading.

Why: a plate is seen in many frames of a track at different sizes/blur/angles; single-frame reads err on different
characters, so pooling the per-slot character distributions beats picking one frame.

Methods (``vote(reads, method=...)``):
  pooled   (default)  weighted log-linear pooling of the per-slot probability rows of all usable reads, then (optionally)
                      format-constrained decoding of the pooled distribution. Weight per read = conf**gamma * quality
                      (quality e.g. sqrt of the plate area in pixels, clipped).  Reads with conf < min_conf are ignored
                      unless that would leave nothing.
  best                the single read with the highest weight (baseline).
  majority            weighted majority vote over whole strings (baseline).
Inputs are ``Observation`` objects; ``reader.Reading`` converts with ``Observation.from_reading``.
Everything is plain numpy; no model code.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from lanes.ident import uz_format as uz


@dataclass
class Observation:
    probs: np.ndarray            # (slots, vocab) softmax rows
    text: str                    # the single-frame reading
    conf: float
    quality: float = 1.0         # e.g. sqrt(plate area px); larger = more trustworthy
    frame: int | None = None

    @staticmethod
    def from_reading(r, quality: float = 1.0, frame: int | None = None) -> "Observation":
        return Observation(r.probs, r.text, r.conf, quality, frame)


@dataclass
class Vote:
    text: str
    conf: float
    format: str
    valid: bool
    n_used: int
    method: str
    support: float = 0.0   # fraction of used reads whose own text equals the voted text
    extra: dict = field(default_factory=dict)


def _weights(obs: list[Observation], gamma: float, qcap: float) -> np.ndarray:
    return np.array([max(o.conf, 1e-3) ** gamma * min(max(o.quality, 1e-3), qcap) for o in obs], dtype=np.float64)


def vote(obs: list[Observation], method: str = "pooled", alphabet: str = uz.ALPHABET, constrained: bool = True,
         min_conf: float = 0.25, gamma: float = 2.0, qcap: float = 200.0, margin: float = 6.0) -> Vote | None:
    if not obs:
        return None
    use = [o for o in obs if o.conf >= min_conf] or list(obs)
    w = _weights(use, gamma, qcap)
    support = lambda t: float(np.mean([o.text == t for o in use]))  # noqa: E731
    if method == "best":
        o = use[int(np.argmax(w))]
        p = uz.parse(o.text)
        return Vote(o.text, o.conf, p.format, p.valid, 1, "best", support(o.text))
    if method == "majority":
        score: dict[str, float] = defaultdict(float)
        for o, wi in zip(use, w):
            score[o.text] += wi
        t = max(score, key=score.get)
        p = uz.parse(t)
        return Vote(t, score[t] / w.sum(), p.format, p.valid, len(use), "majority", support(t))
    assert method == "pooled"
    L = np.log(np.clip(np.stack([o.probs for o in use]), 1e-9, 1.0))       # (n, slots, vocab)
    A = (w[:, None, None] * L).sum(0) / w.sum()                            # weighted geometric mean, still per-slot
    P = np.exp(A - A.max(-1, keepdims=True))
    P /= P.sum(-1, keepdims=True)
    free_idx = P.argmax(-1)
    pad = alphabet.index(uz.PAD)
    free = "".join(alphabet[i] for i in free_idx).rstrip(uz.PAD)
    fscore = float(np.log(P.max(-1)).sum())
    if constrained:
        best = uz.constrained_decode(P, alphabet)
        if best is not None and (fscore - best[1]) <= margin:
            text, sc, layout = best
            conf = float(math.exp(np.mean([math.log(max(P[i, alphabet.index(c)], 1e-9)) for i, c in enumerate(text)])))
            return Vote(text, conf, uz.LAYOUT_TO_FORMAT[layout], True, len(use), "pooled+constrained", support(text))
    conf = float(math.exp(np.mean(np.log(np.clip(P.max(-1)[:max(len(free), 1)], 1e-9, 1)))))
    p = uz.parse(free)
    return Vote(free, conf, p.format if p.valid else uz.FOREIGN, p.valid, len(use), "pooled", support(free))


class TrackVoter:
    """Stateful helper: ``add(track_id, obs)`` then ``result(track_id)``; keeps at most ``max_obs`` best reads per track."""

    def __init__(self, method: str = "pooled", max_obs: int = 30, **kw):
        self.method, self.max_obs, self.kw = method, max_obs, kw
        self.tracks: dict[int, list[Observation]] = defaultdict(list)

    def add(self, track_id: int, o: Observation) -> None:
        lst = self.tracks[track_id]
        lst.append(o)
        if len(lst) > self.max_obs:
            lst.sort(key=lambda x: -(x.conf * min(x.quality, 200.0)))
            del lst[self.max_obs:]

    def result(self, track_id: int) -> Vote | None:
        return vote(self.tracks.get(track_id, []), self.method, **self.kw)
