"""Internal interfaces every lane implements.

Each lane exposes ``lanes/<lane>/predict.py::get_predictor(**kw) -> Predictor``.
The pipeline (seen/pipeline.py) composes predictors and merges their FramePred by frame index.
Detectors run first; state and inspection predictors receive the merged detector output as ``context``.
Bounding boxes are always normalised [x1, y1, x2, y2] in 0..1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

BBox = tuple[float, float, float, float]


@dataclass
class Frame:
    index: int
    t_s: float
    image: np.ndarray  # H x W x 3, uint8, RGB


@dataclass
class Det:
    cls: str  # see seen/vocab.py
    confidence: float
    bbox: BBox
    track_id: int | None = None
    text: str | None = None  # OCR text for plate / vin / dot
    part: str | None = None  # car part a damage lies on
    extra: dict = field(default_factory=dict)


@dataclass
class FramePred:
    index: int
    t_s: float
    detections: list[Det] = field(default_factory=list)
    states: dict[str, float] = field(default_factory=dict)  # BayState -> probability
    scene: dict[str, float] = field(default_factory=dict)  # router scene -> probability
    timings_ms: dict[str, float] = field(default_factory=dict)

    def merge(self, other: "FramePred") -> "FramePred":
        assert other.index == self.index
        self.detections.extend(other.detections)
        self.states.update(other.states)
        self.scene.update(other.scene)
        self.timings_ms.update(other.timings_ms)
        return self


@runtime_checkable
class Predictor(Protocol):
    name: str
    version: str
    # Optional class attribute ``role``: "vehicle" | "plate" | "router" | "states" | "inspection".
    # The pipeline uses it so each event names the model that produced it.

    def load(self, device: str = "cuda:0") -> None: ...

    def predict(self, frames: list[Frame], context: list[FramePred] | None = None) -> list[FramePred]:
        """``context`` = merged predictions from earlier stages (detectors), same order as ``frames``.
        Detectors ignore it; state/inspection predictors use its vehicle/person/plate boxes."""
        ...


def empty_preds(frames: list[Frame]) -> list[FramePred]:
    return [FramePred(index=f.index, t_s=f.t_s) for f in frames]
