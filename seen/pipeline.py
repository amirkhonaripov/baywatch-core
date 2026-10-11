"""Compose predictors into one analysis call. Owned by the integrator.

Order: detectors (vehicle/person/plate/...) -> router decision -> face blur on evidence frames ->
state predictors (only if routed) -> visit engine -> work-order matcher -> optional simulated continuation.
"""
from __future__ import annotations

import importlib
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable

import numpy as np

from seen import blur, router, visit_engine, workorder
from seen.contracts.events import AnalysisResult, Detection, FrameResult, Source, Zone
from seen.contracts.workorder import WorkOrder
from seen.predictor import Frame, FramePred, Predictor

EvidenceWriter = Callable[[str, np.ndarray], str]  # (name, rgb image) -> uri


def _merge(base: list[FramePred], extra: list[FramePred]) -> None:
    by_idx = {p.index: p for p in base}
    for e in extra:
        by_idx[e.index].merge(e)


def _drop_orphan_plates(preds: list[FramePred], pad: float = 0.1) -> None:
    """A plate only counts if its centre lies inside a (slightly padded) vehicle box: removes plate hallucinations
    on non-vehicle images (e.g. the plate detector firing on a flower)."""
    from seen.vocab import PLATE, VEHICLE_CLASSES

    for p in preds:
        vs = [d.bbox for d in p.detections if d.cls in VEHICLE_CLASSES]
        keep = []
        for d in p.detections:
            if d.cls == PLATE:
                cx, cy = (d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2
                if not any(x1 - pad * (x2 - x1) <= cx <= x2 + pad * (x2 - x1) and y1 - pad * (y2 - y1) <= cy <= y2 + pad * (y2 - y1)
                           for x1, y1, x2, y2 in vs):
                    continue
            keep.append(d)
        p.detections = keep


def _load(spec: str, **kw) -> Predictor:
    """spec = 'module.path:factory'."""
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)(**kw)


class Pipeline:
    def __init__(self, detectors: list[Predictor], state_predictors: list[Predictor] | None = None,
                 inspection: list[Predictor] | None = None, device: str = "cuda:0"):
        self.detectors = detectors
        self.state_predictors = state_predictors or []
        self.inspection = inspection or []
        for p in [*self.detectors, *self.state_predictors, *self.inspection]:
            p.load(device)

    def _role_versions(self) -> dict[str, str]:
        """Map pipeline roles -> model name/version so every event names the model that produced it.

        Predictors may declare ``role`` in {"vehicle","plate","router","states","inspection"}; otherwise the first
        detector is the vehicle detector and the first state predictor is the state model."""
        mv: dict[str, str] = {}
        groups = [(self.detectors, "vehicle"), (self.state_predictors, "states"), (self.inspection, "inspection")]
        for preds, default_role in groups:
            for i, p in enumerate(preds):
                role = getattr(p, "role", None) or (default_role if i == 0 else None)
                if role and role not in mv:
                    mv[role], mv[f"{role}_version"] = p.name, p.version
        return mv

    @property
    def model_versions(self) -> dict[str, str]:
        return {p.name: p.version for p in [*self.detectors, *self.state_predictors, *self.inspection]}

    def analyze(self, frames: list[Frame], *, zones: list[Zone] | None = None, work_order: WorkOrder | None = None,
                simulate: bool = False, evidence_writer: EvidenceWriter | None = None,
                source: Source = Source.observed_real, start_time: datetime | None = None) -> AnalysisResult:
        t_all = time.perf_counter()
        input_id = uuid.uuid4().hex[:10]
        preds = [FramePred(index=f.index, t_s=f.t_s) for f in frames]
        for d in self.detectors:
            _merge(preds, d.predict(frames))
        available = ["plate", "visit_engine"]
        if self.state_predictors:
            available.append("bay_states")
        if self.inspection:
            available.append("inspection")
        _drop_orphan_plates(preds)
        route = router.decide(preds, available)
        # Plate predictors are detectors and run on every frame: report that truthfully.
        if any(getattr(d, "role", "") == "plate" for d in self.detectors) and "plate" not in route.ran:
            route.ran.append("plate")
            route.skipped = [s for s in route.skipped if s.module != "plate"]
        if "bay_states" in route.ran:
            for s in self.state_predictors:
                _merge(preds, s.predict(frames, context=preds))
        if "inspection" in route.ran:
            for s in self.inspection:
                _merge(preds, s.predict(frames, context=preds))

        # Evidence frames: blurred before anything is written.
        uris: dict[int, str] = {}
        if evidence_writer:
            for f, p in zip(frames, preds):
                uris[f.index] = evidence_writer(f"{input_id}_{f.index}.jpg", blur.blur_faces(f.image, p.detections))

        mv = self._role_versions()
        visit = None
        if "visit_engine" in route.ran:
            visit = visit_engine.build_visit(preds, start_time=start_time or datetime.now(timezone.utc), zones=zones,
                                             source=source, model_versions=mv, evidence_uri=uris)
            if visit and work_order:
                visit.work_order_id = work_order.id
                end_ts = ((start_time or visit.created_at) + timedelta(seconds=max(f.t_s for f in frames))).timestamp()
                visit.verifications = workorder.verify(work_order, visit.events, end_ts=end_ts)
        continuation = None
        if simulate and visit is not None:
            try:
                from sim.scenario import continue_visit  # lane E
                now = (start_time or visit.created_at) + timedelta(seconds=max(f.t_s for f in frames))
                continuation = continue_visit(visit, work_order, now=now)
            except ImportError:
                continuation = None

        timings: dict[str, float] = {}
        for p in preds:
            for k, v in p.timings_ms.items():
                timings[k] = timings.get(k, 0.0) + v
        timings["total"] = (time.perf_counter() - t_all) * 1000
        return AnalysisResult(
            input_id=input_id,
            input_kind="image" if len(frames) == 1 else "frames",
            route=route,
            frames=[FrameResult(frame_index=p.index, t_s=p.t_s, frame_uri=uris.get(p.index),
                                detections=[Detection(cls=d.cls, confidence=d.confidence, bbox=list(d.bbox),
                                                      track_id=d.track_id, text=d.text, part=d.part) for d in p.detections],
                                states=p.states or None) for p in preds],
            visit=visit,
            simulated_continuation=continuation,
            timings_ms=timings,
            model_versions=self.model_versions,
            warnings=[] if visit else ["No vehicle detected: no visit created."],
        )


def default_pipeline(device: str = "cuda:0", vlm=None, inspection: bool = True, baseline: bool = False) -> Pipeline:
    """Production pipeline from the lanes' shipped artifacts.

    detectors: flow vehicle/person detector (RF-DETR-L fine-tuned) + plate predictor (ident) + scene router (SigLIP2).
    states: cascade (ops fast heads on every frame + Qwen3-VL on uncertain/periodic frames). ``vlm`` = a VLMClient,
    "local" (in-process Qwen), or None/"none" (fast heads only); default from $SEEN_VLM (default "none").
    inspection: damage/parts/tyre predictor (lane D) if its artifacts load.
    ``baseline=True`` (or $SEEN_BASELINE=1) returns the day-1 pretrained pipeline for before/after comparisons.
    """
    import os

    if baseline or os.environ.get("SEEN_BASELINE") == "1":
        from seen.baseline import PretrainedALPR, PretrainedDetector

        return Pipeline(detectors=[PretrainedDetector(), PretrainedALPR()], device=device)
    from lanes.flow.predict import get_predictor as flow_predictor, get_router
    from lanes.ident.predict import get_predictor as plate_predictor
    from lanes.ops.cascade import get_predictor as cascade_predictor

    if vlm is None:
        vlm = os.environ.get("SEEN_VLM", "none")
    detectors = [flow_predictor(runtime="torch"), plate_predictor(), get_router()]
    states = [cascade_predictor(vlm=vlm)]
    insp = []
    if inspection:
        try:
            from lanes.inspect.predict import get_predictor as inspect_predictor

            insp = [inspect_predictor()]
        except Exception as e:  # noqa: BLE001 - lane D optional until shipped
            print(f"[pipeline] inspection disabled: {e}")
    return Pipeline(detectors=detectors, state_predictors=states, inspection=insp, device=device)
