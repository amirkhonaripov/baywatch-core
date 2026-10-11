"""Turn per-frame predictions into a Visit with events.

Lane E implementation. ``build_visit`` keeps its original signature and contract:
returns None when no vehicle was detected; every event carries the given ``source``.

Steps
1. ``arrived`` at the first frame with a vehicle detection; ``plate_read`` from the best OCR'd plate (unchanged).
2. State smoothing over ``FramePred.states`` (frames sorted by time; irregular sampling is fine):
   - ``DECODER = "hmm"`` (default): composite-state HMM / Viterbi with scenario-engine duration priors
     (seen/hmm.py), posterior label marginals as event confidence.
   - ``DECODER = "baseline"``: the original per-label threshold + hysteresis (ON 0.6 / OFF 0.4).
   Select with the env var ``SEEN_DECODER`` or by setting ``visit_engine.DECODER``.
3. Events from the smoothed label sequences (``label_events``; the scenario engine uses the same function to
   build ground truth, so decoded and true events have identical semantics):
   - ``state_started`` / ``state_ended`` for every observed bay state except ``idle``;
   - ``idle_started`` / ``idle_ended`` for spans with the car present and no work for >= ``IDLE_THRESHOLD_S``
     (bay / frame / other zones only, and only when work or idle evidence exists);
   - ``entered_zone`` / ``left_zone`` when the car appears / leaves in a named zone (not "frame"/"other");
   - ``ready`` when a car arrives in a ``ready`` zone; ``handed_over`` when it leaves the ``ready`` zone.
   A label only produces events if the predictors supplied it in >= ``MIN_LABEL_COVERAGE`` of frames.
   If no predictor supplied ``car_present``, it is taken from the vehicle detector's best box confidence
   (``CAR_PRESENT_FROM_DETECTIONS``; when the zone has a polygon, only boxes whose bottom-centre lies inside it
   count) with emission weight ``DETECTOR_CAR_WEIGHT`` (stronger than the tempered
   classifier weight; a fixed choice, not tuned).
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np

from seen import hmm
from seen.contracts.events import (
    BayState, Event, EventType, Evidence, Source, Vehicle, VehicleType, Visit, Zone, ZoneKind,
)
from seen.predictor import FramePred
from seen.vocab import BAY_STATES, PLATE, VEHICLE_CLASSES

ON, OFF = 0.6, 0.4  # hysteresis thresholds (baseline decoder)
DECODER = os.environ.get("SEEN_DECODER", "hmm")  # "hmm" | "baseline"
HMM_RATES = os.environ.get("SEEN_HMM_RATES", "fitted")  # "fitted" | "generic"
IDLE_THRESHOLD_S = float(os.environ.get("SEEN_IDLE_THRESHOLD_S", "300"))
MIN_LABEL_COVERAGE = 0.2
CAR_PRESENT_FROM_DETECTIONS = True
DETECTOR_CAR_WEIGHT = 1.0
IDLE_ZONES = ("bay", "frame", "other")
UNKNOWN_MODEL = "unknown-model"  # evidence model/version when the pipeline did not supply one
NAMED_ZONE_EXCLUDE = ("frame", "other")
_K = {s: i for i, s in enumerate(BAY_STATES)}


def _eid() -> str:
    return uuid.uuid4().hex[:12]


def _best_plate(preds: list[FramePred]):
    best = None
    for p in preds:
        for d in p.detections:
            if d.cls == PLATE and d.text and (best is None or d.confidence > best[1].confidence):
                best = (p, d)
    return best


def _kind(zone: Zone) -> str:
    return zone.kind.value if hasattr(zone.kind, "value") else str(zone.kind)


def label_events(times: np.ndarray, labels: np.ndarray, conf: np.ndarray, observed: np.ndarray, *, zone: Zone,
                 visit_id: str, ts, source: Source, evidence=None,
                 idle_threshold_s: float | None = None) -> list[Event]:
    """Events from decoded per-frame label sequences.

    times (T,) seconds; labels (T, K) bool; conf (T, K) probability each label is on; observed (K,) bool.
    ``ts(i)`` -> datetime of frame i. ``evidence(i, conf) -> Evidence | None``.
    """
    thr = IDLE_THRESHOLD_S if idle_threshold_s is None else idle_threshold_s
    ev = evidence or (lambda i, c: None)
    kind = _kind(zone)
    out: list[Event] = []

    def add(etype: EventType, i: int, c: float, state: str | None = None, payload: dict | None = None):
        out.append(Event(event_id=_eid(), visit_id=visit_id, type=etype, t=ts(i), source=source, zone=zone,
                         state=BayState(state) if state else None, evidence=ev(i, float(np.clip(c, 0, 1))),
                         payload=payload))

    for k, s in enumerate(BAY_STATES):
        if s == "idle" or not observed[k]:
            continue
        for a, b in hmm.segments(labels[:, k]):
            add(EventType.state_started, a, conf[a, k], s)
            if b is not None:
                add(EventType.state_ended, b, 1 - conf[b, k], s)

    car = labels[:, _K["car_present"]]
    if observed[_K["car_present"]]:
        cc = conf[:, _K["car_present"]]
        for a, b in hmm.segments(car):
            if kind not in NAMED_ZONE_EXCLUDE:
                add(EventType.entered_zone, a, cc[a])
                if b is not None:
                    add(EventType.left_zone, b, 1 - cc[b])
            if kind == "ready":
                add(EventType.ready, a, cc[a], payload={"rule": "car_in_ready_zone"})
                if b is not None:
                    add(EventType.handed_over, b, 1 - cc[b], payload={"rule": "car_left_ready_zone"})

    # Idle: car present, no work, not washing; only spans >= threshold become events.
    if kind in IDLE_ZONES and (observed[_K["work"]] or observed[_K["idle"]]):
        if observed[_K["work"]]:
            idle = car & ~labels[:, _K["work"]] & ~labels[:, _K["washing"]]
            iconf = np.minimum(conf[:, _K["car_present"]], 1 - conf[:, _K["work"]])
        else:
            idle = labels[:, _K["idle"]]
            iconf = conf[:, _K["idle"]]
        for a, b in hmm.segments(idle):
            t_end = times[b] if b is not None else times[-1]
            dur = float(t_end - times[a])
            if dur < thr:
                continue
            pay = {"duration_s": round(dur, 1), "threshold_s": thr, "ongoing": b is None}
            add(EventType.idle_started, a, iconf[a], payload=pay)
            if b is not None:
                add(EventType.idle_ended, b, 1 - iconf[b], payload=pay)
    out.sort(key=lambda e: e.t)
    return out


def _in_polygon(x: float, y: float, poly) -> bool:
    """Ray casting; poly = [[x, y], ...] normalised image coordinates."""
    inside, n = False, len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _vehicle_conf(p: FramePred, polygon) -> float | None:
    confs = [d.confidence for d in p.detections if d.cls in VEHICLE_CLASSES
             and (not polygon or _in_polygon((d.bbox[0] + d.bbox[2]) / 2, d.bbox[3], polygon))]
    return max(confs) if confs else None


def _state_matrix(preds: list[FramePred], polygon=None) -> tuple[np.ndarray, np.ndarray]:
    T, K = len(preds), len(BAY_STATES)
    P = np.full((T, K), np.nan)
    M = np.zeros((T, K), dtype=bool)
    for i, p in enumerate(preds):
        for s, v in (p.states or {}).items():
            k = _K.get(s)
            if k is not None and v is not None:
                P[i, k] = float(v)
                M[i, k] = True
    k = _K["car_present"]
    if CAR_PRESENT_FROM_DETECTIONS and not M[:, k].any():
        for i, p in enumerate(preds):
            c = _vehicle_conf(p, polygon)
            P[i, k] = c if c is not None else 0.05
            M[i, k] = True
    return P, M


def decode_states(times: np.ndarray, P: np.ndarray, M: np.ndarray, zone_kind: str | None,
                  decoder: str | None = None, **kw) -> hmm.Decoded:
    decoder = decoder or DECODER
    if decoder == "baseline":
        return hmm.hysteresis(P, M, ON, OFF)
    return hmm.decode(times, P, M, zone_kind=zone_kind, rates=kw.pop("rates", HMM_RATES), **kw)


def build_visit(
    preds: list[FramePred],
    *,
    start_time: datetime | None = None,
    zones: list[Zone] | None = None,
    source: Source = Source.observed_real,
    model_versions: dict[str, str] | None = None,
    evidence_uri: dict[int, str] | None = None,
    visit_id: str | None = None,
) -> Visit | None:
    """Return None if no vehicle was seen (no visit without a vehicle)."""
    start_time = start_time or datetime.now(timezone.utc)
    mv = model_versions or {}
    ev_uri = evidence_uri or {}
    zone = (zones or [Zone(id="frame", kind=ZoneKind.frame)])[0]
    preds = sorted(preds, key=lambda p: p.t_s)
    veh_frames = [p for p in preds if any(d.cls in VEHICLE_CLASSES for d in p.detections)]
    if not veh_frames:
        return None
    vid = visit_id or uuid.uuid4().hex[:10]
    ts = lambda p: start_time + timedelta(seconds=p.t_s)  # noqa: E731
    events: list[Event] = []

    # With a zone polygon, the arrival is the first frame with a vehicle inside the zone (any vehicle otherwise).
    in_zone = [p for p in veh_frames if _vehicle_conf(p, zone.polygon) is not None] if zone.polygon else []
    first = (in_zone or veh_frames)[0]
    vdet = max((d for d in first.detections if d.cls in VEHICLE_CLASSES
                and (not in_zone or _in_polygon((d.bbox[0] + d.bbox[2]) / 2, d.bbox[3], zone.polygon))),
               key=lambda d: d.confidence)
    events.append(Event(event_id=_eid(), visit_id=vid, type=EventType.arrived, t=ts(first), source=source, zone=zone,
                        evidence=Evidence(frame_uri=ev_uri.get(first.index), video_time_s=first.t_s, bbox=list(vdet.bbox),
                                          model=mv.get("vehicle", UNKNOWN_MODEL), model_version=mv.get("vehicle_version", UNKNOWN_MODEL),
                                          confidence=vdet.confidence)))
    vehicle = Vehicle(vehicle_type=VehicleType(vdet.cls) if vdet.cls in ("car", "motorcycle", "truck", "bus") else VehicleType.unknown)

    bp = _best_plate(preds)
    if bp:
        p, d = bp
        vehicle.plate_text = d.text
        vehicle.plate_confidence = d.confidence
        vehicle.plate_format = d.extra.get("format", "unknown")
        vehicle.plate_format_valid = d.extra.get("format_valid")
        events.append(Event(event_id=_eid(), visit_id=vid, type=EventType.plate_read, t=ts(p), source=source, zone=zone,
                            evidence=Evidence(frame_uri=ev_uri.get(p.index), video_time_s=p.t_s, bbox=list(d.bbox),
                                              model=mv.get("plate", UNKNOWN_MODEL), model_version=mv.get("plate_version", UNKNOWN_MODEL),
                                              confidence=d.confidence),
                            payload={"text": d.text}))

    P, M = _state_matrix(preds, zone.polygon)
    observed = M.mean(0) >= MIN_LABEL_COVERAGE
    if observed.any():
        times = np.array([p.t_s for p in preds], dtype=float)
        detector_car = CAR_PRESENT_FROM_DETECTIONS and not any("car_present" in (p.states or {}) for p in preds)
        kw = {"label_weight": {"car_present": DETECTOR_CAR_WEIGHT}} if (detector_car and DECODER != "baseline") else {}
        dec = decode_states(times, P, M, _kind(zone), **kw)
        decoder_name = "hysteresis" if (DECODER == "baseline") else f"hmm-{HMM_RATES}"
        if DECODER != "baseline":
            # Guard: the HMM's duration priors come from the scenario engine. If it contradicts strong per-frame evidence
            # (>=3 frames with p(car_present) >= 0.8 but no car decoded), fall back to hysteresis and say so.
            kc = _K["car_present"]
            strong = int(np.nansum(P[:, kc] >= 0.8))
            if strong >= 3 and not dec.labels[:, kc].any():
                dec = decode_states(times, P, M, _kind(zone), decoder="baseline")
                decoder_name = "hysteresis(fallback: hmm contradicted evidence)"

        # Role-keyed model versions from seen.pipeline: car_present without a state predictor comes from the
        # vehicle detector, everything else from the state predictor(s).
        if detector_car and not any(len(p.states or {}) for p in preds):
            src_model, src_version = mv.get("vehicle", UNKNOWN_MODEL), mv.get("vehicle_version", UNKNOWN_MODEL)
        else:
            src_model, src_version = mv.get("states", UNKNOWN_MODEL), mv.get("states_version", UNKNOWN_MODEL)

        def evidence(i: int, c: float) -> Evidence:
            return Evidence(frame_uri=ev_uri.get(preds[i].index), video_time_s=preds[i].t_s,
                            model=src_model + f"+{decoder_name}", model_version=src_version, confidence=c)

        events.extend(label_events(times, dec.labels, dec.conf, observed, zone=zone, visit_id=vid,
                                   ts=lambda i: ts(preds[i]), source=source, evidence=evidence))
    events.sort(key=lambda e: e.t)
    return Visit(visit_id=vid, created_at=start_time, vehicle=vehicle, events=events,
                 current_stage=events[-1].type.value if events else None)
