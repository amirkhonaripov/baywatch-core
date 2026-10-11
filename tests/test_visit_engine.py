"""Unit tests for seen/visit_engine.py (CPU only)."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from seen import visit_engine
from seen.contracts.events import EventType, Source, Zone, ZoneKind
from seen.predictor import Det, FramePred
from seen.vocab import BAY_STATES

T0 = datetime(2026, 10, 12, 9, 0, tzinfo=timezone.utc)
CAR = Det(cls="car", confidence=0.9, bbox=(0.2, 0.2, 0.8, 0.9))


@pytest.fixture(autouse=True)
def _restore_decoder():
    old = visit_engine.DECODER
    yield
    visit_engine.DECODER = old


def _bay_clip(seed=0, acc=0.85, n=900):
    """1 fps bay clip: car 60-840 s, raised 200-700 s, work 100-300 s and 650-760 s (idle 300-650 s)."""
    rng = np.random.default_rng(seed)
    preds = []
    for i in range(n):
        t = float(i)
        truth = {"car_present": 60 <= t < 840, "raised": 200 <= t < 700,
                 "work": (100 <= t < 300) or (650 <= t < 760), "person_at_car": (100 <= t < 300) or (650 <= t < 760)}
        st = {k: (0.85 if v else 0.15) if rng.random() < acc else (0.15 if v else 0.85) for k, v in truth.items()}
        p = FramePred(index=i, t_s=t, states=st)
        if truth["car_present"]:
            p.detections.append(CAR)
        preds.append(p)
    return preds


def _times(visit, etype, state=None):
    return [(e.t - T0).total_seconds() for e in visit.events
            if e.type == etype and (state is None or (e.state and e.state.value == state))]


def test_no_vehicle_no_visit():
    preds = [FramePred(index=i, t_s=float(i), states={"car_present": 0.9}) for i in range(5)]
    for dec in ("hmm", "baseline"):
        visit_engine.DECODER = dec
        assert visit_engine.build_visit(preds, start_time=T0) is None


@pytest.mark.parametrize("dec", ["hmm", "baseline"])
def test_events_carry_source(dec):
    visit_engine.DECODER = dec
    v = visit_engine.build_visit(_bay_clip(), start_time=T0, source=Source.observed_synthetic,
                                 zones=[Zone(id="bay1", kind=ZoneKind.bay)])
    assert v is not None and v.events
    assert all(e.source == Source.observed_synthetic for e in v.events)
    assert sum(e.type == EventType.arrived for e in v.events) == 1
    assert [e.t for e in v.events] == sorted(e.t for e in v.events)


def test_hmm_decodes_states_and_idle():
    visit_engine.DECODER = "hmm"
    v = visit_engine.build_visit(_bay_clip(), start_time=T0, zones=[Zone(id="bay1", kind=ZoneKind.bay)])
    up, down = _times(v, EventType.state_started, "raised"), _times(v, EventType.state_ended, "raised")
    assert len(up) == 1 and abs(up[0] - 200) <= 10 and len(down) == 1 and abs(down[0] - 700) <= 10
    assert len(_times(v, EventType.entered_zone)) == 1 and len(_times(v, EventType.left_zone)) == 1
    idle = _times(v, EventType.idle_started)
    assert len(idle) == 1 and abs(idle[0] - 300) <= 15  # 350 s without work >= 300 s threshold
    assert not any(e.state is not None and e.state.value == "idle" for e in v.events)
    for e in v.events:
        if e.evidence is not None:
            assert 0 <= e.evidence.confidence <= 1


def test_hmm_is_cleaner_than_baseline_on_noise():
    n_events = {}
    for dec in ("hmm", "baseline"):
        visit_engine.DECODER = dec
        v = visit_engine.build_visit(_bay_clip(acc=0.8), start_time=T0, zones=[Zone(id="bay1", kind=ZoneKind.bay)])
        n_events[dec] = len(_times(v, EventType.state_started, "raised"))
    assert n_events["hmm"] == 1 and n_events["baseline"] > 1


def _old_baseline_state_events(preds):
    """The original hysteresis loop (before Lane E) for the regression check."""
    out, active = [], set()
    for p in preds:
        for s in BAY_STATES:
            prob = p.states.get(s)
            if prob is None:
                continue
            if s not in active and prob >= 0.6:
                active.add(s)
                out.append(("state_started", s, p.t_s))
            elif s in active and prob <= 0.4:
                active.discard(s)
                out.append(("state_ended", s, p.t_s))
    return sorted(out, key=lambda x: (x[2], x[0], x[1]))


def test_baseline_flag_reproduces_original_state_events():
    visit_engine.DECODER = "baseline"
    preds = _bay_clip(acc=0.9, n=300)
    v = visit_engine.build_visit(preds, start_time=T0)
    got = sorted([(e.type.value, e.state.value, (e.t - T0).total_seconds()) for e in v.events
                  if e.type in (EventType.state_started, EventType.state_ended)], key=lambda x: (x[2], x[0], x[1]))
    assert got == _old_baseline_state_events(preds)


def test_ready_zone_emits_ready_and_handover():
    visit_engine.DECODER = "hmm"
    preds = []
    for i in range(0, 600, 2):
        on = 100 <= i < 500
        p = FramePred(index=i, t_s=float(i), states={"car_present": 0.9 if on else 0.1})
        if on:
            p.detections.append(CAR)
        preds.append(p)
    v = visit_engine.build_visit(preds, start_time=T0, zones=[Zone(id="ready", kind=ZoneKind.ready)])
    assert [round(t) for t in _times(v, EventType.ready)] == [100]
    assert [round(t) for t in _times(v, EventType.handed_over)] == [500]
    assert not _times(v, EventType.idle_started)  # no idle alarms outside bays


def test_car_present_from_detections_when_no_state_predictor():
    visit_engine.DECODER = "hmm"
    preds = [FramePred(index=i, t_s=float(i), detections=[CAR] if 10 <= i < 50 else []) for i in range(60)]
    v = visit_engine.build_visit(preds, start_time=T0)  # unconfigured camera: generic grammar
    starts = _times(v, EventType.state_started, "car_present")
    ends = _times(v, EventType.state_ended, "car_present")
    assert len(starts) == 1 and abs(starts[0] - 10) <= 2 and len(ends) == 1 and abs(ends[0] - 50) <= 2
    assert _times(v, EventType.arrived) == [10.0]  # first vehicle inside the zone, not the one outside at t=0
    assert not _times(v, EventType.state_started, "raised")  # unobserved labels produce no events


def test_bay_prior_suppresses_short_weak_blip():
    """In a configured bay, a 2 s 'car present' blip is below the scenario duration prior and is not decoded."""
    visit_engine.DECODER = "hmm"
    preds = [FramePred(index=i, t_s=float(i), detections=[CAR] if i in (10, 11) else []) for i in range(60)]
    v = visit_engine.build_visit(preds, start_time=T0, zones=[Zone(id="bay1", kind=ZoneKind.bay)])
    assert not _times(v, EventType.state_started, "car_present")
    assert _times(v, EventType.arrived)  # the arrival (detector) event is unaffected


def test_hmm_guard_keeps_strong_evidence():
    """When the HMM prior would drop 40 frames of confident detections, the decoder falls back to hysteresis and says so
    (added after the HMM decoded 'empty bay' on real timelapse CCTV; see docs/deck_notes.md)."""
    visit_engine.DECODER = "hmm"
    preds = [FramePred(index=i, t_s=float(i), detections=[CAR] if 10 <= i < 50 else []) for i in range(60)]
    v = visit_engine.build_visit(preds, start_time=T0, zones=[Zone(id="bay1", kind=ZoneKind.bay)])
    assert _times(v, EventType.state_started, "car_present")
    assert "fallback" in str(v)


def test_single_frame_input():
    visit_engine.DECODER = "hmm"
    v = visit_engine.build_visit([FramePred(index=0, t_s=0.0, detections=[CAR])], start_time=T0)
    assert v is not None and v.events[0].type == EventType.arrived


def test_label_events_matches_ground_truth_semantics():
    times = np.arange(0, 1000, 1.0)
    lab = np.zeros((len(times), len(BAY_STATES)), dtype=bool)
    k = {s: i for i, s in enumerate(BAY_STATES)}
    lab[100:900, k["car_present"]] = True
    lab[100:200, k["work"]] = lab[100:200, k["person_at_car"]] = True
    obs = np.ones(len(BAY_STATES), dtype=bool)
    ev = visit_engine.label_events(times, lab, lab.astype(float), obs, zone=Zone(id="bay1", kind=ZoneKind.bay),
                                   visit_id="v", ts=lambda i: T0 + timedelta(seconds=float(times[i])),
                                   source=Source.simulated, idle_threshold_s=300)
    idle = [e for e in ev if e.type == EventType.idle_started]
    assert len(idle) == 1 and idle[0].payload["duration_s"] == 700.0


def test_zone_polygon_filters_vehicles_outside():
    visit_engine.DECODER = "hmm"
    other = Det(cls="car", confidence=0.9, bbox=(0.7, 0.2, 0.95, 0.5))  # bottom-centre (0.825, 0.5): outside
    preds = [FramePred(index=i, t_s=float(i), detections=[CAR if 10 <= i < 50 else other]) for i in range(60)]
    poly = [[0.0, 0.3], [0.6, 0.3], [0.6, 1.0], [0.0, 1.0]]
    v = visit_engine.build_visit(preds, start_time=T0, zones=[Zone(id="z", kind=ZoneKind.other, polygon=poly)])
    starts, ends = _times(v, EventType.state_started, "car_present"), _times(v, EventType.state_ended, "car_present")
    assert len(starts) == 1 and abs(starts[0] - 10) <= 2 and len(ends) == 1 and abs(ends[0] - 50) <= 2
    assert _times(v, EventType.arrived) == [10.0]  # first vehicle inside the zone, not the one outside at t=0
