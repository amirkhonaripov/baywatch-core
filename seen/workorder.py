"""Work-order catalogue and evidence matching.

A work-order line gets one of three statuses (never "verified"/"done"):
  evidence_found         every required operation was observed for at least its min duration
  not_observed           at least one required operation was never observed
  not_camera_verifiable  the item has no camera-observable operation (verifiability == "none")
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from seen.contracts.events import Event, LineStatus, LineVerification, Source
from seen.contracts.workorder import Catalogue, WorkOrder

CATALOGUE_PATH = Path(__file__).resolve().parent.parent / "contracts" / "fixtures" / "catalogue.json"

# Which bay states count as which operation.
STATE_TO_OPERATION = {
    "car_present": "car_in_bay",
    "raised": "lift_raise",
    "wheel_off": "wheel_removal",
    "hood_open": "hood_open",
    "person_under_car": "under_car_work",
    "person_at_car": "person_at_car",
    "work": "person_at_car",
    "washing": "wash",
}


@lru_cache(maxsize=1)
def load_catalogue(path: str | None = None) -> Catalogue:
    return Catalogue.model_validate(json.loads(Path(path or CATALOGUE_PATH).read_text()))


# Zones where "car_present" counts as the car being in a work area.
WORK_ZONE_KINDS = {"bay", "wash", "frame"}


def operation_spans(events: list[Event], end_ts: float | None = None) -> dict[str, list[tuple[float, float, list[str]]]]:
    """Return operation -> list of (start_ts, end_ts, event_ids) from state_started/state_ended pairs.

    Open spans are keyed by (operation, zone, state), so overlapping spans of one operation from two zones or from
    two states that map to the same operation ("work" and "person_at_car") do not overwrite each other.
    Only observed events count as evidence (simulated events never verify anything). ``car_present`` only counts
    as ``car_in_bay`` inside bay/wash zones (or the whole-frame default zone).
    """
    spans: dict[str, list[tuple[float, float, list[str]]]] = {}
    open_: dict[tuple, tuple[float, str]] = {}
    for ev in sorted(events, key=lambda e: e.t):
        if ev.source == Source.simulated or ev.state is None:
            continue
        op = STATE_TO_OPERATION.get(ev.state.value)
        if op is None:
            continue
        zkind = ev.zone.kind.value if ev.zone is not None else "frame"
        if op == "car_in_bay" and zkind not in WORK_ZONE_KINDS:
            continue
        key = (op, ev.zone.id if ev.zone is not None else "", ev.state.value)
        ts = ev.t.timestamp()
        if ev.type.value == "state_started":
            open_[key] = (ts, ev.event_id)
        elif ev.type.value == "state_ended" and key in open_:
            t0, eid0 = open_.pop(key)
            spans.setdefault(op, []).append((t0, ts, [eid0, ev.event_id]))
    if events:
        last = max([e.t.timestamp() for e in events] + ([end_ts] if end_ts is not None else []))
        for (op, _, _), (t0, eid0) in open_.items():  # still open at end of input
            spans.setdefault(op, []).append((t0, last, [eid0]))
    return spans


def verify(work_order: WorkOrder, events: list[Event], catalogue: Catalogue | None = None,
           end_ts: float | None = None) -> list[LineVerification]:
    """``end_ts`` = timestamp of the last analysed frame; states still on at the end count until then."""
    cat = catalogue or load_catalogue()
    items = {i.code: i for i in cat.items}
    min_dur = {o.code.value if hasattr(o.code, "value") else o.code: (o.min_duration_s or 0) for o in cat.operations}
    spans = operation_spans(events, end_ts)
    out: list[LineVerification] = []
    for code in work_order.lines:
        item = items.get(code)
        if item is None:
            out.append(LineVerification(line_code=code, status=LineStatus.not_observed, note="unknown catalogue code"))
            continue
        if item.verifiability.value == "none" or not item.operations_required:
            out.append(LineVerification(line_code=code, title=item.title_en, status=LineStatus.not_camera_verifiable, note=item.note))
            continue
        evidence: list[str] = []
        missing: list[str] = []
        for op in item.operations_required:
            ok = [s for s in spans.get(op, []) if (s[1] - s[0]) >= min_dur.get(op, 0)]
            if ok:
                evidence.extend(ok[0][2])
            else:
                missing.append(op)
        status = LineStatus.evidence_found if not missing else LineStatus.not_observed
        note = item.note if not missing else f"not observed: {', '.join(missing)}. {item.note or ''}".strip()
        out.append(LineVerification(line_code=code, title=item.title_en, status=status, evidence_event_ids=evidence, note=note))
    return out


def camera_verifiable_share(catalogue: Catalogue | None = None) -> dict[str, float]:
    cat = catalogue or load_catalogue()
    n = len(cat.items)
    counts: dict[str, int] = {}
    for i in cat.items:
        counts[i.verifiability.value] = counts.get(i.verifiability.value, 0) + 1
    return {k: v / n for k, v in counts.items()}
