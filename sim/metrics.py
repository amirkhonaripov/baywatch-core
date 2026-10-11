"""Scoring decoded visits against scenario ground truth (shared by sim/noise_test.py and sim/eval_synthetic.py).

Event timing  : events are matched one-to-one per category (type, state, zone kind) by smallest |dt| within a
                tolerance (greedy on sorted |dt|, optimal for 1-D times in practice). TP/FP/FN at +-10 s and +-30 s;
                onset errors of the +-30 s matches.
Idle          : spans from idle_started/idle_ended (an ongoing span ends at the end of the clip). Segment-level
                P/R (temporal IoU >= 0.5) and time-level P/R (seconds flagged idle).
Work order    : line status from seen.workorder.verify on decoded bay/wash events vs the same function on
                ground-truth events ("oracle perception"), over camera-verifiable lines; also vs "performed".
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import numpy as np

from seen import workorder
from seen.contracts.events import Event, LineStatus, Source
from seen.contracts.workorder import WorkOrder

# Key events = slow physical states (plus idle, zone, ready and handover events). Person/work states toggle at every
# short pause in the ground truth (seconds-long), so they are scored separately as "fine" events.
KEY_STATES = ("car_present", "raised", "wheel_off", "hood_open", "washing")
FINE_STATES = ("person_at_car", "person_under_car", "work")
SKIP_TYPES = ("arrived", "plate_read")
TOLS = (10.0, 30.0)


def category(e: Event) -> str:
    z = e.zone.kind.value if e.zone is not None else "none"
    return f"{e.type.value}:{e.state.value if e.state else ''}@{z}"


def is_key(cat: str) -> bool:
    typ, rest = cat.split(":", 1)
    state = rest.split("@")[0]
    if typ in ("state_started", "state_ended"):
        return state in KEY_STATES
    return typ in ("idle_started", "idle_ended", "ready", "handed_over", "entered_zone", "left_zone")


def _rel(events: list[Event], t0: datetime) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for e in events:
        if e.type.value in SKIP_TYPES:
            continue
        out[category(e)].append((e.t - t0).total_seconds())
    return out


def match_times(gt: list[float], pr: list[float], tol: float) -> list[tuple[int, int, float]]:
    pairs = sorted((abs(g - p), i, j) for i, g in enumerate(gt) for j, p in enumerate(pr) if abs(g - p) <= tol)
    ug, up, out = set(), set(), []
    for d, i, j in pairs:
        if i not in ug and j not in up:
            ug.add(i)
            up.add(j)
            out.append((i, j, pr[j] - gt[i]))
    return out


def event_counts(gt_events: list[Event], pr_events: list[Event], t0: datetime) -> dict[str, dict]:
    """category -> {tp10, fp10, fn10, tp30, fp30, fn30, err30: [signed seconds]}."""
    G, Pr = _rel(gt_events, t0), _rel(pr_events, t0)
    out = {}
    for cat in set(G) | set(Pr):
        g, p = G.get(cat, []), Pr.get(cat, [])
        rec: dict = {}
        for tol in TOLS:
            m = match_times(g, p, tol)
            tag = int(tol)
            rec[f"tp{tag}"], rec[f"fp{tag}"], rec[f"fn{tag}"] = len(m), len(p) - len(m), len(g) - len(m)
            if tol == 30.0:
                rec["err30"] = [d for _, _, d in m]
        out[cat] = rec
    return out


def idle_spans(events: list[Event], t0: datetime, t_end: float) -> list[tuple[float, float]]:
    spans, start = [], None
    for e in sorted(events, key=lambda e: e.t):
        if e.type.value == "idle_started":
            start = (e.t - t0).total_seconds()
        elif e.type.value == "idle_ended" and start is not None:
            spans.append((start, (e.t - t0).total_seconds()))
            start = None
    if start is not None:
        spans.append((start, t_end))
    return spans


def _iou(a, b) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def idle_counts(gt: list[tuple[float, float]], pr: list[tuple[float, float]], iou_thr: float = 0.5) -> dict:
    used, tp = set(), 0
    for g in gt:
        best = max(((j, _iou(g, p)) for j, p in enumerate(pr) if j not in used), key=lambda x: x[1], default=None)
        if best is not None and best[1] >= iou_thr:
            used.add(best[0])
            tp += 1
    ov = sum(max(0.0, min(g[1], p[1]) - max(g[0], p[0])) for g in gt for p in pr)
    return {"seg_tp": tp, "seg_pred": len(pr), "seg_gt": len(gt), "time_overlap_s": ov,
            "time_pred_s": sum(b - a for a, b in pr), "time_gt_s": sum(b - a for a, b in gt)}


def operation_spans_keyed(events: list[Event]) -> dict[str, list[tuple[float, float, list[str]]]]:
    """Like seen.workorder.operation_spans, but open spans are keyed by (operation, zone id, state).

    The shipped function keys open spans by operation only, so overlapping spans of one operation from two
    zones (bay camera still reports the car while the wash camera already does) or two states mapped to the same
    operation ("work" and "person_at_car") overwrite each other. Reported here as a separate, labelled variant
    until the integrator fixes seen/workorder.py.
    """
    spans: dict[str, list] = {}
    open_: dict[tuple, tuple[float, str]] = {}
    for ev in sorted(events, key=lambda e: e.t):
        if ev.source == Source.simulated or ev.state is None:
            continue
        op = workorder.STATE_TO_OPERATION.get(ev.state.value)
        if op is None:
            continue
        key = (op, ev.zone.id if ev.zone is not None else "", ev.state.value)
        ts = ev.t.timestamp()
        if ev.type.value == "state_started":
            open_[key] = (ts, ev.event_id)
        elif ev.type.value == "state_ended" and key in open_:
            t0, eid0 = open_.pop(key)
            spans.setdefault(op, []).append((t0, ts, [eid0, ev.event_id]))
    if events:
        last = max(e.t.timestamp() for e in events)
        for (op, _, _), (t0, eid0) in open_.items():
            spans.setdefault(op, []).append((t0, last, [eid0]))
    return spans


def verify_keyed(wo: WorkOrder, events: list[Event]) -> dict[str, str]:
    """seen.workorder.verify rules with operation_spans_keyed (status only)."""
    cat = workorder.load_catalogue()
    items = {i.code: i for i in cat.items}
    min_dur = {(o.code.value if hasattr(o.code, "value") else o.code): (o.min_duration_s or 0) for o in cat.operations}
    spans = operation_spans_keyed(events)
    out = {}
    for code in wo.lines:
        it = items.get(code)
        if it is None:
            out[code] = LineStatus.not_observed.value
        elif it.verifiability.value == "none" or not it.operations_required:
            out[code] = LineStatus.not_camera_verifiable.value
        else:
            ok = all(any(b - a >= min_dur.get(op, 0) for a, b, _ in spans.get(op, [])) for op in it.operations_required)
            out[code] = (LineStatus.evidence_found if ok else LineStatus.not_observed).value
    return out


def line_records(wo: WorkOrder, gt_events: list[Event], pr_events: list[Event], performed: list[str]) -> list[dict]:
    """Per camera-verifiable line: oracle status (gt), predicted status with shipped verify (pred) and with
    zone/state-keyed spans (pred_keyed, gt_keyed), and whether the line was actually performed."""
    def bw(evs, relabel):
        out = [e for e in evs if e.zone is not None and e.zone.kind.value in ("bay", "wash")]
        return [e.model_copy(update={"source": Source.observed_synthetic}) for e in out] if relabel else out

    gt_e, pr_e = bw(gt_events, True), bw(pr_events, False)
    g = {v.line_code: v.status for v in workorder.verify(wo, gt_e)}
    p = {v.line_code: v.status for v in workorder.verify(wo, pr_e)}
    gk, pk = verify_keyed(wo, gt_e), verify_keyed(wo, pr_e)
    return [{"code": c, "gt": g[c].value, "pred": p[c].value, "gt_keyed": gk[c], "pred_keyed": pk[c],
             "performed": c in performed}
            for c in wo.lines if g[c] != LineStatus.not_camera_verifiable]


# ------------------------------------------------------------------------------------------- aggregation
def sum_events(recs: list[dict], key_only: bool | None) -> dict:
    tot = defaultdict(float)
    errs: list[float] = []
    for r in recs:
        for cat, c in r.items():
            if key_only is not None and is_key(cat) != key_only:
                continue
            for k, v in c.items():
                if k == "err30":
                    errs.extend(v)
                else:
                    tot[k] += v
    tot["errs"] = errs
    return tot


def f1(tp, fp, fn) -> float:
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else float("nan")


def event_f1(visit_event_recs: list[dict], tol: int, key_only: bool | None = True) -> float:
    t = sum_events(visit_event_recs, key_only)
    return f1(t[f"tp{tol}"], t[f"fp{tol}"], t[f"fn{tol}"])


def median_abs_err(visit_event_recs: list[dict], key_only: bool | None = True) -> float:
    e = sum_events(visit_event_recs, key_only)["errs"]
    return float(np.median(np.abs(e))) if e else float("nan")


def idle_pr(idle_recs: list[dict]) -> dict[str, float]:
    s = defaultdict(float)
    for r in idle_recs:
        for k, v in r.items():
            s[k] += v
    nan = float("nan")
    return {"seg_precision": s["seg_tp"] / s["seg_pred"] if s["seg_pred"] else nan,
            "seg_recall": s["seg_tp"] / s["seg_gt"] if s["seg_gt"] else nan,
            "time_precision": s["time_overlap_s"] / s["time_pred_s"] if s["time_pred_s"] else nan,
            "time_recall": s["time_overlap_s"] / s["time_gt_s"] if s["time_gt_s"] else nan}


def line_acc(line_recs: list[list[dict]], ref: str = "gt", keyed: bool = False, only: str | None = None) -> float:
    """Share of lines whose predicted status matches the reference.

    ref="gt": oracle status (same verify on ground-truth events); ref="performed": evidence_found <-> performed.
    keyed: use the zone/state-keyed span variant for both sides. only: restrict to lines whose reference is
    "evidence_found" / "not_observed" (per-class recall).
    """
    pk, gk = ("pred_keyed", "gt_keyed") if keyed else ("pred", "gt")
    k = n = 0
    for recs in line_recs:
        for r in recs:
            ref_val = r[gk] if ref == "gt" else ("evidence_found" if r["performed"] else "not_observed")
            if only is not None and ref_val != only:
                continue
            n += 1
            k += r[pk] == ref_val
    return k / n if n else float("nan")
