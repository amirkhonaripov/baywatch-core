"""Scenario engine: synthetic service visits with exact ground truth (Lane E).

A visit walks the stage grammar  gate -> reception -> waiting -> bay -> [wash] -> ready -> handover/exit.
In the bay a *job* is planned from work-order lines: each line contributes steps that need a physical
configuration (car raised, wheel off, bonnet open, person at / under the car, work or no work); a planner inserts
the transitions between them (raise, lower, wheels off/on, bonnet), adds short pauses, occasional long technician
absences and, for repair jobs, a parts wait where the car sits idle in the bay.

Every duration is drawn from a documented prior (lognormal unless noted). The priors are ASSUMPTIONS, loosely
anchored on public figures (research/service_ops_kpis.md: ~1 h express service, 22 min average handover in the
J.D. Power Vietnam study, "about half of servicing time is the car stagnating between processes" (Toyota TSM),
~20 % workshop idle). They are not measured at any dealer; pilot footage is needed to calibrate them.

Public API
    generate_visits(n, seed, config=None) -> list[(Visit ground truth, WorkOrder, Timeline)]
    continue_visit(visit, work_order) -> Visit   (observed events untouched, added events source="simulated")
    fit_hmm_priors(n, seed) -> dict              (CTMC dwell/transition counts for seen/hmm.py)

Ground-truth labels are what a perfect *camera* labeller would mark: work done inside the cabin (cabin filter)
is invisible, so it counts as "no visible work" (idle) - a real limit of camera evidence, kept on purpose.

CLI
    python -m sim.scenario --fit-priors            -> sim/priors/hmm_rates.json
    python -m sim.scenario --export N SEED OUT_DIR -> scenario JSON per visit (Blender input)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from seen import hmm
from seen.contracts.events import (
    Event, EventType, Source, Vehicle, VehicleType, Visit, Zone, ZoneKind,
)
from seen.contracts.workorder import WorkOrder
from seen.vocab import BAY_STATES

SCENARIO_VERSION = "0.1.0"
LOCATIONS = ("gate", "reception", "waiting", "bay", "wash", "ready", "exit")
LOC = {k: i for i, k in enumerate(LOCATIONS)}
K = len(BAY_STATES)
KI = {s: i for i, s in enumerate(BAY_STATES)}
PRIORS_PATH = Path(__file__).resolve().parent / "priors" / "hmm_rates.json"
TZ = timezone(timedelta(hours=5))  # Tashkent


SLOW_LABELS = ("car_present", "raised", "wheel_off", "hood_open", "washing")
SLOW_OFF_FALLBACK_S = 120.0


# --------------------------------------------------------------------------------------------------- priors
@dataclass(frozen=True)
class Prior:
    """Lognormal duration prior: median seconds and log-sd; clipped to [lo, hi]."""
    median_s: float
    sigma: float
    lo: float = 1.0
    hi: float = 6 * 3600.0

    def sample(self, rng: np.random.Generator, scale: float = 1.0) -> float:
        return float(np.clip(self.median_s * scale * math.exp(self.sigma * rng.standard_normal()), self.lo, self.hi))

    def residual(self, rng: np.random.Generator, elapsed: float, scale: float = 1.0, tries: int = 64) -> float:
        """Remaining duration given the step has already lasted ``elapsed`` seconds (rejection sampling)."""
        for _ in range(tries):
            d = self.sample(rng, scale)
            if d > elapsed:
                return d - elapsed
        return max(self.lo, 0.25 * self.median_s * scale)


@dataclass
class ScenarioConfig:
    duration_scale: float = 1.0  # multiplies every median (prior-mismatch tests)
    work_chunk_mean_s: float = 150.0  # mean uninterrupted work before a short pause (exponential)
    pause: Prior = Prior(20, 0.6, 3, 120)  # short pause inside a work step
    pause_person_leaves_p: float = 0.4
    absence_p: float = 0.12  # per pausable step: technician leaves for a long time
    absence: Prior = Prior(600, 0.6, 120, 3600)
    parts_wait: Prior = Prior(2700, 0.7, 600, 4 * 3600)
    job_mix: dict = field(default_factory=lambda: {
        "express_maintenance": 0.35, "tyre_change": 0.20, "brake_pads": 0.15,
        "chassis_diagnostics": 0.15, "repair_parts_wait": 0.15})
    ev_share: float = 0.30  # BYD-like EVs: no oil change
    wash_p: float = 0.60
    wash_on_order_p: float = 0.70  # a wash that happens is on the work order with this probability
    phantom_line_p: float = 0.25  # work order carries a paid line that is NOT performed
    wait_tech_p: float = 0.5
    paperwork_p: float = 0.5


# Fixed step priors (seconds). All are assumptions, see module docstring.
P = {
    "gate": Prior(20, 0.3, 5, 120),
    "reception_walkaround": Prior(240, 0.4, 60, 1200),
    "reception_wait": Prior(120, 0.8, 10, 1800),
    "waiting": Prior(900, 1.0, 60, 4 * 3600),
    "drive_in": Prior(35, 0.3, 10, 120),
    "wait_tech": Prior(240, 0.8, 20, 3600),
    "walkaround": Prior(90, 0.4, 20, 600),
    "raise": Prior(40, 0.3, 15, 120),
    "lower": Prior(40, 0.3, 15, 120),
    "wheels_off": Prior(200, 0.35, 60, 900),
    "wheels_on": Prior(240, 0.35, 60, 900),
    "close_hood": Prior(10, 0.3, 3, 60),
    "paperwork": Prior(150, 0.8, 20, 1800),
    "drive_out": Prior(30, 0.3, 10, 120),
    "wash_drive_in": Prior(30, 0.3, 10, 120),
    "washing": Prior(480, 0.35, 120, 1800),
    "drying": Prior(240, 0.4, 60, 1200),
    "wash_wait": Prior(60, 0.8, 5, 900),
    "ready_wait": Prior(1800, 0.9, 120, 6 * 3600),
    "handover": Prior(240, 0.5, 60, 1800),
    "exit": Prior(10, 0.2, 3, 60),
}


@dataclass(frozen=True)
class Step:
    """One bay step. ``needs`` = required factor values (raised, wheel, hood); person/work while doing it."""
    name: str
    prior: Prior
    person: int = 1  # 0 none, 1 at the car, 2 under the car
    work: int = 1
    phase: float = 1.0  # 0 = down pre-checks, 1 = raised work, 2 = down after
    raised: int | None = None
    wheel: int | None = None
    hood: int | None = None
    pauses: bool = False
    line: str | None = None


def _line_steps(line: str, rng: np.random.Generator) -> list[Step]:
    """Steps a work-order line contributes (catalogue codes from contracts/fixtures/catalogue.json)."""
    tyre_in_bay = rng.random() < 0.5
    blocks = {
        "oil_filter_change": [
            Step("remove_filler_cap", Prior(60, 0.4, 15, 300), phase=0, raised=0, hood=1, line=line),
            Step("drain_oil_replace_filter", Prior(660, 0.35, 240, 2400), person=2, phase=1, raised=1, pauses=True, line=line),
            Step("refill_oil", Prior(300, 0.3, 120, 1200), phase=2, raised=0, hood=1, line=line)],
        "air_filter": [Step("replace_air_filter", Prior(240, 0.35, 60, 900), phase=0, raised=0, hood=1, line=line)],
        "coolant_check": [Step("check_coolant", Prior(120, 0.4, 30, 600), phase=0, raised=0, hood=1, line=line)],
        "battery_12v_check": [Step("test_12v_battery", Prior(150, 0.4, 30, 600), phase=0, raised=0, hood=1, line=line)],
        "brake_pads": [Step("replace_brake_pads", Prior(1500, 0.35, 600, 4800), phase=1.2, raised=1, wheel=1, pauses=True, line=line)],
        "brake_fluid": [Step("bleed_brakes", Prior(600, 0.35, 240, 2400), phase=1.3, raised=1, wheel=1, hood=1, pauses=True, line=line)],
        "seasonal_tyre_change": [Step("mount_balance_tyres" if tyre_in_bay else "tyres_at_offbay_machine",
                                      Prior(900, 0.35, 300, 3600), person=1 if tyre_in_bay else 0,
                                      work=1 if tyre_in_bay else 0, phase=1.2, raised=1, wheel=1, pauses=tyre_in_bay, line=line)],
        "tyre_rotation": [Step("swap_wheel_positions", Prior(300, 0.35, 90, 1200), phase=1.2, raised=1, wheel=1, line=line)],
        "wheel_alignment": [Step("wheel_alignment", Prior(1200, 0.35, 400, 3600), phase=2.5, raised=0, pauses=True, line=line)],
        "chassis_diagnostics": [Step("inspect_under_car", Prior(900, 0.4, 300, 3600), person=2, phase=1, raised=1, pauses=True, line=line)],
        "suspension_tightening": [Step("tighten_suspension", Prior(900, 0.4, 300, 3600), person=2, phase=1.6, raised=1, pauses=True, line=line)],
        "hv_battery_inspection": [Step("inspect_hv_battery", Prior(600, 0.35, 200, 2400), person=2, phase=1, raised=1, pauses=True, line=line)],
        "cabin_filter": [Step("cabin_filter_inside_car", Prior(300, 0.4, 90, 1200), person=0, work=0, phase=2.6, raised=0, line=line)],
        "computer_diagnostics": [Step("obd_scan_at_driver_door", Prior(480, 0.4, 120, 1800), phase=2.7, raised=0, line=line)],
        "wiper_blades": [Step("replace_wipers", Prior(180, 0.4, 60, 600), phase=2.8, raised=0, line=line)],
    }
    return list(blocks.get(line, []))


def _choose_job(rng: np.random.Generator, cfg: ScenarioConfig, ev: bool) -> tuple[str, list[str]]:
    kinds, probs = zip(*cfg.job_mix.items())
    job = str(rng.choice(kinds, p=np.asarray(probs) / sum(probs)))
    add = lambda p: rng.random() < p  # noqa: E731
    if job == "express_maintenance":
        lines = (["hv_battery_inspection", "coolant_check"] if ev else ["oil_filter_change"])
        lines += [ln for ln, p in (("air_filter", 0.0 if ev else 0.4), ("coolant_check", 0.0 if ev else 0.4),
                                   ("battery_12v_check", 0.3), ("cabin_filter", 0.3), ("computer_diagnostics", 0.3),
                                   ("wiper_blades", 0.15), ("brake_fluid", 0.15)) if add(p)]
    elif job == "tyre_change":
        lines = ["seasonal_tyre_change" if add(0.7) else "tyre_rotation"] + (["wheel_alignment"] if add(0.2) else [])
    elif job == "brake_pads":
        lines = ["brake_pads"] + (["brake_fluid"] if add(0.35) else [])
    elif job == "chassis_diagnostics":
        lines = ["chassis_diagnostics"] + [ln for ln, p in (("suspension_tightening", 0.35), ("computer_diagnostics", 0.3),
                                                             ("hv_battery_inspection", 1.0 if ev else 0.0)) if add(p)]
    else:  # repair with a parts wait between diagnosis and repair
        lines = ["chassis_diagnostics", "parts_wait", "brake_pads" if add(0.4) else "suspension_tightening"]
    seen_: list[str] = []
    for ln in lines:
        if ln not in seen_:
            seen_.append(ln)
    return job, seen_


# ------------------------------------------------------------------------------------------------- planner
@dataclass
class Piece:
    loc: str
    dur: float
    raised: int = 0
    wheel: int = 0
    hood: int = 0
    person: int = 0
    work: int = 0
    wash: int = 0
    step: str = ""
    line: str | None = None

    def cstate(self) -> hmm.CState:
        return hmm.CState(1, self.raised, self.wheel, self.hood, self.person, self.work if self.person else 0, self.wash)


class _Builder:
    def __init__(self, rng: np.random.Generator, cfg: ScenarioConfig, f: dict | None = None):
        self.rng, self.cfg = rng, cfg
        self.f = dict(raised=0, wheel=0, hood=0) if f is None else dict(f)
        self.pieces: list[Piece] = []

    def emit(self, loc: str, dur: float, person: int, work: int, step: str, line: str | None = None, wash: int = 0):
        if dur <= 0:
            return
        self.pieces.append(Piece(loc, dur, self.f["raised"], self.f["wheel"], self.f["hood"], person,
                                 work if person else 0, wash, step, line))

    def sample(self, key_or_prior, first_elapsed: float | None = None) -> float:
        pr = P[key_or_prior] if isinstance(key_or_prior, str) else key_or_prior
        if first_elapsed is not None:
            return pr.residual(self.rng, first_elapsed, self.cfg.duration_scale)
        return pr.sample(self.rng, self.cfg.duration_scale)

    def work_step(self, loc: str, dur: float, person: int, work: int, step: str, line: str | None, pauses: bool):
        """Work with short pauses (person stays or steps away) and possibly one long absence."""
        if not pauses or not work:
            self.emit(loc, dur, person, work, step, line)
            return
        absence_at = self.rng.uniform(0.2, 0.8) * dur if self.rng.random() < self.cfg.absence_p else None
        t = 0.0
        while t < dur:
            chunk = min(self.rng.exponential(self.cfg.work_chunk_mean_s), dur - t)
            if absence_at is not None and t <= absence_at < t + chunk:
                self.emit(loc, absence_at - t, person, 1, step, line)
                self.emit(loc, self.cfg.absence.sample(self.rng), 0, 0, "technician_absent", line)
                chunk -= absence_at - t
                t = absence_at
                absence_at = None
            self.emit(loc, chunk, person, 1, step, line)
            t += chunk
            if t < dur:
                pz = min(self.cfg.pause.sample(self.rng), dur - t)
                leaves = self.rng.random() < self.cfg.pause_person_leaves_p
                self.emit(loc, pz, 0 if leaves else person, 0, "short_pause", line)
                t += pz

    # transitions inserted by the planner
    def ensure(self, raised: int | None, wheel: int | None, hood: int | None):
        if raised == 0 and self.f["raised"]:
            if self.f["wheel"]:
                self.emit("bay", self.sample("wheels_on"), 1, 1, "wheels_on")
                self.f["wheel"] = 0
            self.f["raised"] = 0
            self.emit("bay", self.sample("lower"), 1, 1, "lower_lift")
        if (raised == 1 or wheel == 1) and not self.f["raised"]:
            self.f["raised"] = 1
            self.emit("bay", self.sample("raise"), 1, 1, "raise_lift")
        if wheel == 1 and not self.f["wheel"]:
            self.f["wheel"] = 1
            self.emit("bay", self.sample("wheels_off"), 1, 1, "wheels_off")
        if wheel == 0 and self.f["wheel"]:
            self.emit("bay", self.sample("wheels_on"), 1, 1, "wheels_on")
            self.f["wheel"] = 0
        if hood is not None:
            self.f["hood"] = hood

    def bay_steps(self, steps: list[Step], first_elapsed: float | None = None):
        for i, st in enumerate(steps):
            if st.name == "parts_wait":
                if self.f["raised"] and not self.f["wheel"] and i > 0 and self.rng.random() < 0.5:
                    self.ensure(0, None, None)  # car lowered for the wait; the next step raises it again
                self.emit("bay", self.sample(self.cfg.parts_wait, first_elapsed if i == 0 else None), 0, 0, "parts_wait")
                continue
            if st.hood is None and st.phase >= 1 and self.f["hood"] and self.rng.random() < 0.5:
                self.f["hood"] = 0  # bonnet closed before going up
            self.ensure(st.raised, st.wheel, st.hood)
            dur = self.sample(st.prior, first_elapsed if i == 0 else None)
            self.work_step("bay", dur, st.person, st.work, st.name, st.line, st.pauses)

    def close_bay(self):
        self.ensure(0, 0, None)
        if self.f["hood"]:
            self.emit("bay", self.sample("close_hood"), 1, 1, "close_hood")
            self.f["hood"] = 0
        if self.rng.random() < self.cfg.paperwork_p:
            self.emit("bay", self.sample("paperwork"), 0, 0, "paperwork_away")
        self.emit("bay", self.sample("drive_out"), 0, 0, "drive_out")

    def wash(self, first: str = "drive_in", first_elapsed: float | None = None):
        order = ["drive_in", "washing", "drying", "wait"]
        for k in order[order.index(first):]:
            fe = first_elapsed if k == first else None
            if k == "drive_in":
                self.emit("wash", self.sample("wash_drive_in", fe), 0, 0, "wash_drive_in")
            elif k == "washing":
                self.emit("wash", self.sample("washing", fe), 1, 1, "washing", "wash", wash=1)
            elif k == "drying":
                self.emit("wash", self.sample("drying", fe), 1, 1, "drying", "wash")
            else:
                self.emit("wash", self.sample("wash_wait", fe), 0, 0, "wash_wait")

    def ready(self, in_wait: bool = True, first_elapsed: float | None = None):
        if in_wait:
            self.emit("ready", self.sample("ready_wait", first_elapsed), 0, 0, "ready_wait")
            first_elapsed = None
        self.emit("ready", self.sample("handover", first_elapsed), 1, 0, "handover")
        self.emit("exit", self.sample("exit"), 0, 0, "exit")


def plan_steps(lines: list[str], rng: np.random.Generator) -> list[Step]:
    """Order line steps by phase (pre-checks down, raised work, work after lowering); lines listed after a
    parts wait are planned after it."""
    steps: list[Step] = []
    offset = 0.0
    for ln in lines:
        if ln == "parts_wait":
            steps.append(Step("parts_wait", P["waiting"], person=0, work=0, phase=1.5))
            offset = 10.0
        else:
            steps.extend(replace(st, phase=st.phase + offset) for st in _line_steps(ln, rng))
    return sorted(steps, key=lambda s: s.phase)  # stable: keeps line order within a phase


# ------------------------------------------------------------------------------------------------ timeline
@dataclass
class Timeline:
    """Per-second ground truth. ``labels`` are relative to the car's current location (car_present=1 in every zone)."""
    visit_id: str
    t0: datetime
    location: np.ndarray  # (T,) int8 index into LOCATIONS
    labels: np.ndarray  # (T, K) uint8, columns = BAY_STATES
    step: np.ndarray  # (T,) int16 index into step_names
    step_names: list[str]
    meta: dict

    @property
    def T(self) -> int:
        return len(self.location)

    def span(self, kind: str) -> tuple[int, int] | None:
        idx = np.flatnonzero(self.location == LOC[kind])
        return (int(idx[0]), int(idx[-1]) + 1) if len(idx) else None

    def view(self, kind: str, pad_before: int = 0, pad_after: int = 0) -> tuple[np.ndarray, np.ndarray] | None:
        """What a perfect camera watching one zone of this kind sees, on a 1-s grid (empty zone outside the visit)."""
        sp = self.span(kind)
        if sp is None:
            return None
        a, b = sp[0] - pad_before, sp[1] + pad_after
        idx = np.arange(a, b)
        lab = np.zeros((len(idx), K), dtype=np.uint8)
        ok = (idx >= 0) & (idx < self.T)
        inside = ok.copy()
        inside[ok] = self.location[idx[ok]] == LOC[kind]
        lab[inside] = self.labels[idx[inside]]
        return idx.astype(float), lab

    def to_json(self) -> dict:
        return {"visit_id": self.visit_id, "t0": self.t0.isoformat(), "scenario_version": SCENARIO_VERSION,
                "labels": list(BAY_STATES), "locations": list(LOCATIONS), "meta": self.meta}


def _build_timeline(pieces: list[Piece], visit_id: str, t0: datetime, meta: dict) -> Timeline:
    loc, lab, stp, names = [], [], [], []
    nidx: dict[str, int] = {}
    t = 0
    meta["pieces"] = []
    for pc in pieces:
        n = max(1, int(round(pc.dur)))
        cs = pc.cstate()
        loc.append(np.full(n, LOC[pc.loc], dtype=np.int8))
        lab.append(np.tile(np.array(cs.labels(), dtype=np.uint8), (n, 1)))
        if pc.step not in nidx:
            nidx[pc.step] = len(names)
            names.append(pc.step)
        stp.append(np.full(n, nidx[pc.step], dtype=np.int16))
        meta["pieces"].append({"loc": pc.loc, "start_s": t, "dur_s": n, "raised": pc.raised, "wheel": pc.wheel,
                               "hood": pc.hood, "person": pc.person, "work": int(bool(pc.work and pc.person)),
                               "wash": pc.wash, "step": pc.step, "line": pc.line})
        t += n
    return Timeline(visit_id, t0, np.concatenate(loc), np.concatenate(lab), np.concatenate(stp), names, meta)


def _plate(rng: np.random.Generator) -> str:
    region = rng.choice(["01", "10", "20", "25", "30", "40", "50", "60", "70", "75", "80", "85", "90", "95"])
    L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return f"{region}{rng.choice(list(L))}{rng.integers(0, 1000):03d}{''.join(rng.choice(list(L), 2))}"


def simulate_visit(rng: np.random.Generator, cfg: ScenarioConfig, visit_id: str, t0: datetime,
                   bay_id: str = "bay1") -> tuple[Timeline, WorkOrder]:
    ev = rng.random() < cfg.ev_share
    job, lines = _choose_job(rng, cfg, ev)
    b = _Builder(rng, cfg)
    b.emit("gate", b.sample("gate"), 0, 0, "gate")
    b.emit("reception", b.sample("reception_walkaround"), 1, 1, "reception_walkaround")
    b.emit("reception", b.sample("reception_wait"), 0, 0, "reception_wait")
    b.emit("waiting", b.sample("waiting"), 0, 0, "waiting_for_bay")
    b.emit("bay", b.sample("drive_in"), 0, 0, "drive_in")
    if rng.random() < cfg.wait_tech_p:
        b.emit("bay", b.sample("wait_tech"), 0, 0, "wait_for_technician")
    b.emit("bay", b.sample("walkaround"), 1, 1, "bay_walkaround")
    b.bay_steps(plan_steps(lines, rng))
    b.close_bay()
    washed = rng.random() < cfg.wash_p
    if washed:
        b.wash()
    b.ready()
    performed = [ln for ln in lines if ln != "parts_wait"] + (["wash"] if washed else [])
    wo_lines = [ln for ln in performed if ln != "wash" or rng.random() < cfg.wash_on_order_p]
    phantom: list[str] = []
    if rng.random() < cfg.phantom_line_p:
        cands = [c for c in ("brake_pads", "tyre_rotation", "air_filter", "wheel_alignment", "chassis_diagnostics",
                             "wash", "oil_filter_change", "wiper_blades") if c not in performed]
        phantom = [str(rng.choice(cands))]
        wo_lines += phantom
    plate = _plate(rng)
    meta = {"job_type": job, "powertrain": "ev" if ev else "ice", "bay_id": bay_id, "plate": plate,
            "lines_performed": performed, "phantom_lines": phantom, "washed": washed,
            "config": asdict(cfg)}
    tl = _build_timeline(b.pieces, visit_id, t0, meta)
    return tl, WorkOrder(id=f"wo_{visit_id}", plate_text=plate, lines=wo_lines)


# ----------------------------------------------------------------------------------------- ground truth
def zone_for(kind: str, bay_id: str = "bay1") -> Zone:
    if kind == "bay":
        return Zone(id=bay_id, kind=ZoneKind.bay, label=bay_id.replace("bay", "Bay "))
    return Zone(id=kind, kind=ZoneKind(kind), label=kind.capitalize())


# Which labels a camera in each zone kind is assumed to provide (states outside these are not reported there).
ZONE_LABELS = {
    "bay": [s for s in BAY_STATES if s != "washing"],
    "wash": ["car_present", "person_at_car", "work", "idle", "washing"],
    "ready": ["car_present"],
    "reception": ["car_present"],
    "waiting": ["car_present"],
}


def view_events(tl: Timeline, kind: str, visit_id: str, source: Source = Source.simulated,
                idle_threshold_s: float | None = None) -> list[Event]:
    from seen.visit_engine import label_events

    v = tl.view(kind, 1, 1)
    if v is None:
        return []
    times, lab = v
    observed = np.array([s in ZONE_LABELS[kind] for s in BAY_STATES])
    evs = label_events(times, lab.astype(bool), lab.astype(float), observed, zone=zone_for(kind, tl.meta["bay_id"]),
                       visit_id=visit_id, ts=lambda i: tl.t0 + timedelta(seconds=float(times[i])), source=source,
                       idle_threshold_s=idle_threshold_s)
    return [e.model_copy(update={"event_id": f"{visit_id}_{kind}_{i:03d}"}) for i, e in enumerate(evs)]


def gt_visit(tl: Timeline, wo: WorkOrder | None = None, idle_threshold_s: float | None = None) -> Visit:
    from seen import workorder

    vid = tl.visit_id
    gate = zone_for("gate")
    evs = [Event(event_id=f"{vid}_arr", visit_id=vid, type=EventType.arrived, t=tl.t0, source=Source.simulated, zone=gate),
           Event(event_id=f"{vid}_plate", visit_id=vid, type=EventType.plate_read, t=tl.t0 + timedelta(seconds=3),
                 source=Source.simulated, zone=gate, payload={"text": tl.meta["plate"]})]
    for kind in ("reception", "waiting", "bay", "wash", "ready"):
        evs += view_events(tl, kind, vid, idle_threshold_s=idle_threshold_s)
    evs.sort(key=lambda e: e.t)
    visit = Visit(visit_id=vid, created_at=tl.t0, events=evs, current_stage=evs[-1].type.value,
                  vehicle=Vehicle(plate_text=tl.meta["plate"], plate_confidence=1.0, plate_format="uz_private",
                                  plate_format_valid=True, vehicle_type=VehicleType.car))
    if wo is not None:
        visit.work_order_id = wo.id
        visit.verifications = workorder.verify(wo, oracle_events(evs))
    return visit


def oracle_events(events: list[Event]) -> list[Event]:
    """Ground-truth events as if observed by perfect perception, restricted to bay and wash zones.

    workorder.verify ignores simulated events, so ground truth is relabelled observed_synthetic here (only for
    computing the reference line status). Reception/ready zones are excluded because verify() is zone-agnostic.
    """
    return [e.model_copy(update={"source": Source.observed_synthetic}) for e in events
            if e.zone is not None and e.zone.kind.value in ("bay", "wash")]


def generate_visits(n: int, seed: int, config: ScenarioConfig | None = None,
                    day: datetime | None = None) -> list[tuple[Visit, WorkOrder, Timeline]]:
    cfg = config or ScenarioConfig()
    rng = np.random.default_rng(seed)
    day = day or datetime(2026, 10, 12, 8, 0, tzinfo=TZ)
    out = []
    for i in range(n):
        t0 = day + timedelta(days=i // 40, seconds=float(rng.uniform(0, 8 * 3600)))
        tl, wo = simulate_visit(rng, cfg, f"sim{seed}_{i:04d}", t0, bay_id=f"bay{1 + i % 2}")
        out.append((gt_visit(tl, wo), wo, tl))
    return out


# ------------------------------------------------------------------------------------------ prior fitting
def fit_hmm_priors(n: int = 600, seed: int = 1000, config: ScenarioConfig | None = None) -> dict:
    """CTMC sufficient statistics (seconds in each composite state, transition counts) per zone kind."""
    rng = np.random.default_rng(seed + 7)
    stats = {k: (np.zeros(hmm.S), np.zeros((hmm.S, hmm.S))) for k in ("bay", "wash", "ready", "pooled")}
    segs = {k: {lab: ([], []) for lab in BAY_STATES} for k in stats}
    for _, _, tl in generate_visits(n, seed, config):
        for kind in ("bay", "wash", "ready"):
            v = tl.view(kind, int(rng.uniform(30, 600)), int(rng.uniform(30, 600)))
            if v is None:
                continue
            _, lab = v
            st = np.array([hmm.labels_to_state(r) for r in lab])
            for li, name in enumerate(BAY_STATES):
                for a, b, val in hmm._runs(lab[:, li].astype(bool)):
                    if a > 0 and b < len(lab):  # interior segments only
                        for key in (kind, "pooled"):
                            segs[key][name][0 if val else 1].append(b - a)
            for key in (kind, "pooled"):
                dw, cn = stats[key]
                np.add.at(dw, st, 1.0)
                ch = np.flatnonzero(st[1:] != st[:-1])
                np.add.at(cn, (st[ch], st[ch + 1]), 1.0)
    zones = {}
    for kind, (dw, cn) in stats.items():
        ii, jj = np.nonzero(cn)
        q01 = lambda v: float(np.percentile(v, 1)) if len(v) >= 20 else 0.0  # noqa: E731
        # Slow labels never toggle off mid-visit in the scenario; assume a gap < 120 s is not real (documented).
        fb = lambda lab, v: q01(v) if len(v) >= 20 else (SLOW_OFF_FALLBACK_S if lab in SLOW_LABELS else 0.0)  # noqa: E731
        zones[kind] = {"dwell_s": dw.tolist(), "counts": [[int(i), int(j), float(cn[i, j])] for i, j in zip(ii, jj)],
                       "min_seg_s": {lab: [q01(on), fb(lab, off)] for lab, (on, off) in segs[kind].items()},
                       "median_seg_s": {lab: [float(np.median(on)) if on else None, float(np.median(off)) if off else None]
                                        for lab, (on, off) in segs[kind].items()}}
    return {"version": SCENARIO_VERSION, "fitted_on": {"n_visits": n, "seed": seed, "generator": "sim.scenario"},
            "alpha": 0.5, "beta_s": 600.0, "states": [s.as_list() for s in hmm.STATES],
            "state_factors": ["car", "raised", "wheel", "hood", "person", "work", "wash"], "zones": zones}


# --------------------------------------------------------------------------------------- continuation
def _phase_of(visit: Visit, now: datetime | None = None) -> tuple[str, dict, float]:
    """(phase, open factor values, seconds since the last observed state change)."""
    evs = sorted(visit.events, key=lambda e: e.t)
    now = now or evs[-1].t
    if any(e.type in (EventType.handed_over,) for e in evs):
        return "done", {}, 0.0
    open_: dict[str, datetime] = {}
    last_change = evs[0].t
    for e in evs:
        if e.type == EventType.state_started and e.state is not None:
            open_[e.state.value] = e.t
            last_change = e.t
        elif e.type == EventType.state_ended and e.state is not None:
            open_.pop(e.state.value, None)
            last_change = e.t
    kinds = [e.zone.kind.value for e in evs if e.zone is not None]
    last_kind = kinds[-1] if kinds else "frame"
    if last_kind == "ready" or any(e.type == EventType.ready for e in evs):
        phase = "ready"
    elif last_kind == "wash" or "washing" in open_:
        phase = "wash"
    elif last_kind == "bay" or (set(open_) - {"car_present"}):
        phase = "bay"
    else:
        phase = "arrival"
    if phase == "bay" and "car_present" not in open_ and any(e.type == EventType.state_ended and e.state is not None
                                                             and e.state.value == "car_present" for e in evs):
        phase = "after_bay"
    f = {"raised": int("raised" in open_), "wheel": int("wheel_off" in open_), "hood": int("hood_open" in open_),
         "person": 2 if "person_under_car" in open_ else int("person_at_car" in open_ or "work" in open_),
         "work": int("work" in open_), "washing": int("washing" in open_)}
    return phase, f, max(0.0, (now - last_change).total_seconds())


def _job_lines(visit: Visit, wo: WorkOrder | None) -> list[str]:
    """Work-order lines (minus wash), or a job guessed from the observed states when there is no work order."""
    if wo is not None:
        return [ln for ln in wo.lines if ln != "wash"]
    st = {e.state.value for e in visit.events if e.state is not None}
    if "wheel_off" in st:
        return ["brake_pads"]
    if "person_under_car" in st or "raised" in st:
        return ["chassis_diagnostics"]
    return ["oil_filter_change"]


def _config_history(visit: Visit, now: datetime) -> list[tuple[tuple[int, int, int, int], float]]:
    """Observed bay configurations (raised, wheel, hood, person) with durations, from bay/frame/other events."""
    evs = sorted((e for e in visit.events if e.state is not None and e.zone is not None
                  and e.zone.kind.value in ("bay", "frame", "other")), key=lambda e: e.t)
    open_: set[str] = set()
    hist: list[list] = []
    i = 0
    while i < len(evs):
        t = evs[i].t
        while i < len(evs) and evs[i].t == t:
            e = evs[i]
            (open_.add if e.type == EventType.state_started else open_.discard)(e.state.value)
            i += 1
        cfg = (int("raised" in open_), int("wheel_off" in open_), int("hood_open" in open_),
               2 if "person_under_car" in open_ else int("person_at_car" in open_ or "work" in open_))
        if hist and hist[-1][0] == cfg:
            continue
        hist.append([cfg, t])
    out = []
    for j, (cfg, t) in enumerate(hist):
        t_next = hist[j + 1][1] if j + 1 < len(hist) else now
        out.append((cfg, (t_next - t).total_seconds()))
    return out


def _matches(st: Step, cfg: tuple[int, int, int, int]) -> bool:
    raised, wheel, hood, person = cfg
    return ((st.raised is None or st.raised == raised) and (st.wheel is None or st.wheel == wheel)
            and (st.hood is None or st.hood == hood) and st.person == person)


def _progress(steps: list[Step], hist) -> tuple[int, bool, float]:
    """Greedy in-order matching of planned steps to observed configurations (short no-person gaps = pauses).

    Returns (index of the first step not yet started, whether the step before it is still in progress,
    seconds since that step started). Consecutive steps with the same configuration (e.g. two under-car
    inspections) are told apart only by time: the elapsed time is spent across them at their median durations."""
    idx = 0
    started_at = 0
    for k, (cfg, dur) in enumerate(hist):
        last = k == len(hist) - 1
        if cfg[3] == 0 and dur < 120 and not last:
            continue  # a short pause, not a step
        if idx > 0 and _matches(steps[idx - 1], cfg):
            continue
        j = idx
        while j < len(steps):  # only steps without a visible person (waits, in-cabin work) may be skipped
            st = steps[j]
            if _matches(st, cfg) and (last or dur >= 0.5 * st.prior.lo):
                idx = j + 1
                started_at = k
                break
            if st.person != 0:
                break
            j += 1
    in_progress = bool(hist) and idx > 0 and _matches(steps[idx - 1], hist[-1][0])
    elapsed = float(sum(d for _, d in hist[started_at:])) if in_progress else 0.0
    if in_progress:
        while idx < len(steps) and _matches(steps[idx], hist[-1][0]) and elapsed > steps[idx - 1].prior.median_s:
            elapsed -= steps[idx - 1].prior.median_s
            idx += 1
    return idx, in_progress, elapsed


def _p_parts_wait(lines: list[str], cfg: ScenarioConfig) -> float:
    """P(a parts wait | lines) implied by the scenario job mix (repair jobs = diagnosis + wait + repair)."""
    if "chassis_diagnostics" not in lines:
        return 0.0
    if "brake_pads" in lines:
        return 1.0
    if "suspension_tightening" in lines:
        rep = cfg.job_mix.get("repair_parts_wait", 0) * 0.6
        diag = cfg.job_mix.get("chassis_diagnostics", 0) * 0.35
        return rep / (rep + diag) if rep + diag else 0.0
    return 0.0


def _continuation_pieces(rng: np.random.Generator, cfg: ScenarioConfig, phase: str, f: dict, elapsed: float,
                         lines: list[str], wash_p: float, hist) -> list[Piece]:
    lines = list(lines)
    if "parts_wait" not in lines and rng.random() < _p_parts_wait(lines, cfg):
        lines.insert(lines.index("chassis_diagnostics") + 1, "parts_wait")
    wash = rng.random() < wash_p
    b = _Builder(rng, cfg, {"raised": f.get("raised", 0), "wheel": f.get("wheel", 0), "hood": f.get("hood", 0)})
    if phase == "arrival":
        b.emit("reception", b.sample("reception_walkaround"), 1, 1, "reception_walkaround")
        b.emit("reception", b.sample("reception_wait"), 0, 0, "reception_wait")
        b.emit("waiting", b.sample("waiting"), 0, 0, "waiting_for_bay")
        b.emit("bay", b.sample("drive_in"), 0, 0, "drive_in")
        if rng.random() < cfg.wait_tech_p:
            b.emit("bay", b.sample("wait_tech"), 0, 0, "wait_for_technician")
        b.emit("bay", b.sample("walkaround"), 1, 1, "bay_walkaround")
        b.bay_steps(plan_steps(lines, rng))
        b.close_bay()
    elif phase == "bay":
        steps = plan_steps(lines, rng)
        idx, in_prog, step_elapsed = _progress(steps, hist)
        rest = steps[idx - 1:] if in_prog else steps[idx:]
        if rest:
            b.bay_steps(rest, first_elapsed=step_elapsed if in_prog else None)
        elif f.get("person", 0) == 0 and not any(f.get(k) for k in ("raised", "wheel", "hood")):
            b.emit("bay", P["paperwork"].residual(rng, elapsed, cfg.duration_scale), 0, 0, "paperwork_away")
        b.close_bay()
    if phase in ("arrival", "bay", "after_bay") and wash:
        b.wash()
    elif phase == "wash":
        b.wash("washing" if f.get("washing") else "drying", first_elapsed=elapsed)
    if phase == "ready":
        b.ready(in_wait=f.get("person", 0) == 0, first_elapsed=elapsed)
    elif phase != "done":
        b.ready()
    return b.pieces


def continue_visit(visit: Visit, work_order: WorkOrder | None = None, *, n_samples: int = 200,
                   config: ScenarioConfig | None = None, seed: int | None = None, now: datetime | None = None) -> Visit:
    """Extend an observed visit with plausible future stages.

    Observed events are copied unchanged; every added event has source="simulated" and payload.simulated=True.
    The added path is the Monte-Carlo sample whose ready time is closest to the median; the ``ready`` event
    carries the ETA quantiles (p10/p50/p90) over ``n_samples`` continuations. ``now`` = end of the observed
    input (default: time of the last observed event); simulated events start after it.
    """
    cfg = config or ScenarioConfig()
    out = visit.model_copy(deep=True)
    if not visit.events:
        return out
    last = max(e.t for e in visit.events)
    now = max(now, last) if now is not None else last
    phase, f, elapsed = _phase_of(visit, now)
    if phase == "done":
        return out
    lines = _job_lines(visit, work_order)
    hist = _config_history(visit, now)
    observed_wash = any(e.state is not None and e.state.value == "washing" for e in visit.events)
    if observed_wash:
        wash_p = 0.0
    elif work_order is not None and "wash" in work_order.lines:
        wash_p = 1.0
    else:  # washes happen without being on the order (courtesy wash)
        q = cfg.wash_p * (1 - cfg.wash_on_order_p)
        wash_p = q / (1 - cfg.wash_p * cfg.wash_on_order_p) if work_order is not None else cfg.wash_p
    s = seed if seed is not None else int(hashlib.sha1(visit.visit_id.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(s)
    samples = []
    for _ in range(n_samples):
        pcs = _continuation_pieces(rng, cfg, phase, f, elapsed, lines, wash_p, hist)
        t, t_ready = 0.0, None
        for pc in pcs:
            if pc.loc == "ready" and t_ready is None:
                t_ready = t
            t += max(1, round(pc.dur))
        samples.append((t_ready if t_ready is not None else 0.0, pcs))
    readys = np.array([r for r, _ in samples])
    p10, p50, p90 = np.quantile(readys, [0.1, 0.5, 0.9])
    pcs = samples[int(np.argmin(np.abs(readys - p50)))][1]
    bay_id = next((e.zone.id for e in visit.events if e.zone is not None and e.zone.kind.value == "bay"), "bay1")
    # Prefix one second of the current configuration so ongoing states do not produce "started" events.
    cur = Piece("bay" if phase == "bay" else {"arrival": "gate", "after_bay": "bay", "wash": "wash", "ready": "ready"}[phase],
                1, f.get("raised", 0), f.get("wheel", 0), f.get("hood", 0), f.get("person", 0), f.get("work", 0),
                f.get("washing", 0), "observed_now")
    if phase == "after_bay":
        cur.loc = "waiting"
    tl = _build_timeline([cur] + pcs, visit.visit_id, now, {"bay_id": bay_id, "plate": ""})
    obs_zone = {e.zone.kind.value: e.zone for e in visit.events if e.zone is not None}
    if "bay" not in obs_zone and phase == "bay":  # car observed in the bay on an unnamed camera zone
        generic = [e.zone for e in visit.events if e.zone is not None and e.zone.kind.value in ("frame", "other")]
        if generic:
            obs_zone["bay"] = generic[-1]
    added: list[Event] = []
    for kind in ("reception", "waiting", "bay", "wash", "ready"):
        for e in view_events(tl, kind, visit.visit_id, Source.simulated):
            if e.t <= now:
                continue
            z = obs_zone.get(kind, e.zone)
            pay = dict(e.payload or {})
            pay.update({"simulated": True, "method": "scenario_engine", "scenario_version": SCENARIO_VERSION,
                        "phase_at_cut": phase})
            if e.type == EventType.ready:
                pay.update({"eta_p10": (now + timedelta(seconds=float(p10))).isoformat(),
                            "eta_p50": (now + timedelta(seconds=float(p50))).isoformat(),
                            "eta_p90": (now + timedelta(seconds=float(p90))).isoformat(), "n_samples": n_samples})
            added.append(e.model_copy(update={"event_id": f"sim_{e.event_id}_{now.timestamp():.0f}", "zone": z, "payload": pay,
                                              "source": Source.simulated, "evidence": None}))
    added.sort(key=lambda e: e.t)
    out.events = out.events + added
    return out


# ------------------------------------------------------------------------------------------------------ CLI
def _export(n: int, seed: int, out_dir: Path, config: ScenarioConfig | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for visit, wo, tl in generate_visits(n, seed, config):
        d = tl.to_json()
        d["work_order"] = wo.model_dump()
        d["gt_events"] = json.loads(visit.model_dump_json())["events"]
        d["gt_verifications"] = json.loads(visit.model_dump_json()).get("verifications")
        (out_dir / f"{tl.visit_id}.json").write_text(json.dumps(d, indent=1))
    print(f"wrote {n} scenarios to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-priors", action="store_true")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--export", nargs=3, metavar=("N", "SEED", "OUT_DIR"))
    a = ap.parse_args()
    if a.fit_priors:
        PRIORS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PRIORS_PATH.write_text(json.dumps(fit_hmm_priors(a.n, a.seed)))
        print(f"wrote {PRIORS_PATH}")
    if a.export:
        _export(int(a.export[0]), int(a.export[1]), Path(a.export[2]))


if __name__ == "__main__":
    main()
