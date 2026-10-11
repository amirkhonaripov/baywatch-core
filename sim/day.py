"""'A day on the service floor': the same simulated day run twice, without and with DNA.

Visits (job mix, bay work steps, parts waits, washes, unperformed paid lines) come from the scenario engine
(sim/scenario.py). A simple dispatcher assigns waiting cars to free bays (FIFO). DNA changes only what the dealer
*knows* and therefore how fast staff react; every effect is an explicit, editable assumption (ASSUMPTIONS below):

  bay_turnover_min   time from "work finished" until the bay is freed for the next car
                     (without DNA nobody notices the finished car at once; with DNA an alert fires)
  parts_wait_move_min  a car waiting for parts blocks its bay; with DNA the idle alert gets it moved to the lot
  ready_notice_min   time until the customer learns the car is ready (without: someone calls eventually / customer
                     calls; with: automatic message when the car reaches the ready zone)
  calls_per_visit    customer status calls without DNA (with DNA: status pushed automatically)
  flag_recall        share of unperformed paid lines flagged before handover with DNA (camera evidence; capped by
                     the measured camera ceiling, see docs/lanes/sim_notes.md)

Outputs JSON for the web (both runs: per-car schedule + metrics + assumptions). Everything is SIMULATED.
    python -m sim.day --out web/public/day/day.json
"""
from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from sim.scenario import LOC, ScenarioConfig, TZ, simulate_visit


@dataclass
class Assumptions:
    bays: int = 6
    open_hours: float = 9.0
    arrivals: int = 56  # cars wanting service per day: a busy dealer where demand exceeds bay capacity (queue complaints)
    labour_rate_usd_h: float = 20.0  # REPLACE with the dealer's labour-hour rate
    working_days_month: int = 26
    bay_turnover_min_without: tuple = (10.0, 40.0)  # uniform range
    bay_turnover_min_with: float = 4.0
    parts_wait_move_min_without: float | None = None  # None = car blocks the bay for the whole wait
    parts_wait_move_min_with: float = 10.0
    ready_notice_min_without: float = 45.0  # mean (exponential)
    ready_notice_min_with: float = 1.0
    calls_per_visit_without: float = 1.5
    flag_recall_with: float = 0.8
    seed: int = 7


def _bay_profile(tl) -> tuple[float, float, float, bool]:
    """Seconds of bay occupancy, of which parts wait; wash seconds; and whether a parts wait happened."""
    loc = tl.location
    bay_s = float((loc == LOC["bay"]).sum())
    wash_s = float((loc == LOC["wash"]).sum()) if "wash" in LOC else 0.0
    names = tl.step_names
    pw = 0.0
    if "parts_wait" in names:
        pw = float((tl.step == names.index("parts_wait")).sum())
    return bay_s, pw, wash_s, pw > 0


@dataclass
class CarPlan:
    car: str
    plate: str
    job: str
    lines: list
    phantom_lines: list
    arrival_s: float
    bay_work_s: float
    parts_wait_s: float
    wash_s: float
    vehicle: str


def make_cars(a: Assumptions) -> list[CarPlan]:
    rng = np.random.default_rng(a.seed)
    cfg = ScenarioConfig()
    day = datetime(2026, 10, 12, 8, 0, tzinfo=TZ)
    cars = []
    arrivals = np.sort(rng.uniform(0, a.open_hours * 3600 * 0.8, a.arrivals))
    for i, t in enumerate(arrivals):
        tl, wo = simulate_visit(rng, cfg, f"day_{i:02d}", day + timedelta(seconds=float(t)))
        bay_s, pw, wash_s, _ = _bay_profile(tl)
        m = tl.meta
        cars.append(CarPlan(car=f"car{i:02d}", plate=m.get("plate", f"01 A {100 + i} AA"), job=m.get("job_type", "service"),
                            lines=list(wo.lines), phantom_lines=list(m.get("phantom_lines", [])), arrival_s=float(t),
                            bay_work_s=max(bay_s - pw, 600.0), parts_wait_s=pw, wash_s=wash_s,
                            vehicle=["sedan", "suv", "hatchback-sports", "taxi"][int(rng.integers(0, 4))]))
    return cars


def run(cars: list[CarPlan], a: Assumptions, with_dna: bool) -> dict:
    rng = np.random.default_rng(a.seed + (1 if with_dna else 0))
    close = a.open_hours * 3600
    free_at = [0.0] * a.bays
    queue: list[tuple[float, int]] = []
    order = sorted(range(len(cars)), key=lambda i: cars[i].arrival_s)
    sched = []
    productive = blocked = 0.0
    for i in order:
        c = cars[i]
        b = int(np.argmin(free_at))
        start = max(c.arrival_s + 300.0, free_at[b])  # 5 min reception before the bay
        if start >= close:
            sched.append({"car": c.car, "plate": c.plate, "vehicle": c.vehicle, "arrival_s": c.arrival_s, "served": False})
            continue
        work = c.bay_work_s
        if c.parts_wait_s > 0:
            mv = a.parts_wait_move_min_with if with_dna else a.parts_wait_move_min_without
            in_bay_wait = c.parts_wait_s if mv is None else min(c.parts_wait_s, mv * 60)
        else:
            in_bay_wait = 0.0
        work_end = start + work + in_bay_wait
        turnover = (a.bay_turnover_min_with if with_dna else rng.uniform(*a.bay_turnover_min_without)) * 60
        free_at[b] = work_end + turnover
        productive += work
        blocked += in_bay_wait + turnover
        # if the car left the bay to wait for parts, it still needs the parts before it is ready
        parts_done = start + work + c.parts_wait_s
        wash_end = max(work_end, parts_done) + c.wash_s
        notice = (a.ready_notice_min_with * 60) if with_dna else float(rng.exponential(a.ready_notice_min_without * 60))
        pickup = wash_end + notice + 20 * 60
        flagged = [l for l in c.phantom_lines if with_dna and rng.random() < a.flag_recall_with]
        sched.append({"car": c.car, "plate": c.plate, "vehicle": c.vehicle, "job": c.job, "lines": c.lines,
                      "arrival_s": c.arrival_s, "bay": b + 1, "bay_start_s": start, "work_end_s": start + work,
                      "parts_wait_s": c.parts_wait_s, "in_bay_wait_s": in_bay_wait, "bay_free_s": free_at[b],
                      "ready_s": wash_end, "notified_s": wash_end + notice, "pickup_s": pickup,
                      "phantom_lines": c.phantom_lines, "flagged_before_handover": flagged, "served": True})
    served = [s for s in sched if s["served"]]
    open_bay_s = a.bays * close
    m = {
        "cars_served": len(served),
        "cars_not_served": len(sched) - len(served),
        "bay_productive_share": round(productive / open_bay_s, 3),
        "bay_blocked_hours": round(blocked / 3600, 1),
        "avg_time_at_dealer_h": round(float(np.mean([s["pickup_s"] - s["arrival_s"] for s in served])) / 3600, 2) if served else 0,
        "avg_ready_to_notified_min": round(float(np.mean([s["notified_s"] - s["ready_s"] for s in served])) / 60, 1) if served else 0,
        "status_calls": round(len(served) * (0.0 if with_dna else a.calls_per_visit_without)),
        "unperformed_paid_lines": sum(len(s["phantom_lines"]) for s in served),
        "flagged_before_handover": sum(len(s["flagged_before_handover"]) for s in served),
    }
    return {"with_dna": with_dna, "schedule": sched, "metrics": m}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="web/public/day/day.json")
    ap.add_argument("--bays", type=int, default=None, help="background bays simulated (the console reserves the rest for real footage)")
    ap.add_argument("--arrivals", type=int, default=None)
    ap.add_argument("--bay-ids", default=None, help="comma list: physical bay numbers for simulated bays 1..N (e.g. 1,3,5,6)")
    args = ap.parse_args()
    a = Assumptions()
    if args.bays: a.bays = args.bays
    if args.arrivals: a.arrivals = args.arrivals
    cars = make_cars(a)
    without, with_ = run(cars, a, False), run(cars, a, True)
    if args.bay_ids:  # map simulated bay k (1..N) to a physical bay number on the console floor
        ids = [int(x) for x in args.bay_ids.split(",")]
        for r in (without, with_):
            for s_ in r["schedule"]:
                if s_.get("bay"):
                    s_["bay"] = ids[s_["bay"] - 1]
    extra_cars = with_["metrics"]["cars_served"] - without["metrics"]["cars_served"]
    freed_h = without["metrics"]["bay_blocked_hours"] - with_["metrics"]["bay_blocked_hours"]
    avg_job_h = float(np.mean([c.bay_work_s for c in cars])) / 3600
    out = {
        "kind": "SIMULATION",
        "note": "Simulated day from the scenario engine. DNA's effect is modelled through the stated assumptions; edit them to see the impact.",
        "assumptions": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(a).items()},
        "cars": [asdict(c) for c in cars],
        "without": without, "with": with_,
        "summary": {
            "extra_cars_per_day": extra_cars,
            "bay_hours_freed_per_day": round(freed_h, 1),
            "avg_bay_job_hours": round(avg_job_h, 2),
            "extra_revenue_per_month_usd": round(max(extra_cars, 0) * avg_job_h * a.labour_rate_usd_h * a.working_days_month),
            "revenue_formula": "extra_cars_per_day x avg_bay_job_hours (billed) x labour_rate_usd_h x working_days_month; counts only freed time that actually became served cars",
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"without": without["metrics"], "with": with_["metrics"], "summary": out["summary"]}, indent=1))


if __name__ == "__main__":
    main()
