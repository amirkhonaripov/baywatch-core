"""Perception-free noise test (Lane E): how good must per-frame state recognition be?

Ground-truth state timelines from the scenario engine are turned into fake per-frame predictor output with a
controlled per-frame accuracy, then decoded by the visit engine (baseline hysteresis vs HMM) and scored against
the ground truth: event onset error and F1, idle-detection precision/recall, and work-order line status accuracy
(seen.workorder.verify on decoded vs ground-truth events).

Noise model (per camera zone: bay, wash, ready; frames sampled at a random 0.5-2 fps, a share of frames dropped):
    for each label k the predictor score is  s = (2y-1) * m + n_t,  m = Phi^-1(a),
    n_t an AR(1) Gaussian with time constant ``corr_s`` (0 = independent per frame = flicker; 10-60 s = sustained
    errors such as occlusion), so P(s > 0 | y) = a for every frame. The reported probability is the calibrated
    posterior for equal priors, p = sigmoid(2 m s). Measured per-frame accuracy (p > 0.5 vs truth) is reported too.
    Labels per zone: bay = car_present, raised, wheel_off, hood_open, person_at_car, person_under_car, work;
    wash = car_present, person_at_car, work, washing; ready = car_present. A vehicle detection is added to frames
    whose car_present probability is > 0.5.

Everything here is synthetic. The decoder's priors come from the same generator (optimistic); the "mismatch"
condition tests visits drawn with different durations, pause rates and job mix.

    python -m sim.noise_test --tune            # choose HMM eps/weight on dev seeds
    python -m sim.noise_test                   # full grid -> runs/sim/noise_test/
    python -m sim.noise_test --quick           # small smoke run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path
from statistics import NormalDist

import numpy as np

from eval.stats import boot_metric, metric
from seen import hmm, visit_engine
from seen.contracts.events import EventType, Source
from seen.predictor import Det, FramePred
from seen.vocab import BAY_STATES
from sim import metrics as M
from sim.scenario import ScenarioConfig, Timeline, continue_visit, generate_visits, zone_for

NOISE_LABELS = {
    "bay": ["car_present", "raised", "wheel_off", "hood_open", "person_at_car", "person_under_car", "work"],
    "wash": ["car_present", "person_at_car", "work", "washing"],
    "ready": ["car_present"],
}
KI = {s: i for i, s in enumerate(BAY_STATES)}
DECODERS = {  # name -> (visit_engine.DECODER, HMM_RATES)
    "baseline_hysteresis": ("baseline", "fitted"),
    "hmm_generic": ("hmm", "generic"),
    "hmm_fitted": ("hmm", "fitted"),
}
TEST_SEED, DEV_SEED = 2000, 1500  # priors are fitted on seed 1000 (sim/priors/hmm_rates.json)
MISMATCH = ScenarioConfig(duration_scale=1.6, work_chunk_mean_s=80.0, absence_p=0.25, wash_p=0.8,
                          job_mix={"express_maintenance": 0.2, "tyre_change": 0.2, "brake_pads": 0.2,
                                   "chassis_diagnostics": 0.2, "repair_parts_wait": 0.2})


@dataclass(frozen=True)
class Noise:
    acc: float  # nominal per-frame, per-label accuracy
    corr_s: float = 10.0  # AR(1) time constant of the error process (0 = white)
    drop: float = 0.1  # share of frames missing
    fps_lo: float = 0.5
    fps_hi: float = 2.0


def make_preds(tl: Timeline, kind: str, nz: Noise, rng: np.random.Generator):
    """Fake per-frame predictor output for one zone camera. Returns (preds, t_end, frame_correct, frame_total)."""
    pad_b, pad_a = int(rng.uniform(60, 300)), int(rng.uniform(60, 300))
    v = tl.view(kind, pad_b, pad_a)
    if v is None:
        return None
    g_t, g_lab = v
    t, t_end = float(g_t[0]), float(g_t[-1])
    times = []
    while t <= t_end:
        times.append(t)
        t += 1.0 / rng.uniform(nz.fps_lo, nz.fps_hi)
    times = np.array([x for x in times if rng.random() >= nz.drop] or [times[0]])
    labels = NOISE_LABELS[kind]
    m = NormalDist().inv_cdf(min(max(nz.acc, 0.5 + 1e-6), 1 - 1e-6))
    noise = np.zeros((len(times), len(labels)))
    n = rng.standard_normal(len(labels))
    for i in range(len(times)):
        if i and nz.corr_s > 0:
            rho = math.exp(-(times[i] - times[i - 1]) / nz.corr_s)
            n = rho * n + math.sqrt(1 - rho * rho) * rng.standard_normal(len(labels))
        elif i:
            n = rng.standard_normal(len(labels))
        noise[i] = n
    gi = np.clip(np.searchsorted(g_t, times, side="right") - 1, 0, len(g_t) - 1)
    y = g_lab[gi][:, [KI[s] for s in labels]].astype(float)
    score = (2 * y - 1) * m + noise
    prob = 1 / (1 + np.exp(-np.clip(2 * m * score, -40, 40)))
    correct = int(((prob > 0.5) == (y > 0.5)).sum())
    preds = []
    ci = labels.index("car_present")
    for i, ts in enumerate(times):
        p = FramePred(index=i, t_s=float(ts), states={s: float(prob[i, j]) for j, s in enumerate(labels)})
        if prob[i, ci] > 0.5:
            p.detections.append(Det(cls="car", confidence=float(prob[i, ci]), bbox=(0.3, 0.3, 0.7, 0.8)))
        preds.append(p)
    return preds, t_end, correct, y.size


def decode_events(preds, tl: Timeline, kind: str, decoder: str, eps: float | None = None, weight: float | None = None,
                  prune: float | None = None):
    visit_engine.DECODER, visit_engine.HMM_RATES = DECODERS[decoder][:2]
    old = (hmm.DEFAULT_EPS, hmm.DEFAULT_WEIGHT, hmm.DEFAULT_PRUNE_SCALE)
    if eps is not None:
        hmm.DEFAULT_EPS, hmm.DEFAULT_WEIGHT = eps, weight
    if prune is not None:
        hmm.DEFAULT_PRUNE_SCALE = prune
    elif len(DECODERS[decoder]) > 2:
        hmm.DEFAULT_PRUNE_SCALE = DECODERS[decoder][2]
    try:  # decode() reads the module defaults at call time
        v = visit_engine.build_visit(preds, start_time=tl.t0, zones=[zone_for(kind, tl.meta["bay_id"])],
                                     source=Source.observed_synthetic, visit_id=tl.visit_id)
    finally:
        hmm.DEFAULT_EPS, hmm.DEFAULT_WEIGHT, hmm.DEFAULT_PRUNE_SCALE = old
    return [] if v is None else [e for e in v.events if e.type not in (EventType.arrived, EventType.plate_read)]


def eval_visit(args) -> dict:
    """Score every decoder on one visit under one noise setting."""
    (visit, wo, tl), nz, decoders, seed, hp = args
    rng = np.random.default_rng(seed)
    out = {"visit_id": tl.visit_id, "job": tl.meta["job_type"], "frame_correct": 0, "frame_total": 0,
           "dec": {d: {"events": {}, "idle": {}, "lines": []} for d in decoders}}
    gt = [e for e in visit.events if e.zone is not None and e.zone.kind.value in NOISE_LABELS]
    pr_all = {d: [] for d in decoders}
    gt_idle_all, pr_idle_all = [], {d: [] for d in decoders}
    for kind in NOISE_LABELS:
        r = make_preds(tl, kind, nz, rng)
        if r is None:
            continue
        preds, t_end, c, n = r
        out["frame_correct"] += c
        out["frame_total"] += n
        gk = [e for e in gt if e.zone.kind.value == kind]
        gt_idle_all += M.idle_spans(gk, tl.t0, t_end)
        for d in decoders:
            ev = decode_events(preds, tl, kind, d, *(hp or (None, None, None)))
            pr_all[d] += ev
            pr_idle_all[d] += M.idle_spans(ev, tl.t0, t_end)
    for d in decoders:
        out["dec"][d]["events"] = M.event_counts(gt, pr_all[d], tl.t0)
        out["dec"][d]["idle"] = M.idle_counts(gt_idle_all, pr_idle_all[d])
        out["dec"][d]["lines"] = M.line_records(wo, gt, pr_all[d], tl.meta["lines_performed"])
        gi = sum(b - a for a, b in gt_idle_all)
        pi = sum(b - a for a, b in pr_idle_all[d])
        out["dec"][d]["idle_minutes_abs_err"] = abs(gi - pi) / 60
    return out


def summarize(recs: list[dict], decoder: str) -> dict:
    ev = [r["dec"][decoder]["events"] for r in recs]
    idl = [r["dec"][decoder]["idle"] for r in recs]
    ln = [r["dec"][decoder]["lines"] for r in recs]
    ip = M.idle_pr(idl)
    return {"n_visits": len(recs),
            "frame_acc_measured": sum(r["frame_correct"] for r in recs) / max(1, sum(r["frame_total"] for r in recs)),
            "key_f1_10s": M.event_f1(ev, 10), "key_f1_30s": M.event_f1(ev, 30),
            "fine_f1_10s": M.event_f1(ev, 10, False), "fine_f1_30s": M.event_f1(ev, 30, False),
            "key_median_abs_onset_err_s": M.median_abs_err(ev), **{f"idle_{k}": v for k, v in ip.items()},
            "idle_minutes_mae": float(np.mean([r["dec"][decoder]["idle_minutes_abs_err"] for r in recs])),
            "line_acc_vs_oracle": M.line_acc(ln, "gt"), "line_acc_vs_performed": M.line_acc(ln, "performed"),
            "line_evidence_recall": M.line_acc(ln, "gt", only="evidence_found"),
            "line_not_observed_recall": M.line_acc(ln, "gt", only="not_observed"),
            "line_acc_vs_oracle_keyed": M.line_acc(ln, "gt", keyed=True),
            "line_evidence_recall_keyed": M.line_acc(ln, "gt", keyed=True, only="evidence_found"),
            "line_not_observed_recall_keyed": M.line_acc(ln, "gt", keyed=True, only="not_observed"),
            "line_acc_vs_performed_keyed": M.line_acc(ln, "performed", keyed=True),
            "oracle_share_evidence_found": M.line_acc([[{**r, "pred": "evidence_found"} for r in x] for x in ln], "gt"),
            "n_lines": sum(len(x) for x in ln)}


def per_category(recs: list[dict], decoder: str) -> dict:
    agg: dict = {}
    for r in recs:
        for cat, c in r["dec"][decoder]["events"].items():
            a = agg.setdefault(cat, {"tp10": 0, "fp10": 0, "fn10": 0, "tp30": 0, "fp30": 0, "fn30": 0, "err30": []})
            for k, v in c.items():
                if k == "err30":
                    a[k] += v
                else:
                    a[k] += v
    return {cat: {"f1_10s": M.f1(a["tp10"], a["fp10"], a["fn10"]), "f1_30s": M.f1(a["tp30"], a["fp30"], a["fn30"]),
                  "n_gt": a["tp30"] + a["fn30"], "n_pred": a["tp30"] + a["fp30"],
                  "median_abs_err_s": float(np.median(np.abs(a["err30"]))) if a["err30"] else None}
            for cat, a in sorted(agg.items())}


def run_condition(visits, nz: Noise, decoders, pool, seed: int, hp=None) -> list[dict]:
    args = [(v, nz, decoders, seed * 100003 + i, hp) for i, v in enumerate(visits)]
    return pool.map(eval_visit, args, chunksize=1) if pool else [eval_visit(a) for a in args]


# ------------------------------------------------------------------------------------------------- metrics
def component_results(recs: list[dict], decoder: str, nz: Noise, cond: str, dataset: str) -> dict:
    """ComponentResult with visit-level bootstrap CIs (eval.stats)."""
    def bm(name, fn, unit=None):
        try:
            return boot_metric(name, recs, fn, unit, n_boot=500)
        except ValueError:
            return None

    dec = decoder
    ms = [
        metric("frame_accuracy_measured", summarize(recs, dec)["frame_acc_measured"], unit="fraction"),
        bm("key_event_f1_at_10s", lambda rs: M.event_f1([r["dec"][dec]["events"] for r in rs], 10), "fraction"),
        bm("key_event_f1_at_30s", lambda rs: M.event_f1([r["dec"][dec]["events"] for r in rs], 30), "fraction"),
        bm("key_event_median_abs_onset_error", lambda rs: M.median_abs_err([r["dec"][dec]["events"] for r in rs]), "s"),
        bm("fine_event_f1_at_10s", lambda rs: M.event_f1([r["dec"][dec]["events"] for r in rs], 10, False), "fraction"),
        bm("idle_segment_precision_iou50", lambda rs: M.idle_pr([r["dec"][dec]["idle"] for r in rs])["seg_precision"], "fraction"),
        bm("idle_segment_recall_iou50", lambda rs: M.idle_pr([r["dec"][dec]["idle"] for r in rs])["seg_recall"], "fraction"),
        bm("idle_time_precision", lambda rs: M.idle_pr([r["dec"][dec]["idle"] for r in rs])["time_precision"], "fraction"),
        bm("idle_time_recall", lambda rs: M.idle_pr([r["dec"][dec]["idle"] for r in rs])["time_recall"], "fraction"),
        bm("idle_minutes_per_visit_mae", lambda rs: float(np.mean([r["dec"][dec]["idle_minutes_abs_err"] for r in rs])), "min"),
        bm("workorder_line_status_accuracy_vs_oracle", lambda rs: M.line_acc([r["dec"][dec]["lines"] for r in rs], "gt"), "fraction"),
        bm("workorder_evidence_found_recall", lambda rs: M.line_acc([r["dec"][dec]["lines"] for r in rs], "gt", only="evidence_found"), "fraction"),
        bm("workorder_not_observed_recall", lambda rs: M.line_acc([r["dec"][dec]["lines"] for r in rs], "gt", only="not_observed"), "fraction"),
        bm("workorder_evidence_vs_performed_accuracy", lambda rs: M.line_acc([r["dec"][dec]["lines"] for r in rs], "performed"), "fraction"),
        bm("workorder_line_status_accuracy_vs_oracle_keyed_spans", lambda rs: M.line_acc([r["dec"][dec]["lines"] for r in rs], "gt", keyed=True), "fraction"),
        bm("workorder_not_observed_recall_keyed_spans", lambda rs: M.line_acc([r["dec"][dec]["lines"] for r in rs], "gt", keyed=True, only="not_observed"), "fraction"),
    ]
    s = summarize(recs, dec)
    return {"id": f"sim.noise.{cond}.{dec}", "name": f"Noise test {cond}: {dec}", "lane": "sim", "data_kind": "synthetic",
            "domain": "perception-free simulation (simulated per-frame noise on scenario-engine timelines)",
            "dataset": dataset, "split": "test (seed %d)" % TEST_SEED, "n": len(recs), "model": f"visit_engine:{dec}",
            "metrics": [m for m in ms if m is not None],
            "notes": (f"noise acc={nz.acc} corr_s={nz.corr_s} drop={nz.drop} fps={nz.fps_lo}-{nz.fps_hi}; "
                      f"n_lines={s['n_lines']}; CIs = visit-level percentile bootstrap (500). Same generator for HMM "
                      "priors (seed 1000) and test visits (different seed): optimistic for the fitted HMM.")}


# --------------------------------------------------------------------------------------------------- ETA
def eta_test(visits, seed: int = 7) -> dict:
    """Cut each visit inside the bay, give the perfectly observed prefix to continue_visit, compare ETA to truth."""
    rng = np.random.default_rng(seed)
    errs, naive_errs, covered, phantom = [], [], [], []
    rem_all = []
    for visit, wo, tl in visits:
        sp = tl.span("bay")
        cut = sp[0] + rng.uniform(0.2, 0.8) * (sp[1] - sp[0])
        t_ready = tl.span("ready")[0]
        rem_all.append(t_ready - cut)
    naive = float(np.median(rem_all))
    rng = np.random.default_rng(seed)
    for (visit, wo, tl), rem in zip(visits, rem_all):
        sp = tl.span("bay")
        cut_t = tl.t0 + timedelta(seconds=float(tl.span("ready")[0] - rem))
        obs = [e.model_copy(update={"source": Source.observed_synthetic}) for e in visit.events
               if e.t <= cut_t and e.type not in (EventType.idle_started, EventType.idle_ended)]
        cut_visit = visit.model_copy(update={"events": obs, "verifications": None})
        cont = continue_visit(cut_visit, wo, n_samples=100, now=cut_t)
        rd = [e for e in cont.events if e.type == EventType.ready and e.source == Source.simulated]
        if not rd:
            continue
        pay = rd[0].payload
        p50 = (datetime.fromisoformat(pay["eta_p50"]) - cut_t).total_seconds()
        p10 = (datetime.fromisoformat(pay["eta_p10"]) - cut_t).total_seconds()
        p90 = (datetime.fromisoformat(pay["eta_p90"]) - cut_t).total_seconds()
        errs.append((p50 - rem) / 60)
        naive_errs.append((naive - rem) / 60)
        covered.append(p10 <= rem <= p90)
        phantom.append(bool(tl.meta["phantom_lines"]))
    ph = np.array(phantom)
    e, ne = np.abs(errs), np.abs(naive_errs)
    return {"n": len(errs), "mae_min": float(e.mean()), "naive_mae_min": float(ne.mean()),
            "coverage_p10_p90": float(np.mean(covered)), "bias_min": float(np.mean(errs)),
            "n_no_phantom": int((~ph).sum()), "mae_min_no_phantom": float(e[~ph].mean()),
            "naive_mae_min_no_phantom": float(ne[~ph].mean()),
            "coverage_no_phantom": float(np.array(covered)[~ph].mean()),
            "errs": errs, "naive_errs": naive_errs, "covered": covered, "phantom": phantom}


# ---------------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tune", action="store_true")
    a = ap.parse_args()
    from seen.paths import RUNS

    out = Path(a.out) if a.out else RUNS / "sim" / "noise_test"
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    pool = Pool(a.workers) if a.workers > 1 else None
    decoders = list(DECODERS)

    if a.tune:
        dev = generate_visits(40, DEV_SEED)
        res = []
        for eps, w, pr in [(0.1, 0.3, 0.5), (0.3, 0.3, 0.5), (0.3, 0.2, 0.5), (0.3, 0.1, 0.5), (0.5, 0.3, 0.5),
                           (0.5, 0.2, 0.5), (0.3, 0.2, 0.0), (0.3, 0.2, 1.0), (0.3, 0.3, 1.0)]:
            sc = []
            for nz in (Noise(0.7, 10), Noise(0.85, 10), Noise(0.85, 0), Noise(0.85, 60)):
                recs = run_condition(dev, nz, ["hmm_fitted"], pool, 11, (eps, w, pr))
                s = summarize(recs, "hmm_fitted")
                sc.append((s["key_f1_10s"], s["line_acc_vs_oracle"], s["idle_seg_recall"]))
            res.append({"eps": eps, "weight": w, "prune_scale": pr, "key_f1_10s_mean": float(np.mean([x[0] for x in sc])),
                        "line_acc_mean": float(np.mean([x[1] for x in sc])), "detail": sc})
            print(res[-1], flush=True)
        (out / "tune.json").write_text(json.dumps(res, indent=1))
        best = max(res, key=lambda r: r["key_f1_10s_mean"] + r["line_acc_mean"])
        print("BEST", best)
        return

    n = 12 if a.quick else a.n
    accs = (0.7, 0.9) if a.quick else (0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)
    corrs = (10.0,) if a.quick else (0.0, 10.0, 60.0)
    test = generate_visits(n, TEST_SEED)
    dataset = f"sim.scenario v{test[0][2].to_json()['scenario_version']} default config, {n} visits"
    grid, results = [], []
    for corr in corrs:
        for acc in accs:
            nz = Noise(acc, corr)
            recs = run_condition(test, nz, decoders, pool, int(acc * 1000 + corr))
            for d in decoders:
                row = {"cond": "grid", **asdict(nz), "decoder": d, **summarize(recs, d)}
                grid.append(row)
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            if corr == 10.0 and acc in (0.7, 0.8, 0.9):
                tag = f"a{int(acc * 100)}_c{int(corr)}"
                results += [component_results(recs, d, nz, tag, dataset) for d in decoders]
                if acc == 0.8:
                    (out / "per_category_a80_c10.json").write_text(json.dumps({d: per_category(recs, d) for d in decoders}, indent=1))
    if not a.quick:
        for drop in (0.0, 0.3, 0.5):
            nz = Noise(0.85, 10.0, drop)
            recs = run_condition(test, nz, decoders, pool, 777 + int(drop * 10))
            for d in decoders:
                grid.append({"cond": "drop", **asdict(nz), "decoder": d, **summarize(recs, d)})
        for fps in ((0.5, 0.5), (2.0, 2.0)):
            nz = Noise(0.85, 10.0, 0.1, *fps)
            recs = run_condition(test, nz, decoders, pool, 888 + int(fps[0] * 10))
            for d in decoders:
                grid.append({"cond": "fps", **asdict(nz), "decoder": d, **summarize(recs, d)})
        mism = generate_visits(n, TEST_SEED + 1, MISMATCH)
        mds = f"sim.scenario MISMATCH config (durations x1.6, pauses x1.9, absences x2, uniform job mix), {n} visits"
        for acc in (0.7, 0.8, 0.9):
            nz = Noise(acc, 10.0)
            recs = run_condition(mism, nz, decoders, pool, 999 + int(acc * 100))
            for d in decoders:
                grid.append({"cond": "mismatch", **asdict(nz), "decoder": d, **summarize(recs, d)})
            if acc == 0.8:
                results += [component_results(recs, d, nz, "mismatch_a80_c10", mds) for d in decoders]
        eta = eta_test(test)
        results.append({"id": "sim.continuation.eta", "name": "Scenario continuation: ready-time ETA from a bay cut",
                        "lane": "sim", "data_kind": "synthetic", "domain": "perception-free simulation (perfect observation of the prefix)",
                        "dataset": dataset, "split": "test (seed %d)" % TEST_SEED, "n": eta["n"], "model": "sim.scenario.continue_visit",
                        "metrics": [boot_metric("eta_ready_mae", np.abs(eta["errs"]), np.mean, "min", n_boot=1000),
                                    boot_metric("naive_median_eta_mae", np.abs(eta["naive_errs"]), np.mean, "min", n_boot=1000),
                                    boot_metric("eta_p10_p90_coverage", np.array(eta["covered"], float), np.mean, "fraction", n_boot=1000)],
                        "notes": "Cut uniformly in the 20-80% of the bay stay; prefix events observed perfectly; ETA from 100 Monte-Carlo "
                                 "continuations with the same generator that produced the truth (optimistic). Naive = one global median "
                                 "remaining time of the same test visits (uses test information). "
                                 f"Without visits that carry a not-performed (phantom) line: n={eta['n_no_phantom']}, "
                                 f"MAE {eta['mae_min_no_phantom']:.1f} min vs naive {eta['naive_mae_min_no_phantom']:.1f} min, "
                                 f"p10-p90 coverage {eta['coverage_no_phantom']:.2f}; bias {eta['bias_min']:+.1f} min overall."})
        (out / "eta.json").write_text(json.dumps({k: v for k, v in eta.items()}, indent=1))
    (out / "grid.json").write_text(json.dumps(grid, indent=1))
    (out / "component_results.json").write_text(json.dumps(results, indent=1))
    print(f"done in {time.time() - t_start:.0f}s -> {out}")


if __name__ == "__main__":
    main()
