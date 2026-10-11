"""Cost model from *measured* speed (docs/lanes/eval_speed.json + eval_bench_details.json). Lane G.

    python -m eval.cost [--root <dir with docs/lanes/eval_*.json>] [--out <output root>]

Writes docs/lanes/eval_cost.json (list of CostLine). Every price below is cited; every throughput comes from
eval/bench.py. Where a number is a scaled estimate (another GPU than the one measured) the line says so.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------------------------- inputs (cited)
MODAL = {
    "l4_usd_per_s": 0.000222,          # https://modal.com/pricing (read 2026-10-10): Nvidia L4 $0.000222 / sec
    "cpu_usd_per_core_s": 0.0000131,   # same page: CPU $0.0000131 / physical core / sec
    "mem_usd_per_gib_s": 0.00000222,   # same page: memory $0.00000222 / GiB / sec
    "scaledown_window_s": 120,         # deployed value (serve/modal_app.py SEEN_MODAL_SCALEDOWN default); Modal's own default
                                       # is 60 s and idle time is billed (https://modal.com/docs/guide/cold-start)
    "source": "modal.com/pricing and modal.com/docs/guide/cold-start, read 2026-10-10; base region, preemptible",
}
CONTAINER = {"cores": 2.0, "mem_gib": 8.0}  # our assumption for the GPU container request (serve/ may differ)

EDGE = {
    "pc_usd_low": 925, "pc_usd_high": 1345, "pc_usd_new_i5": 1020,
    "source_pc": "research/competitors_legal.md: OLX.uz, 10 Oct 2026: used gaming PC with RTX 4060 Ti 11-16M UZS "
                 "(~$925-1,345); new i5-13400F + RTX 4060 Ti PC 12.1M UZS (~$1,020)",
    "months": 36,
    "power_w": 250,  # assumption: GPU TGP 160-165 W (4060 Ti) + ~85 W CPU, board, PSU loss at sustained load
    "hours_per_month": 730,  # 24 h x 365 / 12: cameras run around the clock
    "tariff_uzs_per_kwh": 1100,
    "source_tariff": "kun.uz 15 May 2026 'Uzbekistan to raise electricity and natural gas tariffs from June 1': legal "
                     "entities in groups I, II, IV pay UZS 1,100 per kWh from 1 Jun 2026 (Interfax 15 May 2026 agrees)",
    "uzs_per_usd": 11778.45,
    "source_fx": "Central Bank of Uzbekistan official rate for 6 Oct 2026 (via zamin.uz)",
}

# GPU specs used for the proxy scaling (FP32 TFLOPS, memory GB/s). 4060 Ti FP32 is derived (4,352 cores x 2 x 2.535 GHz).
GPU_SPECS = {
    "RTX PRO 5000 Blackwell": {"fp32": 66.9, "bw": 1344, "src": "14,080 CUDA cores x 2 x 2.377 GHz boost (reseller specs); 1,344 GB/s (NVIDIA/PNY datasheet)"},
    "L4": {"fp32": 30.3, "bw": 300, "src": "NVIDIA L4 datasheet: 30.3 TFLOPS FP32, 300 GB/s, 72 W"},
    "RTX 4060 Ti": {"fp32": 22.06, "bw": 288, "src": "4,352 CUDA cores x 2 x 2.535 GHz (derived); 288 GB/s; 160-165 W TGP"},
}


def _gpu_key(hw: str) -> str | None:
    h = hw.lower()
    if "l4" in h.split() or "nvidia l4" in h or h.startswith("l4"):
        return "L4"
    if "pro 5000" in h:
        return "RTX PRO 5000 Blackwell"
    if "4060" in h:
        return "RTX 4060 Ti"
    return None


def scale_to(src: str, dst: str) -> tuple[float, float, str]:
    """(conservative, optimistic, explanation) throughput multiplier from GPU ``src`` to ``dst``."""
    a, b = GPU_SPECS[src], GPU_SPECS[dst]
    rc, rb = b["fp32"] / a["fp32"], b["bw"] / a["bw"]
    lo, hi = min(rc, rb), max(rc, rb)
    return lo, hi, (f"{dst}/{src}: FP32 ratio {rc:.2f}, memory-bandwidth ratio {rb:.2f}; conservative = min, optimistic = max. "
                    "Assumes the pipeline is GPU-bound; CPU-side Python work does not scale with the GPU.")


def _load(name: str, roots: list[Path]):
    hit = None
    for r in roots:
        p = r / "docs" / "lanes" / name
        if p.exists():
            hit = p
    return json.loads(hit.read_text()) if hit else None


def _row(speed: list[dict], prefix: str) -> dict | None:
    rows = [s for s in speed if s["stage"].startswith(prefix)]
    return rows[0] if rows else None


def best_cameras(details: dict | None, speed: list[dict], profile: str) -> tuple[int | None, str | None]:
    trials = (details or {}).get("details", {}).get("streams")
    if isinstance(trials, list) and trials:
        ok = [t for t in trials if t.get("profile") == profile and t.get("ok")]
        if ok:
            b = max(ok, key=lambda t: (t["n_cams"], -t["workers"]))
            return int(b["n_cams"]), f"{b['workers']} worker thread(s)"
        return 0, None
    for s in speed:
        if s["stage"].startswith(f"streams: {profile}") and "max sustained cameras = " in s.get("notes", ""):
            return int(s["notes"].split("max sustained cameras = ")[1].split()[0]), s.get("runtime")
    return None, None


def line(item: str, value: float, unit: str, assumptions: str) -> dict:
    return {"item": item, "value": round(float(value), 6), "unit": unit, "assumptions": assumptions}


def compute(speed: list[dict], details: dict | None) -> list[dict]:
    out: list[dict] = []
    gpu_s_price = MODAL["l4_usd_per_s"]
    cont_s_price = gpu_s_price + CONTAINER["cores"] * MODAL["cpu_usd_per_core_s"] + CONTAINER["mem_gib"] * MODAL["mem_usd_per_gib_s"]
    out.append(line("Modal L4 container price (GPU + 2 cores + 8 GiB)", cont_s_price * 3600, "USD/hour",
                    f"L4 ${gpu_s_price}/s + CPU ${MODAL['cpu_usd_per_core_s']}/core/s x {CONTAINER['cores']:g} + memory "
                    f"${MODAL['mem_usd_per_gib_s']}/GiB/s x {CONTAINER['mem_gib']:g}. {MODAL['source']}."))

    pipe_l4 = [s for s in speed if s["stage"].startswith("pipeline total") and _gpu_key(s["hardware"]) == "L4"]
    pipe = pipe_l4[0] if pipe_l4 else _row(speed, "pipeline total")
    cold = _row(speed, "deployed API /analyze/image, first call") or _row(speed, "pipeline cold start")
    if not pipe:
        out.append(line("Per-frame cost", 0, "USD", "NOT COMPUTED: no pipeline latency measured yet (run eval/bench.py)"))
        return out
    hw = pipe["hardware"]
    key = _gpu_key(hw)
    if key == "L4":
        l4_ms, l4_note = pipe["p50_ms"], f"measured on {hw}"
    elif key:
        lo, hi, why = scale_to(key, "L4")
        l4_ms = pipe["p50_ms"] / lo
        l4_note = (f"ESTIMATE: L4 not measured yet; box p50 {pipe['p50_ms']:.0f} ms on {hw} divided by the conservative "
                   f"factor {lo:.2f} ({why})")
    else:
        l4_ms, l4_note = pipe["p50_ms"], f"measured on {hw} (unknown GPU; used as-is)"
    per_frame = cont_s_price * l4_ms / 1000
    out.append(line("Modal L4: marginal cost per photo (warm container, 1 frame)", per_frame, "USD",
                    f"warm pipeline latency {l4_ms:.0f} ms/frame x container price; {l4_note}"))
    out.append(line("Modal L4: marginal cost per 1,000 frames (warm)", per_frame * 1000, "USD", "as above x 1,000"))
    vid_frames = 120
    out.append(line("Modal L4: marginal cost per 60 s phone video (2 fps = 120 frames)", per_frame * vid_frames, "USD",
                    "browser samples phone video at 2 fps (plan); frames processed sequentially in one warm container"))
    cold_s = cold["cold_start_s"] if cold and cold.get("cold_start_s") else None
    if cold_s is not None:
        iso = cont_s_price * (cold_s + l4_ms / 1000 + MODAL["scaledown_window_s"])
        out.append(line("Modal L4: cost of one isolated photo request (cold start + idle tail)", iso, "USD",
                        f"first-call time {cold_s:.1f}s ({cold['stage']} on {cold['hardware']}) + inference + "
                        f"{MODAL['scaledown_window_s']}s idle window (deployed setting), all billed"))
    n_gate_box, _ = best_cameras(details, speed, "gate")
    n_bay_box, _ = best_cameras(details, speed, "bay")
    stream_rows = [s for s in speed if s["stage"].startswith("streams:")]
    stream_hw = stream_rows[0]["hardware"] if stream_rows else None
    skey = _gpu_key(stream_hw) if stream_hw else None  # the GPU the camera counts were measured on
    # cloud continuous streaming (why we don't run cameras in the cloud)
    if skey and n_bay_box:
        lo, _, _ = scale_to(skey, "L4") if skey != "L4" else (1.0, 1.0, "")
        n_bay_l4 = max(1, math.floor(n_bay_box * lo))
        out.append(line("Modal L4 running 24/7 for bay cameras: cost per camera-month", cont_s_price * 3600 * EDGE["hours_per_month"] / n_bay_l4,
                        "USD/camera/month", f"{n_bay_l4} bay cameras at 1 fps per L4 ("
                        + ("measured" if skey == "L4" else f"ESTIMATE: {n_bay_box} measured on {stream_hw} x {lo:.2f}, conservative spec ratio")
                        + f"); {EDGE['hours_per_month']} h/month; shows why continuous video stays on site"))

    # ------------------------------------------------------------------ edge box
    pc_mid = (EDGE["pc_usd_low"] + EDGE["pc_usd_high"]) / 2
    amort = pc_mid / EDGE["months"]
    kwh = EDGE["power_w"] / 1000 * EDGE["hours_per_month"]
    power_usd = kwh * EDGE["tariff_uzs_per_kwh"] / EDGE["uzs_per_usd"]
    out.append(line("Edge box (RTX 4060 Ti PC, Tashkent) purchase price, midpoint", pc_mid, "USD",
                    f"range ${EDGE['pc_usd_low']}-{EDGE['pc_usd_high']}; {EDGE['source_pc']}"))
    out.append(line("Edge box amortisation", amort, "USD/month", f"midpoint price / {EDGE['months']} months, no residual value, no financing cost"))
    out.append(line("Edge box electricity", power_usd, "USD/month",
                    f"{EDGE['power_w']} W assumed average wall power x {EDGE['hours_per_month']} h = {kwh:.0f} kWh x "
                    f"{EDGE['tariff_uzs_per_kwh']} UZS/kWh / {EDGE['uzs_per_usd']} UZS/USD. {EDGE['source_tariff']}. {EDGE['source_fx']}"))
    box_month = amort + power_usd
    out.append(line("Edge box total running cost", box_month, "USD/month", "amortisation + electricity; excludes cameras, NVR, installation, internet, support"))
    if not skey:
        return out
    src = skey
    lo, hi, why = scale_to(src, "RTX 4060 Ti")
    for prof, fps, n_box in (("bay", 1, n_bay_box), ("gate", 5, n_gate_box)):
        if n_box is None:
            out.append(line(f"Edge box: cost per {prof} camera-month", 0, "USD/camera/month", f"NOT COMPUTED: {prof} stream test missing"))
            continue
        if n_box == 0:
            out.append(line(f"Edge box: cameras per box ({prof} @ {fps} fps)", 0, "cameras",
                            f"measured {src} could not sustain 1 camera; no per-camera cost"))
            continue
        n_lo = max(0, math.floor(n_box * lo))
        n_hi = max(0, math.floor(n_box * hi))
        mem = ((details or {}).get("details", {}).get("pipeline") or {})
        mem = mem.get("gpu_mem_mib") if isinstance(mem, dict) else None
        _, how = best_cameras(details, speed, prof)
        proxy = (f"PROXY ESTIMATE: {n_box} cameras measured on {src} ({how or 'see speed table'}) x {lo:.2f} (conservative) / "
                 f"x {hi:.2f} (optimistic). {why} The 4060 Ti itself was not measured; one pipeline instance held "
                 f"{mem if mem is not None else 'n/a'} MiB of GPU memory on the box (4060 Ti: 8 or 16 GB), which also caps the "
                 "number of workers.")
        out.append(line(f"Edge box: cameras per box ({prof} @ {fps} fps), conservative", n_lo, "cameras", proxy))
        out.append(line(f"Edge box: cameras per box ({prof} @ {fps} fps), optimistic", n_hi, "cameras", proxy))
        if n_lo >= 1:
            out.append(line(f"Edge box: cost per {prof} camera-month (conservative)", box_month / n_lo, "USD/camera/month",
                            f"${box_month:.2f}/month / {n_lo} cameras"))
        if n_hi >= 1:
            out.append(line(f"Edge box: cost per {prof} camera-month (optimistic)", box_month / n_hi, "USD/camera/month",
                            f"${box_month:.2f}/month / {n_hi} cameras"))
    if n_bay_box and n_gate_box:
        bay_lo = max(1, math.floor(n_bay_box * lo))
        gate_lo = max(1, math.floor(n_gate_box * lo))
        load = 6 / bay_lo + 1 / gate_lo  # 12-bay site: 1 camera per 2 bays + 1 gate camera
        boxes = math.ceil(load)
        out.append(line("Example 12-bay site (6 bay cameras + 1 gate camera): edge boxes needed (conservative)", boxes, "boxes",
                        f"load = 6/{bay_lo} + 1/{gate_lo} = {load:.2f} box-equivalents"))
        out.append(line("Example 12-bay site: running cost (conservative)", boxes * box_month, "USD/month",
                        f"{boxes} box(es) x ${box_month:.2f}; compare: Russian cloud video $2.4-10 per camera-month "
                        "(research/competitors_legal.md), which is storage only, no analytics"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--out", default=str(REPO))
    a = ap.parse_args(argv)
    roots = [REPO] + [Path(r) for r in a.root]
    speed = (_load("eval_speed.json", roots) or []) + (_load("eval_speed_modal.json", roots) or [])
    details = _load("eval_bench_details.json", roots)
    lines = compute(speed, details)
    d = Path(a.out) / "docs" / "lanes"
    d.mkdir(parents=True, exist_ok=True)
    (d / "eval_cost.json").write_text(json.dumps(lines, indent=2))
    for l in lines:
        print(f"{l['item']}: {l['value']} {l['unit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
