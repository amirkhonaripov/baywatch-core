"""Assemble lane C results: component results (handheld, hand-check, CCTV, VLM), speed and cost.

  python -m lanes.ops.report --exp heads_v1   -> runs/ops/<exp>/ops_results.json (ComponentResult list) + report.json

Cost model (stated, not measured on an L4 — we have no L4):
  * measured: frames/s and per-frame latency of each system on the RTX PRO 5000 Blackwell (GPU1 of the box);
  * L4 estimate: scale throughput by the memory-bandwidth ratio L4 / RTX PRO 5000 (300 / 1344 GB/s). Autoregressive VLM
    decoding is bandwidth-bound, so this is a reasonable first-order estimate for the VLM; for the small CNN/ViT student it
    is pessimistic-to-fair (compute ratio is similar);
  * L4 price: USD 0.71 per GPU-hour (assumed on-demand cloud price for one L4; edit L4_USD_PER_H to re-cost).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from seen.paths import RUNS

L4_BW, PRO5000_BW = 300.0, 1344.0
L4_USD_PER_H = 0.71


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    a = ap.parse_args()
    d = RUNS / "ops" / a.exp
    comps = []
    for f in ("component_results.json", "handcheck_component_results.json", "cctv_component_results.json"):
        comps += _load(d / f) or []
    speed = _load(d / "artifact" / "speed.json") or []
    vlm_stats = _load(RUNS / "ops" / "vlm" / "cctv_vlm.parquet.stats.json") or {}
    # student per-frame cost: backbone (batch 16) + head + part detectors (measured by eval_cctv timings if present)
    stu_ms = None
    for s in speed:
        if "backbone" in s["stage"] and s.get("batch") == 16 and "torch" in s.get("runtime", ""):
            stu_ms = s["p50_ms"]
    head_ms = next((s["p50_ms"] for s in speed if "head" in s["stage"]), 0.0)
    pipe = _load(d / "cctv_timing.json") or {}
    cost = []
    if vlm_stats:
        fps = vlm_stats["frames_per_s"]
        fps_l4 = fps * L4_BW / PRO5000_BW
        cost += [
            dict(item="Qwen3-VL-8B-FP8 throughput, measured (transformers, batch 8)", value=fps, unit="frames/s",
                 assumptions=f"{vlm_stats['gpu']}; ~{vlm_stats['mean_prompt_tokens']:.0f} prompt + {vlm_stats['mean_output_tokens']:.0f} output tokens per frame"),
            dict(item="Qwen3-VL-8B-FP8 single-frame latency, measured", value=vlm_stats["single_latency_ms_p50"], unit="ms (p50)",
                 assumptions=vlm_stats["gpu"]),
            dict(item="Qwen3-VL-8B-FP8 cost per 1k frames on an L4 (estimate)", value=1000 / fps_l4 / 3600 * L4_USD_PER_H, unit="USD",
                 assumptions=f"throughput scaled by bandwidth {L4_BW}/{PRO5000_BW}; L4 at USD {L4_USD_PER_H}/h; transformers (not vLLM) so an upper bound"),
        ]
    if stu_ms:
        per = stu_ms + head_ms + pipe.get("parts_ms", 0.0)
        fps_s = 1000.0 / per
        fps_s_l4 = fps_s * L4_BW / PRO5000_BW
        cost += [
            dict(item="Student per-frame GPU time, measured (DINOv2-L bf16 batch 16 + ONNX head + part detectors)", value=per,
                 unit="ms", assumptions=f"{speed[0]['hardware'] if speed else ''}; excludes the shared vehicle/person detector"),
            dict(item="Student cost per 1k frames on an L4 (estimate)", value=1000 / fps_s_l4 / 3600 * L4_USD_PER_H, unit="USD",
                 assumptions=f"same bandwidth scaling and price as the VLM line"),
        ]
    rep = dict(exp=a.exp, n_component_results=len(comps), speed=speed, cost=cost)
    (d / "ops_results.json").write_text(json.dumps(comps, indent=1))
    (d / "report.json").write_text(json.dumps(rep, indent=1))
    for c in cost:
        print(f"{c['item']}: {c['value']:.4g} {c['unit']}")
    print("components", len(comps))


if __name__ == "__main__":
    main()
