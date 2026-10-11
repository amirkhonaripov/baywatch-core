"""Breakdown of the synthetic-UZ OCR test accuracy by crop width, scenario and plate format (SYNTHETIC data, reported separately).

  python -m lanes.ident.analyze_synth --exp final --mode auto
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from lanes.ident import evalkit as ek
from seen.paths import RUNS, fast_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--mode", default="auto")
    ap.add_argument("--model", default="")
    a = ap.parse_args()
    meta = [json.loads(l) for l in (fast_dir("ident") / "ocr" / "test_synth_uz.meta.jsonl").read_text().splitlines()]
    import csv
    rows = list(csv.DictReader(open(RUNS / "ident" / a.exp / f"preds_test_synth_uz_{a.mode}.csv")))
    assert len(rows) == len(meta)
    groups: dict[str, list[bool]] = defaultdict(list)
    for m, r in zip(meta, rows):
        ok = ek.norm(r["pred"]) == ek.norm(r["gt"])
        w = m["out_w"]
        groups[f"width {'<80' if w < 80 else '80-119' if w < 120 else '120-199' if w < 200 else '>=200'} px"].append(ok)
        groups[f"scenario {m['scenario']}"].append(ok)
        groups[f"format {m['format']}"].append(ok)
        for op in ("glare", "dirt", "rain", "occlusion", "shadow"):
            if op in m["ops"]:
                groups[f"degradation {op}"].append(ok)
    comps = []
    for k, v in sorted(groups.items()):
        comps.append(ek.component(f"ident.ocr.{a.exp}.synth_breakdown.{k.replace(' ', '_')}.{a.mode}", f"Synthetic UZ OCR exact match, subset: {k}", "synthetic", "rendered Uzbek plates + augmentations",
                                  "synthetic UZ (ours)", "held-out seed", len(v), a.model or a.exp, [ek.prop_metric("exact_match", int(sum(v)), len(v))],
                                  notes=f"decoding={a.mode}; subsets overlap (a crop can be in several degradation groups)"))
        em = comps[-1]["metrics"][0]
        print(f"{k:28s} n={len(v):5d} EM={em['value']:.3f} [{em['ci_low']:.3f},{em['ci_high']:.3f}]")
    ek.write_results(RUNS / "ident" / a.exp / "component_results_synth_breakdown.json", comps)


if __name__ == "__main__":
    main()
