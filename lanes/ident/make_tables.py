"""Print the markdown tables used in docs/lanes/ident_notes.md from docs/lanes/ident_results.json (so every table is generated, not typed).

  python -m lanes.ident.make_tables docs/lanes/ident_results.json > /tmp/ident_tables.md
"""
from __future__ import annotations

import json
import sys

R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "docs/lanes/ident_results.json"))
by = {c["id"]: c for c in R}


def metric(c, name_prefix):
    for m in c["metrics"]:
        if m["name"].startswith(name_prefix):
            return m
    return None


def cell(c, name="exact_match", pct=True):
    if c is None:
        return "n/a"
    m = metric(c, name)
    if m is None:
        return "n/a"
    f = (lambda v: f"{100 * v:.1f}") if pct else (lambda v: f"{v:.3f}")
    ci = f" [{f(m['ci_low'])}, {f(m['ci_high'])}]" if "ci_low" in m else ""
    return f"{f(m['value'])}{ci}"


def get(exp, sub, mode, tag=""):
    return by.get(f"ident.ocr.{exp}.{sub}.{mode}{tag}")


SETS = [("test_autoria_kz", "AUTO.RIA KZ test", "real"), ("test_autoria_kg", "AUTO.RIA KG test", "real"), ("test_autoria_ru", "AUTO.RIA RU test", "real"),
        ("test_synth_uz", "synthetic UZ test", "SYNTHETIC"), ("test_uz_real", "**real UZ test (upright crops)**", "real")]
print("### OCR exact match, % [95% Wilson CI]; n in the first column\n")
print("| Subset (data kind, n) | fast-plate-ocr cct-s-v2 (pretrained) | mobile-vit-v2 (pretrained) | ours, AUTO.RIA only (no synthetic) | ours (AUTO.RIA + synthetic UZ), argmax | ours, auto decoding |")
print("|---|---|---|---|---|---|")
for sub, label, kind in SETS:
    c0 = get("baseline_cct_s_v2_global", sub, "argmax", "_upright" if sub == "test_uz_real" else "")
    n = (c0 or get("final", sub, "argmax", "") or {}).get("n", "")
    print(f"| {label} ({kind}, n={n}) | {cell(c0)} | {cell(get('baseline_global_mobile_vit_v2_ocr', sub, 'argmax', '_upright' if sub == 'test_uz_real' else ''))} | "
          f"{cell(get('real_only', sub, 'argmax'))} | {cell(get('final', sub, 'argmax'))} | {cell(get('final', sub, 'auto'))} |")
print()
print("### Character accuracy (mean per crop, 1 - normalised edit distance), % [95% bootstrap CI]\n")
print("| Subset | cct-s-v2 pretrained | ours, argmax | ours, auto |\n|---|---|---|---|")
for sub, label, kind in SETS:
    print(f"| {label} | {cell(get('baseline_cct_s_v2_global', sub, 'argmax', '_upright' if sub == 'test_uz_real' else ''), 'char_accuracy')} | {cell(get('final', sub, 'argmax'), 'char_accuracy')} | {cell(get('final', sub, 'auto'), 'char_accuracy')} |")
print()
print("### Ablation: format-constrained decoding (final model), exact match %\n")
print("| Subset | argmax | constrained (forced) | auto (constrained with fall-back) |\n|---|---|---|---|")
for sub, label, kind in SETS:
    print(f"| {label} | {cell(get('final', sub, 'argmax'))} | {cell(get('final', sub, 'constrained'))} | {cell(get('final', sub, 'auto'))} |")
print()
print("### Orientation search on raw-orientation real-UZ crops (about 31 % of the verified rotations differed from the first proposal; roughly 40 % of photos are upside-down)\n")
print("| Condition | exact match % [CI] |\n|---|---|")
for lab, c in (("pretrained cct-s-v2, no rotation search", get("baseline_cct_s_v2_global", "test_uz_real_raw", "argmax", "_orient_search")),
               ("ours, no rotation search", get("final", "test_uz_real_raw", "argmax", "_no_orient_search")),
               ("ours, rotation search (argmax)", get("final", "test_uz_real_raw", "argmax", "_orient_search")),
               ("ours, rotation search (auto)", get("final", "test_uz_real_raw", "auto", "_orient_search"))):
    print(f"| {lab} | {cell(c)} |")
print()
print("### Plate detection, IoU 0.5 (recall at conf >= 0.25, Wilson CI; AP50)\n")
print("| Subset (n images) | fast-alpr YOLOv9-t-384 | ours, Uzbek-only | ours, full (+CCPD/RodoSol/composites) |\n|---|---|---|---|")
for sub, label in (("uz_real_test", "real UZ test"), ("rodosol_test", "RodoSol test (BR/Mercosur, day+night, rain)"), ("ccpd_db", "CCPD-DB (illumination)"), ("ccpd_weather", "CCPD-Weather"), ("ccpd_challenge", "CCPD-Challenge")):
    cs = [by.get(f"ident.det.{e}.{sub}") for e in ("baseline_fastalpr", "ours_plate_det_uz_only", "ours_plate_det")]
    n = next((c["n"] for c in cs if c), "")
    def d(c):
        if c is None:
            return "n/a"
        ap = metric(c, "AP@IoU0.5")
        pr = metric(c, "precision@IoU0.5")
        return f"{cell(c, 'recall@IoU0.5')} / P {100 * pr['value']:.1f} / AP50 {ap['value']:.3f}" if pr else f"{cell(c, 'recall@IoU0.5')} / AP50 {ap['value']:.3f}"
    print(f"| {label} (n={n}) | {d(cs[0])} | {d(cs[1])} | {d(cs[2])} |")
print()
print("### End-to-end on real UZ photos (detector + OCR, exact match over all GT plates)\n")
print("| System | e2e exact match % [CI] | detected % | exact match given detected % |\n|---|---|---|---|")
for lab, k in (("fast-alpr as shipped (YOLOv9-t-384 + cct-s-v2, no rotation search)", "ident.e2e.baseline_e2e.uz_real"), ("ours (RF-DETR full + fine-tuned OCR, auto decoding + rotation search)", "ident.e2e.ours_e2e.uz_real")):
    c = by.get(k)
    print(f"| {lab} | {cell(c, 'e2e_exact_match')} | {cell(c, 'detected@')} | {cell(c, 'exact_match|detected')} |")
print()
print("### Track voting (synthetic tracks, 400 tracks x 10 frames), exact match % over tracks\n")
print("| Frames used | best-confidence frame | majority vote | pooled | pooled + constrained |\n|---|---|---|---|---|")
for k in (3, 5, 10):
    row = [cell(by.get(f"ident.vote.final.first{k}_{m}")) for m in ("best_conf_frame", "majority", "pooled", "pooled+constrained")]
    print(f"| first {k} | " + " | ".join(row) + " |")
print(f"| (single frame, mean over all frames, unit = frame) | {cell(by.get('ident.vote.final.single_frame_argmax(all_frames)'))} | | | |")
print()
print("### Stress test: real UZ crops degraded artificially (auto decoding), exact match %\n")
print("| Degradation | EM % [CI] |\n|---|---|")
for c in R:
    if ".stress_uz_real." in c["id"]:
        print(f"| {c['name'].split(': ')[1]} | {cell(c)} |")
print()
print("### Latency\n")
print("| Stage | runtime | p50 ms | p95 ms |\n|---|---|---|---|")
try:
    for s in json.load(open(sys.argv[2])):
        print(f"| {s['stage']} | {s['runtime']} | {s['p50_ms']:.1f} | {s['p95_ms']:.1f} |")
except Exception:
    pass
