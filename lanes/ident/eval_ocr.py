"""OCR evaluation on the prepared test sets (exact match + character accuracy with CIs; ablations by decoding mode).

  python -m lanes.ident.eval_ocr --exp baseline_cct_s --model-dir $SEEN_ROOT/data/raw/ident_pretrained --model cct_s_v2_global \
      --sets test_autoria_kz test_autoria_kg test_autoria_ru test_synth_uz test_uz_real --modes argmax
Model dir must contain <model>.onnx and <model>_plate_config.yaml (hub naming) or model.onnx + plate_config.yaml.
Results: runs/ident/<exp>/component_results.json (+ predictions CSV per set).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from lanes.ident import evalkit as ek
from lanes.ident.ocr_onnx import PlateOcrOnnx, default_providers
from lanes.ident.reader import PlateReader
from seen.paths import PREP, RUNS

SET_INFO = {
    "test_autoria_kz": ("real", "AUTO.RIA crops, KZ plates (marketplace photos)", "AUTO.RIA OCR KZ", "test (native split)"),
    "test_autoria_kg": ("real", "AUTO.RIA crops, KG plates (marketplace photos)", "AUTO.RIA OCR KG", "test (native split)"),
    "test_autoria_ru": ("real", "AUTO.RIA crops, RU plates (marketplace photos)", "AUTO.RIA OCR RU", "test (native split)"),
    "test_synth_uz": ("synthetic", "rendered Uzbek plates + augmentations", "synthetic UZ (ours)", "held-out seed, disjoint texts"),
    "test_uz_real": ("real", "Uzbek vehicle-inspection station photos, GT-box crops upright", "Roboflow Uzbek_Plates valid originals", "held-out cars (test)"),
    "test_uz_real_raw": ("real", "Uzbek vehicle-inspection station photos, GT-box crops in raw orientation", "Roboflow Uzbek_Plates valid originals", "held-out cars (test)"),
}


def load_set(name: str, limit: int | None = None):
    from lanes.ident.data.pack import load_test_pack

    imgs, gts, regs, files = load_test_pack(name)
    if limit:
        imgs, gts, regs, files = imgs[:limit], gts[:limit], regs[:limit], files[:limit]
    return imgs, gts, regs, files


def resolve_model(model_dir: Path, model: str):
    for m, c in ((f"{model}.onnx", f"{model}_plate_config.yaml"), (f"{model}.onnx", f"{model}_config.yaml"), (f"{model}.onnx", "plate_config.yaml"), (f"{model}/model.onnx", f"{model}/plate_config.yaml")):
        if (model_dir / m).exists() and (model_dir / c).exists():
            return model_dir / m, model_dir / c
    raise FileNotFoundError(f"{model} in {model_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--modes", nargs="+", default=["argmax"], choices=["argmax", "constrained", "auto"])
    ap.add_argument("--orient", action="store_true", help="rotation search (use for raw-orientation sets)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    onnx, cfg = resolve_model(Path(a.model_dir), a.model)
    ocr = PlateOcrOnnx(onnx, cfg, providers=default_providers(a.device))
    out = RUNS / "ident" / a.exp
    comps = []
    for name in a.sets:
        imgs, gts, regs, files = load_set(name, a.limit)
        if not imgs:
            print("skip (empty)", name)
            continue
        kind, domain, dataset, split = SET_INFO.get(name, ("real", name, name, "test"))
        for mode in a.modes:
            rd = PlateReader(ocr, mode=mode, orient=a.orient)
            t0 = time.time()
            res = []
            for i in range(0, len(imgs), 256):
                res += rd.read(imgs[i:i + 256])
            dt = (time.time() - t0) / len(imgs) * 1000
            preds = [r.text for r in res]
            m = ek.ocr_metrics(preds, gts)
            # visual format head (uz_private / uz_legal / uz_ev / uz_moto / foreign) and text-layout format accuracy
            if res and res[0].region_probs is not None and ocr.regions:
                norm_reg = ["foreign" if g in ("kz", "kg", "ru", "su") else g for g in regs]
                ok = [(ocr.regions[int(np.argmax(r.region_probs))] == g) for r, g in zip(res, norm_reg) if g in ocr.regions]
                if ok:
                    m.append(ek.prop_metric("format_head_acc(5-class visual head)", int(sum(ok)), len(ok)))
            uzr = [(r.format, g) for r, g in zip(res, regs) if g in ("uz_private", "uz_legal", "uz_moto")]
            if uzr:
                m.append(ek.prop_metric("text_layout_format_acc(private/legal/moto from decoded text)", int(sum(f == g for f, g in uzr)), len(uzr)))
            cid = f"ident.ocr.{a.exp}.{name}.{mode}{a.tag}"
            comps.append(ek.component(cid, f"Plate OCR exact-match/char-acc [{mode}]", kind, domain, dataset, split, len(gts), f"{a.model}", m,
                                      notes=f"decoding={mode}; orientation_search={a.orient}; wall time {dt:.2f} ms/crop incl. decode (batch 256, {a.device})",
                                      predictions_csv=f"runs/ident/{a.exp}/preds_{name}_{mode}.csv"))
            ek.write_preds_csv(out / f"preds_{name}_{mode}.csv", [{"file": f, "gt": g, "pred": p, "conf": round(r.conf, 4), "rot": r.rot, "format": r.format}
                                                                  for f, g, p, r in zip(files, gts, preds, res)])
            em = [x for x in m if x["name"] == "exact_match"][0]
            ca = [x for x in m if x["name"] == "char_accuracy"][0]
            print(f"{name:20s} {mode:12s} n={len(gts):5d}  EM={em['value']:.4f} [{em['ci_low']:.3f},{em['ci_high']:.3f}]  CA={ca['value']:.4f}  {dt:.2f} ms/crop", flush=True)
    ek.write_results(out / "component_results.json", comps)


if __name__ == "__main__":
    main()
