"""Export a trained OCR run to artifacts/ident/plate_ocr/ (ONNX + plate_config.yaml + model_card.json).

  python -m lanes.ident.export_ocr --run ocr_main [--metrics-ref docs/lanes/ident_results.json] [--version 0.1.0]
Uses the fast-plate-ocr exporter (Keras 3 -> ONNX, uint8 NHWC input, outputs 'plate' (slots x vocab softmax) and 'region' (format head)),
then re-checks the ONNX against the exported Keras model on random inputs and against onnxruntime CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml

from seen.paths import ARTIFACTS, RUNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="plate_ocr")
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--metrics-ref", default="docs/lanes/ident_results.json")
    ap.add_argument("--auto-margin", type=float, default=6.0)
    ap.add_argument("--notes", default="")
    ap.add_argument("--static-batch", type=int, default=0, help="export with a fixed batch size (0 = dynamic batch)")
    ap.add_argument("--ckpt", default=None, help="explicit .keras file (default: run/best_path.txt)")
    a = ap.parse_args()
    run = RUNS / "ident" / a.run
    best = Path(a.ckpt) if a.ckpt else Path((run / "best_path.txt").read_text().strip())
    out = ARTIFACTS / "ident" / a.out
    out.mkdir(parents=True, exist_ok=True)
    # The trained .keras stores a mixed_bfloat16 policy; ONNX export needs float32 -> rebuild the same architecture in float32 and load the weights.
    os.environ["KERAS_BACKEND"] = "torch"
    import keras

    keras.mixed_precision.set_global_policy("float32")
    from fast_plate_ocr.cli.export import export_onnx
    from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
    from fast_plate_ocr.train.model.model_builders import build_model
    from fast_plate_ocr.train.model.model_schema import load_model_config_from_yaml

    pcfg = load_plate_config_from_yaml(run / "plate_config.yaml")
    mcfg = load_model_config_from_yaml(run / "model_config.yaml")
    m32 = build_model(mcfg, pcfg, enable_region_head=True)
    m32.load_weights(str(best))
    tmp = run / "onnx"
    tmp.mkdir(exist_ok=True)
    onnx = tmp / (best.stem + ".onnx")
    export_onnx(model=m32, plate_config=pcfg, out_file=onnx, simplify=True, dynamic_batch=(a.static_batch == 0), skip_validation=False, onnx_input_dtype="uint8",
                onnx_data_format="channels_last")
    shutil.copy(onnx, out / "model.onnx")
    shutil.copy(run / "plate_config.yaml", out / "plate_config.yaml")
    pc = yaml.safe_load((run / "plate_config.yaml").read_text())
    summ = json.loads((run / "data_summary.json").read_text())
    card = {
        "name": "ident-plate-ocr", "version": a.version, "arch": "fast-plate-ocr CCT-S-v2 (Keras 3, torch backend), fine-tuned from cct_s_v2_global",
        "classes": {"alphabet": pc["alphabet"], "pad_char": pc["pad_char"], "max_plate_slots": pc["max_plate_slots"], "format_head": pc["plate_regions"]},
        "input_size": [pc["img_height"], pc["img_width"], 3], "input_dtype": "uint8 RGB NHWC, resized with cv2 INTER_LINEAR (no letter-boxing)",
        "outputs": {"plate": "(B, slots, vocab) softmax, decode per slot; pad '_'", "region": "(B, 5) visual format head: uz_private, uz_legal, uz_ev, uz_moto, foreign"},
        "decoding": "lanes/ident/reader.py: argmax | constrained (valid Uzbek layouts, regions 01-99) | auto (constrained with fall-back); track voting in lanes/ident/voting.py",
        "metrics_ref": a.metrics_ref, "training_data": {"variant": summ["variant"], "n_train": summ["n_train"], "by_source": summ.get("train_by_source"), "by_format_label": summ.get("train_by_format_label")},
        "licence_notes": ("Initialised from fast-plate-ocr cct_s_v2_global (MIT). Trained on AUTO.RIA KZ/KG/RU crops (licence not stated by the publisher; real plates), "
                          "synthetic Uzbek plates rendered with OFL fonts (ours), and optionally pseudo-labelled crops of the CC BY 4.0 Roboflow Uzbek_Plates set. "
                          "See docs/lanes/ident_sources.md. Weights are research artefacts; do not redistribute commercially without clearing the AUTO.RIA terms."),
        "auto_margin": a.auto_margin, "notes": a.notes, "source_run": str(run),
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=1))
    # sanity: onnxruntime CPU run
    import numpy as np
    import onnxruntime as ort
    s = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])
    x = np.random.randint(0, 256, (s.get_inputs()[0].shape[0] if isinstance(s.get_inputs()[0].shape[0], int) else 2, pc["img_height"], pc["img_width"], 3), dtype=np.uint8)
    y = s.run(None, {s.get_inputs()[0].name: x})
    print("exported", out, [o.name for o in s.get_outputs()], [t.shape for t in y])


if __name__ == "__main__":
    main()
