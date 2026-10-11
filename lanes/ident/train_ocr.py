"""Fine-tune the pretrained fast-plate-ocr CCT-S-v2 (global) on AUTO.RIA KZ+KG+RU (real) + synthetic Uzbek plates
(+ optionally pseudo-labelled real Uzbek crops), re-using the fast-plate-ocr model builder, loss, metrics and augmentation
(Keras 3, torch backend) with an in-RAM packed dataset (the shared drive is too I/O-starved for per-sample PNG reads).

  scripts/box.sh --gpu 0 --lock -- python -m lanes.ident.train_ocr --name ocr_main --variant main --epochs 30

Variants (ablation "synthetic on/off"):
  real_only   AUTO.RIA KZ+KG+RU only                        (no Uzbek-specific data at all)
  main        AUTO.RIA + synthetic UZ (+ composite crops)   (headline model)
  main_pl     main + real-UZ crops pseudo-labelled by reader/VLM agreement (train split only; declared as pseudo-labels)
Outputs in runs/ident/<name>/: plate_config.yaml, model_config.yaml, data_summary.json, train_files.txt, keras/<ts>/best.keras, training_log.csv,
exported ONNX in runs/ident/<name>/onnx/.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import yaml

from lanes.ident.data.pack import OCR, load_train_pack
from seen.paths import RAW, RUNS

PRE = RAW / "ident_pretrained"
FORMATS = ["uz_private", "uz_legal", "uz_ev", "uz_moto", "foreign"]


def take(name: str, n: int | None, rng: random.Random, region: str | None = None):
    if not (OCR / f"{name}.npy").exists():
        return None
    arr, texts, regs, srcs = load_train_pack(name)
    idx = list(range(len(texts)))
    if n is not None and n < len(idx):
        idx = rng.sample(idx, n)
    idx.sort()
    return arr[idx], [texts[i] for i in idx], [region or regs[i] for i in idx], [srcs[i] for i in idx]


def build(variant: str, seed: int, ru_cap: int, synth_n: int, comp_n: int):
    rng = random.Random(seed)
    parts = [take("autoria_kz_train", None, rng, "foreign"), take("autoria_kg_train", None, rng, "foreign"), take("autoria_ru_train", ru_cap, rng, "foreign")]
    if variant in ("main", "main_pl"):
        parts += [take("synth_uz_train", synth_n, rng), take("synth_comp_train", comp_n, rng)]
    if variant == "main_pl":
        parts += [take("uz_real_pl_train", None, rng)]
    parts = [p for p in parts if p is not None]
    X = np.concatenate([p[0] for p in parts])
    T = sum((p[1] for p in parts), [])
    R = sum((p[2] for p in parts), [])
    S = sum((p[3] for p in parts), [])
    vparts = [take("autoria_kz_val", None, rng, "foreign"), take("autoria_kg_val", 1000, rng, "foreign"), take("autoria_ru_val", 1000, rng, "foreign"),
              take("synth_uz_val", None, rng), take("uz_real_pl_val", None, rng)]
    vparts = [p for p in vparts if p is not None]
    VX = np.concatenate([p[0] for p in vparts])
    VT = sum((p[1] for p in vparts), [])
    VR = sum((p[2] for p in vparts), [])
    return (X, T, R, S), (VX, VT, VR)


class RamPlates:
    """torch Dataset over packed uint8 arrays; fast-plate-ocr albumentations pipeline for training."""

    def __init__(self, X, T, R, plate_cfg, regions, train: bool, aug=None):
        self.X, self.T, self.R, self.cfg, self.reg, self.train, self.aug = X, T, R, plate_cfg, regions, train, aug
        self.alpha = plate_cfg["alphabet"]
        self.slots = plate_cfg["max_plate_slots"]
        self.pad = plate_cfg["pad_char"]
        self.ri = {r: i for i, r in enumerate(regions)}

    def __len__(self):
        return len(self.T)

    def __getitem__(self, i):
        x = np.array(self.X[i])
        if self.aug is not None:
            x = self.aug(image=x)["image"]
        t = self.T[i].ljust(self.slots, self.pad)[: self.slots]
        y = np.zeros((self.slots, len(self.alpha)), np.float32)
        for k, c in enumerate(t):
            y[k, self.alpha.index(c)] = 1.0
        r = np.zeros(len(self.reg), np.float32)
        r[self.ri[self.R[i]]] = 1.0
        return x.astype(np.float32), {"plate": y, "region": r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--variant", choices=["real_only", "main", "main_pl"], default="main")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ru-cap", type=int, default=25000)
    ap.add_argument("--synth-n", type=int, default=60000)
    ap.add_argument("--comp-n", type=int, default=8000)
    ap.add_argument("--base", default="cct_s_v2_global")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline")
    ap.add_argument("--ema", action="store_true", help="optimizer EMA of the weights (off by default: it costs ~1.2 s/step of Python overhead on the loaded box)")
    a = ap.parse_args()

    out = RUNS / "ident" / a.name
    out.mkdir(parents=True, exist_ok=True)
    (X, T, R, S), (VX, VT, VR) = build(a.variant, a.seed, a.ru_cap, a.synth_n, a.comp_n)
    if a.smoke:
        X, T, R, S = X[:2048], T[:2048], R[:2048], S[:2048]
        VX, VT, VR = VX[:512], VT[:512], VR[:512]
        a.epochs = 1
    # leakage guard against published held-out manifests (Lane G); we record what was checked
    leak = {"manifests_present": False}
    try:
        from eval.manifest import components
        leak["manifests_present"] = bool(components())
    except Exception as e:  # noqa: BLE001
        leak["error"] = str(e)
    (out / "leakage_note.json").write_text(json.dumps(leak))
    # packed sources: <dataset>:<native id>; composite crops list the REAL background photo they were built on (so Lane G's pHash audit sees it)
    from seen.paths import fast_dir
    cmeta = {}
    cm = fast_dir("ident") / "det" / "synth_comp" / "meta.jsonl"
    if cm.exists():
        cmeta = {m["name"]: m["src"] for m in map(json.loads, cm.read_text().splitlines())}
    (out / "train_files.txt").write_text("\n".join(cmeta.get(s.rsplit("_", 1)[0], s) for s in S) + "\n")
    pc = yaml.safe_load((PRE / f"{a.base}_plate_config.yaml").read_text())
    pc["plate_regions"] = FORMATS
    (out / "plate_config.yaml").write_text(yaml.safe_dump(pc, sort_keys=False))
    shutil.copy(PRE / f"{a.base}_model_config.yaml", out / "model_config.yaml")
    comp = Counter(r for r in R)
    srcs = Counter(("synth_comp" if (":" not in s and "_" in s and s.split("_")[0] in ("rodosol", "ccpd", "photos")) else s.split(":")[0] + (":" + s.split(":")[1] if s.startswith("synth") else "")) if ":" in s else "synth_comp" for s in S)
    (out / "data_summary.json").write_text(json.dumps({"variant": a.variant, "n_train": len(T), "n_val": len(VT), "train_by_format_label": comp,
                                                       "train_by_source": srcs, "args": vars(a)}, indent=1))
    print("train", len(T), "val", len(VT), dict(comp), dict(srcs), flush=True)

    import albumentations as A
    import keras
    import torch
    from fast_plate_ocr.train.data.augmentation import default_train_augmentation
    from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
    from fast_plate_ocr.train.model.loss import cce_loss
    from fast_plate_ocr.train.model.metric import cat_acc_metric, plate_acc_metric
    from fast_plate_ocr.train.model.model_builders import build_model
    from fast_plate_ocr.train.model.model_schema import load_model_config_from_yaml
    from keras.src.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, SwapEMAWeights, TerminateOnNaN
    from keras.src.optimizers import AdamW

    keras.utils.set_random_seed(a.seed)
    keras.mixed_precision.set_global_policy("mixed_bfloat16")
    plate_cfg = load_plate_config_from_yaml(out / "plate_config.yaml")
    model_cfg = load_model_config_from_yaml(out / "model_config.yaml")
    aug = default_train_augmentation(img_color_mode=plate_cfg.image_color_mode)
    A.save(aug, out / "train_augmentation.yaml", "yaml")
    tr = RamPlates(X, T, R, pc, FORMATS, True, aug)
    va = RamPlates(VX, VT, VR, pc, FORMATS, False, None)
    dl = torch.utils.data.DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0,
                                     prefetch_factor=4 if a.workers > 0 else None)
    vdl = torch.utils.data.DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=min(4, a.workers))
    model = build_model(model_cfg, plate_cfg, enable_region_head=True)
    if not a.no_pretrained:
        model.load_weights(str(PRE / f"{a.base}.keras"), skip_mismatch=True)
        print("loaded pretrained", a.base, flush=True)
    steps = a.epochs * len(dl)
    warm = max(1, int(0.05 * steps))
    sched = keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.0, decay_steps=steps - warm, alpha=1e-2, warmup_steps=warm, warmup_target=a.lr)
    opt = AdamW(sched, weight_decay=0.01, clipnorm=1.0, use_ema=a.ema)
    opt.exclude_from_weight_decay(var_names=["bias", "layer_norm"])
    V, N = plate_cfg.vocabulary_size, plate_cfg.max_plate_slots
    model.compile(loss={"plate": cce_loss(V, 0.01), "region": keras.losses.CategoricalCrossentropy()}, loss_weights={"plate": 0.9, "region": 0.1},
                  jit_compile=False, optimizer=opt,
                  metrics={"plate": [cat_acc_metric(N, V), plate_acc_metric(N, V)], "region": [keras.metrics.CategoricalAccuracy(name="acc")]})
    kdir = out / "keras" / time.strftime("%Y-%m-%d_%H-%M-%S")
    kdir.mkdir(parents=True, exist_ok=True)
    cbs = [EarlyStopping(monitor="val_plate_acc", mode="max", patience=10, verbose=1), *([SwapEMAWeights(swap_on_epoch=True)] if a.ema else []),
           ModelCheckpoint(kdir / "last.keras", save_best_only=False), ModelCheckpoint(kdir / "best.keras", monitor="val_plate_acc", mode="max", save_best_only=True, verbose=1),
           TerminateOnNaN(), CSVLogger(str(kdir / "training_log.csv"))]
    model.fit(dl, epochs=a.epochs, validation_data=vdl, callbacks=cbs)
    (out / "best_path.txt").write_text(str(kdir / "best.keras"))
    print("done", kdir, flush=True)


if __name__ == "__main__":
    main()
