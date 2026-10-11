"""Pack the trained weights BayWatch needs at inference into one archive (run on the GPU box).

    python scripts/pack_weights.py --out /tmp/baywatch_weights_v1.tar.gz

The file list is exactly what the deployed pipeline opens (traced with strace, see scripts/trace_weights.py).
Public base models (SigLIP2, Qwen3-VL) are NOT packed: scripts/download_weights.py fetches them from Hugging Face.
Writes weights/manifest.json (per-file sha256 + archive sha256) next to the code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from seen.paths import ARTIFACTS

ARTIFACT_FILES = [
    "flow/rfdetr-large-flow/checkpoint.pth", "flow/rfdetr-large-flow/classes.json", "flow/rfdetr-large-flow/model_card.json",
    "flow/router/probe.json", "flow/router/probe.npz",
    "ident/plate_det/model.onnx", "ident/plate_det/model_card.json",
    "ident/plate_ocr/model.onnx", "ident/plate_ocr/model_card.json", "ident/plate_ocr/plate_config.yaml",
    "inspect/damage_seg/model.onnx", "inspect/damage_seg/model_card.json",
    "inspect/dot_chars/model.onnx", "inspect/dot_chars/model_card.json",
    "inspect/parts_seg/model.onnx", "inspect/parts_seg/model_card.json",
    "inspect/tyre_cond/model.onnx", "inspect/tyre_cond/model_card.json",
    "ops/states/backbone_hf/config.json", "ops/states/backbone_hf/model.safetensors", "ops/states/head.onnx",
    "ops/states/hood_yolo.pt", "ops/states/model_card.json", "ops/states/parts_yolo.pt",
]
# third-party cache the plate predictor reads (worldwide plate detector, open-image-models, MIT)
CACHE_DIRS = {"cache/open-image-models": Path.home() / ".cache" / "open-image-models"}
HF_MODELS = {  # public, fetched by download_weights.py; revisions we tested with
    "google/siglip2-so400m-patch14-384": "e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
    "Qwen/Qwen3-VL-8B-Instruct": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    entries = [(ARTIFACTS / rel, f"artifacts/{rel}") for rel in ARTIFACT_FILES]
    for arc, src in CACHE_DIRS.items():
        entries += [(p, f"{arc}/{p.relative_to(src).as_posix()}") for p in sorted(src.rglob("*")) if p.is_file()]
    files = []
    with tarfile.open(a.out, "w:gz", compresslevel=1) as tar:
        for src, arc in entries:
            tar.add(src, arcname=arc)
            files.append({"path": arc, "bytes": src.stat().st_size, "sha256": sha256(src)})
            print(f"{src.stat().st_size / 1e6:9.1f} MB  {arc}")
    out = Path(a.out)
    manifest = {
        "version": "v1",
        "archive": {"name": out.name, "bytes": out.stat().st_size, "sha256": sha256(out),
                    "url": "REPLACE_WITH_GOOGLE_DRIVE_SHARE_LINK"},
        "files": files,
        "huggingface": HF_MODELS,
    }
    Path("weights/manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"archive {out} {out.stat().st_size / 1e9:.2f} GB, {len(files)} files -> weights/manifest.json")


if __name__ == "__main__":
    main()
