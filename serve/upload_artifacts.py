"""Push deployable model weights from the box to the Modal Volume "seen-artifacts".

Run on the box (Modal is authenticated there):
    scripts/box.sh -- python -m serve.upload_artifacts --dry-run     # list what would be uploaded
    scripts/box.sh -- python -m serve.upload_artifacts               # upload

Volume layout (mounted at /artifacts in the GPU container, see serve/modal_app.py):
    artifacts/<lane>/<model>/{model.onnx|weights.pt, model_card.json}    <- $SEEN_ARTIFACTS (lane deliverables)
    cache/roboflow/...            <- ~/.roboflow/models          (RF-DETR pretrained weights; RF_HOME in the container)
    cache/fast-plate-ocr/...      <- ~/.cache/fast-plate-ocr     (fast-alpr OCR)
    cache/open-image-models/...   <- ~/.cache/open-image-models  (fast-alpr plate detector)
Only files under these roots are sent; training runs/checkpoints elsewhere are never uploaded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from seen.paths import ARTIFACTS

VOLUME = "seen-artifacts"
SKIP_SUFFIXES = {".ckpt", ".log", ".tmp", ".lock"}


def sources() -> list[tuple[Path, str]]:
    home = Path.home()
    return [
        (ARTIFACTS, "artifacts"),
        (home / ".roboflow" / "models", "cache/roboflow"),
        (home / ".cache" / "fast-plate-ocr", "cache/fast-plate-ocr"),
        (home / ".cache" / "open-image-models", "cache/open-image-models"),
    ]


def plan(only: list[str] | None = None) -> list[tuple[Path, str, int]]:
    out = []
    for root, remote in sources():
        if only and not any(remote.startswith(o) or o.startswith(remote + "/") for o in only):
            continue
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            rp = f"{remote}/{p.relative_to(root).as_posix()}"
            if only and not any(rp.startswith(o) for o in only):
                continue
            if p.is_file() and p.suffix not in SKIP_SUFFIXES and not p.name.startswith("."):
                out.append((p, f"{remote}/{p.relative_to(root).as_posix()}", p.stat().st_size))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", help="remote prefixes to include, e.g. artifacts/ident cache/roboflow")
    ap.add_argument("--max-gb", type=float, default=6.0, help="refuse to upload more than this")
    args = ap.parse_args()

    files = plan(args.only)
    total = sum(s for _, _, s in files)
    for p, remote, size in files:
        print(f"{size / 1e6:9.1f} MB  {remote}  <- {p}")
    print(f"{len(files)} files, {total / 1e9:.2f} GB -> modal volume {VOLUME}")
    if args.dry_run or not files:
        return
    if total > args.max_gb * 1e9:
        raise SystemExit(f"refusing: {total / 1e9:.2f} GB > --max-gb {args.max_gb}")

    import modal

    vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
    with vol.batch_upload(force=True) as batch:
        for p, remote, _ in files:
            batch.put_file(str(p), "/" + remote)
    print("uploaded; listing:")
    for e in vol.listdir("/", recursive=False):
        print("  ", e.path)


if __name__ == "__main__":
    main()
