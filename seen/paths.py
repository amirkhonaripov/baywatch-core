"""Filesystem conventions (box defaults; override with env vars on other machines).

SEEN_ROOT/
  code/        rsync of the Mac repo (never write outputs here)
  data/raw/<dataset>/      downloaded archives / extracted raw data
  data/prep/<lane>/...     prepared crops, features, splits
  runs/<lane>/<exp>/       training runs (weights, logs)
  artifacts/<lane>/<model>/{model.onnx|weights.pt, model_card.json}   deployable models
  renders/                 Blender outputs
  locks/gpu0.lock          flock for training jobs on GPU0
SEEN_FAST (/home/amirkhon/seen_fast/<lane>/)  NVMe staging for prepared data (budget per lane in docs/LANES.md)
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("SEEN_ROOT", "/mnt/bigdrive/amirkhon/seen"))
DATA = Path(os.environ.get("SEEN_DATA", ROOT / "data"))
RAW = DATA / "raw"
PREP = DATA / "prep"
RUNS = Path(os.environ.get("SEEN_RUNS", ROOT / "runs"))
ARTIFACTS = Path(os.environ.get("SEEN_ARTIFACTS", ROOT / "artifacts"))
RENDERS = ROOT / "renders"
LOCKS = ROOT / "locks"
# NVMe staging for prepared/active training data (user-approved, 60 GB total budget). Raw data stays on the HDD.
FAST = Path(os.environ.get("SEEN_FAST", "/home/amirkhon/seen_fast"))


def fast_dir(lane: str) -> Path:
    p = FAST / lane
    p.mkdir(parents=True, exist_ok=True)
    return p


def artifact_dir(lane: str, model: str) -> Path:
    p = ARTIFACTS / lane / model
    p.mkdir(parents=True, exist_ok=True)
    return p
