"""Jury-set helpers shared by fetch.py, jury_run.py and bench.py.

``eval/jury/jury.jsonl`` is the curated, labelled list (one JSON object per line)::

    {"id": "jury_017", "kind": "image" | "video" | "video_frame", "category": "garage",
     "title": "File:....jpg", "page": "https://commons.wikimedia.org/wiki/File:...", "download_url": "...",
     "file": "jury_017.jpg", "licence": "CC BY-SA 4.0", "licence_url": "...", "author": "...",
     "frame_of": "jury_090", "frame_t_s": 10.0,            # video_frame only: cut from that video by fetch.py
     "expected": {"scene": "bay", "acceptable_scenes": ["bay"], "vehicle": true, "plate_visible": false,
                  "inspection_target": false, "notes": "..."}}

Media live outside the repo in ``SEEN_DATA/raw/eval_jury/<file>`` (fetched by ``python -m eval.jury.fetch``).
"""
from __future__ import annotations

import json
from pathlib import Path

JURY_DIR = Path(__file__).resolve().parent
JURY_LIST = JURY_DIR / "jury.jsonl"
RAW_NAME = "eval_jury"
SCENES = ("gate", "bay", "wash", "street", "closeup", "motorcycle", "unknown")


def raw_dir() -> Path:
    from seen.paths import RAW

    return RAW / RAW_NAME


def fast_jury_dir() -> Path | None:
    """NVMe staging copy on the box (seen.paths.FAST/eval/jury); None where FAST does not exist (e.g. the Mac)."""
    try:
        from seen.paths import FAST
    except ImportError:
        return None
    return FAST / "eval" / "jury" if FAST.exists() else None


def jury_items(path: Path | None = None) -> list[dict]:
    p = path or JURY_LIST
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip() and not l.startswith("//")]


def local_path(item: dict) -> Path:
    """The NVMe copy if staged (the HDD is IO-saturated), else the raw copy."""
    f = fast_jury_dir()
    if f is not None and (f / item["file"]).exists():
        return f / item["file"]
    return raw_dir() / item["file"]


def raw_path(item: dict) -> Path:
    return raw_dir() / item["file"]


def expected_modules(exp: dict) -> list[str]:
    """What the product spec (plan: Architecture) says should run for this input. Independent of seen/router.py.

    - no vehicle: nothing beyond the router and the detector; no visit, the answer is 'unknown' or a scene without a visit
    - vehicle: plate reading and the visit engine
    - vehicle in a bay or wash scene: bay-state scoring
    - close-up of a vehicle or part, a motorcycle, or a single prominent vehicle photo: light inspection
    """
    mods = ["router", "vehicle_detector"]
    if exp.get("vehicle"):
        mods += ["plate", "visit_engine"]
        if exp.get("scene") in ("bay", "wash"):
            mods.append("bay_states")
        if exp.get("scene") in ("closeup", "motorcycle") or exp.get("inspection_target"):
            mods.append("inspection")
    return mods
