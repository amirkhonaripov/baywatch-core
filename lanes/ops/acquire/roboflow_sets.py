"""Roboflow Universe sets for hood / wash / wheel cues (CC BY 4.0 by uploader label).

Run with the key loaded from the workspace .env (never printed):
  bash -c 'set -a; source $SEEN_ROOT/.env; set +a; python -m lanes.ops.acquire.roboflow_sets'
"""
from __future__ import annotations

import json
import os

from seen.paths import RAW

# (workspace, project, export format, local dir)
SETS = [
    ("carhoodopen", "car-hood-open", "yolov8", "ops_rf_car_hood_open"),
    ("vehicles-ztmmv", "car-open-hood", "folder", "ops_rf_car_open_hood"),
    ("deep-learning-ocos", "car-condition-detection", "yolov8", "ops_rf_car_condition_det"),
    ("deep-learning-ocos", "car-condition-ue963", "yolov8", "ops_rf_car_condition"),
    ("um-84bee", "washing-car", "yolov8", "ops_rf_washing_car"),
    ("bimmerwheels", "wheel-tire-detection-ceyns", "yolov8", "ops_rf_wheel_tire"),
]


def main() -> None:
    import argparse

    from roboflow import Roboflow

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(RAW), help="download root (default data/raw; NVMe when the HDD is IO-bound)")
    root = type(RAW)(ap.parse_args().root)

    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    log = []
    for ws, proj, fmt, d in SETS:
        out = root / d
        if out.exists() and any(out.iterdir()):
            log.append(dict(project=f"{ws}/{proj}", status="have"))
            continue
        try:
            p = rf.workspace(ws).project(proj)
            versions = sorted(int(v.version.split("/")[-1]) for v in p.versions())
            v = versions[-1]
            p.version(v).download(fmt, location=str(out), overwrite=True)
            log.append(dict(project=f"{ws}/{proj}", version=v, format=fmt, status="ok", dir=str(out)))
        except Exception as e:  # noqa: BLE001
            log.append(dict(project=f"{ws}/{proj}", status=f"error: {type(e).__name__}: {str(e)[:200]}"))
        print(log[-1], flush=True)
    (root / "ops_rf_download_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
