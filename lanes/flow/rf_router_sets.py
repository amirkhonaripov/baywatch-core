"""Roboflow Universe images for the router's 'gate' / 'wash' classes (CC BY 4.0 by uploader label; provenance unknown, plates visible).

Run on the box with the key from the workspace .env (never printed):
  scripts/box.sh -- bash -c 'set -a; source $SEEN_ROOT/.env; set +a; python -m lanes.flow.rf_router_sets download'
  scripts/box.sh -- python -m lanes.flow.rf_router_sets pick      # resized subset + records_rf.jsonl (no key needed)
The 'washing car' set was already downloaded by the ops lane (read-only reuse of data/raw/ops_rf_washing_car).
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import cv2

from lanes.flow.common import prep_dir, raw_dir
from seen.paths import RAW

SETS = [  # (workspace, project, source name, scene)
    ("hssafty", "vehicle-gate-counter", "rf_gate_counter", "gate"),
    ("feyza-axujh", "vehicle-gate-detection", "rf_gate_detection", "gate"),
]


def download():
    from roboflow import Roboflow

    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    for ws, proj, name, _ in SETS:
        out = raw_dir(name)
        if out.exists() and any(out.iterdir()):
            print("have", name)
            continue
        try:
            p = rf.workspace(ws).project(proj)
            v = sorted(int(x.version.split("/")[-1]) for x in p.versions())[-1]
            p.version(v).download("yolov8", location=str(out), overwrite=True)
            print("ok", name, v, flush=True)
        except Exception as e:  # noqa: BLE001
            print("error", name, type(e).__name__, str(e)[:200], flush=True)


def _images(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png") and "images" in p.parts)


def pick(n_gate_each=250, n_wash=250):
    outd = prep_dir("router", "images_rf")
    recs = []
    rng = random.Random(0)

    def add(source, files, scene, dataset, lic):
        rng.shuffle(files)
        for f in files[:n_gate_each if scene == "gate" else n_wash]:
            im = cv2.imread(str(f))
            if im is None:
                continue
            s = 800 / max(im.shape[:2])
            if s < 1:
                im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            dst = outd / source / (f.stem[:40] + ".jpg")
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
            recs.append({"id": f"{source}:{dst.stem}", "path": str(dst), "label": scene, "source": source, "dataset": dataset, "license": lic, "url": ""})
        print(source, len(files))

    for ws, proj, name, scene in SETS:
        if raw_dir(name).exists():
            add(name, _images(raw_dir(name)), scene, f"roboflow:{ws}/{proj}", "CC BY 4.0 (uploader label)")
    # wash: only images that contain the 'washing car' class (ops lane download)
    wr = RAW / "ops_rf_washing_car"
    if wr.exists():
        import yaml

        names = yaml.safe_load(open(wr / "data.yaml"))["names"]
        names = list(names.values()) if isinstance(names, dict) else names
        wash_ids = {i for i, n in enumerate(names) if "wash" in n.lower()}
        files = []
        for lf in wr.rglob("labels/*.txt"):
            if any(int(l.split()[0]) in wash_ids for l in lf.read_text().splitlines() if l.strip()):
                im = lf.parent.parent / "images" / (lf.stem + ".jpg")
                if im.exists():
                    files.append(im)
        add("rf_washing_car", files, "wash", "roboflow:um-84bee/washing-car (via ops lane)", "CC BY 4.0 (uploader label)")
    (prep_dir("router") / "records_rf.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    print("records", len(recs))


if __name__ == "__main__":
    {"download": download, "pick": pick}[sys.argv[1]]()
