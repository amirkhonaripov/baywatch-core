"""Re-run eval/bench.py on a Modal L4 (the deployment target and the proxy for an RTX 4060 Ti edge box).

    modal run eval/modal_bench.py                 # quick: pipeline, runtimes, cold start, streams (short)
    modal run eval/modal_bench.py --full          # same settings as the box run
Writes the box-format results to ./runs_modal/<stamp>/docs/lanes/eval_speed.json; then
    python -m eval.cost --root runs_modal/<stamp> && python -m eval.report --root runs_modal/<stamp>
NOTE: not run yet (Modal token not configured on 2026-10-10). Costs about $0.80/h of L4 time; a quick run is ~10 min.

Jury images are read from SEEN_DATA/raw/eval_jury if that folder exists locally; otherwise the remote function
downloads the first 24 vehicle photos from Wikimedia Commons (URLs in eval/jury/jury.jsonl).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
LOCAL_JURY = Path(os.environ.get("SEEN_DATA", "/mnt/bigdrive/amirkhon/seen/data")) / "raw" / "eval_jury"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", "rfdetr==1.11.2", "fast-alpr[onnx-gpu]==0.4.0",
                 "onnxruntime-gpu==1.22.0", "av", "opencv-python-headless", "pydantic>=2.7", "jsonschema",
                 "supervision", "numpy", "pillow")
    .env({"SEEN_ROOT": "/root/seen", "SEEN_DATA": "/root/seen/data", "PYTHONPATH": "/root/repo"})
)
for sub in ("seen", "eval", "contracts", "lanes", "sim"):
    if (REPO / sub).exists():
        image = image.add_local_dir(str(REPO / sub), f"/root/repo/{sub}", ignore=["**/__pycache__/**", "**/*.pyc"])
if LOCAL_JURY.exists():
    image = image.add_local_dir(str(LOCAL_JURY), "/root/seen/data/raw/eval_jury")

app = modal.App("seen-eval-bench", image=image)


@app.function(gpu="L4", timeout=3600, cpu=2.0, memory=8192)
def run_bench(full: bool = False) -> dict:
    import sys

    sys.path.insert(0, "/root/repo")
    from eval.jury.common import jury_items, local_path

    items = [i for i in jury_items() if i["kind"] == "image" and i["expected"]["vehicle"]][:24]
    missing = [i for i in items if not local_path(i).exists()]
    if missing:
        from eval.jury.fetch import fetch

        fetch(missing)
    from eval import bench

    out = Path("/root/out")
    args = ["--out", str(out), "--hardware-label", "NVIDIA L4 (Modal)"]
    args += ["--workers", "1,2"] if full else ["--quick"]
    bench.main(args)
    d = out / "docs" / "lanes"
    return {p.name: p.read_text() for p in d.glob("eval_*.json")}


@app.local_entrypoint()
def main(full: bool = False):
    files = run_bench.remote(full)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = REPO / "runs_modal" / stamp / "docs" / "lanes"
    d.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (d / name).write_text(text)
    print(json.dumps({"written": str(d), "files": sorted(files)}, indent=2))
