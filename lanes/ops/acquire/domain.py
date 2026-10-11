"""Unlabelled fixed-camera workshop / car-wash clips for teacher distillation (TRAINING domain data).

Different shops and YouTube ids from the six CCTV test clips in lanes.ops.acquire.cctv (asserted below). Frames get
Qwen3-VL teacher labels only; they are never used for evaluation.
  python -m lanes.ops.acquire.domain download
  python -m lanes.ops.acquire.domain sample --per-clip 150     # -> FAST/ops/frames/domain/<id>/... + PREP/ops/frames_domain.parquet
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

from lanes.ops.acquire.cctv import TEST_CLIPS
from seen.paths import FAST, PREP, RAW

OUT = RAW / "ops_domain"
FR = FAST / "ops" / "frames" / "domain"
# research/data_inventory_bay.md section 6, list B (+ Bh4IqT2Wq-s, downloaded earlier and kept out of the test set)
DOMAIN = {
    "Bh4IqT2Wq-s": "Williams Auto Group service timelapse",
    "ZPN8nzjUXaE": "tyre-shop CCTV timelapse (R. Pokrovsky)",
    "Qk-FmEMHX5I": "Belka-Service bay timelapse",
    "yEsYfsVj4jI": "Christian Brothers Automotive timelapse",
    "vJc9AIHQJkM": "Stratstone service timelapse",
    "sAOUw1pNUpQ": "Banks Autos service department timelapse",
    "uua317eka8o": "Lik-Lik garage timelapse",
    "SVipIgzzI8k": "Slick Diggs mechanics shop timelapse",
    "vr5u6knKNUk": "BAY Automotive timelapse",
    "E3BJXah51aM": "Bays Complete Auto Repair oil change drive-thru",
    "aNCbUwc3k34": "Lik-Lik garage timelapse 2",
    "cDuW72oagyg": "Bartlett Automotive workshop timelapse 1",
    "DIHVgyb2Lds": "Bartlett Automotive workshop timelapse 2",
    "9G77Txg3oSg": "self-service car wash CCTV",
    "UN7sf_0VKPs": "car wash bay IP CCTV",
    "llk_jzFaws4": "car wash CCTV sample 1",
    "InG57uGKKzs": "car wash CCTV sample 2",
}
assert not set(DOMAIN) & set(TEST_CLIPS), "domain clips must not overlap the CCTV test clips"


def download() -> None:
    import yt_dlp

    OUT.mkdir(parents=True, exist_ok=True)
    opts = dict(format="bv*[height<=720][ext=mp4]/bv*[height<=720]/b", outtmpl=str(OUT / "%(id)s.%(ext)s"), quiet=True,
                no_warnings=True, retries=3)
    for vid in DOMAIN:
        if list(OUT.glob(f"{vid}.*")):
            continue
        try:
            with yt_dlp.YoutubeDL(opts) as y:
                y.download([f"https://www.youtube.com/watch?v={vid}"])
            print("ok", vid, flush=True)
        except Exception as e:  # noqa: BLE001
            print("FAIL", vid, str(e)[:120], flush=True)


def sample(per_clip: int) -> None:
    rows = []
    for vid in DOMAIN:
        src = next(iter(OUT.glob(f"{vid}.mp4")), None) or next(iter(OUT.glob(f"{vid}.*")), None)
        if src is None:
            continue
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)],
                                   capture_output=True, text=True).stdout.strip() or 0)
        if dur <= 0:
            continue
        d = FR / vid
        fps = per_clip / max(dur - 2.0, 1.0)
        if not (d / ".done").exists():
            import shutil

            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-ss", "1", "-i", str(src), "-vf", f"fps={fps:.6f},scale='min(640,iw)':-2",
                            "-q:v", "4", str(d / "%05d.jpg")], check=False)
            (d / ".done").write_text(f"{fps}")
        for f in sorted(p for p in d.glob("*.jpg") if p.stem.isdigit()):
            rows.append(dict(dataset="domain", source=f"yt:{vid}", video=vid, path=str(f.relative_to(FAST / "ops")),
                             t_s=round(1 + (int(f.stem) - 1) / fps, 2), split="train", label_src="teacher_only"))
        print(vid, sum(1 for p in d.glob("*.jpg") if p.stem.isdigit()), flush=True)
    pd.DataFrame(rows).to_parquet(PREP / "ops" / "frames_domain.parquet")
    print("domain frames", len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "sample"])
    ap.add_argument("--per-clip", type=int, default=150)
    a = ap.parse_args()
    download() if a.cmd == "download" else sample(a.per_clip)
