"""Real-domain bay test clips (fixed camera, public YouTube; NEVER used for training).

  python -m lanes.ops.acquire.cctv download
  python -m lanes.ops.acquire.cctv sample --n-total 400   # -> data/prep/ops/cctv/frames/<clip>_<k>.jpg + index.csv
"""
from __future__ import annotations

import argparse
import csv

from seen.paths import PREP, RAW

OUT = RAW / "ops_cctv"
FR = PREP / "ops" / "cctv"

# id -> (licence, note). Source: research/data_inventory_bay.md section 6.
CLIPS = {
    "e4O3X9gli1s": ("CC BY", "Vianor Moscow tyre shop, real CCTV CAM-3 timelapse"),
    "-OJPETY9Bis": ("CC BY", "Revolution Automotive workshop timelapse"),
    "JIEX2ruxZKM": ("Standard YouTube", "Dave Hollinger auto repair timelapse"),
    "wWSNERw5oME": ("Standard YouTube", "Cody Mausolf GoPro shop timelapse"),
    "2mhugD9KYjI": ("Standard YouTube", "Chris Thompson mechanic shop timelapse"),
    "BZsVY9FUUOo": ("Standard YouTube", "1st chance tire shop timelapse"),
    "Bh4IqT2Wq-s": ("Standard YouTube", "Williams Auto Group service timelapse"),
}
# Bh4IqT2Wq-s is a close tripod shot of an engine job, not a CCTV-like bay view: downloaded, not used in the test set.
TEST_CLIPS = [c for c in CLIPS if c != "Bh4IqT2Wq-s"]


def download() -> None:
    import yt_dlp

    OUT.mkdir(parents=True, exist_ok=True)
    opts = dict(format="bv*[height<=720][ext=mp4]/bv*[height<=720]/b", outtmpl=str(OUT / "%(id)s.%(ext)s"),
                quiet=True, no_warnings=True, retries=3)
    for vid in CLIPS:
        if list(OUT.glob(f"{vid}.*")):
            continue
        try:
            with yt_dlp.YoutubeDL(opts) as y:
                y.download([f"https://www.youtube.com/watch?v={vid}"])
            print("ok", vid)
        except Exception as e:  # noqa: BLE001
            print("FAIL", vid, e)


def sample(n_total: int) -> None:
    import cv2
    import numpy as np

    FR.joinpath("frames").mkdir(parents=True, exist_ok=True)
    files = {v: next(iter(OUT.glob(f"{v}.mp4")), None) for v in TEST_CLIPS}
    files = {v: f for v, f in files.items() if f is not None}
    durs = {}
    for v, f in files.items():
        cap = cv2.VideoCapture(str(f))
        durs[v] = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS))
        cap.release()
    per = max(1, n_total // len(files))
    rows = []
    for v, f in files.items():
        n, fps = durs[v]
        # skip the first/last 3 % (title cards); evenly spaced
        idx = np.linspace(0.03 * n, 0.97 * n, per).astype(int)
        cap = cv2.VideoCapture(str(f))
        for k, i in enumerate(idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if not ok:
                continue
            name = f"{v}_{k:03d}.jpg"
            cv2.imwrite(str(FR / "frames" / name), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append(dict(frame=name, clip=v, k=k, frame_idx=int(i), t_s=round(i / fps, 3), w=fr.shape[1], h=fr.shape[0]))
        cap.release()
        print(v, n, fps, len(idx))
    with (FR / "index.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("frames", len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "sample"])
    ap.add_argument("--n-total", type=int, default=420)
    a = ap.parse_args()
    download() if a.cmd == "download" else sample(a.n_total)
