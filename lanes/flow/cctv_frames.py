"""Frames from the ops lane's public workshop / tyre-shop CCTV-style videos (data/raw/ops_cctv/*.mp4, read-only) for the router's 'bay' class.

One source per video (so source-wise splits are possible). Run on the box:  python -m lanes.flow.cctv_frames --every 5
Videos are YouTube clips (licences per clip in docs/lanes/ops_sources.md); staff faces are visible in some frames, so these images are
used for training/evaluating the router only and never shown.
"""
import argparse
import json
from pathlib import Path

import cv2

from lanes.flow.common import prep_dir
from seen.paths import RAW

ap = argparse.ArgumentParser()
ap.add_argument("--every", type=float, default=5.0)
a = ap.parse_args()
outd = prep_dir("router", "images_cctv")
recs = []
for mp4 in sorted((RAW / "ops_cctv").glob("*.mp4")):
    cap = cv2.VideoCapture(str(mp4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(a.every * fps)
    k = 0
    for fi in range(0, n, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, im = cap.read()
        if not ok:
            break
        s = 800 / max(im.shape[:2])
        if s < 1:
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        dst = outd / mp4.stem / f"{k:04d}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
        recs.append({"id": f"cctv_{mp4.stem}:{k:04d}", "path": str(dst), "label": "bay", "source": f"cctv_{mp4.stem}", "dataset": "youtube_public_cctv(ops_cctv)",
                     "license": "YouTube (see ops_sources.md)", "url": f"https://www.youtube.com/watch?v={mp4.stem}"})
        k += 1
    print(mp4.name, n, fps, k)
(prep_dir("router") / "records_cctv.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
