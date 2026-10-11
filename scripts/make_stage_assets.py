"""Build web/public/stages/: face-blurred, resized frame sequences per service stage for the guided demo.

bay  : real public workshop CCTV, CC BY clips only (Vianor tyre shop e4O3X9gli1s, two-post lift bay -OJPETY9Bis).
wash/ready/handover/bay_synth : Blender renders of our synthetic service centre (sim/blender), labelled synthetic.
Run on the box: python scripts/make_stage_assets.py --out /tmp/stages
"""
import argparse, json, glob
from pathlib import Path
import cv2
from seen import blur
from seen.predictor import Det, Frame
from lanes.flow.predict import get_predictor
DET = get_predictor(); DET.load("cuda:0")

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
F = "/home/amirkhon/seen_fast/ops/cctv/frames"
manifest = {"stages": {}}

def put(stage, name, img_bgr, meta):
    h, w = img_bgr.shape[:2]; s = 960 / max(h, w)
    if s < 1: img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    dets = [d for d in DET.predict([Frame(0, 0.0, rgb)])[0].detections if d.cls == "person"]
    rgb = blur.blur_faces(rgb, dets)  # YuNet on the frame + per-person head crops + head-box fallback
    d = out / stage; d.mkdir(exist_ok=True)
    cv2.imwrite(str(d / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
    manifest["stages"].setdefault(stage, []).append({"file": f"stages/{stage}/{name}", **meta})

clips = {"e4O3X9gli1s": ("Vianor tyre shop CCTV (YouTube, CC BY)", "https://www.youtube.com/watch?v=e4O3X9gli1s"),
         "-OJPETY9Bis": ("Two-post lift bay timelapse (YouTube, CC BY)", "https://www.youtube.com/watch?v=-OJPETY9Bis")}
for clip, (title, url) in clips.items():
    fs = sorted(glob.glob(f"{F}/{clip}_*.jpg"))[::3][:20]
    for i, f in enumerate(fs):
        put("bay_real", f"{clip.strip('-')}_{i:02d}.jpg", cv2.imread(f),
            {"source": "real_public_cctv", "label": "Real CCTV (public)", "clip": clip, "title": title, "url": url, "licence": "CC BY", "t_s": i * 3.0})
R = "/home/amirkhon/seen_fast/sim/bay_v1"
eps = sorted(glob.glob(f"{R}/ep*_test"))[:2] or sorted(glob.glob(f"{R}/ep*"))[:2]
for e in eps:
    fs = sorted(glob.glob(f"{e}/frames/*.jpg") + glob.glob(f"{e}/frames/*.png"))[::12][:24]
    for i, f in enumerate(fs):
        put("bay_synth", f"{Path(e).name}_{i:02d}.jpg", cv2.imread(f),
            {"source": "synthetic_render", "label": "Synthetic render (Blender)", "episode": Path(e).name, "licence": "own render, CC0 assets", "t_s": i * 48.0})
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))
print({k: len(v) for k, v in manifest["stages"].items()})
