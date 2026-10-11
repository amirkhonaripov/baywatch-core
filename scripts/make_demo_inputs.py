"""Build demo_inputs/: real, licensed photos and short videos to upload live during the demo (faces blurred).
Run on the box: python scripts/make_demo_inputs.py --out /tmp/demo_inputs"""
import argparse, glob, json, subprocess
from pathlib import Path
import cv2
from seen import blur
from seen.predictor import Frame
from lanes.flow.predict import get_predictor

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
out = Path(a.out); (out / "videos").mkdir(parents=True, exist_ok=True)
DET = get_predictor(); DET.load("cuda:0")
RAW = "/mnt/bigdrive/amirkhon/seen/data/raw/ops_cctv"
J = "/home/amirkhon/seen_fast/eval/jury"

def cut(src, dst, start, dur, fps=10, width=960):
    cap = cv2.VideoCapture(src)
    sfps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    step = max(1, round(sfps / fps))
    w = None; i = 0; n = 0
    tmp = str(dst) + ".raw.mp4"
    while n < dur * fps:
        ok, f = cap.read()
        if not ok: break
        i += 1
        if i % step: continue
        h0, w0 = f.shape[:2]; s = width / w0
        f = cv2.resize(f, (width, int(h0 * s) // 2 * 2))
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        dets = [d for d in DET.predict([Frame(0, 0.0, rgb)])[0].detections if d.cls == "person"]
        f = cv2.cvtColor(blur.blur_faces(rgb, dets), cv2.COLOR_RGB2BGR)
        if w is None: w = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (f.shape[1], f.shape[0]))
        w.write(f); n += 1
    w.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-movflags", "+faststart", str(dst)], check=True)
    Path(tmp).unlink()
    return n

clips = {
    "bay_lift_cctv.mp4": (f"{RAW}/-OJPETY9Bis.mp4", 13, 25, "Two-post lift bay timelapse, YouTube -OJPETY9Bis, CC BY"),
    "bay_tyre_shop_cctv.mp4": (f"{RAW}/e4O3X9gli1s.mp4", 5.5, 27.5, "Vianor tyre shop CCTV, YouTube e4O3X9gli1s, CC BY"),
    "gate_car_park.mp4": (f"{J}/jury_087.webm", 0, 20, "Estacionamiento de un Supermercado Ley, Mazatlán (Wikimedia Commons), CC BY-SA 4.0"),
    "wash_timelapse.mp4": (f"{J}/jury_085.webm", 0, 20, "Timelapse-wasstraat van Slingelandtstraat (Wikimedia Commons), CC BY-SA 4.0"),
}
meta = {}
for name, (src, st, dur, credit) in clips.items():
    if not Path(src).exists():
        src2 = glob.glob(src.replace(".webm", ".*"))
        if not src2: print("missing", src); continue
        src = src2[0]
    n = cut(src, out / "videos" / name, st, dur)
    meta[name] = {"credit": credit, "frames": n}
    print(name, n)
(out / "videos" / "CREDITS.json").write_text(json.dumps(meta, indent=1))
