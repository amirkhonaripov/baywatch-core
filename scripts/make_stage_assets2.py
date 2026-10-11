"""Add inspection / wash / ready-zone stage frames (real public datasets, face-blurred) to web/public/stages.
Run on the box: python scripts/make_stage_assets2.py --out /tmp/stages2"""
import argparse, glob, json, random
from pathlib import Path
import cv2
from seen import blur
from seen.predictor import Frame
from lanes.flow.predict import get_predictor

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
R = "/mnt/bigdrive/amirkhon/seen/data/raw"
DET = get_predictor(); DET.load("cuda:0")
random.seed(3)
sets = {
    "inspect": (sorted(glob.glob(f"{R}/inspect_hitl/Car damages dataset/File1/img/*")), 8,
                {"source": "real_public_dataset", "label": "Real photo (public dataset)", "title": "Humans in the Loop car damage", "licence": "CC0"}),
    "tyre": (sorted(glob.glob(f"{R}/inspect_tyrenet/extracted/Tyre dataset/*/*")), 6,
             {"source": "real_public_dataset", "label": "Real photo (public dataset)", "title": "TyreNet", "licence": "CC BY 4.0"}),
    "wash": (sorted(glob.glob(f"{R}/ops_rf_washing_car/valid/images/*")), 8,
             {"source": "real_public_dataset", "label": "Real photo (public dataset)", "title": "Roboflow washing-car", "licence": "CC BY 4.0 (uploader label)"}),
    "ready": (sorted(glob.glob(f"{R}/flow_pklot/PKLot/PKLot/UFPR04/Sunny/*/*.jpg")), 6,
              {"source": "real_public_dataset", "label": "Real parking camera (public dataset)", "title": "PKLot UFPR04", "licence": "CC BY 4.0"}),
}
man = {"stages": {}}
for st, (files, n, meta) in sets.items():
    pick = random.sample(files, min(n, len(files))) if files else []
    for i, f in enumerate(pick):
        img = cv2.imread(f)
        if img is None: continue
        h, w = img.shape[:2]; s = 960 / max(h, w)
        if s < 1: img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = [d for d in DET.predict([Frame(0, 0.0, rgb)])[0].detections if d.cls == "person"]
        rgb = blur.blur_faces(rgb, dets)
        (out / st).mkdir(exist_ok=True)
        name = f"{st}_{i:02d}.jpg"
        cv2.imwrite(str(out / st / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
        man["stages"].setdefault(st, []).append({"file": f"stages/{st}/{name}", **meta, "t_s": i * 30.0})
(out / "manifest2.json").write_text(json.dumps(man, indent=1))
print({k: len(v) for k, v in man["stages"].items()})
