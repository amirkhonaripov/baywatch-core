"""Crops of YOLO-labelled plates from a Roboflow export dir, as a captioned contact sheet (for design calibration)."""
import argparse, random
from pathlib import Path
from PIL import Image, ImageDraw
from lanes.ident.tools.sheet import font

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--split", default="valid"); ap.add_argument("--n", type=int, default=40)
ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--cols", type=int, default=4)
ap.add_argument("--start", type=int, default=0)
a = ap.parse_args()
root = Path(a.root) / a.split
imgs = sorted((root / "images").glob("*.jpg"))
random.Random(a.seed).shuffle(imgs)
imgs = imgs[a.start:a.start + a.n]
CW, CH = 400, 130
rows = (len(imgs) + a.cols - 1) // a.cols
sheet = Image.new("RGB", (a.cols * CW, rows * (CH + 20)), "white")
d = ImageDraw.Draw(sheet); f = font(12)
for i, p in enumerate(imgs):
    im = Image.open(p).convert("RGB"); W, H = im.size
    lab = (root / "labels" / (p.stem + ".txt")).read_text().split()
    cx, cy, w, h = map(float, lab[1:5])
    pad = 0.15
    box = [(cx - w / 2 * (1 + pad)) * W, (cy - h / 2 * (1 + pad)) * H, (cx + w / 2 * (1 + pad)) * W, (cy + h / 2 * (1 + pad)) * H]
    cr = im.crop(tuple(int(round(v)) for v in box))
    if cr.height > cr.width * 1.2:  # vertical plate: rotate for viewing
        cr = cr.rotate(90, expand=True)
    cr.thumbnail((CW - 4, CH))
    x, y = (i % a.cols) * CW, (i // a.cols) * (CH + 20)
    sheet.paste(cr, (x + 2, y))
    d.text((x + 3, y + CH + 3), f"{i + a.start} {p.name[:34]}", fill=(0, 0, 0), font=f)
sheet.save(a.out, quality=90)
