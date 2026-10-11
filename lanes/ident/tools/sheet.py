"""Contact-sheet helper: render N images (optionally with YOLO boxes / captions) into one JPEG for visual review.

python -m lanes.ident.tools.sheet --out sheet.jpg --cols 5 --cell 320 img1 img2 ...
Captions: pass "path::caption". Boxes: if <dir>/../labels/<stem>.txt exists and --yolo, draw it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(sz=14):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", "/System/Library/Fonts/Menlo.ttc"):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def draw_yolo(im: Image.Image, label_file: Path):
    if not label_file.exists():
        return im
    d = ImageDraw.Draw(im)
    W, H = im.size
    for ln in label_file.read_text().splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        cx, cy, w, h = map(float, p[1:5])
        d.rectangle([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], outline=(255, 0, 0), width=3)
    return im


def make_sheet(items, out, cols=5, cell=320, yolo=False, cap_h=22, keep_aspect=True):
    f = font(13)
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + cap_h)), "white")
    d = ImageDraw.Draw(sheet)
    for i, it in enumerate(items):
        path, _, cap = it.partition("::")
        im = Image.open(path).convert("RGB")
        if yolo:
            p = Path(path)
            im = draw_yolo(im, p.parent.parent / "labels" / (p.stem + ".txt"))
        if keep_aspect:
            im.thumbnail((cell, cell))
        else:
            im = im.resize((cell, cell))
        x, y = (i % cols) * cell, (i // cols) * (cell + cap_h)
        sheet.paste(im, (x, y))
        d.text((x + 3, y + cell + 3), cap or Path(path).name[:40], fill=(0, 0, 0), font=f)
    sheet.save(out, quality=88)
    return sheet


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--cell", type=int, default=320)
    ap.add_argument("--yolo", action="store_true")
    ap.add_argument("items", nargs="+")
    a = ap.parse_args()
    make_sheet(a.items, a.out, a.cols, a.cell, a.yolo)
