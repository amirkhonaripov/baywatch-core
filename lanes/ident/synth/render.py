"""Clean Uzbek licence-plate renderer (no degradation; see augment.py for that).

Plate geometry follows the formats in lanes/ident/uz_format.py. Unit = millimetres at PX_PER_MM pixels/mm.
  car plates      520 x 110 mm: black rim, region field at the left in a thin rectangle (GREEN for electric vehicles),
                  main field = 6 character slots on a fixed grid (1 letter + 3 digits + 2 letters, or 3 digits + 3 letters),
                  flag + "UZ" block at the right.
  motorcycle      square 2-row plate (layout from Wikipedia/goldenpages text, not verified on photos): top row three digits and a
                  small region code at the upper right, bottom row two letters, flag/UZ at the lower right.
Fonts: OFL-licensed DIN-like faces from google/fonts (Barlow family, Saira Semi Condensed, Share Tech); NO verified free
digitisation of the real FE-Schrift was found, so glyph details of the real plates are approximated, not reproduced.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lanes.ident import uz_format as uz

FONT_DIR = Path(__file__).parent / "fonts"
FONT_FILES = [  # (file, weight_bias) all SIL OFL 1.1, see fonts/OFL-*.txt
    ("Barlow-Bold.ttf", 3), ("BarlowSemiCondensed-Bold.ttf", 3), ("BarlowSemiCondensed-SemiBold.ttf", 2),
    ("BarlowCondensed-SemiBold.ttf", 1), ("SairaSemiCondensed-Bold.ttf", 2), ("ShareTech-Regular.ttf", 1),
]
PX = 2.0  # pixels per mm


@lru_cache(maxsize=64)
def _font(name: str, size: int):
    return ImageFont.truetype(str(FONT_DIR / name), size)


def pick_font(rng: random.Random) -> str:
    return rng.choices([f for f, _ in FONT_FILES], weights=[w for _, w in FONT_FILES])[0]


@dataclass
class PlateStyle:
    font: str = "Barlow-Bold.ttf"
    ev: bool = False
    paper: tuple = (244, 244, 240)
    ink: tuple = (18, 18, 20)
    green: tuple = (0, 128, 72)
    cap_mm: float = 64.0
    cell_mm: float = 51.0
    gap_mm: float = 15.0
    hscale: float = 1.0
    bold_mm: float = 0.0
    region_cap_mm: float = 44.0
    rim_mm: float = 3.2
    corner_mm: float = 9.0
    flag: bool = True
    region_box: bool = True
    emboss: bool = True
    jitter_mm: float = 1.2

    @staticmethod
    def sample(rng: random.Random, ev: bool = False) -> "PlateStyle":
        paper = rng.choice([(244, 244, 240), (240, 240, 236), (235, 236, 232), (248, 247, 242), (228, 230, 226)])
        return PlateStyle(
            font=pick_font(rng), ev=ev, paper=paper, ink=rng.choice([(18, 18, 20), (10, 10, 12), (28, 28, 30), (35, 33, 32)]),
            green=rng.choice([(0, 128, 72), (0, 112, 60), (10, 140, 80), (0, 150, 90)]),
            cap_mm=rng.uniform(58, 68), cell_mm=rng.uniform(47, 54), gap_mm=rng.uniform(10, 20), hscale=rng.uniform(0.88, 1.08),
            bold_mm=rng.choice([0, 0, 0.5, 1.0, 1.5]), region_cap_mm=rng.uniform(38, 48), rim_mm=rng.uniform(2.5, 4.0),
            corner_mm=rng.uniform(6, 12), flag=rng.random() < 0.95, region_box=rng.random() < 0.9, emboss=rng.random() < 0.7,
        )


def _glyph(draw_img: Image.Image, ch: str, cx: float, cy_base: float, style: PlateStyle, cap_px: float, fill, hscale: float, bold_px: float):
    """Draw one character centred horizontally at cx with its baseline at cy_base (pixels). Horizontal scaling via an
    intermediate image so condensed/wide variants are possible."""
    # pick the font size so that the cap height ('H') == cap_px
    f0 = _font(style.font, 100)
    bb = f0.getbbox("H")
    cap100 = bb[3] - bb[1]
    size = max(8, int(round(100 * cap_px / cap100)))
    f = _font(style.font, size)
    l, t, r, b = f.getbbox(ch, anchor="ls")
    w = int(r - l + 2 * bold_px + 6)
    h = int(size * 1.3)
    tmp = Image.new("L", (max(w, 4), h), 0)
    d = ImageDraw.Draw(tmp)
    d.text((3 + bold_px - l, h * 0.8), ch, font=f, fill=255, anchor="ls", stroke_width=int(round(bold_px)), stroke_fill=255)
    if abs(hscale - 1.0) > 1e-3:
        tmp = tmp.resize((max(2, int(tmp.width * hscale)), tmp.height), Image.BILINEAR)
    # paste so that glyph centre (of its ink box) is at cx and baseline at cy_base
    bbox = tmp.getbbox() or (0, 0, tmp.width, tmp.height)
    gx = cx - (bbox[0] + bbox[2]) / 2
    gy = cy_base - h * 0.8
    mask = Image.new("L", draw_img.size, 0)
    mask.paste(tmp, (int(round(gx)), int(round(gy))))
    return mask


def _compose_mask(size, glyphs):
    m = Image.new("L", size, 0)
    for g in glyphs:
        m = Image.fromarray(np.maximum(np.asarray(m), np.asarray(g)))
    return m


def _flag(draw: ImageDraw.ImageDraw, x, y, w, h):
    """Tiny Uzbek flag: blue / white / green with thin red separators; the crescent/stars are too small to matter."""
    s = h / 3.0
    draw.rectangle([x, y, x + w, y + s], fill=(0, 154, 220))
    draw.rectangle([x, y + s, x + w, y + 2 * s], fill=(250, 250, 250))
    draw.rectangle([x, y + 2 * s, x + w, y + h], fill=(0, 150, 70))
    draw.line([x, y + s, x + w, y + s], fill=(210, 40, 40), width=max(1, int(PX * 0.5)))
    draw.line([x, y + 2 * s, x + w, y + 2 * s], fill=(210, 40, 40), width=max(1, int(PX * 0.5)))
    draw.ellipse([x + w * 0.12, y + s * 0.2, x + w * 0.12 + s * 0.6, y + s * 0.8], fill=(250, 250, 250))


def render_car_plate(text: str, style: PlateStyle, rng: random.Random | None = None) -> Image.Image:
    """520x110 mm plate for private / legal-entity layouts (text = canonical region+series string). Returns RGB."""
    rng = rng or random.Random(0)
    p = uz.parse(text)
    if p.format not in (uz.UZ_PRIVATE, uz.UZ_LEGAL):
        raise ValueError(f"not a car plate: {text}")
    Wm, Hm = 520.0, 110.0
    W, H = int(Wm * PX), int(Hm * PX)
    img = Image.new("RGB", (W, H), style.paper)
    d = ImageDraw.Draw(img)
    # rim
    rim = style.rim_mm * PX
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=style.corner_mm * PX, outline=style.ink, width=int(rim * 1.1))
    # region field
    rx0, rx1 = 11 * PX, 86 * PX
    ry0, ry1 = 13 * PX, H - 13 * PX
    region = text[:2] if text[:2].isdigit() else p.region
    if style.ev:
        d.rectangle([rx0, ry0, rx1, ry1], fill=style.green)
        reg_fill = (245, 245, 245)
    else:
        reg_fill = style.ink
    if style.region_box:
        d.rectangle([rx0, ry0, rx1, ry1], outline=style.ink, width=max(1, int(1.8 * PX)))
    glyphs_ink, glyphs_reg = [], []
    base_reg = (H + style.region_cap_mm * PX) / 2.0 + rng.uniform(-1, 1) * PX
    for i, ch in enumerate(region):
        cx = rx0 + (rx1 - rx0) * (0.28 + 0.44 * i) + rng.uniform(-1, 1) * style.jitter_mm * PX
        glyphs_reg.append(_glyph(img, ch, cx, base_reg, style, style.region_cap_mm * PX, reg_fill, style.hscale * 0.95, style.bold_mm * PX * 0.7))
    # main field: 6 slots
    chars = text[2:]
    if p.format == uz.UZ_PRIVATE:
        groups = [1, 3, 2]
    else:
        groups = [3, 3]
    total = len(chars) * style.cell_mm + (len(groups) - 1) * style.gap_mm
    x_start = 95 + ((455 - 95) - total) / 2.0
    x = x_start
    k = 0
    base = (H + style.cap_mm * PX) / 2.0 + rng.uniform(-1, 1) * PX
    for gi, g in enumerate(groups):
        for j in range(g):
            cx = (x + style.cell_mm / 2.0) * PX + rng.uniform(-1, 1) * style.jitter_mm * PX
            glyphs_ink.append(_glyph(img, chars[k], cx, base, style, style.cap_mm * PX, style.ink, style.hscale, style.bold_mm * PX))
            x += style.cell_mm
            k += 1
        x += style.gap_mm
    ink_mask = _compose_mask((W, H), glyphs_ink)
    ink_arr = np.asarray(ink_mask, np.float32)[..., None] / 255.0
    arr = np.asarray(img, np.float32)
    arr = arr * (1 - ink_arr) + np.array(style.ink, np.float32) * ink_arr
    reg_mask = np.asarray(_compose_mask((W, H), glyphs_reg), np.float32)[..., None] / 255.0
    arr = arr * (1 - reg_mask) + np.array(reg_fill, np.float32) * reg_mask
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    # flag + UZ block
    if style.flag:
        fx, fy, fw, fh = 466 * PX, 20 * PX, 30 * PX, 20 * PX
        _flag(d, fx, fy, fw, fh)
        f = _font("Barlow-Bold.ttf", int(21 * PX))
        d.text((481 * PX, 70 * PX), "UZ", font=f, fill=(20, 40, 120), anchor="mm")
    out = np.asarray(img, np.float32)
    if style.emboss:  # subtle embossing: darker rim shadow around ink, lighter top-left edge
        g = (np.asarray(ink_mask, np.float32) / 255.0)
        import cv2
        sh = cv2.GaussianBlur(g, (0, 0), 1.2 * PX)
        out *= (1 - 0.10 * np.clip(sh - g, 0, 1))[..., None]
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def render_moto_plate(text: str, style: PlateStyle, rng: random.Random | None = None) -> Image.Image:
    """Square two-row motorcycle plate: top row 3 digits (+ small region at the upper right), bottom row 2 letters."""
    rng = rng or random.Random(0)
    p = uz.parse(text)
    if p.format != uz.UZ_MOTO:
        raise ValueError(text)
    Wm, Hm = 190.0, 150.0
    W, H = int(Wm * PX), int(Hm * PX)
    img = Image.new("RGB", (W, H), style.paper)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=style.corner_mm * PX, outline=style.ink, width=int(style.rim_mm * PX * 1.1))
    region, digits, letters = text[:2], text[2:5], text[5:]
    glyphs, reg_glyphs = [], []
    cap = 52 * PX
    cell = 40 * PX
    base1 = 66 * PX
    for i, ch in enumerate(digits):
        glyphs.append(_glyph(img, ch, (18 * PX + cell * (i + 0.5)), base1, style, cap, style.ink, style.hscale, style.bold_mm * PX))
    base2 = 130 * PX
    for i, ch in enumerate(letters):
        glyphs.append(_glyph(img, ch, (38 * PX + cell * (i + 0.5)), base2, style, cap, style.ink, style.hscale, style.bold_mm * PX))
    # region at upper right
    rx0, rx1, ry0, ry1 = 140 * PX, 182 * PX, 10 * PX, 38 * PX
    if style.ev:
        d.rectangle([rx0, ry0, rx1, ry1], fill=style.green)
        rf = (245, 245, 245)
    else:
        rf = style.ink
    d.rectangle([rx0, ry0, rx1, ry1], outline=style.ink, width=max(1, int(1.5 * PX)))
    for i, ch in enumerate(region):
        reg_glyphs.append(_glyph(img, ch, rx0 + (rx1 - rx0) * (0.28 + 0.44 * i), 33 * PX, style, 22 * PX, rf, 0.95, 0))
    m = np.asarray(_compose_mask((W, H), glyphs), np.float32)[..., None] / 255.0
    arr = np.asarray(img, np.float32) * (1 - m) + np.array(style.ink, np.float32) * m
    rm = np.asarray(_compose_mask((W, H), reg_glyphs), np.float32)[..., None] / 255.0
    arr = arr * (1 - rm) + np.array(rf, np.float32) * rm
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    if style.flag:
        _flag(d, 150 * PX, 110 * PX, 26 * PX, 17 * PX)
        d.text((163 * PX, 140 * PX), "UZ", font=_font("Barlow-Bold.ttf", int(16 * PX)), fill=(20, 40, 120), anchor="mm")
    return img


def render_plate(text: str, rng: random.Random, fmt: str | None = None, style: PlateStyle | None = None) -> tuple[Image.Image, str, PlateStyle]:
    """Render ``text``; ``fmt`` in {uz_private, uz_legal, uz_ev, uz_moto}; returns (RGB image, format, style)."""
    p = uz.parse(text)
    fmt = fmt or p.format
    ev = fmt == uz.UZ_EV
    style = style or PlateStyle.sample(rng, ev=ev)
    style.ev = ev
    if p.format == uz.UZ_MOTO:
        return render_moto_plate(text, style, rng), (uz.UZ_EV if ev else uz.UZ_MOTO), style
    return render_car_plate(text, style, rng), (uz.UZ_EV if ev else p.format), style


if __name__ == "__main__":
    import sys
    rng = random.Random(1)
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "plate_samples.png")
    rows = []
    for fmt in (uz.UZ_PRIVATE, uz.UZ_PRIVATE, uz.UZ_LEGAL, uz.UZ_EV, uz.UZ_EV, uz.UZ_MOTO):
        t = uz.sample_text(rng, uz.UZ_PRIVATE if fmt == uz.UZ_EV else fmt)
        im, f, _ = render_plate(t, rng, fmt)
        rows.append((t, f, im))
    Wm = max(im.width for _, _, im in rows)
    sheet = Image.new("RGB", (Wm, sum(im.height + 8 for _, _, im in rows)), (90, 90, 90))
    y = 0
    for t, f, im in rows:
        sheet.paste(im, (0, y))
        y += im.height + 8
    sheet.save(out)
    print([(t, f) for t, f, _ in rows])
