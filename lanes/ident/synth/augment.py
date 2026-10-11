"""Degradations that turn a clean rendered plate into a crop that looks like it came from a camera.

Pipeline (``degrade``):  plate on a bumper-like background  ->  perspective / roll / yaw  ->  resolution loss
-> scenario (day / dusk / night / night-IR / overexposed) -> glare, dirt, rain streaks, partial occlusion, shadows
-> blur (gaussian / motion / defocus) -> sensor noise -> JPEG.
All randomness comes from the ``random.Random`` / ``np.random.Generator`` passed in, so every sample is reproducible
from its seed. Everything is plain OpenCV/numpy (fast enough for ~50 crops/s/core).
"""
from __future__ import annotations

import math
import random

import cv2
import numpy as np

SCENARIOS = ("day", "day", "day", "dusk", "night", "night_ir", "overexposed", "overcast")


# ---------------------------------------------------------------------------------------------------- helpers
def _gauss(rng: random.Random, mu, sd, lo=None, hi=None):
    v = rng.gauss(mu, sd)
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def _background(h: int, w: int, rng: random.Random, nprng: np.random.Generator) -> np.ndarray:
    """Bumper / car-body texture: a base colour with a gradient and fine noise."""
    kind = rng.random()
    if kind < 0.35:
        base = np.array([rng.randint(15, 60)] * 3) + np.array([rng.randint(-6, 6) for _ in range(3)])  # black / dark grey
    elif kind < 0.6:
        v = rng.randint(150, 235)
        base = np.array([v, v, v]) + np.array([rng.randint(-8, 8) for _ in range(3)])  # white / silver
    else:  # coloured paint, mostly desaturated (real bumpers are rarely neon)
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.05, 0.55), rng.uniform(0.15, 0.85))
        base = np.array([int(r * 255), int(g * 255), int(b * 255)])
    ang = rng.uniform(0, 2 * math.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = (np.cos(ang) * xx / max(w, 1) + np.sin(ang) * yy / max(h, 1))
    bg = base[None, None, :] + rng.uniform(-35, 35) * grad[..., None]
    bg += nprng.normal(0, rng.uniform(1.5, 7), (h, w, 1))
    return bg.clip(0, 255).astype(np.uint8)


def _perlin_like(h: int, w: int, nprng: np.random.Generator, scales=(4, 8, 16)) -> np.ndarray:
    out = np.zeros((h, w), np.float32)
    for k, s in enumerate(scales):
        small = nprng.random((max(2, h // s), max(2, w // s))).astype(np.float32)
        out += cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * (0.5 ** k)
    out -= out.min()
    return out / max(out.max(), 1e-6)


def _ellipse_glow(h, w, cx, cy, rx, ry, ang):
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (int(cx), int(cy)), (max(1, int(rx)), max(1, int(ry))), ang, 0, 360, 1.0, -1)
    return m


# ------------------------------------------------------------------------------------------------- geometry
def place_with_perspective(plate: np.ndarray, rng: random.Random, nprng: np.random.Generator, out_w: int,
                           margin=(0.0, 0.25), strength: float = 1.0):
    """Paste ``plate`` (HxWx3 RGB) on a background and warp. Returns (crop, quad_in_crop 4x2 TL,TR,BR,BL, plate_mask)."""
    ph, pw = plate.shape[:2]
    # random plate-frame / bumper margin around the plate (fraction of plate height)
    mt, mb = rng.uniform(*margin), rng.uniform(*margin)
    ml, mr = rng.uniform(*margin) * 0.6, rng.uniform(*margin) * 0.6
    top, bot, lef, rig = int(ph * mt), int(ph * mb), int(ph * ml), int(ph * mr)
    H, W = ph + top + bot, pw + lef + rig
    bg = _background(H, W, rng, nprng)
    canvas = bg.copy()
    canvas[top:top + ph, lef:lef + pw] = plate
    mask = np.zeros((H, W), np.uint8)
    mask[top:top + ph, lef:lef + pw] = 255
    # bolts / frame details
    if rng.random() < 0.25:
        for bx in (lef + int(pw * 0.03), lef + pw - int(pw * 0.03)):
            cv2.circle(canvas, (bx, top + ph // 2), max(2, ph // 22), (rng.randint(80, 150),) * 3, -1)
    src = np.float32([[lef, top], [lef + pw, top], [lef + pw, top + ph], [lef, top + ph]])
    # destination quad: roll, yaw (horizontal foreshortening + vertical skew), pitch (keystone)
    roll = math.radians(_gauss(rng, 0, 5 * strength, -14, 14))
    yaw = _gauss(rng, 0, 0.18 * strength, -0.45, 0.45)     # fractional left/right height difference
    pitch = _gauss(rng, 0, 0.08 * strength, -0.2, 0.2)     # fractional top/bottom width difference
    cx, cy = W / 2, H / 2
    pts = []
    for (x, y) in src:
        dx, dy = x - cx, y - cy
        sx = 1 + pitch * (dy / (H / 2))
        sy = 1 + yaw * (dx / (W / 2))
        x2, y2 = dx * sx, dy * sy
        pts.append((x2 * math.cos(roll) - y2 * math.sin(roll), x2 * math.sin(roll) + y2 * math.cos(roll)))
    pts = np.float32(pts)
    allc = np.float32([[0, 0], [W, 0], [W, H], [0, H]]) - [cx, cy]
    allp = []
    for (dx, dy) in allc:
        sx = 1 + pitch * (dy / (H / 2))
        sy = 1 + yaw * (dx / (W / 2))
        x2, y2 = dx * sx, dy * sy
        allp.append((x2 * math.cos(roll) - y2 * math.sin(roll), x2 * math.sin(roll) + y2 * math.cos(roll)))
    allp = np.float32(allp)
    mn, mx = allp.min(0), allp.max(0)
    scale = out_w / max(mx[0] - mn[0], 1)
    dst = (pts - mn) * scale
    outsize = (max(8, int(round((mx[0] - mn[0]) * scale))), max(8, int(round((mx[1] - mn[1]) * scale))))
    # pre-shrink with area filtering so the warp does not alias
    shrink = max(1.0, (W / outsize[0]) / 1.5)
    if shrink > 1.0:
        sw, sh = int(W / shrink), int(H / shrink)
        canvas = cv2.resize(canvas, (sw, sh), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_AREA)
        src = src / shrink
    Mx = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
    out = cv2.warpPerspective(canvas, Mx, outsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    pmask = cv2.warpPerspective(mask, Mx, outsize, flags=cv2.INTER_LINEAR, borderValue=0)
    return out, dst, pmask


# ------------------------------------------------------------------------------------------- photometric
def apply_scenario(img: np.ndarray, scen: str, rng: random.Random, nprng: np.random.Generator) -> np.ndarray:
    f = img.astype(np.float32)
    h, w = f.shape[:2]
    if scen == "day":
        f = f * rng.uniform(0.75, 1.1) + rng.uniform(-10, 10)
    elif scen == "overcast":
        m = f.mean()
        f = (f - m) * rng.uniform(0.6, 0.9) + m * rng.uniform(0.8, 1.0)
    elif scen == "dusk":
        f = f * rng.uniform(0.35, 0.65)
        f[..., 2] *= rng.uniform(1.0, 1.15)
        f[..., 0] *= rng.uniform(0.85, 1.0)
    elif scen == "night":
        g = rng.uniform(0.08, 0.35)
        f = (f / 255.0) ** rng.uniform(1.2, 1.8) * 255.0 * g * 2.0
        f[..., 2] *= rng.uniform(1.0, 1.25)   # cool cast
        f += nprng.normal(0, rng.uniform(2, 9), f.shape)
        # headlight/streetlight bloom
        if rng.random() < 0.6:
            bloom = _ellipse_glow(h, w, rng.uniform(0, w), rng.uniform(0, h), rng.uniform(w * 0.2, w * 0.7), rng.uniform(h * 0.3, h), rng.uniform(0, 180))
            bloom = cv2.GaussianBlur(bloom, (0, 0), max(2, w * 0.08))
            f += bloom[..., None] * rng.uniform(40, 160) * np.array([1.0, 0.95, 0.8])[None, None, :]
    elif scen == "night_ir":
        g = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        g = np.clip((g - g.mean()) * rng.uniform(1.0, 1.8) + rng.uniform(90, 150), 0, 255)
        # IR illuminators: centre hot spot, retro-reflective plate bright
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        hot = np.exp(-(((xx - w * rng.uniform(0.3, 0.7)) / (w * rng.uniform(0.4, 0.9))) ** 2 + ((yy - h * 0.5) / (h * 1.2)) ** 2))
        g = np.clip(g * (0.7 + 0.5 * hot), 0, 255)
        f = np.repeat(g[..., None], 3, axis=2)
        f += nprng.normal(0, rng.uniform(2, 8), f.shape)
    elif scen == "overexposed":
        f = f * rng.uniform(1.25, 1.9) + rng.uniform(5, 40)
        f = np.clip(f, 0, 255)
        f = (f - 128) * rng.uniform(0.8, 1.0) + 128
    return np.clip(f, 0, 255).astype(np.uint8)


def add_glare(img: np.ndarray, rng: random.Random, nprng: np.random.Generator) -> np.ndarray:
    f = img.astype(np.float32)
    h, w = f.shape[:2]
    n = rng.randint(1, 2)
    for _ in range(n):
        m = _ellipse_glow(h, w, rng.uniform(-0.1 * w, 1.1 * w), rng.uniform(-0.2 * h, 1.2 * h), rng.uniform(w * 0.08, w * 0.45), rng.uniform(h * 0.15, h * 0.9), rng.uniform(0, 180))
        m = cv2.GaussianBlur(m, (0, 0), rng.uniform(1.5, max(2.0, w * 0.06)))
        f = f + m[..., None] * rng.uniform(60, 230)
    return np.clip(f, 0, 255).astype(np.uint8)


def add_shadow(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, math.pi)
    d = (xx - rng.uniform(0.2, 0.8) * w) * math.cos(ang) + (yy - h * rng.uniform(0.2, 0.8)) * math.sin(ang)
    soft = rng.uniform(1.5, max(2.0, 0.08 * w))
    m = 1 / (1 + np.exp(-d / soft))
    return np.clip(img.astype(np.float32) * (1 - rng.uniform(0.25, 0.65) * m)[..., None], 0, 255).astype(np.uint8)


def add_dirt(img: np.ndarray, rng: random.Random, nprng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    n = _perlin_like(h, w, nprng, scales=(3, 6, 12))
    thr = rng.uniform(0.55, 0.8)
    m = np.clip((n - thr) / (1 - thr), 0, 1)
    # concentrate dirt at the bottom / sides (spray)
    yy = np.linspace(0, 1, h)[:, None]
    m = m * (0.35 + 0.65 * yy ** rng.uniform(0.5, 2.0))
    colour = np.array(rng.choice([(95, 80, 60), (70, 62, 50), (120, 110, 95), (40, 38, 35)]), np.float32)
    a = np.clip(m * rng.uniform(0.4, 0.95), 0, 0.85)[..., None]
    f = f * (1 - a) + colour * a
    # a few scratches
    for _ in range(rng.randint(0, 4)):
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        ang = rng.uniform(0, math.pi)
        L = rng.uniform(0.1, 0.5) * w
        cv2.line(f, (int(x0), int(y0)), (int(x0 + L * math.cos(ang)), int(y0 + L * math.sin(ang))), (rng.randint(110, 200),) * 3, 1, cv2.LINE_AA)
    return np.clip(f, 0, 255).astype(np.uint8)


def add_rain(img: np.ndarray, rng: random.Random, nprng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    layer = np.zeros((h, w), np.float32)
    n = int(rng.uniform(0.01, 0.06) * w)
    ang = rng.uniform(-25, 25)
    for _ in range(n):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        L = rng.uniform(0.15, 0.6) * h
        dx, dy = L * math.sin(math.radians(ang)), L * math.cos(math.radians(ang))
        cv2.line(layer, (int(x), int(y)), (int(x + dx), int(y + dy)), 1.0, 1, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), 0.7)
    f = img.astype(np.float32)
    f = f * (1 - 0.25 * layer[..., None]) + 255 * 0.45 * layer[..., None]
    # water droplets: small blurred bright/dark blobs
    for _ in range(rng.randint(0, 12)):
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.01, 0.05) * w
        drop = np.zeros((h, w), np.float32)
        cv2.circle(drop, (int(cx), int(cy)), max(1, int(r)), 1.0, -1)
        drop = cv2.GaussianBlur(drop, (0, 0), max(0.6, r * 0.35))
        f = f * (1 - 0.18 * drop[..., None]) + 255 * 0.22 * np.roll(drop, 1, 0)[..., None]
    if rng.random() < 0.5:  # wet low-contrast haze
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.4, 1.0)) * 0.92 + 20
    return np.clip(f, 0, 255).astype(np.uint8)


def add_occlusion(img: np.ndarray, rng: random.Random, plate_mask: np.ndarray | None = None) -> np.ndarray:
    """Partial occlusion: bars (tow hook / frame / wiper), blobs (snow, tape, finger). Kept thin so the label stays readable."""
    h, w = img.shape[:2]
    out = img.copy()
    kind = rng.choice(["bar_v", "bar_h", "blob", "corner", "edge"])
    col = tuple(rng.randint(10, 230) for _ in range(3))
    if kind == "bar_v":
        x = rng.randint(0, w - 1)
        bw = max(1, int(w * rng.uniform(0.008, 0.03)))
        out[:, x:x + bw] = col
    elif kind == "bar_h":
        y = rng.randint(0, h - 1)
        bh = max(1, int(h * rng.uniform(0.03, 0.10)))
        out[y:y + bh, :] = col
    elif kind == "blob":
        cv2.ellipse(out, (rng.randint(0, w), rng.randint(0, h)), (max(1, int(w * rng.uniform(0.02, 0.07))), max(1, int(h * rng.uniform(0.08, 0.25)))), rng.uniform(0, 180), 0, 360, col, -1, cv2.LINE_AA)
    elif kind == "corner":
        pts = np.array([[0, 0], [int(w * rng.uniform(0.03, 0.1)), 0], [0, int(h * rng.uniform(0.15, 0.4))]])
        if rng.random() < 0.5:
            pts[:, 0] = w - 1 - pts[:, 0]
        cv2.fillPoly(out, [pts.astype(np.int32)], col)
    else:  # lower edge strip (snow / mud line)
        y = int(h * rng.uniform(0.86, 0.97))
        out[y:, :] = col
    return out


def blur(img: np.ndarray, rng: random.Random, strength: float = 1.0) -> np.ndarray:
    k = rng.random()
    if k < 0.45:
        s = _gauss(rng, 0.7 * strength, 0.7 * strength, 0.0, 2.8)
        return cv2.GaussianBlur(img, (0, 0), s) if s > 0.15 else img
    if k < 0.8:  # motion blur
        L = rng.choice([3, 5, 7, 9, 11, 15])
        if rng.random() < 0.5:
            L = max(3, int(L * strength))
        ker = np.zeros((L, L), np.float32)
        cv2.line(ker, (0, L // 2), (L - 1, L // 2), 1.0, 1)
        ang = rng.uniform(-25, 25) if rng.random() < 0.8 else rng.uniform(60, 120)
        M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), ang, 1.0)
        ker = cv2.warpAffine(ker, M, (L, L))
        ker /= max(ker.sum(), 1e-6)
        return cv2.filter2D(img, -1, ker)
    r = rng.choice([2, 3, 4])  # defocus
    ker = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    cv2.circle(ker, (r, r), r, 1.0, -1)
    ker /= ker.sum()
    return cv2.filter2D(img, -1, ker)


def add_noise(img: np.ndarray, rng: random.Random, nprng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    f = img.astype(np.float32)
    sd = rng.uniform(0, 7) * scale
    if sd > 0.5:
        f += nprng.normal(0, sd, f.shape) if rng.random() < 0.6 else nprng.normal(0, sd, f.shape[:2])[..., None]
    if rng.random() < 0.15:  # poisson-ish
        f = nprng.poisson(np.clip(f, 0, 255) / 4.0).astype(np.float32) * 4.0
    return np.clip(f, 0, 255).astype(np.uint8)


def jpeg(img: np.ndarray, rng: random.Random) -> np.ndarray:
    q = int(_gauss(rng, 65, 22, 18, 98))
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def colour_cast(img: np.ndarray, rng: random.Random) -> np.ndarray:
    f = img.astype(np.float32)
    f *= np.array([rng.uniform(0.9, 1.1), rng.uniform(0.92, 1.08), rng.uniform(0.88, 1.12)])[None, None, :]
    if rng.random() < 0.15:  # desaturate
        g = f.mean(-1, keepdims=True)
        f = g + (f - g) * rng.uniform(0.2, 0.7)
    return np.clip(f, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------------------------- the pipeline
def degrade(plate_rgb: np.ndarray, seed: int, out_w: int | None = None, intensity: float = 1.0) -> tuple[np.ndarray, dict]:
    """plate_rgb: clean rendered plate (HxWx3 uint8). Returns (degraded crop RGB uint8, parameter dict)."""
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    info: dict = {}
    out_w = out_w or int(math.exp(rng.uniform(math.log(64), math.log(400))))
    info["out_w"] = out_w
    img, quad, pmask = place_with_perspective(plate_rgb, rng, nprng, out_w, margin=(0.0, 0.22 * intensity), strength=intensity)
    scen = rng.choice(SCENARIOS)
    info["scenario"] = scen
    img = apply_scenario(img, scen, rng, nprng)
    ops = []
    if rng.random() < 0.28 * intensity:
        img = add_shadow(img, rng); ops.append("shadow")
    if rng.random() < 0.22 * intensity:
        img = add_glare(img, rng, nprng); ops.append("glare")
    if rng.random() < 0.20 * intensity:
        img = add_dirt(img, rng, nprng); ops.append("dirt")
    if rng.random() < 0.12 * intensity:
        img = add_rain(img, rng, nprng); ops.append("rain")
    if rng.random() < 0.12 * intensity:
        img = add_occlusion(img, rng, pmask); ops.append("occlusion")
    if rng.random() < 0.5:
        img = colour_cast(img, rng); ops.append("cast")
    if rng.random() < 0.7:
        img = blur(img, rng, strength=1.0 if out_w > 140 else 0.5 if out_w < 90 else 0.75); ops.append("blur")
    # down/up-scale blockiness for tiny crops
    if out_w < 90 and rng.random() < 0.5:
        s = rng.uniform(0.5, 0.85)
        sm = cv2.resize(img, (max(8, int(img.shape[1] * s)), max(8, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(sm, (img.shape[1], img.shape[0]), interpolation=rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST]))
        ops.append("rescale")
    img = add_noise(img, rng, nprng, scale=1.0 if scen not in ("night", "night_ir") else 1.4); ops.append("noise")
    if rng.random() < 0.8:
        img = jpeg(img, rng); ops.append("jpeg")
    # detector-box jitter: tight boxes may shave a few percent off the plate edges; Roboflow-style non-uniform stretch
    h, w = img.shape[:2]
    if rng.random() < 0.5:
        cx0, cx1 = int(w * rng.uniform(0, 0.03)), int(w * rng.uniform(0, 0.03))
        cy0, cy1 = int(h * rng.uniform(0, 0.04)), int(h * rng.uniform(0, 0.04))
        img = img[cy0:h - cy1, cx0:w - cx1]
        ops.append("boxjitter")
    if rng.random() < 0.3:
        sx = rng.uniform(0.75, 1.3)
        img = cv2.resize(img, (max(8, int(img.shape[1] * sx)), img.shape[0]), interpolation=cv2.INTER_LINEAR)
        ops.append("stretch")
    info["ops"] = ops
    return img, info
