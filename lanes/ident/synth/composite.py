"""Composite rendered Uzbek plates onto real vehicle photos in place of the original plate (homography).

Declared method (docs/lanes/ident_notes.md): the original plate quad (given as 4 corners in the dataset, or an axis-aligned
detector/GT box) is replaced as follows
  1. the unit rectangle -> quad homography H of the old plate is estimated; the NEW plate occupies a horizontally stretched
     rectangle in the same plane (Uzbek plates are 520x110 mm = 4.73:1, wider than BR/CN plates), so perspective is preserved;
  2. the old quad is filled by inpainting (Telea) so no old characters peek out;
  3. a clean plate (render.py) is warped with H' and alpha-blended with an anti-aliased mask;
  4. photometric matching to the replaced region: white/black level, colour tint, blur (searched over a small sigma grid by
     Laplacian-variance ratio) and sensor-noise level, so the pasted plate is not unrealistically crisp;
  5. the label is the NEW quad's bounding box + the invented plate text.
Only the replaced plate is labelled. Synthetic-composite data is always reported as synthetic.
"""
from __future__ import annotations

import math
import random

import cv2
import numpy as np

from lanes.ident import uz_format as uz
from lanes.ident.synth.render import render_plate

CAR_ASPECT = 520.0 / 110.0


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Return corners as TL, TR, BR, BL for 4 arbitrary points."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(1)
    d = np.diff(pts, axis=1)[:, 0]
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(d)], pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], np.float32)


def box_to_quad(b, rotate90: int = 0) -> np.ndarray:
    x1, y1, x2, y2 = b
    q = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    return np.roll(q, -rotate90, axis=0) if rotate90 else q


def _luma(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _lapvar(y, mask):
    lap = cv2.Laplacian(y, cv2.CV_32F, ksize=3)
    v = lap[mask > 0]
    return float(v.var()) if v.size > 10 else 0.0


def paste_plate(img: np.ndarray, quad: np.ndarray, text: str, fmt: str, rng: random.Random, match: bool = True,
                match_blur: bool = True, scale_h: tuple[float, float] = (0.95, 1.15), moto: bool = False):
    """img RGB uint8; quad TL,TR,BR,BL of the plate being replaced. Returns (new_img, new_quad, info)."""
    H_img, W_img = img.shape[:2]
    quad = order_quad(quad)
    w0 = float(np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    h0 = float(np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    if w0 < 6 or h0 < 3:
        return None
    unit = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    Hq = cv2.getPerspectiveTransform(unit, quad)
    a_target = (190.0 / 150.0) if moto else CAR_ASPECT
    a0 = w0 / h0
    k = a_target / a0 * rng.uniform(*scale_h) if not moto else rng.uniform(0.9, 1.1)
    s_h = rng.uniform(*scale_h)
    # new rectangle in the unit plane, centred on the old one (width stretched by k, height by s_h)
    nx0, nx1 = 0.5 - k / 2, 0.5 + k / 2
    ny0, ny1 = 0.5 - s_h / 2, 0.5 + s_h / 2
    new_unit = np.float32([[nx0, ny0], [nx1, ny0], [nx1, ny1], [nx0, ny1]])
    new_quad = cv2.perspectiveTransform(new_unit[None], Hq)[0]
    if (new_quad[:, 0].min() < -0.15 * W_img or new_quad[:, 0].max() > 1.15 * W_img or new_quad[:, 1].min() < -0.15 * H_img
            or new_quad[:, 1].max() > 1.15 * H_img):
        return None
    out = img.copy()
    # (2) inpaint the old plate quad
    m_old = np.zeros((H_img, W_img), np.uint8)
    cv2.fillConvexPoly(m_old, np.round(quad).astype(np.int32), 255)
    m_old_d = cv2.dilate(m_old, np.ones((3, 3), np.uint8), iterations=2)
    out = cv2.inpaint(cv2.cvtColor(out, cv2.COLOR_RGB2BGR), m_old_d, 3, cv2.INPAINT_TELEA)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    # (3) render + warp
    plate, f, style = render_plate(text, rng, fmt)
    pl = np.asarray(plate)
    ph, pw = pl.shape[:2]
    tw = int(np.clip(max(np.linalg.norm(new_quad[1] - new_quad[0]), np.linalg.norm(new_quad[2] - new_quad[3])) * 1.5, 96, pw))
    th = max(8, int(round(tw * ph / pw)))
    pl = cv2.resize(pl, (tw, th), interpolation=cv2.INTER_AREA)
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    Mx = cv2.getPerspectiveTransform(src, np.float32(new_quad))
    warped = cv2.warpPerspective(pl, Mx, (W_img, H_img), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = np.zeros((H_img, W_img), np.float32)
    ss = 3  # supersampled mask for anti-aliased edges
    big = np.zeros((H_img * ss, W_img * ss), np.uint8)
    cv2.fillConvexPoly(big, np.round(new_quad * ss).astype(np.int32), 255)
    mask = cv2.resize(big, (W_img, H_img), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    info: dict = {"format": f}
    wf = warped.astype(np.float32)
    # (4) photometric matching
    if match:
        y_old = _luma(img)
        reg = y_old[m_old > 0]
        if reg.size > 20:
            lw, lb = float(np.percentile(reg, 92)), float(np.percentile(reg, 6))
            lw = max(lw, lb + 25.0)
            wf = lb + (wf / 255.0) * (lw - lb)
            # tint: match mean chroma of the old region
            mean_old = img[m_old > 0].astype(np.float32).mean(0)
            g = mean_old / max(mean_old.mean(), 1e-3)
            wf = wf * (0.6 + 0.4 * g)[None, None, :]
            info["white"], info["black"] = lw, lb
        sig = 0.0
        if match_blur and reg.size > 50:
            target = _lapvar(y_old, m_old)
            best = (1e18, 0.0)
            for s in (0.0, 0.6, 1.0, 1.5, 2.2, 3.0):
                t = cv2.GaussianBlur(wf, (0, 0), s) if s > 0 else wf
                v = _lapvar(_luma(np.clip(t, 0, 255).astype(np.uint8)), (mask > 0.5).astype(np.uint8))
                err = abs(math.log((v + 1) / (target + 1)))
                if err < best[0]:
                    best = (err, s)
            sig = best[1]
        else:
            sig = rng.choice([0.0, 0.6, 1.0, 1.5])
        if sig > 0:
            wf = cv2.GaussianBlur(wf, (0, 0), sig)
        info["blur_sigma"] = sig
        hp = y_old - cv2.GaussianBlur(y_old, (0, 0), 1.5)
        nstd = float(hp[m_old > 0].std()) if reg.size > 20 else 2.0
        nstd = min(nstd, 8.0) * 0.5
        wf = wf + np.random.default_rng(rng.randint(0, 2**31)).normal(0, nstd, wf.shape)
        info["noise_std"] = nstd
    else:
        sig = rng.choice([0.0, 0.8, 1.5])
        if sig:
            wf = cv2.GaussianBlur(wf, (0, 0), sig)
    a = mask[..., None]
    res = out.astype(np.float32) * (1 - a) + np.clip(wf, 0, 255) * a
    res = np.clip(res, 0, 255).astype(np.uint8)
    xs, ys = new_quad[:, 0], new_quad[:, 1]
    box = np.array([max(0, xs.min()), max(0, ys.min()), min(W_img, xs.max()), min(H_img, ys.max())], np.float32)
    return res, new_quad, box, info


def random_text(rng: random.Random) -> tuple[str, str]:
    r = rng.random()
    if r < 0.60:
        return uz.sample_text(rng, uz.UZ_PRIVATE), uz.UZ_PRIVATE
    if r < 0.82:
        return uz.sample_text(rng, uz.UZ_LEGAL), uz.UZ_LEGAL
    if r < 0.93:
        return uz.sample_text(rng, uz.UZ_PRIVATE if rng.random() < 0.5 else uz.UZ_LEGAL), uz.UZ_EV
    return uz.sample_text(rng, uz.UZ_MOTO), uz.UZ_MOTO
