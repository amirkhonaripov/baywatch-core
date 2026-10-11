"""Scene constants of sim/blender/render_bay.py and a pure-numpy replica of its camera (no bpy needed).

Used by sim/eval_synthetic.py to project each bay's floor rectangle into the image (zone polygons).
tests/test_scenario.py checks that the constants match render_bay.py.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

BAY_X = {"bay1": -2.75, "bay2": 2.75}
CAMERAS = {  # location, look-at, horizontal FOV (deg)
    "cam_A": ((-5.2, -8.4, 4.6), (0.8, 1.2, 0.9), 78.0),
    "cam_B": ((5.4, -8.2, 4.4), (-0.8, 1.2, 0.9), 76.0),
    "cam_C": ((0.3, -9.2, 5.0), (0.0, 1.0, 0.9), 72.0),
}
BAY_HALF_WIDTH = 2.3
BAY_Y = (-3.6, 3.8)


def h01(*parts) -> float:
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) / 2 ** 32


def camera_pose(episode_id: str, camera: str):
    loc, look, fov = CAMERAS[camera]
    jit = np.array([(h01(episode_id, "cam", i) - 0.5) * 0.3 for i in range(3)])
    return np.array(loc) + jit, np.array(look, dtype=float), math.radians(fov)


def project(points: np.ndarray, episode_id: str, camera: str, resolution=(1280, 720)) -> np.ndarray:
    """World points (N,3) -> normalised image coords (N,2), x right, y down (Blender track-to -Z, up Y)."""
    cam, look, fov = camera_pose(episode_id, camera)
    fwd = look - cam
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    d = np.asarray(points, dtype=float) - cam
    xc, yc, zc = d @ right, d @ up, d @ fwd
    t = math.tan(fov / 2)
    aspect = resolution[1] / resolution[0]
    u = 0.5 + 0.5 * (xc / zc) / t
    v = 0.5 - 0.5 * (yc / zc) / (t * aspect)
    return np.stack([u, v], 1)


def bay_floor_polygon(episode_id: str, camera: str, bay: str) -> np.ndarray:
    c = BAY_X[bay]
    pts = np.array([[c - BAY_HALF_WIDTH, BAY_Y[0], 0], [c + BAY_HALF_WIDTH, BAY_Y[0], 0],
                    [c + BAY_HALF_WIDTH, BAY_Y[1], 0], [c - BAY_HALF_WIDTH, BAY_Y[1], 0]])
    return project(pts, episode_id, camera)


def bay_zone_polygon(episode_id: str, camera: str, bay: str, z_top: float = 2.0) -> np.ndarray:
    """Zone polygon for 'a car is in this bay': convex hull of the bay footprint projected at floor level and at
    lifted-car height, so the bottom-centre of a raised car's box still falls inside."""
    c = BAY_X[bay]
    pts = np.array([[c + dx, y, z] for dx in (-BAY_HALF_WIDTH, BAY_HALF_WIDTH) for y in BAY_Y for z in (0.0, z_top)])
    uv = project(pts, episode_id, camera)
    return _hull(uv)


def _hull(pts: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull (counter-clockwise)."""
    P = sorted(map(tuple, pts))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for q in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1])


def bay_roi(episode_id: str, camera: str, bay: str) -> list[float]:
    """Same box as render_bay.py writes to rois.json (projected bay volume, clipped to the image)."""
    c = BAY_X[bay]
    pts = np.array([[c + dx, y, z] for dx in (-BAY_HALF_WIDTH, BAY_HALF_WIDTH) for y in (-3.6, 3.8) for z in (0.0, 3.3)])
    uv = project(pts, episode_id, camera)
    return [max(0.0, uv[:, 0].min()), max(0.0, uv[:, 1].min()), min(1.0, uv[:, 0].max()), min(1.0, uv[:, 1].max())]
