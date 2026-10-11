"""Detector-box geometry for bay states (pure numpy; shared by training and the predictor).

Boxes are normalised [x1, y1, x2, y2]. The "primary" vehicle is the one the bay states refer to: with a bay ROI it is
the vehicle with the largest area inside the ROI, otherwise the largest (area x confidence) vehicle.
"""
from __future__ import annotations

import numpy as np

VEH = ("car", "truck", "bus")
GEOM_NAMES = [
    "has_car", "car_conf", "car_area", "car_aspect", "car_cx", "car_cy", "car_y2", "n_vehicles",
    "n_person", "n_person_near", "max_person_ioa", "min_person_gap",
    "near_feet_below", "near_top_above_bottom", "near_lying", "near_center_in_x", "near_h_rel",
    "under_geom", "at_geom", "n_wheels", "wheel_low_rel", "wheel_conf_max", "hood_open_conf", "hood_closed_conf",
]


def area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def primary_vehicle(vehicles: list[tuple[float, tuple]], roi=None):
    """vehicles: [(conf, box)]. Returns (conf, box) or None."""
    if not vehicles:
        return None
    if roi is not None:
        # vehicle with the largest area inside the bay; at least 30 % of it must be inside
        cands = [(inter(b, roi) * (0.5 + 0.5 * c), c, b) for c, b in vehicles if inter(b, roi) >= 0.3 * area(b)]
        if not cands:
            return None
        _, c, b = max(cands, key=lambda s: s[0])
        return c, b
    return max(vehicles, key=lambda v: area(v[1]) * v[0])


def crop_box(box, margin: float = 0.25, img_aspect: float = 16 / 9):
    """Expand a vehicle box by ``margin`` of its size on every side and make it square in pixels."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, x2 = x1 - margin * w, x2 + margin * w
    y1, y2 = y1 - margin * h, y2 + margin * h
    # square in pixel space: width in px = w*img_aspect (relative to height units)
    wp, hp = (x2 - x1) * img_aspect, (y2 - y1)
    if wp > hp:
        d = (wp - hp) / 2
        y1, y2 = y1 - d, y2 + d
    else:
        d = (hp - wp) / 2 / img_aspect
        x1, x2 = x1 - d, x2 + d
    return (max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2))


def geom_features(dets: list[tuple[str, float, tuple]], roi=None, parts: list[tuple[str, float, tuple]] | None = None) -> np.ndarray:
    """dets: [(cls, conf, box)] from the vehicle/person detector; parts: optional [(cls, conf, box)] from the part
    detector (wheel / hood_open / hood_closed), in full-frame normalised coordinates."""
    f = dict.fromkeys(GEOM_NAMES, 0.0)
    f["min_person_gap"] = 1.0
    vehicles = [(c, b) for k, c, b in dets if k in VEH]
    persons = [(c, b) for k, c, b in dets if k == "person"]
    f["n_vehicles"] = float(len(vehicles))
    f["n_person"] = float(len(persons))
    pv = primary_vehicle(vehicles, roi)
    if pv is None:
        return np.array([f[n] for n in GEOM_NAMES], np.float32)
    c, b = pv
    W, H = b[2] - b[0], b[3] - b[1]
    f.update(has_car=1.0, car_conf=c, car_area=area(b), car_aspect=W / max(H, 1e-6), car_cx=(b[0] + b[2]) / 2,
             car_cy=(b[1] + b[3]) / 2, car_y2=b[3])
    near = []
    for pc, p in persons:
        pw, ph = p[2] - p[0], p[3] - p[1]
        ioa = inter(p, b) / max(area(p), 1e-9)
        gx = max(0.0, max(b[0] - p[2], p[0] - b[2])) / max(W, 1e-6)
        gy = max(0.0, max(b[1] - p[3], p[1] - b[3])) / max(H, 1e-6)
        gap = max(gx, gy)
        f["max_person_ioa"] = max(f["max_person_ioa"], ioa)
        f["min_person_gap"] = min(f["min_person_gap"], gap)
        if gap < 0.3:
            near.append((pc, p, ioa, gap, pw, ph))
    f["n_person_near"] = float(len(near))
    if near:
        pc, p, ioa, gap, pw, ph = max(near, key=lambda n: n[2] - n[3])
        cx = (p[0] + p[2]) / 2
        f["near_feet_below"] = (p[3] - b[3]) / max(H, 1e-6)        # >0: feet lower than the car's lowest point
        f["near_top_above_bottom"] = (b[3] - p[1]) / max(H, 1e-6)  # >0: head above the car's lowest point
        f["near_lying"] = pw / max(ph, 1e-6)
        f["near_center_in_x"] = float(b[0] <= cx <= b[2])
        f["near_h_rel"] = ph / max(H, 1e-6)
        # under the car: centred under the body, head above the car's bottom edge, feet below it
        f["under_geom"] = float(f["near_center_in_x"] and f["near_top_above_bottom"] > 0.05 and f["near_feet_below"] > 0.05)
        f["at_geom"] = float(ioa > 0.05 or gap < 0.1)
    if parts:
        wheels = [(pc, p) for k, pc, p in parts if k == "wheel" and inter(p, b) / max(area(p), 1e-9) > 0.5]
        f["n_wheels"] = float(len(wheels))
        if wheels:
            f["wheel_conf_max"] = max(pc for pc, _ in wheels)
            f["wheel_low_rel"] = (b[3] - max(p[3] for _, p in wheels)) / max(H, 1e-6)
        for k, key in (("hood_open", "hood_open_conf"), ("hood_closed", "hood_closed_conf")):
            hs = [pc for kk, pc, p in parts if kk == k and inter(p, b) / max(area(p), 1e-9) > 0.3]
            f[key] = max(hs) if hs else 0.0
    return np.array([f[n] for n in GEOM_NAMES], np.float32)
