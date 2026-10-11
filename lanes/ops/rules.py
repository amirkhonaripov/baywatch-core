"""Post-processing shared by the predictor and the offline evaluation (vectorised over frames)."""
from __future__ import annotations

import numpy as np

CAR_STATES = ("raised", "wheel_off", "hood_open", "person_at_car", "person_under_car", "work", "washing")


def apply_rules(p: dict[str, np.ndarray], car_conf: np.ndarray) -> dict[str, np.ndarray]:
    """p: head probabilities per state (arrays); car_conf: detector confidence of the primary vehicle (0 = none).
    car_present = car_conf; no vehicle -> car states capped at 0.05; person_under_car gated by raised; idle derived."""
    out = {k: np.asarray(v, dtype=float).copy() for k, v in p.items()}
    car_conf = np.asarray(car_conf, dtype=float)
    out["car_present"] = car_conf
    none = car_conf <= 0
    for k in CAR_STATES:
        if k in out:
            out[k] = np.where(none, np.minimum(out[k], 0.05), out[k])
    if "person_under_car" in out and "raised" in out:
        out["person_under_car"] = out["person_under_car"] * np.minimum(1.0, out["raised"] / 0.5)
    out["idle"] = out["car_present"] * (1.0 - out.get("work", np.zeros_like(car_conf)))
    return out
