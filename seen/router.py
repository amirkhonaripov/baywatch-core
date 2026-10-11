"""Scene router: decides which modules run on an input and explains what was skipped.

Lane B fills ``FramePred.scene`` with a SigLIP2 zero-shot + linear-probe predictor (``lanes/flow/predict.py::get_router``); with no
scene probabilities the detection-only ``heuristic_scene`` is used. ``decide`` stays the single place that turns probabilities into a
Route: low confidence -> scene "unknown", and every module skipped because of that says why (top candidate, probability, threshold).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from seen.contracts.events import Route, SkippedModule
from seen.predictor import FramePred
from seen.vocab import PERSON, VEHICLE_CLASSES

UNKNOWN_THRESHOLD = 0.45  # default; the calibrated value is read from the router model card when present


def unknown_threshold() -> float:
    """Calibrated rejection threshold: $SEEN_ROUTER_UNKNOWN, else artifacts/flow/router/probe.json["unknown_threshold"], else default."""
    if os.environ.get("SEEN_ROUTER_UNKNOWN"):
        return float(os.environ["SEEN_ROUTER_UNKNOWN"])
    cards = [Path(__file__).resolve().parents[1] / "lanes" / "flow" / "assets" / "router_probe.json"]
    try:
        from seen.paths import ARTIFACTS

        cards.insert(0, Path(ARTIFACTS) / "flow" / "router" / "probe.json")
    except Exception:  # noqa: BLE001
        pass
    for card in cards:
        try:
            if card.exists():
                v = json.loads(card.read_text()).get("unknown_threshold")
                if v is not None:
                    return float(v)
        except Exception:  # noqa: BLE001
            continue
    return UNKNOWN_THRESHOLD


HEURISTIC_UNKNOWN_CONF = 0.6  # nominal: the detection-only fallback has no calibrated confidence, so it never reports 100%


def heuristic_scene(preds: list[FramePred]) -> dict[str, float]:
    n = max(len(preds), 1)
    veh = sum(any(d.cls in VEHICLE_CLASSES for d in p.detections) for p in preds) / n
    moto = sum(any(d.cls == "motorcycle" for d in p.detections) for p in preds) / n
    if moto > 0.5:
        return {"motorcycle": 0.6 + 0.4 * moto}
    if veh > 0:
        return {"street": 0.5 + 0.5 * veh}
    return {"unknown": HEURISTIC_UNKNOWN_CONF}


def decide(preds: list[FramePred], available: list[str]) -> Route:
    """Aggregate per-frame scene probabilities into a Route and the list of modules to run."""
    agg: dict[str, float] = {}
    for p in preds:
        for k, v in (p.scene or {}).items():
            agg[k] = agg.get(k, 0.0) + v / max(len(preds), 1)
    from_model = bool(agg)
    if not agg:
        agg = heuristic_scene(preds)
    ranked = sorted(agg.items(), key=lambda kv: -kv[1])
    scene, conf = ranked[0]
    thr = unknown_threshold()
    unknown_reason = None
    if scene != "unknown" and conf < thr:
        unknown_reason = (f"scene unknown: top candidate '{scene}' p={conf:.2f} is below the confidence threshold {thr:.2f}"
                          + (f" (runner-up '{ranked[1][0]}' p={ranked[1][1]:.2f})" if len(ranked) > 1 else ""))
        scene = "unknown"
    elif scene == "unknown":
        unknown_reason = (f"scene unknown: the router found no service-flow scene (p_unknown={conf:.2f})" if from_model
                          else "scene unknown: no vehicle detected and no scene model output")
    has_vehicle = any(d.cls in VEHICLE_CLASSES for p in preds for d in p.detections)
    has_person = any(d.cls == PERSON for p in preds for d in p.detections)

    # Reported confidence: the model's probability of the chosen scene; for "unknown" it is p(unknown) when that is the arg-max, else
    # 1 - (best known-scene probability) = how sure we are that no single known scene applies. Never a hard-coded 1.0.
    if scene == "unknown":
        if from_model:
            known_best = max((v for k, v in agg.items() if k != "unknown"), default=0.0)
            conf = agg.get("unknown", 0.0) if ranked[0][0] == "unknown" else 1.0 - known_best
        else:
            conf = min(conf, HEURISTIC_UNKNOWN_CONF)
    note = None if from_model else "no scene model output: scene from the detection-only heuristic, confidence is nominal"

    ran, skipped = ["router", "vehicle_detector"], []
    want = {
        "plate": has_vehicle,
        "bay_states": has_vehicle and scene in ("bay", "street", "gate", "wash"),
        # close-ups (a tyre, a dent) often show no whole vehicle: inspection still runs on them
        "inspection": scene == "closeup" or (has_vehicle and scene in ("street", "gate", "motorcycle")),
        "visit_engine": has_vehicle,
    }
    reasons = {
        "plate": "no vehicle detected",
        "bay_states": "no vehicle in a bay-like scene",
        "inspection": "no vehicle and not a close-up",
        "visit_engine": "no vehicle detected, so no visit is created",
    }
    for m, ok in want.items():
        if m not in available:
            skipped.append(SkippedModule(module=m, reason="module not deployed"))
        elif ok:
            ran.append(m)
        elif m in ("bay_states", "inspection") and has_vehicle and unknown_reason:
            skipped.append(SkippedModule(module=m, reason=unknown_reason))
        elif m in ("bay_states", "inspection") and has_vehicle:
            skipped.append(SkippedModule(module=m, reason=f"scene '{scene}' does not use this module"))
        else:
            skipped.append(SkippedModule(module=m, reason=reasons[m]))
    if unknown_reason and not has_vehicle:
        skipped.append(SkippedModule(module="scene_modules", reason=unknown_reason))
    if note and scene == "unknown":
        skipped.append(SkippedModule(module="scene_model", reason=note))
    if not has_person:
        skipped.append(SkippedModule(module="face_blur", reason="no person detected"))
    return Route(scene=scene, confidence=float(conf), ran=ran, skipped=skipped)
