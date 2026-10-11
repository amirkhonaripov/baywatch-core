"""CPU smoke tests for lane C (ops). GPU/model-dependent parts are skipped when the artifact is not present."""
from __future__ import annotations

import numpy as np
import pytest

from lanes.ops.geometry import GEOM_NAMES, crop_box, geom_features, primary_vehicle
from lanes.ops.heads import HeadModel
from lanes.ops.metrics import auroc, binary_ece, f1
from seen.paths import ARTIFACTS
from seen.predictor import Det, Frame, FramePred
from seen.vocab import BAY_STATES


def test_geometry_shapes_and_under_rule():
    car = ("car", 0.9, (0.3, 0.2, 0.7, 0.5))  # raised car: bottom edge at y=0.5
    person_under = ("person", 0.8, (0.45, 0.3, 0.55, 0.8))  # head above the car's bottom edge, feet below it
    f = geom_features([car, person_under])
    assert f.shape == (len(GEOM_NAMES),)
    g = dict(zip(GEOM_NAMES, f))
    assert g["has_car"] == 1 and g["under_geom"] == 1 and g["at_geom"] == 1
    f2 = dict(zip(GEOM_NAMES, geom_features([car, ("person", 0.8, (0.9, 0.2, 0.95, 0.6))])))
    assert f2["under_geom"] == 0
    assert dict(zip(GEOM_NAMES, geom_features([])))["has_car"] == 0


def test_primary_vehicle_roi():
    v = [(0.9, (0.0, 0.0, 0.2, 0.2)), (0.6, (0.5, 0.5, 0.9, 0.9))]
    assert primary_vehicle(v, roi=(0.45, 0.45, 1.0, 1.0))[1] == (0.5, 0.5, 0.9, 0.9)
    assert primary_vehicle(v, roi=(0.95, 0.95, 1.0, 1.0)) is None
    x1, y1, x2, y2 = crop_box((0.4, 0.4, 0.6, 0.6), 0.25)
    assert 0 <= x1 < 0.4 and x2 > 0.6 and y2 <= 1


def test_head_folding_matches():
    rng = np.random.default_rng(0)
    D, S = 7, 3
    m = HeadModel(rng.normal(size=D), rng.uniform(0.5, 2, D), rng.normal(size=(D, S)), rng.normal(size=S),
                  rng.uniform(0.5, 1.5, S), rng.normal(size=S), ["a", "b", "c"])
    X = rng.normal(size=(5, D))
    W, b = m.folded()
    np.testing.assert_allclose(1 / (1 + np.exp(-(X @ W + b))), m.predict_proba(X), rtol=1e-4, atol=1e-5)


def test_metrics():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert f1(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert binary_ece(y, np.array([0.0, 0.0, 1.0, 1.0])) == 0.0


@pytest.mark.skipif(not (ARTIFACTS / "ops" / "states" / "model_card.json").exists(), reason="ops artifact not present")
def test_predictor_end_to_end_cpu():
    from lanes.ops.predict import get_predictor

    p = get_predictor(use_parts=False)
    p.load("cpu")
    img = np.zeros((240, 320, 3), np.uint8)
    img[100:200, 80:260] = 128
    frames = [Frame(i, float(i), img) for i in range(3)]
    ctx = [FramePred(i, float(i), detections=[Det("car", 0.9, (0.25, 0.4, 0.8, 0.85)), Det("person", 0.7, (0.1, 0.3, 0.2, 0.9))])
           for i in range(3)]
    out = p.predict(frames, ctx)
    assert len(out) == 3
    for o in out:
        assert set(BAY_STATES) - {"washing"} <= set(o.states)
        assert all(0.0 <= v <= 1.0 for v in o.states.values())
    empty = p.predict(frames[:1], [FramePred(0, 0.0)])[0]
    assert empty.states["car_present"] == 0.0 and empty.states["idle"] == 0.0
