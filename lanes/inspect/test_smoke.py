"""CPU smoke tests for lane D. Run:  pytest lanes/inspect/test_smoke.py -q
Pure-logic tests always run; the end-to-end predictor test runs only if the ONNX artifacts exist (artifacts/inspect/...)."""
from __future__ import annotations

import numpy as np
import pytest

from lanes.inspect import dot as DOT
from lanes.inspect.assign import assign, box_to_poly
from lanes.inspect.evaldet import Evaluator
from lanes.inspect.onnx_yolo import letterbox, nms_classwise


def test_dot_validator():
    assert DOT.validate_wwyy("2922")["valid"] is True
    assert DOT.validate_wwyy("2922")["year"] == 2022
    assert DOT.validate_wwyy("5422")["valid"] is False     # week 54
    assert DOT.validate_wwyy("0022")["valid"] is False     # week 0
    assert DOT.validate_wwyy("1099")["valid"] is False     # year 2099 in the future
    assert DOT.validate_wwyy("12a4")["valid"] is False
    assert DOT.validate_wwyy(None)["valid"] is False
    assert DOT.date_code("40055932922") == "2922"
    assert DOT.date_code("195/65R15") is None


def test_reading_order_rotated():
    # five characters along a 45 degree line, shuffled
    t = np.arange(5)
    c = np.stack([10 + 8 * t, 10 + 8 * t], 1).astype(float)
    perm = np.array([3, 0, 4, 2, 1])
    order = DOT.reading_order(c[perm])
    assert list(perm[order]) == [0, 1, 2, 3, 4]


def test_assemble_drops_boundary_and_maps_symbols():
    boxes = np.array([[0, 0, 4, 10], [5, 0, 9, 10], [10, 0, 14, 10], [15, 0, 19, 10]], float)
    assert DOT.assemble(boxes, ["1", "slash", "Boundary", "7"]) == "1/7"


def test_edit_distance():
    assert DOT.edit_distance("2922", "2822") == 1
    assert DOT.edit_distance("", "ab") == 2


def test_assign_max_overlap_and_unknown():
    square = lambda x, y, s: [[[x, y], [x + s, y], [x + s, y + s], [x, y + s]]]  # noqa: E731
    parts = [square(0, 0, 50), square(50, 0, 50)]
    # damage straddling both parts, mostly in the second
    r = assign([square(40, 10, 40)], parts, ["door", "hood"], None, 100, 100)
    assert r[0][0] == "hood" and 0.5 < r[0][1] <= 1.0
    # damage outside every part
    r = assign([square(60, 70, 10)], parts, ["door", "hood"], None, 100, 100)
    assert r[0][0] is None


def test_evaluator_perfect_and_empty():
    gt = [{"cls": "dent", "box": [10, 10, 50, 50], "poly": [[[10, 10], [50, 10], [50, 50], [10, 50]]]}]
    good = {"id": "a", "w": 100, "h": 100, "gt": gt, "pred": [{"cls": "dent", "score": 0.9, "box": [10, 10, 50, 50], "poly": gt[0]["poly"]}]}
    miss = {"id": "b", "w": 100, "h": 100, "gt": gt, "pred": []}
    ev = Evaluator([good], ["dent"], "mask", 0.5)
    assert ev.ap("dent", n_boot=10)[0] == pytest.approx(1.0)
    ev = Evaluator([good, miss], ["dent"], "box", 0.5)
    assert ev.ap("dent", n_boot=10)[0] == pytest.approx(0.5, abs=0.01)
    assert ev.recall_at_fppi("*", 0.1, n_boot=10)[0] == pytest.approx(0.5)


def test_cluster_bootstrap_uses_groups():
    gt = [{"cls": "dent", "box": [10, 10, 50, 50], "poly": None}]
    hit = {"id": "a", "w": 100, "h": 100, "gt": gt, "pred": [{"cls": "dent", "score": 0.9, "box": [10, 10, 50, 50], "poly": None}]}
    miss = {"id": "b", "w": 100, "h": 100, "gt": gt, "pred": []}
    recs = [hit, hit, hit, miss]          # three copies of one image + one other image: two clusters
    ev = Evaluator(recs, ["dent"], "box", 0.5, groups=[0, 0, 0, 1])
    p, lo, hi = ev.ap("dent", n_boot=300)
    assert 0.0 <= lo <= p <= hi <= 1.0
    # resampling 2 clusters gives {0,0}->1.0, {0,1}->0.75 AP-ish, {1,1}->0.0: the interval must be wider than for 4 independent images
    ev_ind = Evaluator(recs, ["dent"], "box", 0.5)
    _, lo2, hi2 = ev_ind.ap("dent", n_boot=300)
    assert (hi - lo) >= (hi2 - lo2) - 1e-9


def test_dot_cluster_prop():
    from lanes.inspect.dot import cluster_prop
    m = cluster_prop("x", [1, 1, 1, 0], [0, 0, 0, 1], n_boot=200)
    assert m["value"] == 0.75 and m["ci_low"] <= 0.75 <= m["ci_high"]


def test_letterbox_and_nms():
    img = np.zeros((100, 200, 3), np.uint8)
    lb, r, left, top = letterbox(img, 64)
    assert lb.shape == (64, 64, 3) and r == pytest.approx(0.32) and left == 0 and top == 16
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10]], float)
    keep = nms_classwise(boxes, np.array([0.9, 0.8, 0.7]), np.array([0, 0, 1]), 0.5, 10)
    assert sorted(keep.tolist()) == [0, 2]   # same-class overlap suppressed, other class kept


def test_predictor_requires_artifacts(tmp_path):
    from lanes.inspect.predict import get_predictor
    p = get_predictor(artifacts=tmp_path)
    with pytest.raises(FileNotFoundError):
        p.load("cpu")


def test_end_to_end_if_artifacts_present():
    from seen.paths import ARTIFACTS
    from seen.predictor import Frame
    if not (ARTIFACTS / "inspect" / "damage_seg" / "model.onnx").exists():
        pytest.skip("inspect artifacts not built on this machine")
    from lanes.inspect.predict import get_predictor
    p = get_predictor()
    p.load("cuda:0")
    img = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype(np.uint8)
    out = p.predict([Frame(index=0, t_s=0.0, image=img)], None)
    assert len(out) == 1 and out[0].index == 0
    for d in out[0].detections:
        assert all(0.0 <= v <= 1.0 for v in d.bbox)
