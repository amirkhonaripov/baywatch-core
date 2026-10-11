"""CPU smoke tests for Lane B (run: ``pytest lanes/flow/test_smoke.py``). GPU / model-download tests skip themselves."""
from __future__ import annotations

import numpy as np
import pytest

from seen.predictor import Det, Frame, FramePred
from seen.router import decide
from seen.zones import DwellTracker, assign_to_zones, box_zone_overlap, point_in_polygon, slot_occupancy_score


def test_zones_geometry():
    sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert point_in_polygon((5, 5), sq) and not point_in_polygon((11, 5), sq)
    o = box_zone_overlap((5, 5, 15, 15), sq)
    assert o.inter == pytest.approx(25.0) and o.frac_of_box == pytest.approx(0.25)
    assert slot_occupancy_score([(1, 1, 9, 9)], sq) == pytest.approx(0.64)
    assert slot_occupancy_score([(9, 9, 30, 30)], sq) == 0.0  # mostly outside the slot: neighbour overhang does not count


def test_assign_and_dwell():
    zones = [{"id": "bay1", "polygon": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]}, {"id": "bay2", "polygon": [[0.5, 0], [1, 0], [1, 1], [0.5, 1]]}]
    dets = [Det("car", 0.9, (0.1, 0.2, 0.3, 0.8)), Det("person", 0.8, (0.6, 0.2, 0.7, 0.9))]
    got = assign_to_zones(dets, zones, rule="foot")
    assert got == {"bay1": [0], "bay2": [1]}
    d = DwellTracker(max_gap_s=2.0)
    ev = []
    for t in [0, 1, 2, 3, 10, 11]:
        ev += d.update(t, [("a", "bay1")] if t != 3 else [])
    ev += d.flush()
    assert [e.kind for e in ev] == ["entered", "left", "entered", "left"]
    assert ev[1].dwell_s == pytest.approx(2.0)


def _tracks(name, fps):
    from lanes.flow.tracker import make_tracker

    rng = np.random.default_rng(0)
    starts = rng.uniform(100, 600, (4, 2))
    vel = rng.uniform(-8, 8, (4, 2))
    trk = make_tracker(name, fps=fps)
    ids = {k: set() for k in range(4)}
    for t in range(40):
        c = starts + vel * t
        boxes = np.concatenate([c - [30, 20], c + [30, 20], np.full((4, 1), 0.9)], 1)
        out = trk.update(boxes, t / fps)
        for row in out:
            k = int(np.argmin(np.linalg.norm(c - (row[:2] + row[2:4]) / 2, axis=1)))
            ids[k].add(int(row[4]))
    return ids


@pytest.mark.parametrize("name", ["ocsort", "iou"])
def test_tracker_keeps_identities(name):
    ids = _tracks(name, 5.0)
    assert all(len(v) == 1 for v in ids.values())


def test_mot_metrics_perfect_and_swap():
    from lanes.flow.mot_metrics import all_metrics, seq_counts

    gt = {t: (np.array([1, 2]), np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float) + t) for t in range(20)}
    m = all_metrics(seq_counts(gt, gt))
    assert m["HOTA"] == pytest.approx(1.0) and m["IDF1"] == pytest.approx(1.0) and m["MOTA"] == pytest.approx(1.0)
    pr = {t: (np.array([1, 2]) if t < 10 else np.array([7, 8]), gt[t][1]) for t in gt}
    m2 = all_metrics(seq_counts(gt, pr))
    assert m2["IDF1"] < 0.6 and m2["IDSW"] == 2 and m2["DetA"] == pytest.approx(1.0)


def test_router_decide_reasons():
    v = Det("car", 0.9, (0.1, 0.1, 0.5, 0.5))
    avail = ["plate", "bay_states", "inspection", "visit_engine"]
    r = decide([FramePred(0, 0.0, [v], scene={"bay": 0.8, "unknown": 0.2})], avail)
    assert r.scene == "bay" and "bay_states" in r.ran
    r = decide([FramePred(0, 0.0, [v], scene={"bay": 0.3, "street": 0.28, "gate": 0.22, "unknown": 0.2})], avail)
    assert r.scene == "unknown" and "bay_states" not in r.ran
    assert any("below the confidence threshold" in s.reason for s in r.skipped if s.module == "bay_states")
    r = decide([FramePred(0, 0.0, [], scene={"unknown": 0.9})], avail)
    assert r.scene == "unknown" and r.ran == ["router", "vehicle_detector"]


def test_blur_changes_pixels_and_never_noops_on_person():
    from seen import blur

    img = np.full((240, 320, 3), 127, np.uint8)
    img[40:100, 120:180] = (200, 150, 120)  # a textured patch so blur is detectable
    img[60:80, 130:170:4] = 20
    person = Det("person", 0.9, (0.3, 0.1, 0.7, 0.95))
    boxes = blur.face_boxes(img, [person])
    assert boxes, "a person box must always produce a blur region (face model or heuristic fallback)"
    out = blur.blur_faces(img, [person])
    assert out.shape == img.shape and (out != img).any() and img[0, 0].tolist() == out[0, 0].tolist()
    assert blur.backend() in ("yunet", "heuristic")


def test_predictor_with_stub_backend_sets_track_ids():
    from lanes.flow.predict import get_predictor

    class Stub:
        def load(self, device):  # noqa: D401
            pass

        def infer(self, images, conf):
            h, w = images[0].shape[:2]
            return [np.array([[0.1 * w + 5 * k, 0.2 * h, 0.4 * w + 5 * k, 0.6 * h, 0.9, 1]], dtype=np.float32) for k, _ in enumerate(images)]

    det = get_predictor(model="stub", tracker="ocsort", backend=Stub())
    det.load("cpu")
    frames = [Frame(i, i * 0.2, np.zeros((100, 200, 3), np.uint8)) for i in range(6)]
    out = det.predict(frames)
    ids = {d.track_id for p in out for d in p.detections}
    assert len(ids) == 1 and None not in ids
    assert all(0 <= v <= 1 for p in out for d in p.detections for v in d.bbox) and out[0].detections[0].cls == "car"


def test_router_zero_shot_gpu_optional():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from lanes.flow.predict import get_router

    r = get_router(mode="zeroshot")
    r.load("cuda:0")
    f = Frame(0, 0.0, np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8))
    p = r.predict([f])[0]
    assert abs(sum(p.scene.values()) - 1.0) < 1e-3
