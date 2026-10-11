"""CPU tests for Lane G shared tooling: pytest eval/test_eval.py"""
from __future__ import annotations

import json

import numpy as np
import pytest

from eval import manifest as M
from eval import stats as S


def test_wilson_known_values():
    p, lo, hi = S.wilson_ci(81, 100)
    assert p == pytest.approx(0.81)
    assert lo == pytest.approx(0.7222, abs=1e-3) and hi == pytest.approx(0.8749, abs=1e-3)
    p, lo, hi = S.wilson_ci(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    assert S.wilson_ci(10, 10)[2] == 1.0


def test_prop_metric_schema_shape():
    m = S.prop_metric("recall", 5, 20)
    assert set(m) == {"name", "value", "ci_low", "ci_high", "unit"}
    with pytest.raises(ValueError):
        S.prop_metric("x", 0, 0)


def test_bootstrap_mean_covers_truth():
    rng = np.random.default_rng(1)
    x = rng.normal(3.0, 1.0, 400)
    p, lo, hi = S.bootstrap_ci(x, np.mean, n_boot=500)
    assert lo < 3.0 < hi and lo < p < hi


def test_ece_perfect_and_bad():
    assert S.ece([1.0, 1.0, 0.0], [1, 1, 0]) == pytest.approx(0.0)
    assert S.ece([0.9] * 10, [0] * 10) == pytest.approx(0.9)


def test_ap_perfect_and_half():
    ims = [{"gt": [("car", 0, 0, 10, 10)], "pred": [("car", 0.9, 0, 0, 10, 10)]},
           {"gt": [("car", 0, 0, 10, 10)], "pred": [("car", 0.8, 50, 50, 60, 60)]}]
    assert S.average_precision(ims[:1], "car") == pytest.approx(1.0)
    assert S.average_precision(ims, "car") == pytest.approx(0.5)
    m = S.detection_ap(ims, ["car"], n_boot=50)
    assert m["name"] == "mAP@0.5" and 0 <= m["ci_low"] <= m["ci_high"] <= 1


def test_confusion():
    c = S.confusion(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
    assert c["matrix"] == [[1, 1], [0, 1]]


def _img(seed, size=64):
    rng = np.random.default_rng(seed)
    return (rng.random((size, size, 3)) * 255).astype(np.uint8)


def test_phash_near_dup_and_distinct():
    a = _img(0, 128)
    b = np.clip(a.astype(int) + 3, 0, 255).astype(np.uint8)  # brightness shift
    c = _img(1, 128)
    ha, hb, hc = M.phash(a), M.phash(b), M.phash(c)
    assert len(ha) == 16
    assert M.hamming(ha, hb) <= M.STRICT
    assert M.hamming(ha, hc) > M.LOOSE


def test_dedup_and_split(tmp_path, monkeypatch):
    recs = []
    for s in range(6):
        h = M.phash(_img(100 + s, 96))
        recs.append({"id": f"d:{s}", "path": f"x/{s}.jpg", "source": f"src{s % 3}", "phash": h})
    recs.append({"id": "d:dup", "path": "x/dup.jpg", "source": "src9", "phash": recs[0]["phash"]})
    kept, removed = M.dedup(recs)
    assert len(removed) == 1 and removed[0]["dup_of"] == "d:0"
    split = M.split_by_source(recs, test_sources=["src0"])
    by = {r["id"]: r["split"] for r in split}
    assert by["d:0"] == "test" and by["d:3"] == "test"
    assert by["d:dup"] == "quarantine"  # near-duplicate of a test item from another source
    assert by["d:1"] == "train"
    # random-fraction mode never splits a source
    split2 = M.split_by_source(recs, test_frac=0.3)
    for s in {r["source"] for r in split2}:
        assert len({r["split"] for r in split2 if r["source"] == s}) == 1


def test_audit_finds_leak(tmp_path, monkeypatch):
    from PIL import Image

    monkeypatch.setattr(M, "MANIFEST_DIR", tmp_path / "manifests")
    monkeypatch.setenv("SEEN_PHASH_CACHE", str(tmp_path / "cache.jsonl"))
    imgs = [_img(200 + i, 96) for i in range(4)]
    paths = []
    for i, im in enumerate(imgs):
        p = tmp_path / f"{i}.png"
        Image.fromarray(im).save(p)
        paths.append(p)
    # resized copy of test image 0 sneaks into training
    leak = tmp_path / "leak.jpg"
    Image.fromarray(imgs[0]).resize((80, 80)).save(leak, quality=90)
    M.write_manifest("toy", M.add_phash([{"id": "toy:0", "path": str(paths[0]), "source": "a"},
                                         {"id": "toy:1", "path": str(paths[1]), "source": "b"}]))
    assert M.excluded_ids("toy") == {"toy:0", "toy:1"}
    rep = M.leakage_audit(["toy"], [str(paths[2]), str(paths[3]), str(leak)])
    assert rep["verdict"] == "FAIL"
    assert rep["components"]["toy"]["near_dup_loose_train_files"] == 1
    kept, dropped = M.filter_train([str(paths[2]), str(leak), str(paths[1])], ["toy"])
    assert kept == [str(paths[2])] and len(dropped) == 2
    rep2 = M.leakage_audit(["toy"], [str(paths[2]), str(paths[3])])
    assert rep2["verdict"] == "PASS"
    json.dumps(rep)


def test_jury_scoring_with_fake_pipeline():
    """Pipeline + router + visit engine on CPU with a fake detector; checks the jury scoring logic."""
    from seen.pipeline import Pipeline
    from seen.predictor import Det, Frame, empty_preds

    from eval import jury_run

    class FakeDet:
        name, version = "fake", "0"

        def load(self, device="cpu"):
            pass

        def predict(self, frames, context=None):
            out = empty_preds(frames)
            for f, p in zip(frames, out):
                if f.image.mean() > 100:
                    p.detections.append(Det(cls="car", confidence=0.9, bbox=(0.1, 0.1, 0.5, 0.5)))
            return out

    pipe = Pipeline([FakeDet()], device="cpu")
    car = np.full((64, 64, 3), 200, np.uint8)
    dark = np.zeros((64, 64, 3), np.uint8)
    it_car = {"id": "a", "category": "uz_car", "kind": "image", "expected": {"scene": "street", "acceptable_scenes": ["street"],
              "vehicle": True, "plate_visible": True, "inspection_target": False, "hard": False, "notes": ""}}
    it_none = {"id": "b", "category": "non_car", "kind": "image", "expected": {"scene": "unknown", "acceptable_scenes": ["unknown"],
               "vehicle": False, "plate_visible": False, "inspection_target": False, "hard": False, "notes": ""}}
    outs = []
    for it, img in ((it_car, car), (it_none, dark), (it_none, car)):
        r = pipe.analyze([Frame(index=0, t_s=0.0, image=img)])
        outs.append(jury_run.score(it, {"result": r, "latency_ms": 5.0, "n_frames": 1}, None))
    assert outs[0]["visit_ok"] and outs[0]["route_ok"] and outs[0]["plate_ok"] is False
    assert outs[1]["visit_ok"] and outs[1]["unknown_ok"]
    assert not outs[2]["visit_ok"]  # a 'car' on a no-vehicle input = false visit
    outs.append(jury_run.score(it_car, None, "boom"))
    s, metrics, conf = jury_run.summarise(outs)
    assert s["crashes"] == 1 and s["false_visit_rate_no_vehicle"] == 0.5
    assert conf["labels"][0] == "gate" and sum(map(sum, conf["matrix"])) == 3
    assert all("ci_low" in m for m in metrics if m.get("unit") == "fraction")
