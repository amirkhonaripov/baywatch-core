"""CPU tests for the API plumbing (limits, job flow, evidence, contract shape).

Uses the real seen.pipeline.Pipeline with a TEST-ONLY stub detector (bright image => one 'car' box), so the router,
visit engine, work-order matcher and face blur all run for real. Nothing here is served to users.
Run: pytest serve/test_app.py
"""
from __future__ import annotations

import io
import json
import time

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from seen.contracts.events import AnalysisResult, JobStatus  # noqa: E402
from seen.pipeline import Pipeline  # noqa: E402
from seen.predictor import Det, empty_preds  # noqa: E402
from serve import core  # noqa: E402
from serve.app import LocalBackend, create_app  # noqa: E402


class _StubDetector:
    name, version = "stub-detector", "test"

    def load(self, device: str = "cpu") -> None:
        pass

    def predict(self, frames, context=None):
        out = empty_preds(frames)
        for f, p in zip(frames, out):
            if f.image.mean() > 128:
                p.detections.append(Det(cls="car", confidence=0.9, bbox=(0.1, 0.2, 0.8, 0.9)))
            p.timings_ms["vehicle_detector"] = 0.1
        return out


def _jpeg(value: int, size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((size[1], size[0], 3), value, np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path):
    backend = LocalBackend(lambda: Pipeline(detectors=[_StubDetector()], device="cpu"), tmp_path / "ev")
    return TestClient(create_app(backend))


def test_health_and_catalogue(client):
    for _ in range(50):
        h = client.get("/health").json()
        if h["status"] != "loading":
            break
        time.sleep(0.05)
    assert h["status"] == "ok" and h["model_versions"] == {"stub-detector": "test"}
    cat = client.get("/catalogue").json()
    assert cat["items"] and all("verifiability" in i for i in cat["items"])


def test_image_with_vehicle_creates_visit_and_evidence(client):
    wo = {"id": "WO-1", "lines": ["wash", "cabin_filter"]}
    r = client.post("/analyze/image", files={"file": ("car.jpg", _jpeg(220), "image/jpeg")},
                    data={"work_order": json.dumps(wo), "simulate": "false"})
    assert r.status_code == 200, r.text
    res = AnalysisResult.model_validate(r.json())
    assert res.visit is not None and any(e.type.value == "arrived" for e in res.visit.events)
    assert res.visit.events[0].source.value == "observed_real"
    statuses = {v.line_code: v.status.value for v in res.visit.verifications}
    assert statuses == {"wash": "not_observed", "cabin_filter": "not_camera_verifiable"}
    uri = res.frames[0].frame_uri
    assert uri and uri.startswith("evidence/")
    ev = client.get("/" + uri)
    assert ev.status_code == 200 and ev.headers["content-type"] == "image/jpeg"


def test_non_vehicle_image_is_unknown(client):
    r = client.post("/analyze/image", files={"file": ("dark.jpg", _jpeg(20), "image/jpeg")})
    res = AnalysisResult.model_validate(r.json())
    assert res.route.scene == "unknown" and res.visit is None
    assert any("No vehicle" in w for w in res.warnings)


def test_frames_job_flow(client):
    files = [("files", (f"f{i}.jpg", _jpeg(220 if i >= 2 else 20), "image/jpeg")) for i in range(6)]
    r = client.post("/analyze/frames", files=files, data={"timestamps": json.dumps([i * 0.5 for i in range(6)])})
    assert r.status_code == 200, r.text
    job = JobStatus.model_validate(r.json())
    for _ in range(100):
        st = client.get(f"/jobs/{job.job_id}").json()
        if st["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert st["status"] == "done", st
    res = AnalysisResult.model_validate(st["result"])
    assert res.input_kind == "frames" and len(res.frames) == 6
    assert res.visit is not None and any(e.type.value == "arrived" for e in res.visit.events)
    assert all(f.frame_uri and f.frame_uri.startswith("evidence/") for f in res.frames)


def test_limits(client):
    big = b"\xff\xd8\xff" + b"0" * (core.MAX_IMAGE_BYTES + 10)
    assert client.post("/analyze/image", files={"file": ("big.jpg", big, "image/jpeg")}).status_code == 413
    assert client.post("/analyze/image", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 400
    files = [("files", (f"f{i}.jpg", _jpeg(10, (8, 8)), "image/jpeg")) for i in range(core.MAX_FRAMES + 1)]
    ts = json.dumps([i * 0.5 for i in range(core.MAX_FRAMES + 1)])
    assert client.post("/analyze/frames", files=files, data={"timestamps": ts}).status_code == 413
    files = [("files", ("a.jpg", _jpeg(10), "image/jpeg")), ("files", ("b.jpg", _jpeg(10), "image/jpeg"))]
    # clip length is not limited (timelapse CCTV frames can be minutes apart); only the frame count and bytes are
    assert client.post("/analyze/frames", files=files, data={"timestamps": "[0, 1800]"}).status_code == 200
    assert client.post("/analyze/frames", files=files, data={"timestamps": "[0]"}).status_code == 400
    assert client.post("/analyze/frames", files=files, data={"timestamps": "[1, 0]"}).status_code == 400
    bad_zone = json.dumps([{"id": "z", "kind": "bay", "polygon": [[0, 0], [2, 0], [1, 1]]}])
    r = client.post("/analyze/image", files={"file": ("a.jpg", _jpeg(200), "image/jpeg")}, data={"zones": bad_zone})
    assert r.status_code == 400
    assert client.get("/jobs/nope").status_code == 404
    assert client.get("/evidence/..%2Fsecret.jpg").status_code == 404


def test_kill_switch(tmp_path):
    backend = LocalBackend(lambda: Pipeline(detectors=[_StubDetector()], device="cpu"), tmp_path)
    c = TestClient(create_app(backend, disabled=True))
    assert c.post("/analyze/image", files={"file": ("a.jpg", _jpeg(200), "image/jpeg")}).status_code == 503
    assert c.get("/health").json()["disabled"] is True
