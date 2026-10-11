"""Unit tests for sim/scenario.py (CPU only)."""
import inspect
from datetime import timedelta

import numpy as np
import pytest

from seen import hmm, visit_engine
from seen.contracts.events import EventType, LineStatus, Source, Zone, ZoneKind
from seen.predictor import Det, FramePred
from sim import scenario as SC


@pytest.fixture(scope="module")
def visits():
    return SC.generate_visits(30, 123)


def test_deterministic(visits):
    again = SC.generate_visits(30, 123)
    assert [v.model_dump() for v, _, _ in visits] == [v.model_dump() for v, _, _ in again]
    other = SC.generate_visits(3, 124)
    assert other[0][2].meta["pieces"] != visits[0][2].meta["pieces"]


def test_timeline_grammar(visits):
    order = {k: i for i, k in enumerate(SC.LOCATIONS)}
    for _, wo, tl in visits:
        for row in tl.labels:
            hmm.labels_to_state(row)  # raises on an impossible combination
        locs = [SC.LOCATIONS[i] for i in tl.location]
        seq = [l for i, l in enumerate(locs) if i == 0 or locs[i - 1] != l]
        assert seq[0] == "gate" and seq[-1] == "exit" and "bay" in seq and "ready" in seq
        assert [order[l] for l in seq] == sorted(order[l] for l in seq)  # stages only move forward
        bay = tl.labels[tl.location == SC.LOC["bay"]]
        assert bay[:, SC.KI["washing"]].sum() == 0
        assert tl.meta["job_type"] in ScenarioMix


ScenarioMix = SC.ScenarioConfig().job_mix


def test_ground_truth_visit(visits):
    for v, wo, tl in visits:
        assert all(e.source == Source.simulated for e in v.events)
        types = [e.type for e in v.events]
        assert types[0] == EventType.arrived and EventType.ready in types and types[-1] in (EventType.handed_over, EventType.left_zone)
        st = {x.line_code: x.status for x in v.verifications}
        for ln in tl.meta["lines_performed"]:
            if ln in st and st[ln] != LineStatus.not_camera_verifiable:
                assert st[ln] == LineStatus.evidence_found, (tl.visit_id, ln)
        for e in v.events:
            if e.type == EventType.idle_started:
                assert e.payload["duration_s"] >= visit_engine.IDLE_THRESHOLD_S


def test_job_mix_contains_parts_wait_and_phantoms():
    vs = SC.generate_visits(200, 7)
    jobs = {tl.meta["job_type"] for _, _, tl in vs}
    assert jobs == set(ScenarioMix)
    steps = {p["step"] for _, _, tl in vs for p in tl.meta["pieces"]}
    assert {"parts_wait", "technician_absent", "short_pause", "washing"} <= steps
    assert any(tl.meta["phantom_lines"] for _, _, tl in vs)
    for _, _, tl in vs:  # in repair jobs the parts wait comes between diagnosis and repair
        names = [p["step"] for p in tl.meta["pieces"]]
        if "parts_wait" in names:
            w = names.index("parts_wait")
            assert "inspect_under_car" in names[:w]
            assert any(n in names[w:] for n in ("replace_brake_pads", "tighten_suspension"))


def _cut(v, tl, seconds_into_bay):
    cut_t = tl.t0 + timedelta(seconds=tl.span("bay")[0] + seconds_into_bay)
    obs = [e.model_copy(update={"source": Source.observed_synthetic}) for e in v.events
           if e.t <= cut_t and e.type not in (EventType.idle_started, EventType.idle_ended)]
    return v.model_copy(update={"events": obs, "verifications": None}), cut_t


def test_continue_visit_keeps_observed_and_adds_simulated(visits):
    v, wo, tl = visits[0]
    cut, now = _cut(v, tl, 600)
    before = cut.model_dump()
    cont = SC.continue_visit(cut, wo, now=now)
    assert cut.model_dump() == before  # input not mutated
    n = len(cut.events)
    assert [e.model_dump() for e in cont.events[:n]] == [e.model_dump() for e in cut.events]
    added = cont.events[n:]
    assert added and all(e.source == Source.simulated for e in added)
    assert all(e.t > now for e in added) and all(e.payload.get("simulated") for e in added)
    assert added[-1].type in (EventType.handed_over, EventType.left_zone)
    ready = [e for e in added if e.type == EventType.ready]
    assert len(ready) == 1 and {"eta_p10", "eta_p50", "eta_p90"} <= set(ready[0].payload)
    assert cont.visit_id == v.visit_id


def test_continue_visit_default_signature_used_by_pipeline(visits):
    params = list(inspect.signature(SC.continue_visit).parameters)
    assert params[:2] == ["visit", "work_order"]
    v, wo, tl = visits[1]
    cut, _ = _cut(v, tl, 300)
    assert len(SC.continue_visit(cut, None).events) > len(cut.events)


def test_continue_visit_on_finished_visit_is_identity(visits):
    v, wo, _ = visits[2]
    obs = v.model_copy(update={"events": [e.model_copy(update={"source": Source.observed_synthetic}) for e in v.events]})
    assert SC.continue_visit(obs, wo).model_dump() == obs.model_dump()


def test_continue_visit_from_build_visit_output():
    preds = [FramePred(index=i, t_s=float(i), detections=[Det("car", 0.9, (0.2, 0.2, 0.8, 0.9))]) for i in range(10)]
    v = visit_engine.build_visit(preds, zones=[Zone(id="frame", kind=ZoneKind.frame)])
    cont = SC.continue_visit(v, None)
    added = cont.events[len(v.events):]
    kinds = {e.zone.kind.value for e in added if e.zone is not None}
    assert {"bay", "ready"} <= kinds and all(e.source == Source.simulated for e in added)


def test_fit_priors_structure():
    pri = SC.fit_hmm_priors(n=20, seed=5)
    assert pri["states"] == [s.as_list() for s in hmm.STATES]
    for kind in ("bay", "wash", "ready", "pooled"):
        z = pri["zones"][kind]
        assert len(z["dwell_s"]) == hmm.S
        counts = np.zeros((hmm.S, hmm.S))
        for i, j, c in z["counts"]:
            counts[i, j] = c
        Q = hmm.Q_from_counts(np.array(z["dwell_s"]), counts)
        np.testing.assert_allclose(Q.sum(1), 0.0, atol=1e-9)


def test_view_pads_with_empty_zone(visits):
    _, _, tl = visits[0]
    times, lab = tl.view("bay", 50, 50)
    assert lab[:50].sum() == 0 and lab[-50:].sum() == 0 and lab[50:-50, SC.KI["car_present"]].all()


def test_scene_spec_matches_render_script():
    import ast
    from pathlib import Path

    from sim.blender import scene_spec

    src = (Path(__file__).resolve().parent.parent / "sim" / "blender" / "render_bay.py").read_text()
    consts = {t.id: ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and t.id in ("CAMERAS", "BAY_X")}
    assert consts["CAMERAS"] == scene_spec.CAMERAS and consts["BAY_X"] == scene_spec.BAY_X
    poly = scene_spec.bay_zone_polygon("ep00_train", "cam_A", "bay1")
    assert len(poly) >= 4 and (poly >= -0.5).all() and (poly <= 1.5).all()
