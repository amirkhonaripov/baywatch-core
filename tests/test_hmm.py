"""Unit tests for seen/hmm.py (CPU only)."""
import numpy as np
import pytest

from seen import hmm


def test_state_space_is_consistent():
    assert hmm.S == 30 and len(hmm.STATES) == len(set(hmm.STATES))
    L = hmm.LIDX
    for s in hmm.STATES:
        y = s.labels()
        if not y[L["car_present"]]:
            assert sum(y) == 0
        assert y[L["wheel_off"]] <= y[L["raised"]]
        assert y[L["person_under_car"]] <= y[L["raised"]]
        assert y[L["work"]] <= y[L["person_at_car"]] + y[L["person_under_car"]]
        assert not (y[L["washing"]] and (y[L["raised"]] or y[L["hood_open"]]))
        assert hmm.labels_to_state(y) == hmm.state_index(s)


def test_grammar_moves():
    A = hmm.ALLOWED
    empty = hmm.state_index(hmm.CState(0))
    raised_car = hmm.state_index(hmm.CState(1, raised=1))
    down_car = hmm.state_index(hmm.CState(1))
    assert A[empty, down_car] and A[down_car, empty]
    assert not A[empty, raised_car] and not A[raised_car, empty]  # cannot arrive or leave on the lift
    assert not A.diagonal().any()


def test_expm_properties():
    Q = hmm.generic_Q()
    P1, P2, P3 = hmm._expm(Q * 1.0), hmm._expm(Q * 2.0), hmm._expm(Q * 3.0)
    np.testing.assert_allclose(P1.sum(1), 1.0, atol=1e-9)
    np.testing.assert_allclose(P1 @ P2, P3, atol=1e-8)
    assert (P1 >= 0).all()
    np.testing.assert_allclose(hmm.TransitionCache(Q).P(0.0), np.eye(hmm.S))


def _clip(T=600, seed=0, acc=0.8, irregular=False):
    """Car present 60-540 s, raised 150-450 s; per-frame labels flipped with prob 1-acc (white noise)."""
    rng = np.random.default_rng(seed)
    times = np.cumsum(rng.uniform(0.5, 2.0, T)) if irregular else np.arange(T, dtype=float)
    truth = np.zeros((T, hmm.K), dtype=bool)
    car = (times >= 60) & (times < 540)
    up = (times >= 150) & (times < 450)
    truth[:, hmm.LIDX["car_present"]] = car
    truth[:, hmm.LIDX["raised"]] = up
    truth[:, hmm.LIDX["idle"]] = car
    P = np.where(truth, 0.9, 0.1).astype(float)
    flip = rng.random(P.shape) > acc
    P[flip] = 1 - P[flip]
    M = np.zeros_like(truth)
    M[:, [hmm.LIDX["car_present"], hmm.LIDX["raised"]]] = True
    return times, P, M, truth


@pytest.mark.parametrize("rates", ["generic", "fitted"])
def test_decode_beats_hysteresis_on_flicker(rates):
    """Decoder mechanics on white-noise flicker, untempered emissions (the shipped defaults temper evidence for
    temporally correlated errors; see docs/lanes/sim_notes.md)."""
    times, P, M, truth = _clip()
    cols = [hmm.LIDX["car_present"], hmm.LIDX["raised"]]
    d = hmm.decode(times, P, M, zone_kind="bay", rates=rates, eps=0.1, weight=1.0)
    h = hmm.hysteresis(P, M)
    acc_hmm = (d.labels[:, cols] == truth[:, cols]).mean()
    acc_hys = (h.labels[:, cols] == truth[:, cols]).mean()
    assert acc_hmm > 0.95 and acc_hmm > acc_hys
    n_seg = len(hmm.segments(d.labels[:, hmm.LIDX["raised"]]))
    assert n_seg == 1  # one lift cycle, no flicker
    assert ((d.conf >= 0) & (d.conf <= 1)).all()


def test_decode_irregular_sampling_and_missing_labels():
    times, P, M, truth = _clip(T=400, irregular=True)
    M[::7] = False  # some frames carry no state at all
    d = hmm.decode(times, P, M, zone_kind="frame")
    k = hmm.LIDX["car_present"]
    assert (d.labels[:, k] == truth[:, k]).mean() > 0.95


def test_zone_mask_forbids_states():
    times, P, M, _ = _clip(T=200)
    P[:, hmm.LIDX["raised"]] = 0.99
    d = hmm.decode(times, P, M, zone_kind="wash")
    assert not d.labels[:, hmm.LIDX["raised"]].any()


def test_hysteresis_semantics():
    P = np.array([[0.1], [0.65], [0.5], [0.45], [0.39], [0.7]])
    M = np.ones_like(P, dtype=bool)
    d = hmm.hysteresis(P, M, 0.6, 0.4)
    assert d.labels[:, 0].tolist() == [False, True, True, True, False, True]


def test_prune_short_segments_and_grammar():
    times = np.arange(100, dtype=float)
    lab = np.zeros((100, hmm.K), dtype=bool)
    lab[:, hmm.LIDX["car_present"]] = True
    lab[10:90, hmm.LIDX["raised"]] = True
    lab[40:45, hmm.LIDX["raised"]] = False  # a 5 s "lowering" in the middle of a job
    lab[20:30, hmm.LIDX["wheel_off"]] = True
    out = hmm.prune_short_segments(times, lab, {"raised": (30.0, 30.0)}, scale=1.0)
    assert out[10:90, hmm.LIDX["raised"]].all()
    assert out[:, hmm.LIDX["idle"]].all()  # recomputed: car present, no work
    bad = lab.copy()
    bad[:, hmm.LIDX["raised"]] = False
    assert not hmm.enforce_grammar(bad)[:, hmm.LIDX["wheel_off"]].any()


def test_rates_fallback_without_priors(tmp_path):
    R = hmm.load_rates(str(tmp_path / "missing.json"))
    assert set(R) == {"generic"}
    np.testing.assert_allclose(R["generic"].sum(1), 0.0, atol=1e-12)


def test_fitted_rates_file_matches_state_space():
    R = hmm.load_rates()
    if "bay" not in R:
        pytest.skip("no fitted priors (run python -m sim.scenario --fit-priors)")
    for Q in R.values():
        np.testing.assert_allclose(Q.sum(1), 0.0, atol=1e-9)
        off = Q - np.diag(Q.diagonal())
        assert (off >= 0).all()
