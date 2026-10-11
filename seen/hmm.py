"""Temporal decoding of per-frame bay-state probabilities (Lane E).

Per-frame state predictors (``FramePred.states``) are noisy and flicker. This module turns them into clean
state sequences with a hidden Markov model over a *composite* latent state, then the visit engine turns the
sequences into events.

Latent state
    One composite state = a consistent combination of physical factors seen by one camera zone:
    car present, raised (lift or jack), a wheel off, bonnet open, person (none / at the car / under the car),
    work (a person actively working), washing. The grammar forbids impossible combinations
    (wheel off or person under the car only when raised; work only when a person is there; washing only
    with the car down and closed). 30 states in total; zone kinds mask out states that cannot occur there.

Transitions (irregular sampling)
    A continuous-time Markov chain with generator ``Q``; between two frames ``dt`` seconds apart the transition
    matrix is ``expm(Q * dt)``, so 0.5-2 fps input, dropped frames and gaps are handled exactly.
    ``Q`` comes from ``sim/priors/hmm_rates.json`` when present (dwell times and transition counts measured on
    scenario-engine timelines, i.e. duration priors from the scenario engine), else from a generic grammar with
    one time constant. Exponential dwell times are an HMM simplification (no explicit-duration HSMM).

Emissions
    Each available label probability is treated as an independent noisy observation of the label implied by
    the composite state, softened by an epsilon floor so one confident wrong frame cannot dominate:
    ``p' = eps/2 + (1-eps) p``;  ``log L(s) = w * sum_k [y_k(s) log p'_k + (1-y_k(s)) log(1-p'_k)]``.
    Labels missing in a frame are skipped (marginalised).

Duration priors (HSMM-lite)
    Exponential dwell times cannot say "a lift is never lowered for 20 s in the middle of a job", so after Viterbi
    each label's interior segments (on or off) shorter than ``scale x`` the 1st percentile of that label's segment
    durations in scenario-engine timelines are flipped, shortest first, in dependency order (car, washing, raised,
    wheel, bonnet, under, at, work), and the grammar is re-imposed. Requires the fitted priors file.

Outputs: Viterbi path, per-frame decoded labels and posterior label marginals (forward-backward), used as the
event confidence. ``hysteresis`` is the original threshold baseline kept as a fallback.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from seen.vocab import BAY_STATES

LABELS: tuple[str, ...] = BAY_STATES
K = len(LABELS)
LIDX = {s: i for i, s in enumerate(LABELS)}
RATES_PATH = Path(__file__).resolve().parent.parent / "sim" / "priors" / "hmm_rates.json"

# Defaults chosen on development seeds of sim/noise_test.py (see docs/lanes/sim_notes.md).
DEFAULT_EPS = 0.3
DEFAULT_WEIGHT = 0.2
DEFAULT_PRUNE_SCALE = 0.5  # min segment duration = scale * 1st percentile of scenario segment durations (0 = off)
GENERIC_TAU_S = 300.0  # generic grammar: every allowed move has rate 1/300 s
PRUNE_ORDER = ("car_present", "washing", "raised", "wheel_off", "hood_open", "person_under_car", "person_at_car", "work")


@dataclass(frozen=True)
class CState:
    car: int
    raised: int = 0
    wheel: int = 0
    hood: int = 0
    person: int = 0  # 0 none, 1 at the car, 2 under the car
    work: int = 0
    wash: int = 0

    def labels(self) -> tuple[int, ...]:
        idle = int(self.car and not self.work and not self.wash)
        return (self.car, self.raised, self.wheel, self.hood, int(self.person == 1), int(self.person == 2),
                self.work, idle, self.wash)

    def as_list(self) -> list[int]:
        return [self.car, self.raised, self.wheel, self.hood, self.person, self.work, self.wash]


def enumerate_states() -> list[CState]:
    out = [CState(0)]
    for raised in (0, 1):
        for wheel in ((0, 1) if raised else (0,)):
            for hood in (0, 1):
                for person in ((0, 1, 2) if raised else (0, 1)):
                    for work in ((0, 1) if person else (0,)):
                        out.append(CState(1, raised, wheel, hood, person, work, 0))
    for person in (0, 1):
        for work in ((0, 1) if person else (0,)):
            out.append(CState(1, 0, 0, 0, person, work, 1))
    return out


STATES: list[CState] = enumerate_states()
S = len(STATES)
SIDX = {s: i for i, s in enumerate(STATES)}
Y = np.array([s.labels() for s in STATES], dtype=np.float64)  # S x K


def state_index(c: CState) -> int:
    return SIDX[c]


def labels_to_state(row) -> int:
    """Map a ground-truth label vector (K,) to its composite state index (raises if inconsistent)."""
    r = [int(v) for v in row]
    car, raised, wheel, hood, at, under, work, _idle, wash = r
    if not car:
        return 0
    person = 2 if under else (1 if at else 0)
    return SIDX[CState(1, raised, wheel, hood, person, work if person else 0, wash)]


def zone_mask(kind: str | None) -> np.ndarray:
    """Which composite states may occur in a zone of this kind."""
    m = np.ones(S, dtype=bool)
    if kind in (None, "frame", "other"):
        return m
    for i, s in enumerate(STATES):
        if kind == "bay":
            m[i] = s.wash == 0
        elif kind == "wash":
            m[i] = s.raised == 0 and s.hood == 0
        else:  # gate, reception, waiting, ready, exit
            m[i] = s.raised == 0 and s.wash == 0
    return m


def _allowed_move(a: CState, b: CState) -> bool:
    if a == b:
        return False
    if a.car == 0 or b.car == 0:  # arrival / departure: car must be down, closed, not washing
        c = b if a.car == 0 else a
        return c.raised == 0 and c.wheel == 0 and c.hood == 0 and c.wash == 0
    da = a.as_list()
    db = b.as_list()
    diff = [i for i in range(7) if da[i] != db[i]]
    names = {1: "raised", 2: "wheel", 3: "hood", 4: "person", 5: "work", 6: "wash"}
    d = {names[i] for i in diff}
    if d == {"raised"} or d == {"wheel"} or d == {"hood"} or d == {"work"} or d == {"wash"}:
        return True
    if d <= {"person", "work"} and "person" in d:
        return a.work == b.work or a.person == 0 or b.person == 0
    return False


ALLOWED = np.array([[_allowed_move(a, b) for b in STATES] for a in STATES], dtype=bool)


def generic_Q(tau_s: float = GENERIC_TAU_S) -> np.ndarray:
    Q = ALLOWED.astype(np.float64) / tau_s
    np.fill_diagonal(Q, -Q.sum(1))
    return Q


def Q_from_counts(dwell_s: np.ndarray, counts: np.ndarray, alpha: float = 0.5, beta_s: float = 600.0) -> np.ndarray:
    """MLE CTMC generator with pseudo-counts on grammar moves: q_ij = (N_ij + a*allowed_ij) / (T_i + b)."""
    num = counts.astype(np.float64) + alpha * ALLOWED
    np.fill_diagonal(num, 0.0)
    Q = num / (dwell_s[:, None] + beta_s)
    np.fill_diagonal(Q, -Q.sum(1))
    return Q


@lru_cache(maxsize=4)
def load_rates(path: str | None = None) -> dict[str, np.ndarray]:
    """zone kind -> Q. Falls back to the generic grammar when no fitted priors exist."""
    p = Path(path) if path else RATES_PATH
    out: dict[str, np.ndarray] = {"generic": generic_Q()}
    if not p.exists():
        return out
    data = json.loads(p.read_text())
    if [list(s) for s in data["states"]] != [s.as_list() for s in STATES]:
        raise ValueError(f"{p}: state space differs from seen.hmm.STATES; refit with python -m sim.scenario --fit-priors")
    for kind, d in data["zones"].items():
        counts = np.zeros((S, S))
        for i, j, n in d["counts"]:
            counts[i, j] = n
        out[kind] = Q_from_counts(np.asarray(d["dwell_s"], dtype=np.float64), counts,
                                  data.get("alpha", 0.5), data.get("beta_s", 600.0))
    return out


@lru_cache(maxsize=4)
def load_min_durations(path: str | None = None) -> dict[str, dict[str, tuple[float, float]]]:
    """zone kind -> label -> (min on-segment s, min off-segment s) = 1st percentiles from scenario timelines."""
    p = Path(path) if path else RATES_PATH
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    return {kind: {lab: (float(v[0]), float(v[1])) for lab, v in d.get("min_seg_s", {}).items()}
            for kind, d in data["zones"].items()}


def enforce_grammar(lab: np.ndarray) -> np.ndarray:
    """Make a (T, K) label matrix consistent with the composite-state grammar; recompute idle."""
    g = {s: LIDX[s] for s in LABELS}
    car = lab[:, g["car_present"]]
    for s in LABELS:
        if s != "car_present":
            lab[:, g[s]] &= car
    lab[:, g["wheel_off"]] &= lab[:, g["raised"]]
    lab[:, g["person_under_car"]] &= lab[:, g["raised"]]
    lab[:, g["person_at_car"]] &= ~lab[:, g["person_under_car"]]
    lab[:, g["work"]] &= lab[:, g["person_at_car"]] | lab[:, g["person_under_car"]]
    w = lab[:, g["washing"]]
    for s in ("raised", "wheel_off", "hood_open", "person_under_car"):
        lab[:, g[s]] &= ~w
    lab[:, g["idle"]] = car & ~lab[:, g["work"]] & ~w
    return lab


def _runs(x: np.ndarray) -> list[tuple[int, int, bool]]:
    out, a = [], 0
    for i in range(1, len(x) + 1):
        if i == len(x) or x[i] != x[a]:
            out.append((a, i, bool(x[a])))
            a = i
    return out


def prune_short_segments(times: np.ndarray, labels: np.ndarray, min_dur: dict[str, tuple[float, float]],
                         scale: float = DEFAULT_PRUNE_SCALE, observed: np.ndarray | None = None) -> np.ndarray:
    lab = labels.copy()
    T = len(times)
    if scale <= 0 or T < 3:
        return lab
    for name in PRUNE_ORDER:
        k = LIDX[name]
        if name not in min_dur or (observed is not None and not observed[k]):
            continue
        mn = (scale * min_dur[name][0], scale * min_dur[name][1])  # (on, off)
        for _ in range(T):
            cand = [(times[b] - times[a], a, b, v) for a, b, v in _runs(lab[:, k])
                    if a > 0 and b < T and times[b] - times[a] < (mn[0] if v else mn[1])]
            if not cand:
                break
            _, a, b, v = min(cand)
            lab[a:b, k] = not v
        lab = enforce_grammar(lab)
    return lab


def _expm(A: np.ndarray) -> np.ndarray:
    """Matrix exponential by scaling and squaring with a Taylor series (small dense generators only)."""
    n1 = np.abs(A).sum(1).max()
    s = int(np.ceil(np.log2(n1 / 0.25))) if n1 > 0.25 else 0
    A = A / (2.0 ** s)
    E = np.eye(len(A))
    term = np.eye(len(A))
    for k in range(1, 14):
        term = term @ A / k
        E = E + term
    for _ in range(s):
        E = E @ E
    E = np.clip(E, 0.0, None)
    return E / E.sum(1, keepdims=True)


class TransitionCache:
    def __init__(self, Q: np.ndarray, quantum_s: float = 0.05):
        self.Q, self.q = Q, quantum_s
        self._P: dict[int, np.ndarray] = {}
        self._logP: dict[int, np.ndarray] = {}

    def P(self, dt: float) -> np.ndarray:
        key = max(0, int(round(dt / self.q)))
        if key not in self._P:
            self._P[key] = np.eye(len(self.Q)) if key == 0 else _expm(self.Q * key * self.q)
        return self._P[key]

    def logP(self, dt: float) -> np.ndarray:
        key = max(0, int(round(dt / self.q)))
        if key not in self._logP:
            with np.errstate(divide="ignore"):
                self._logP[key] = np.maximum(np.log(self.P(dt)), -1e9)
        return self._logP[key]


def emission_loglik(P: np.ndarray, M: np.ndarray, eps: float = DEFAULT_EPS, weight=DEFAULT_WEIGHT,
                    Ymat: np.ndarray = Y) -> np.ndarray:
    """P, M: (T, K) probabilities and presence mask; weight: scalar or (K,). Returns (T, S) log-likelihoods."""
    p = eps / 2 + (1 - eps) * np.clip(np.nan_to_num(P, nan=0.5), 0.0, 1.0)
    w = np.broadcast_to(np.asarray(weight, dtype=np.float64), (P.shape[1],))
    L1 = np.where(M, np.log(p), 0.0) * w
    L0 = np.where(M, np.log(1 - p), 0.0) * w
    return L0.sum(1, keepdims=True) + (L1 - L0) @ Ymat.T


@dataclass
class Decoded:
    labels: np.ndarray  # (T, K) bool, decoded label sequences
    conf: np.ndarray  # (T, K) posterior probability that each label is on
    path: np.ndarray | None = None  # (T,) composite state index (HMM only)


def decode(times: np.ndarray, P: np.ndarray, M: np.ndarray, *, zone_kind: str | None = None,
           rates: str = "fitted", eps: float | None = None, weight: float | None = None,
           rates_path: str | None = None, prune_scale: float | None = None,
           label_weight: dict[str, float] | None = None) -> Decoded:
    """Viterbi path + forward-backward label marginals over the composite state space.

    rates: "fitted" (scenario-engine priors for this zone kind plus duration pruning; generic if no file) or
    "generic" (grammar only, no duration priors). Unconfigured zones (None / "frame" / "other", e.g. a jury clip)
    always use the generic grammar: the fitted priors describe service-centre zones only.
    label_weight: per-label emission weight overriding ``weight`` (e.g. detector-derived car_present).
    """
    T = len(times)
    eps = DEFAULT_EPS if eps is None else eps
    weight = DEFAULT_WEIGHT if weight is None else weight
    allQ = load_rates(rates_path)
    if zone_kind in (None, "frame", "other"):
        rates = "generic"
    if rates == "generic":
        Q = allQ["generic"]
    else:
        Q = allQ.get(zone_kind, allQ.get("pooled", allQ["generic"]))
    mask = zone_mask(zone_kind)
    idx = np.flatnonzero(mask)
    Qs = Q[np.ix_(idx, idx)].copy()
    np.fill_diagonal(Qs, 0.0)
    np.fill_diagonal(Qs, -Qs.sum(1))
    Ys = Y[idx]
    n = len(idx)
    cache = TransitionCache(Qs)
    wvec = np.full(K, float(weight))
    for name, wv in (label_weight or {}).items():
        wvec[LIDX[name]] = wv
    logE = emission_loglik(P, M, eps, wvec, Ys)
    # Start of a clip: empty zone and occupied zone equally likely; occupied configurations uniform.
    is_empty = Ys[:, LIDX["car_present"]] == 0
    pi = np.where(is_empty, 0.5 / max(1, is_empty.sum()), 0.5 / max(1, (~is_empty).sum()))
    log_pi = np.log(pi / pi.sum())

    # Viterbi
    back = np.zeros((T, n), dtype=np.int32)
    delta = log_pi + logE[0]
    for t in range(1, T):
        sc = delta[:, None] + cache.logP(times[t] - times[t - 1])
        back[t] = sc.argmax(0)
        delta = sc[back[t], np.arange(n)] + logE[t]
    path = np.zeros(T, dtype=np.int32)
    path[-1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]

    # Forward-backward (scaled, linear space) for posterior label marginals
    E = np.exp(logE - logE.max(1, keepdims=True))
    alpha = np.zeros((T, n))
    a = np.exp(log_pi) * E[0]
    alpha[0] = a / a.sum()
    for t in range(1, T):
        a = (alpha[t - 1] @ cache.P(times[t] - times[t - 1])) * E[t]
        s = a.sum()
        alpha[t] = a / s if s > 0 else np.full(n, 1.0 / n)
    beta = np.ones(n)
    post = np.zeros((T, n))
    post[-1] = alpha[-1]
    for t in range(T - 1, 0, -1):
        b = cache.P(times[t] - times[t - 1]) @ (E[t] * beta)
        s = b.sum()
        beta = b / s if s > 0 else np.ones(n)
        g = alpha[t - 1] * beta
        post[t - 1] = g / g.sum() if g.sum() > 0 else alpha[t - 1]
    labels = Ys[path].astype(bool)
    conf = post @ Ys
    scale = DEFAULT_PRUNE_SCALE if prune_scale is None else prune_scale
    if rates != "generic" and scale > 0:
        md = load_min_durations(rates_path)
        md = md.get(zone_kind, md.get("pooled", {}))
        if md:
            labels = prune_short_segments(times, labels, md, scale, M.any(0))
    return Decoded(labels=labels, conf=np.clip(conf, 0.0, 1.0), path=idx[path])


def hysteresis(P: np.ndarray, M: np.ndarray, on: float = 0.6, off: float = 0.4) -> Decoded:
    """Original baseline: per-label threshold with hysteresis; a missing value keeps the previous state."""
    T, Kk = P.shape
    lab = np.zeros((T, Kk), dtype=bool)
    conf = np.zeros((T, Kk))
    cur = np.zeros(Kk, dtype=bool)
    for t in range(T):
        for k in range(Kk):
            if M[t, k]:
                if not cur[k] and P[t, k] >= on:
                    cur[k] = True
                elif cur[k] and P[t, k] <= off:
                    cur[k] = False
        lab[t] = cur
        conf[t] = np.where(M[t], P[t], np.where(cur, 1.0, 0.0))
    return Decoded(labels=lab, conf=conf)


def segments(on: np.ndarray) -> list[tuple[int, int | None]]:
    """Runs of True in a bool array -> [(start_idx, end_idx_exclusive or None if still on at the end)]."""
    out: list[tuple[int, int | None]] = []
    start = None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, None))
    return out
