"""Multi-object trackers for fixed cameras: OC-SORT (re-implemented here), a greedy-IoU baseline, and ByteTrack (supervision).

Design points
  * All timing is in *seconds*, so the same code runs at the gate (5-10 fps) and in the bay (0.5-1 fps): ``max_age_s`` and
    the Kalman velocity are converted with the observed frame interval.
  * OC-SORT = SORT + (OCM) velocity-direction consistency in the association cost, (OCR) a second pass that matches
    leftover detections to the *last observation* of unmatched tracks, (ORU) observation-centric re-update that replays a
    virtual linear trajectory through the occlusion gap when a lost track is found again (Cao et al., CVPR 2023; the
    reference code is MIT-licensed; this file is an independent re-implementation).
  * ``track_predictions(preds, tracker)`` sets ``Det.track_id`` on a list of ``FramePred`` (the pipeline contract).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from seen.predictor import Det, FramePred


# ------------------------------------------------------------------------------------------------ geometry
def iou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def _z(b: np.ndarray) -> np.ndarray:  # xyxy -> [cx, cy, s(area), r(aspect)]
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w * h, w / max(h, 1e-6)])


def _x2bbox(x: np.ndarray) -> np.ndarray:
    s, r = max(x[2], 1e-6), max(x[3], 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


class KalmanBox:
    """Constant-velocity Kalman filter on [cx, cy, s, r, vx, vy, vs] (SORT parametrisation), dt-aware."""

    def __init__(self, bbox: np.ndarray):
        self.x = np.zeros(7)
        self.x[:4] = _z(bbox)
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001])
        self.frozen_state = None

    def F(self, dt: float) -> np.ndarray:
        F = np.eye(7)
        F[0, 4] = F[1, 5] = F[2, 6] = dt
        return F

    def predict(self, dt: float = 1.0) -> np.ndarray:
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0
        F = self.F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q * max(dt, 1e-3)
        return _x2bbox(self.x)

    def update(self, bbox: np.ndarray | None) -> None:
        if bbox is None:
            return
        z = _z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def freeze(self):
        self.frozen_state = (self.x.copy(), self.P.copy())

    def unfreeze(self):
        if self.frozen_state is not None:
            self.x, self.P = self.frozen_state[0], self.frozen_state[1]
            self.frozen_state = None


def _direction(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ca, cb = (a[:2] + a[2:4]) / 2, (b[:2] + b[2:4]) / 2
    d = cb - ca
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.zeros(2)


class _Track:
    _next = 1

    def __init__(self, det: np.ndarray, t: float, delta_t: int):
        self.kf = KalmanBox(det[:4])
        self.id = _Track._next
        _Track._next += 1
        self.hits = 1
        self.t_last_obs = t
        self.last_obs = det[:4].copy()
        self.score = float(det[4])
        self.history_obs: list[tuple[float, np.ndarray]] = [(t, det[:4].copy())]
        self.delta_t = delta_t
        self.velocity = np.zeros(2)
        self.lost_since: float | None = None
        self.t_pred = t
        self.kf.freeze()

    def predict(self, t: float, dt_ref: float) -> np.ndarray:
        dt = (t - self.t_pred) / dt_ref  # in "reference frames"
        self.t_pred = t
        return self.kf.predict(dt)

    def observed_velocity(self) -> np.ndarray:
        # direction between observations ``delta_t`` steps apart (k-th previous observation), see OCM
        h = self.history_obs
        if len(h) < 2:
            return np.zeros(2)
        a = h[max(len(h) - 1 - self.delta_t, 0)][1]
        return _direction(a, h[-1][1])

    def update(self, det: np.ndarray, t: float, dt_ref: float):
        if self.lost_since is not None and len(self.history_obs):
            # ORU: replay a virtual trajectory from the last observation to the new one over the gap
            t0, b0 = self.history_obs[-1]
            n = max(int(round((t - t0) / dt_ref)), 1)
            self.kf.unfreeze()
            tt = t0
            for k in range(1, n):
                frac = k / n
                vb = b0 + (det[:4] - b0) * frac
                self.kf.predict(1.0)
                self.kf.update(vb)
            self.kf.predict(1.0)
            self.t_pred = t
        self.kf.update(det[:4])
        self.last_obs = det[:4].copy()
        self.t_last_obs = t
        self.score = float(det[4])
        self.history_obs.append((t, det[:4].copy()))
        if len(self.history_obs) > 30:
            self.history_obs.pop(0)
        self.hits += 1
        self.lost_since = None
        self.kf.freeze()  # state right after an observation: restored by ORU if the track is lost and found again


class OCSORT:
    """OC-SORT. ``update(dets, t_s)`` with dets (N,5) [x1,y1,x2,y2,score] -> (M,6) [x1,y1,x2,y2,id,score]."""

    name = "ocsort"

    def __init__(self, det_thresh: float = 0.35, max_age_s: float = 2.0, min_hits: int = 1, iou_threshold: float = 0.3,
                 delta_t: int = 3, inertia: float = 0.2, fps: float = 5.0, low_thresh: float = 0.1, use_byte: bool = True):
        self.det_thresh, self.max_age_s, self.min_hits = det_thresh, max_age_s, min_hits
        self.iou_threshold, self.delta_t, self.inertia = iou_threshold, delta_t, inertia
        self.dt_ref = 1.0 / fps
        self.low_thresh, self.use_byte = low_thresh, use_byte
        self.tracks: list[_Track] = []
        self.frame = 0

    def reset(self):
        self.tracks, self.frame = [], 0

    def _assoc(self, dets, preds, trk_dirs, last_obs, min_iou):
        if len(dets) == 0 or len(preds) == 0:
            return [], list(range(len(dets))), list(range(len(preds)))
        iou = iou_batch(dets[:, :4], preds)
        # OCM: penalise candidates whose direction disagrees with the track's observed motion
        cost = -iou
        if self.inertia > 0:
            for j, tr in enumerate(self.tracks_active):
                d = trk_dirs[j]
                if not np.any(d):
                    continue
                for i in range(len(dets)):
                    dd = _direction(last_obs[j], dets[i, :4])
                    if not np.any(dd):
                        continue
                    cosv = float(np.clip(np.dot(d, dd), -1, 1))
                    ang = np.arccos(cosv)
                    cost[i, j] -= self.inertia * (np.pi / 2 - abs(ang)) / np.pi * dets[i, 4] * (iou[i, j] > 0)
        r, c = linear_sum_assignment(cost)
        matches, um_d, um_t = [], set(range(len(dets))), set(range(len(preds)))
        for i, j in zip(r, c):
            if iou[i, j] >= min_iou:
                matches.append((i, j))
                um_d.discard(i)
                um_t.discard(j)
        return matches, sorted(um_d), sorted(um_t)

    def update(self, dets: np.ndarray, t_s: float) -> np.ndarray:
        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 5)
        self.frame += 1
        high = dets[dets[:, 4] >= self.det_thresh]
        low = dets[(dets[:, 4] < self.det_thresh) & (dets[:, 4] >= self.low_thresh)]
        self.tracks_active = self.tracks
        preds = np.array([tr.predict(t_s, self.dt_ref) for tr in self.tracks]).reshape(-1, 4)
        ok = ~np.isnan(preds).any(1) if len(preds) else np.zeros(0, dtype=bool)
        if len(preds) and not ok.all():
            self.tracks = [tr for tr, o in zip(self.tracks, ok) if o]
            preds = preds[ok]
            self.tracks_active = self.tracks
        dirs = [tr.observed_velocity() for tr in self.tracks]
        last_obs = [tr.last_obs for tr in self.tracks]
        matches, um_d, um_t = self._assoc(high, preds, dirs, last_obs, self.iou_threshold)
        for i, j in matches:
            self.tracks[j].update(high[i], t_s, self.dt_ref)
        # BYTE-style second association of low-score detections with still-unmatched tracks (optional, off in vanilla OC-SORT)
        if self.use_byte and len(low) and um_t:
            sub = [self.tracks[j] for j in um_t]
            iou = iou_batch(low[:, :4], preds[um_t])
            r, c = linear_sum_assignment(-iou)
            for i, j in zip(r, c):
                if iou[i, j] >= 0.5:
                    sub[j].update(low[i], t_s, self.dt_ref)
            um_t = [j for k, j in enumerate(um_t) if not any(c_ == k and iou[r_, c_] >= 0.5 for r_, c_ in zip(r, c))]
        # OCR: leftover detections vs last observation of leftover tracks
        if um_d and um_t:
            left_d = high[um_d]
            left_obs = np.array([self.tracks[j].last_obs for j in um_t])
            iou = iou_batch(left_d[:, :4], left_obs)
            if iou.max() >= self.iou_threshold:
                r, c = linear_sum_assignment(-iou)
                used_d, used_t = set(), set()
                for i, j in zip(r, c):
                    if iou[i, j] >= self.iou_threshold:
                        self.tracks[um_t[j]].update(left_d[i], t_s, self.dt_ref)
                        used_d.add(um_d[i])
                        used_t.add(um_t[j])
                um_d = [d for d in um_d if d not in used_d]
                um_t = [t for t in um_t if t not in used_t]
        for j in um_t:
            tr = self.tracks[j]
            if tr.lost_since is None:
                tr.lost_since = tr.t_last_obs
        for i in um_d:
            self.tracks.append(_Track(high[i], t_s, self.delta_t))
        out = []
        keep = []
        for tr in self.tracks:
            age_s = t_s - tr.t_last_obs
            if age_s > self.max_age_s:
                continue
            keep.append(tr)
            if tr.t_last_obs == t_s and (tr.hits >= self.min_hits):
                out.append([*tr.last_obs, tr.id, tr.score])
        self.tracks = keep
        return np.asarray(out, dtype=np.float64).reshape(-1, 6)


class IoUTracker:
    """Greedy/Hungarian IoU tracker, no motion model (baseline; also good for parked cars at 0.5-1 fps)."""

    name = "iou"

    def __init__(self, det_thresh: float = 0.35, max_age_s: float = 2.0, iou_threshold: float = 0.3, **_):
        self.det_thresh, self.max_age_s, self.iou_threshold = det_thresh, max_age_s, iou_threshold
        self.tracks: list[dict] = []
        self.next_id = 1

    def reset(self):
        self.tracks, self.next_id = [], 1

    def update(self, dets: np.ndarray, t_s: float) -> np.ndarray:
        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 5)
        dets = dets[dets[:, 4] >= self.det_thresh]
        boxes = np.array([t["box"] for t in self.tracks]).reshape(-1, 4)
        iou = iou_batch(dets[:, :4], boxes)
        matched_d, matched_t = set(), set()
        if iou.size:
            r, c = linear_sum_assignment(-iou)
            for i, j in zip(r, c):
                if iou[i, j] >= self.iou_threshold:
                    self.tracks[j].update(box=dets[i, :4], t=t_s, score=dets[i, 4])
                    matched_d.add(i)
                    matched_t.add(j)
        for i in range(len(dets)):
            if i not in matched_d:
                self.tracks.append({"id": self.next_id, "box": dets[i, :4], "t": t_s, "score": dets[i, 4]})
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t_s - t["t"] <= self.max_age_s]
        return np.array([[*t["box"], t["id"], t["score"]] for t in self.tracks if t["t"] == t_s]).reshape(-1, 6)


class ByteTrackWrapper:
    """supervision.ByteTrack (MIT) behind the same interface."""

    name = "bytetrack"

    def __init__(self, det_thresh: float = 0.35, max_age_s: float = 2.0, iou_threshold: float = 0.3, fps: float = 5.0, **_):
        import supervision as sv

        self.sv = sv
        self.kw = dict(track_activation_threshold=det_thresh, lost_track_buffer=max(int(max_age_s * fps), 1), minimum_matching_threshold=iou_threshold + 0.4,
                       frame_rate=max(int(round(fps)), 1), minimum_consecutive_frames=1)
        self.reset()

    def reset(self):
        self.bt = self.sv.ByteTrack(**self.kw)

    def update(self, dets: np.ndarray, t_s: float) -> np.ndarray:
        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 5)
        d = self.sv.Detections(xyxy=dets[:, :4].astype(np.float32).reshape(-1, 4), confidence=dets[:, 4].astype(np.float32),
                               class_id=np.zeros(len(dets), dtype=int))
        out = self.bt.update_with_detections(d)
        if len(out) == 0:
            return np.zeros((0, 6))
        return np.concatenate([out.xyxy, out.tracker_id[:, None], out.confidence[:, None]], 1)


def make_tracker(name: str, fps: float, **kw):
    if name == "ocsort":
        return OCSORT(fps=fps, **kw)
    if name == "iou":
        return IoUTracker(**kw)
    if name == "bytetrack":
        return ByteTrackWrapper(fps=fps, **kw)
    raise ValueError(name)


# ------------------------------------------------------------------------------------------------ pipeline glue
def track_predictions(preds: list[FramePred], tracker=None, classes: tuple[str, ...] = ("car", "truck", "bus", "motorcycle"),
                      image_size: tuple[int, int] | None = None) -> list[FramePred]:
    """Set ``Det.track_id`` for vehicle detections in a time-ordered list of ``FramePred`` (bbox normalised 0..1).

    The tracker runs on normalised coordinates scaled to a nominal 1000-px frame so IoU gates behave the same for every
    resolution. Frame interval is read from ``t_s`` so any fps works. Non-vehicle detections keep ``track_id=None``.
    """
    if tracker is None:
        fps = 1.0
        if len(preds) > 1:
            dts = np.diff([p.t_s for p in preds])
            fps = 1.0 / max(float(np.median(dts[dts > 0])) if (dts > 0).any() else 1.0, 1e-3)
        tracker = OCSORT(fps=fps)
    else:
        tracker.reset()
    S = 1000.0
    for p in preds:
        idx = [i for i, d in enumerate(p.detections) if d.cls in classes]
        arr = np.array([[*(np.asarray(p.detections[i].bbox) * S), p.detections[i].confidence] for i in idx]).reshape(-1, 5)
        out = tracker.update(arr, p.t_s)
        if len(out) and len(idx):
            iou = iou_batch(arr[:, :4], out[:, :4])
            for k, i in enumerate(idx):
                j = int(iou[k].argmax())
                if iou[k, j] > 0.9:
                    p.detections[i].track_id = int(out[j, 4])
    return preds
