"""Zone geometry for the flow pipeline: point-in-polygon, box/polygon overlap, zone assignment, dwell time.

Pure numpy/stdlib (no model code). Coordinates are whatever the caller uses as long as boxes and polygons share
one frame: the pipeline uses normalised 0..1 (``Det.bbox``, ``Zone.polygon``); the slot-occupancy evaluation uses
pixels. A zone with no polygon means "whole frame" (``contracts`` Zone semantics).

Used by ``lanes/flow/eval_occupancy.py`` (PKLot / CNRPark-EXT slot occupancy from detector boxes) and by the visit
engine for gate/bay/wash/waiting zones.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

Point = tuple[float, float]
Polygon = Sequence[Sequence[float]]
Box = tuple[float, float, float, float]

FRAME_POLYGON: list[list[float]] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


# ------------------------------------------------------------------------------------------------ primitives
def polygon_area(poly: Polygon) -> float:
    """Unsigned area (shoelace)."""
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i][0], poly[i][1]
        x2, y2 = poly[(i + 1) % n][0], poly[(i + 1) % n][1]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def box_area(b: Box) -> float:
    return max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)


def box_polygon(b: Box) -> list[list[float]]:
    x1, y1, x2, y2 = b
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def point_in_polygon(pt: Point, poly: Polygon) -> bool:
    """Ray casting; points exactly on an edge count as inside. Empty/degenerate polygon -> False."""
    n = len(poly)
    if n < 3:
        return False
    x, y = pt
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        # on-edge test
        cross = (x - xi) * (yj - yi) - (y - yi) * (xj - xi)
        if abs(cross) < 1e-12 and min(xi, xj) - 1e-12 <= x <= max(xi, xj) + 1e-12 and min(yi, yj) - 1e-12 <= y <= max(yi, yj) + 1e-12:
            return True
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _clip_edge(pts: list[Point], inside, intersect) -> list[Point]:
    out: list[Point] = []
    if not pts:
        return out
    prev = pts[-1]
    for cur in pts:
        if inside(cur):
            if not inside(prev):
                out.append(intersect(prev, cur))
            out.append(cur)
        elif inside(prev):
            out.append(intersect(prev, cur))
        prev = cur
    return out


def clip_polygon_to_box(poly: Polygon, box: Box) -> list[Point]:
    """Sutherland-Hodgman clip of ``poly`` (any simple polygon) against an axis-aligned box."""
    x1, y1, x2, y2 = box
    pts: list[Point] = [(float(p[0]), float(p[1])) for p in poly]

    def ix(x):
        return lambda a, b: (x, a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]))

    def iy(y):
        return lambda a, b: (a[0] + (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]), y)

    pts = _clip_edge(pts, lambda p: p[0] >= x1, ix(x1))
    pts = _clip_edge(pts, lambda p: p[0] <= x2, ix(x2))
    pts = _clip_edge(pts, lambda p: p[1] >= y1, iy(y1))
    pts = _clip_edge(pts, lambda p: p[1] <= y2, iy(y2))
    return pts


def intersection_area(box: Box, poly: Polygon) -> float:
    return polygon_area(clip_polygon_to_box(poly, box))


@dataclass
class Overlap:
    inter: float
    frac_of_box: float  # inter / box area: how much of the detection lies in the zone
    frac_of_zone: float  # inter / zone area: how much of the zone the detection covers
    iou: float


def _empty(poly) -> bool:
    return poly is None or len(poly) == 0


def box_zone_overlap(box: Box, poly: Polygon | None) -> Overlap:
    """Overlap between a detection box and a zone polygon (None / empty = whole frame in the same units)."""
    if _empty(poly):
        a = box_area(box)
        return Overlap(a, 1.0 if a > 0 else 0.0, 0.0, 0.0)
    ba, za = box_area(box), polygon_area(poly)
    inter = intersection_area(box, poly)
    union = ba + za - inter
    return Overlap(inter, inter / ba if ba > 0 else 0.0, inter / za if za > 0 else 0.0, inter / union if union > 0 else 0.0)


# ------------------------------------------------------------------------------------------------ anchors
def anchor_point(box: Box, anchor: str = "foot") -> Point:
    """'foot' = bottom-centre (ground contact; right choice for people and vehicles on a floor), 'centre', 'top'."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    if anchor == "foot":
        return (cx, y2)
    if anchor == "centre":
        return (cx, (y1 + y2) / 2)
    if anchor == "top":
        return (cx, y1)
    raise ValueError(anchor)


def in_zone(box: Box, poly: Polygon | None, *, rule: str = "foot", min_frac_of_box: float = 0.5,
            min_frac_of_zone: float = 0.3) -> bool:
    """Decide whether a detection is "in" a zone.

    rule 'foot'/'centre': anchor point inside the polygon. 'box': at least ``min_frac_of_box`` of the box lies in
    the zone. 'slot': (parking slots) the box covers at least ``min_frac_of_zone`` of the zone AND at least
    ``min_frac_of_box`` of the box is inside; a neighbouring car overhanging a slot does not count.
    """
    if _empty(poly):
        return True
    if rule in ("foot", "centre", "top"):
        return point_in_polygon(anchor_point(box, rule), poly)
    o = box_zone_overlap(box, poly)
    if rule == "box":
        return o.frac_of_box >= min_frac_of_box
    if rule == "slot":
        return o.frac_of_zone >= min_frac_of_zone and o.frac_of_box >= min_frac_of_box
    raise ValueError(rule)


def slot_occupancy_score(boxes: Iterable[Box], slot_poly: Polygon) -> float:
    """Evidence that a parking slot is occupied: the largest fraction of the slot covered by any single box.

    Callers threshold this score (the threshold is calibrated on a labelled lot, see lanes/flow/eval_occupancy.py).
    A box that is mostly outside the slot (frac_of_box < 0.2) contributes nothing, which stops a car in the next
    slot (that overhangs this one by perspective) from marking it occupied.
    """
    best = 0.0
    for b in boxes:
        o = box_zone_overlap(b, slot_poly)
        if o.frac_of_box >= 0.2:
            best = max(best, o.frac_of_zone)
    return best


# ------------------------------------------------------------------------------------------------ zones
def _zone_poly(z: Any) -> Polygon | None:
    poly = getattr(z, "polygon", None) if not isinstance(z, dict) else z.get("polygon")
    return None if _empty(poly) else poly


def _zone_id(z: Any) -> str:
    return getattr(z, "id", None) if not isinstance(z, dict) else z.get("id")


def assign_to_zones(dets: Sequence[Any], zones: Sequence[Any], *, rule: str = "foot",
                    classes: Iterable[str] | None = None, **kw) -> dict[str, list[int]]:
    """{zone_id: [indices into dets]} for detections (``Det``-like: .cls, .bbox) inside each zone.

    ``zones`` are ``contracts.events.Zone`` objects or dicts with id/polygon; polygon absent = whole frame.
    """
    keep = set(classes) if classes is not None else None
    out: dict[str, list[int]] = {_zone_id(z): [] for z in zones}
    for zi, z in enumerate(zones):
        poly = _zone_poly(z)
        for i, d in enumerate(dets):
            if keep is not None and d.cls not in keep:
                continue
            if in_zone(tuple(d.bbox), poly, rule=rule, **kw):
                out[_zone_id(z)].append(i)
    return out


# ------------------------------------------------------------------------------------------------ dwell
@dataclass
class DwellEvent:
    kind: str  # "entered" | "left"
    key: Any  # track id (or any hashable)
    zone_id: str
    t_s: float
    dwell_s: float | None = None  # set on "left"


@dataclass
class DwellTracker:
    """Per (key, zone) presence with hysteresis, so a missed detection does not split one visit in two.

    ``update(t_s, present)`` takes the set of (key, zone_id) pairs seen at time ``t_s``. A pair "enters" once it
    has been seen for ``min_enter_s`` and "leaves" when it has not been seen for ``max_gap_s`` (the leave time is
    back-dated to the last sighting). Works at any frame rate; for the bay (0.5-1 fps) use max_gap_s >= 4.
    """

    max_gap_s: float = 3.0
    min_enter_s: float = 0.0
    _first: dict = field(default_factory=dict)
    _last: dict = field(default_factory=dict)
    _active: dict = field(default_factory=dict)  # pair -> t_enter
    history: list = field(default_factory=list)

    def update(self, t_s: float, present: Iterable[tuple[Any, str]]) -> list[DwellEvent]:
        events: list[DwellEvent] = []
        present = set(present)
        for pair in present:  # a pair that reappears after a long gap starts a new visit
            if pair in self._last and t_s - self._last[pair] > self.max_gap_s:
                if pair in self._active:
                    t_in = self._active.pop(pair)
                    events.append(DwellEvent("left", pair[0], pair[1], self._last[pair], self._last[pair] - t_in))
                del self._last[pair]
                self._first.pop(pair, None)
        for pair in present:
            self._first.setdefault(pair, t_s)
            self._last[pair] = t_s
            if pair not in self._active and t_s - self._first[pair] >= self.min_enter_s:
                self._active[pair] = self._first[pair]
                events.append(DwellEvent("entered", pair[0], pair[1], self._first[pair]))
        for pair in list(self._last):
            if pair in present:
                continue
            if t_s - self._last[pair] > self.max_gap_s:
                if pair in self._active:
                    t_in = self._active.pop(pair)
                    events.append(DwellEvent("left", pair[0], pair[1], self._last[pair], self._last[pair] - t_in))
                del self._last[pair]
                self._first.pop(pair, None)
        self.history.extend(events)
        return events

    def flush(self, t_s: float | None = None) -> list[DwellEvent]:
        """Close every open visit (end of stream); leave time = last sighting."""
        events = []
        for pair, t_in in list(self._active.items()):
            t_out = self._last.get(pair, t_in)
            events.append(DwellEvent("left", pair[0], pair[1], t_out, t_out - t_in))
        self._active.clear()
        self._last.clear()
        self._first.clear()
        self.history.extend(events)
        return events

    def dwell_now(self, key: Any, zone_id: str, t_s: float) -> float:
        t_in = self._active.get((key, zone_id))
        return 0.0 if t_in is None else t_s - t_in
