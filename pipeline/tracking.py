"""Tracker comparison, with the seat as the stable identifier (PRD 7, PRD 9).

PRD 7 states the principle this module is built around:

> Tracker ID is not student identity in an occluded 30-60-person room. Anonymous
> seat/desk context is the stable identifier. A track supports a seat only while
> association is sufficiently clear.

So a track here is a *continuity hint*, not a person. It exists to link the box
at 12.0 s to the box at 12.5 s so that a reach can be measured as one movement
rather than two unrelated observations. The moment a track's seat support gets
weak, the track stops supporting anything and the seat stands alone.

Three named configurations, as PRD 5 requires:

    tracker_primary       IoU with constant-velocity prediction; the working
                          default, and the one that produces interpolation
    tracker_challenger    ByteTrack via supervision, which recovers low-score
                          detections a single-threshold matcher drops
    seat_only_no_track    no association at all -- every box is independent and
                          the seat is the only continuity

`seat_only_no_track` is not a strawman. If it scores as well as the trackers on
this footage, that is the finding: tracking added nothing at this crowding and
resolution, and the extra machinery is not justified.

**On ID switches.** A true ID-switch count needs per-person ground-truth
identity, which this corpus does not have and which PRD 1 forbids us to create
by face or name. What is measurable is a *seat-inconsistent transition*: one
track ID whose seat attribution changes mid-track. It is reported under that
name, never as "ID switches", because the two are not the same quantity.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from .person_detection import DETECTED, INTERPOLATED, PersonBox

PRIMARY = "tracker_primary"
CHALLENGER = "tracker_challenger"
SEAT_ONLY = "seat_only_no_track"
CONFIGS = (PRIMARY, CHALLENGER, SEAT_ONLY)


@dataclass
class TrackConfig:
    # Minimum IoU between a prediction and a detection for them to be the same
    # person. In a dense hall neighbouring seated people overlap heavily, so a
    # permissive threshold swaps two adjacent candidates; 0.30 was chosen
    # against the measured box overlap between adjacent occupied seats.
    min_iou: float = 0.30

    # How long a track survives with no detection before it is closed. At the
    # 2 Hz sample rate this is two missed samples, which covers one person
    # leaning out of frame without keeping ghosts alive across an interval.
    max_age_ms: float = 1200.0

    # Gaps shorter than this are filled with labelled interpolated boxes; longer
    # gaps are left as holes. Inventing a trajectory across a two-second absence
    # is fabrication, and PRD 8 forbids inventing duration.
    max_interpolate_ms: float = 1000.0

    # A track shorter than this is a flicker, retained but marked.
    min_track_ms: float = 500.0


@dataclass
class Track:
    """One continuity hypothesis over a sequence of boxes."""

    track_id: int
    boxes: list[PersonBox] = field(default_factory=list)
    closed: bool = False

    @property
    def start_ms(self) -> float:
        return self.boxes[0].pts_ms if self.boxes else 0.0

    @property
    def end_ms(self) -> float:
        return self.boxes[-1].pts_ms if self.boxes else 0.0

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    @property
    def detected_count(self) -> int:
        return sum(1 for b in self.boxes if b.origin == DETECTED)

    def velocity(self) -> tuple[float, float]:
        """Per-millisecond drift of the box centre over the last two boxes.

        Two boxes, not a longer fit: a seated person's motion is not smooth, and
        a longer window makes the predictor confidently wrong when they stop.
        """
        if len(self.boxes) < 2:
            return (0.0, 0.0)
        a, b = self.boxes[-2], self.boxes[-1]
        dt = b.pts_ms - a.pts_ms
        if dt <= 0:
            return (0.0, 0.0)
        ax, ay = (a.x1 + a.x2) / 2.0, (a.y1 + a.y2) / 2.0
        bx, by = (b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0
        return ((bx - ax) / dt, (by - ay) / dt)

    def predict(self, pts_ms: float) -> tuple[float, float, float, float]:
        last = self.boxes[-1]
        vx, vy = self.velocity()
        dt = pts_ms - last.pts_ms
        return (last.x1 + vx * dt, last.y1 + vy * dt,
                last.x2 + vx * dt, last.y2 + vy * dt)

    def to_row(self) -> dict:
        return {
            "track_id": self.track_id,
            "start_pts_ms": round(self.start_ms, 3),
            "end_pts_ms": round(self.end_ms, 3),
            "duration_ms": round(self.duration_ms, 3),
            "boxes": len(self.boxes),
            "detected": self.detected_count,
            "interpolated": len(self.boxes) - self.detected_count,
        }


def iou(a: tuple[float, float, float, float],
        b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class TrackingResult:
    config_id: str
    state: str = "available"
    tracks: list[Track] = field(default_factory=list)
    boxes: list[PersonBox] = field(default_factory=list)   # with track_id set
    seconds: float = 0.0
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.state == "available"

    def metrics(self) -> dict:
        interpolated = sum(1 for b in self.boxes if b.origin == INTERPOLATED)
        durations = [t.duration_ms for t in self.tracks]
        return {
            "config_id": self.config_id,
            "state": self.state,
            "reason": self.reason,
            "tracks": len(self.tracks),
            "boxes": len(self.boxes),
            "detected_boxes": len(self.boxes) - interpolated,
            "interpolated_boxes": interpolated,
            "interpolated_fraction": round(interpolated / len(self.boxes), 4)
            if self.boxes else 0.0,
            "median_track_ms": round(sorted(durations)[len(durations) // 2], 1)
            if durations else None,
            "longest_track_ms": round(max(durations), 1) if durations else None,
            "seconds": round(self.seconds, 3),
        }


def track(boxes: list[PersonBox], config_id: str = PRIMARY,
          cfg: TrackConfig | None = None) -> TrackingResult:
    """Associate person boxes into tracks under one named configuration."""
    cfg = cfg or TrackConfig()
    if config_id == SEAT_ONLY:
        return _seat_only(boxes)
    if config_id == CHALLENGER:
        return _bytetrack(boxes, cfg)
    if config_id != PRIMARY:
        raise ValueError(f"unknown tracker configuration {config_id!r}; "
                         f"expected one of {CONFIGS}")
    return _iou_tracker(boxes, cfg)


def _seat_only(boxes: list[PersonBox]) -> TrackingResult:
    """The no-tracking baseline: every box stands alone.

    Deliberately not a degenerate case of the tracker. Each box gets no track ID
    at all rather than a unique one, because a unique-ID-per-box tracker would
    report a huge track count and look like a failure, when what it actually
    represents is a decision not to claim continuity.
    """
    started = time.perf_counter()
    out = [PersonBox(**{**asdict(b), "track_id": None}) for b in boxes]
    result = TrackingResult(config_id=SEAT_ONLY, boxes=out)
    result.seconds = time.perf_counter() - started
    return result


def _iou_tracker(boxes: list[PersonBox], cfg: TrackConfig) -> TrackingResult:
    started = time.perf_counter()
    result = TrackingResult(config_id=PRIMARY)

    by_pts: dict[float, list[PersonBox]] = {}
    for box in boxes:
        by_pts.setdefault(box.pts_ms, []).append(box)

    live: list[Track] = []
    next_id = 1

    for pts in sorted(by_pts):
        frame_boxes = by_pts[pts]

        # Close tracks that have gone quiet. Done before matching so a stale
        # track cannot claim a detection it has no business claiming.
        still_live = []
        for candidate in live:
            if pts - candidate.end_ms > cfg.max_age_ms:
                candidate.closed = True
                result.tracks.append(candidate)
            else:
                still_live.append(candidate)
        live = still_live

        # Greedy by descending IoU. Each track and each detection is used once,
        # so two overlapping neighbours cannot both be claimed by one track.
        pairs = []
        for ti, candidate in enumerate(live):
            predicted = candidate.predict(pts)
            for bi, box in enumerate(frame_boxes):
                score = iou(predicted, box.box)
                if score >= cfg.min_iou:
                    pairs.append((score, ti, bi))
        pairs.sort(reverse=True)

        used_tracks: set[int] = set()
        used_boxes: set[int] = set()
        for _score, ti, bi in pairs:
            if ti in used_tracks or bi in used_boxes:
                continue
            used_tracks.add(ti)
            used_boxes.add(bi)
            _extend(live[ti], frame_boxes[bi], cfg)

        for bi, box in enumerate(frame_boxes):
            if bi in used_boxes:
                continue
            fresh = Track(track_id=next_id)
            next_id += 1
            box.track_id = fresh.track_id
            fresh.boxes.append(box)
            live.append(fresh)

    for candidate in live:
        candidate.closed = True
        result.tracks.append(candidate)

    result.tracks.sort(key=lambda t: (t.start_ms, t.track_id))
    result.boxes = [b for t in result.tracks for b in t.boxes]
    result.boxes.sort(key=lambda b: (b.pts_ms, b.track_id or 0))
    result.seconds = time.perf_counter() - started
    return result


def _extend(candidate: Track, box: PersonBox, cfg: TrackConfig) -> None:
    """Add a detection to a track, filling a short gap with labelled boxes."""
    last = candidate.boxes[-1]
    gap = box.pts_ms - last.pts_ms
    box.track_id = candidate.track_id

    # A gap longer than one sample interval and shorter than the interpolation
    # limit is filled. The filled boxes are marked INTERPOLATED and carry the
    # row_id of neither endpoint, because no manifest row produced them.
    if 0 < gap <= cfg.max_interpolate_ms:
        steps = int(gap // 500) - 1
        for step in range(1, max(steps, 0) + 1):
            alpha = step / (steps + 1)
            candidate.boxes.append(PersonBox(
                row_id="",
                pts_ms=round(last.pts_ms + alpha * gap, 3),
                x1=last.x1 + alpha * (box.x1 - last.x1),
                y1=last.y1 + alpha * (box.y1 - last.y1),
                x2=last.x2 + alpha * (box.x2 - last.x2),
                y2=last.y2 + alpha * (box.y2 - last.y2),
                score=min(last.score, box.score),
                origin=INTERPOLATED,
                config_id=box.config_id,
                track_id=candidate.track_id))
    candidate.boxes.append(box)


def _bytetrack(boxes: list[PersonBox], cfg: TrackConfig) -> TrackingResult:
    """ByteTrack via supervision, or an honest unavailable record.

    Not silently replaced by the primary tracker when supervision is missing.
    PRD 5: "never silently substitute" -- a comparison table showing two
    identical result rows under different names is worse than one showing an
    absence.
    """
    try:
        import numpy as np
        import supervision as sv
    except Exception as exc:                            # noqa: BLE001
        return TrackingResult(
            config_id=CHALLENGER, state="resource_unavailable",
            reason=f"supervision/ByteTrack not importable ({type(exc).__name__}: "
                   f"{exc}); the challenger is unavailable, not substituted")

    started = time.perf_counter()
    result = TrackingResult(config_id=CHALLENGER)
    tracker = sv.ByteTrack()

    by_pts: dict[float, list[PersonBox]] = {}
    for box in boxes:
        by_pts.setdefault(box.pts_ms, []).append(box)

    tracks: dict[int, Track] = {}
    for pts in sorted(by_pts):
        frame_boxes = by_pts[pts]
        detections = sv.Detections(
            xyxy=np.array([b.box for b in frame_boxes], dtype=np.float32),
            confidence=np.array([b.score for b in frame_boxes], dtype=np.float32),
            class_id=np.zeros(len(frame_boxes), dtype=int))
        tracked = tracker.update_with_detections(detections)

        for xyxy, track_id in zip(tracked.xyxy, tracked.tracker_id):
            if track_id is None:
                continue
            # Match back to the source box so provenance survives the tracker.
            source = max(frame_boxes,
                         key=lambda b: iou(b.box, tuple(float(v) for v in xyxy)))
            source.track_id = int(track_id)
            candidate = tracks.setdefault(int(track_id), Track(int(track_id)))
            if candidate.boxes:
                _extend(candidate, source, cfg)
            else:
                candidate.boxes.append(source)

    result.tracks = sorted(tracks.values(), key=lambda t: (t.start_ms, t.track_id))
    result.boxes = [b for t in result.tracks for b in t.boxes]
    result.boxes.sort(key=lambda b: (b.pts_ms, b.track_id or 0))
    result.seconds = time.perf_counter() - started
    return result


def continuity(result: TrackingResult, seat_of: dict[str, str] | None = None
               ) -> dict:
    """Continuity measures, including the seat-inconsistency proxy.

    `seat_of` maps a box key to the seat that box was attributed to. Where it is
    supplied, a track whose attributed seat changes mid-track is counted as a
    seat-inconsistent transition -- the honest, measurable stand-in for an ID
    switch on a corpus with no identity ground truth.
    """
    out = {
        "config_id": result.config_id,
        "state": result.state,
        "tracks": len(result.tracks),
        "short_tracks": sum(1 for t in result.tracks if t.duration_ms < 500.0),
        "fragments_per_1000ms": None,
        "seat_inconsistent_transitions": None,
        "seat_inconsistency_note":
            "no seat attribution was supplied, so seat consistency is unmeasured. "
            "This is not a zero-switch result.",
    }
    span = sum(t.duration_ms for t in result.tracks)
    if span > 0:
        out["fragments_per_1000ms"] = round(len(result.tracks) / span * 1000.0, 4)

    if seat_of is not None:
        switches = 0
        for candidate in result.tracks:
            seen = []
            for box in candidate.boxes:
                seat = seat_of.get(f"{box.track_id}|{box.pts_ms:.3f}")
                if seat and (not seen or seen[-1] != seat):
                    seen.append(seat)
            switches += max(len(seen) - 1, 0)
        out["seat_inconsistent_transitions"] = switches
        out["seat_inconsistency_note"] = (
            "one track ID whose attributed seat changed mid-track. This is a "
            "proxy for an ID switch, not an ID-switch count: this corpus has no "
            "per-person identity ground truth and PRD 1 forbids creating one.")
    return out
