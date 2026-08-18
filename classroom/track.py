"""Phase 6 - tracking with mandatory seat reconciliation.

PRD 8.3: tracker IDs are session-local and non-biometric, and they are *not* the
system's identity truth. Seats are. A tracker ID that switches between two
neighbouring candidates would otherwise silently reassign evidence from one
person to another, which is the single most damaging failure this system can
produce.

So every track is reconciled to a calibrated seat polygon each frame, by majority
vote over the frames it appears in. Where the vote is not decisive the track is
reported as ambiguous rather than assigned, and the event carries that
uncertainty forward instead of resolving it arbitrarily.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np

from . import db
from .calibration import Calibration

# A track must spend at least this share of its frames in one seat before that
# seat is accepted. Below it the association is ambiguous.
SEAT_MAJORITY = 0.6

# Anchor used to place a person in a seat: the mid-point of the box bottom edge,
# which for a seated person sits at the chair rather than in the air above them.
ANCHOR_Y_RATIO = 0.92


@dataclass
class TrackObservation:
    track_id: int
    t: float
    box: tuple[float, float, float, float]
    seat_label: str | None = None


@dataclass
class TrackSummary:
    track_id: int
    seat_label: str | None          # None when no seat won a majority
    confidence: float               # share of frames in the winning seat
    frames: int
    ambiguous: bool
    seat_votes: dict[str, int] = field(default_factory=dict)

    @property
    def reconciled(self) -> bool:
        return self.seat_label is not None and not self.ambiguous


def anchor_point(box: tuple[float, float, float, float]) -> tuple[float, float]:
    """Where a detection 'stands' for the purpose of seat assignment."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y1 + (y2 - y1) * ANCHOR_Y_RATIO)


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Ray casting. Avoids a cv2 dependency for a few dozen points per frame."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < xin:
                inside = not inside
    return inside


def seat_for_box(
    box: tuple[float, float, float, float], cal: Calibration
) -> str | None:
    """Which calibrated seat this detection belongs to, if any."""
    point = anchor_point(box)
    for seat in cal.seats:
        if point_in_polygon(point, seat.polygon):
            return seat.label
    return None


def reconcile(
    observations: list[TrackObservation], cal: Calibration
) -> dict[int, TrackSummary]:
    """Assign each track to a seat by majority vote over its frames."""
    votes: dict[int, dict[str, int]] = {}
    totals: dict[int, int] = {}

    for obs in observations:
        label = obs.seat_label if obs.seat_label is not None else seat_for_box(obs.box, cal)
        obs.seat_label = label
        totals[obs.track_id] = totals.get(obs.track_id, 0) + 1
        if label is not None:
            bucket = votes.setdefault(obs.track_id, {})
            bucket[label] = bucket.get(label, 0) + 1

    summaries: dict[int, TrackSummary] = {}
    for track_id, frames in totals.items():
        bucket = votes.get(track_id, {})
        if not bucket:
            summaries[track_id] = TrackSummary(track_id, None, 0.0, frames, True, {})
            continue
        label, count = max(bucket.items(), key=lambda kv: kv[1])
        share = count / frames
        summaries[track_id] = TrackSummary(
            track_id=track_id,
            seat_label=label if share >= SEAT_MAJORITY else None,
            confidence=share,
            frames=frames,
            ambiguous=share < SEAT_MAJORITY,
            seat_votes=bucket,
        )
    return summaries


class SeatTracker:
    """ByteTrack over person boxes, reconciled to seats.

    Loaded lazily so the reconciliation logic above stays testable without
    supervision installed.
    """

    def __init__(self, cal: Calibration, frame_rate: int = 25):
        self.cal = cal
        self.frame_rate = frame_rate
        self._tracker = None
        self.observations: list[TrackObservation] = []

    def load(self) -> None:
        if self._tracker is not None:
            return
        import supervision as sv

        self._tracker = sv.ByteTrack(frame_rate=self.frame_rate)

    def reset(self) -> None:
        """Start a new session. Track IDs never span videos."""
        self._tracker = None
        self.observations = []

    def update(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        t: float,
    ) -> list[TrackObservation]:
        """Feed one frame of person boxes and return the tracked observations."""
        import supervision as sv

        self.load()
        if len(boxes) == 0:
            return []

        detections = sv.Detections(
            xyxy=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            confidence=np.asarray(confidences, dtype=np.float32).reshape(-1),
            class_id=np.zeros(len(boxes), dtype=int),
        )
        tracked = self._tracker.update_with_detections(detections)

        out: list[TrackObservation] = []
        for box, track_id in zip(tracked.xyxy, tracked.tracker_id):
            if track_id is None:
                continue
            obs = TrackObservation(int(track_id), t, tuple(float(v) for v in box))
            obs.seat_label = seat_for_box(obs.box, self.cal)
            out.append(obs)
        self.observations.extend(out)
        return out

    def summarize(self) -> dict[int, TrackSummary]:
        return reconcile(self.observations, self.cal)


def boxes_from_poses(
    people: list[np.ndarray], confidence_min: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Derive person boxes from pose keypoints.

    The pose model already localises every person, so no separate detector pass
    is needed to feed the tracker.
    """
    boxes, scores = [], []
    for kp in people:
        visible = kp[kp[:, 2] >= confidence_min]
        if len(visible) < 2:
            continue
        boxes.append([
            float(visible[:, 0].min()), float(visible[:, 1].min()),
            float(visible[:, 0].max()), float(visible[:, 1].max()),
        ])
        scores.append(float(visible[:, 2].mean()))
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32), np.asarray(scores, dtype=np.float32)


def persist(
    conn: sqlite3.Connection,
    source_video_id: int,
    summaries: dict[int, TrackSummary],
    seat_ids: dict[str, int],
) -> int:
    """Record track-to-seat association. Idempotent per video."""
    conn.execute("DELETE FROM track WHERE source_video_id = ?", (source_video_id,))
    rows = [
        (
            source_video_id, summary.track_id,
            seat_ids.get(summary.seat_label) if summary.seat_label else None,
            summary.confidence, summary.frames, int(summary.ambiguous),
        )
        for summary in summaries.values()
    ]
    db.insert_many(
        conn, "track",
        ["source_video_id", "tracker_id", "seat_id", "seat_confidence",
         "frames", "ambiguous"],
        rows,
    )
    db.audit(conn, "track", "source_video", source_video_id,
             detail=f"{len(rows)} tracks, "
                    f"{sum(1 for s in summaries.values() if s.ambiguous)} ambiguous")
    return len(rows)
