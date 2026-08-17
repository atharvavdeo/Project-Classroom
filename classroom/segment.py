"""Phase 4 - per-seat event segmentation.

A Schmitt trigger over the seat's deviation signal: the threshold to *start* an
event is higher than the threshold to *stay in* one. That asymmetry is what
stops a single sustained action from shattering into a dozen clips, which is
the dominant failure mode of naive thresholding and the reason PS2 lists both
over- and under-segmentation as risks.

Each seat runs its own independent state machine, so two candidates acting at
the same time produce two overlapping events rather than one merged blob.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import Enum

from . import db
from .config import SegmentConfig
from .motion import ScoredWindow


class State(Enum):
    QUIET = "quiet"
    CANDIDATE = "candidate"
    ACTIVE = "active"
    COOLING = "cooling"


@dataclass
class Event:
    seat_label: str
    t_start: float
    t_end: float
    peak_t: float
    peak_deviation: float
    driving_zone: str
    below_desk_peak: float
    window_count: int

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


@dataclass
class _Builder:
    """Mutable accumulator for the event currently being assembled."""

    t_start: float
    t_end: float
    peak_t: float
    peak_deviation: float
    driving_zone: str
    below_desk_peak: float
    window_count: int

    def observe(self, window: ScoredWindow, z: float, zone: str, below: float) -> None:
        self.t_end = window.window.t_end
        self.window_count += 1
        self.below_desk_peak = max(self.below_desk_peak, below)
        if z > self.peak_deviation:
            self.peak_deviation = z
            self.peak_t = window.window.t_start
            self.driving_zone = zone

    def finish(self, seat_label: str, t_end: float | None = None) -> Event:
        return Event(
            seat_label=seat_label,
            t_start=self.t_start,
            t_end=t_end if t_end is not None else self.t_end,
            peak_t=self.peak_t,
            peak_deviation=self.peak_deviation,
            driving_zone=self.driving_zone,
            below_desk_peak=self.below_desk_peak,
            window_count=self.window_count,
        )


def _seat_signal(window: ScoredWindow, seat_label: str) -> tuple[float, str, float]:
    """Reduce a seat's zones to one score, remembering which zone drove it."""
    best_z = float("-inf")
    best_zone = "desk"
    below = 0.0
    for (label, zone), z in window.deviation.items():
        if label != seat_label:
            continue
        below = max(below, window.window.zones[(label, zone)].below_desk)
        if z > best_z:
            best_z, best_zone = z, zone
    if best_z == float("-inf"):
        return 0.0, "desk", 0.0
    return best_z, best_zone, below


def segment_seat(
    scored: list[ScoredWindow], seat_label: str, cfg: SegmentConfig
) -> list[Event]:
    """Run the state machine over one seat's timeline."""
    events: list[Event] = []
    state = State.QUIET
    builder: _Builder | None = None
    cooling_since = 0.0

    for window in scored:
        z, zone, below = _seat_signal(window, seat_label)
        t0, t1 = window.window.t_start, window.window.t_end

        if state is State.QUIET:
            if z >= cfg.start_z:
                builder = _Builder(t0, t1, t0, z, zone, below, 1)
                state = State.CANDIDATE

        elif state is State.CANDIDATE:
            assert builder is not None
            if z >= cfg.stop_z:
                builder.observe(window, z, zone, below)
                if builder.t_end - builder.t_start >= cfg.min_duration_s:
                    state = State.ACTIVE
            else:
                # Short impulse: a dropped pen, a compression artefact. Discarded
                # rather than promoted, but the motion window itself is retained
                # in the store for the reviewer.
                builder = None
                state = State.QUIET

        elif state is State.ACTIVE:
            assert builder is not None
            if z >= cfg.stop_z:
                builder.observe(window, z, zone, below)
                if builder.t_end - builder.t_start >= cfg.max_duration_s:
                    events.append(builder.finish(seat_label))
                    builder = None
                    state = State.QUIET
            else:
                cooling_since = t0
                state = State.COOLING

        elif state is State.COOLING:
            assert builder is not None
            if z >= cfg.stop_z:
                builder.observe(window, z, zone, below)
                state = State.ACTIVE
            elif t1 - cooling_since >= cfg.merge_gap_s:
                # End the event where activity actually stopped, not where the
                # cooling window expired.
                events.append(builder.finish(seat_label, t_end=cooling_since))
                builder = None
                state = State.QUIET

    if builder is not None and state in (State.ACTIVE, State.COOLING):
        end = cooling_since if state is State.COOLING else None
        events.append(builder.finish(seat_label, t_end=end))

    return events


def segment(scored: list[ScoredWindow], cfg: SegmentConfig) -> list[Event]:
    """Segment every seat present in the scan, ordered by start time."""
    labels = sorted({label for w in scored for (label, _) in w.deviation})
    events: list[Event] = []
    for label in labels:
        events.extend(segment_seat(scored, label, cfg))
    events.sort(key=lambda e: (e.t_start, e.seat_label))
    return events


def persist(
    conn: sqlite3.Connection,
    source_video_id: int,
    seat_ids: dict[str, int],
    events: list[Event],
) -> int:
    """Write events. Idempotent: replaces any prior segmentation of this video."""
    conn.execute("DELETE FROM event WHERE source_video_id = ?", (source_video_id,))
    rows = [
        (
            source_video_id, seat_ids[e.seat_label], e.t_start, e.t_end,
            e.peak_t, e.peak_deviation,
        )
        for e in events
        if e.seat_label in seat_ids
    ]
    db.insert_many(
        conn, "event",
        ["source_video_id", "seat_id", "t_start", "t_end", "peak_t", "peak_deviation"],
        rows,
    )
    db.audit(conn, "segment", "source_video", source_video_id,
             detail=f"{len(rows)} events")
    return len(rows)
