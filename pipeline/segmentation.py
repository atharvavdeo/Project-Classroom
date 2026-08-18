"""Event segmentation with recorded reasoning (PRD 8).

`QUIET -> CANDIDATE -> ACTIVE -> COOLING -> CLOSED`, one machine per seat-zone,
with every transition storing its reason, timestamp, scores, thresholds and
feature values. A boundary nobody can explain is a boundary nobody can review.

Five rules carry the correctness, and four of them are refusals:

**Starting requires motion, not a z-score.** PRD 8: "Start requires residual
motion and meaningful area fraction, not z-score alone." A near-zero baseline
turns a single block of compression noise into an enormous standardised value,
and that is exactly how an unoccupied desk row produced the twelve
highest-scoring events in a real recording.

**Bridging spans gaps in activity, never manufactures duration.** PRD 8 forbids
rolling-max smoothing outright. The bridge measures *time since activity was
last above the stop threshold*, so a genuine reach that pauses and resumes stays
one event, while a dropped pen never accumulates duration — there is no second
burst to bridge to.

**Nothing is bridged across a boundary.** Decode gap, blackout, uncertain camera
movement, epoch change, calibration change. On the far side of any of those, the
evidence is about a different thing; joining across one produces an event whose
middle nobody observed.

**Short and subtle candidates are kept.** PRD 8: "Keep short/subtle candidates
even if they do not activate; mark lower priority." A candidate that never
reached ACTIVE is retained as `low_priority` and remains searchable. Discarding
it would make a missed subtle event indistinguishable from a span that was never
examined.

**A split keeps its lineage.** Long events split at sustained troughs, quality
interruptions, epoch changes or the maximum duration, and each child names its
parent so the original extent is recoverable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

QUIET = "QUIET"
CANDIDATE = "CANDIDATE"
ACTIVE = "ACTIVE"
COOLING = "COOLING"
CLOSED = "CLOSED"


class State(str, Enum):
    QUIET = QUIET
    CANDIDATE = CANDIDATE
    ACTIVE = ACTIVE
    COOLING = COOLING
    CLOSED = CLOSED


# Why an event ended. Kept distinct because they mean different things to a
# reviewer: a merge-gap close is a complete action, a max-duration close is an
# arbitrary cut, and a boundary close is an observation that stopped.
CLOSE_QUIET = "activity_ceased"
CLOSE_MAX_DURATION = "maximum_duration_reached"
CLOSE_BOUNDARY = "boundary_reached"
CLOSE_TROUGH = "sustained_trough"
CLOSE_END_OF_STREAM = "end_of_stream"

PRIORITY_NORMAL = "normal"
PRIORITY_LOW = "low_priority"


@dataclass
class SegmentConfig:
    """Immutable thresholds (PRD 8, 17.10)."""

    # Hysteresis. Start is deliberately higher than stop: that gap is what stops
    # one action fragmenting into dozens of clips.
    start_z: float = 4.0
    stop_z: float = 2.0

    # Starting also requires real motion, not just an unusual number.
    start_min_area_ratio: float = 0.05
    start_min_moving_blocks: int = 1

    min_duration_ms: float = 1200.0
    candidate_bridge_ms: float = 1500.0
    merge_gap_ms: float = 2500.0
    max_duration_ms: float = 90_000.0

    # A trough this deep and this long inside an ACTIVE event splits it: two
    # actions separated by genuine stillness are two events.
    trough_z: float = 0.5
    trough_ms: float = 4000.0

    pre_roll_ms: float = 3000.0
    post_roll_ms: float = 3000.0


@dataclass
class Transition:
    """One state change, with everything that caused it (PRD 8)."""

    pts_ms: float
    seat_id: str
    zone: str
    from_state: str
    to_state: str
    reason: str
    deviation: float = 0.0
    area_ratio: float = 0.0
    moving_blocks: int = 0
    thresholds: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return asdict(self)


@dataclass
class Event:
    """One segmented event."""

    event_id: str
    seat_id: str
    zone: str
    start_pts_ms: float
    end_pts_ms: float
    peak_pts_ms: float
    peak_deviation: float
    priority: str = PRIORITY_NORMAL
    close_reason: str = CLOSE_QUIET
    parent_event_id: str | None = None
    windows: int = 0
    peak_area_ratio: float = 0.0
    below_desk_peak: float = 0.0

    @property
    def duration_ms(self) -> float:
        return self.end_pts_ms - self.start_pts_ms

    def to_row(self) -> dict:
        return {**asdict(self), "duration_ms": round(self.duration_ms, 3)}


class BoundaryPolicy:
    """Where an event may not be bridged or extended (PRD 8).

    Supplied by the caller so that segmentation does not need to know about
    health timelines, epochs or calibration versions — only that some instants
    are walls.
    """

    def __init__(self, walls: list[tuple[float, float, str]] | None = None):
        # (start_ms, end_ms, reason)
        self.walls = sorted(walls or [])

    def crossing(self, start_ms: float, end_ms: float) -> tuple[float, str] | None:
        """The first wall between two instants, if any."""
        for wall_start, wall_end, reason in self.walls:
            if wall_start < end_ms and wall_end > start_ms:
                return wall_start, reason
        return None


@dataclass
class _Builder:
    seat_id: str
    zone: str
    start_ms: float
    end_ms: float
    peak_ms: float
    peak_deviation: float
    peak_area: float
    below_desk: float
    windows: int
    last_active_ms: float

    def observe(self, window, deviation: float) -> None:
        self.end_ms = window.end_ms
        self.windows += 1
        if deviation > self.peak_deviation:
            self.peak_deviation = deviation
            self.peak_ms = window.start_ms
        self.peak_area = max(self.peak_area, window.area_ratio)
        self.below_desk = max(self.below_desk, window.below_desk)


def segment(scored, cfg: SegmentConfig | None = None,
            boundaries: BoundaryPolicy | None = None,
            event_prefix: str = "evt") -> tuple[list[Event], list[Transition]]:
    """Segment one run's scored windows into events, with a transition log.

    `scored` is `pipeline.baseline.Scored`. Windows are grouped per seat-zone and
    each group runs an independent machine, so simultaneous activity in adjacent
    seats never merges into one event.
    """
    cfg = cfg or SegmentConfig()
    boundaries = boundaries or BoundaryPolicy()

    grouped: dict[tuple[str, str], list] = {}
    for item in scored:
        grouped.setdefault((item.window.seat_id, item.window.zone), []).append(item)

    events: list[Event] = []
    transitions: list[Transition] = []
    counter = 0

    for (seat_id, zone), group in sorted(grouped.items()):
        group.sort(key=lambda s: s.window.start_ms)
        state = State.QUIET
        builder: _Builder | None = None
        cooling_since: float | None = None
        trough_since: float | None = None

        def log(pts_ms, from_state, to_state, reason, item=None):
            transitions.append(Transition(
                pts_ms=pts_ms, seat_id=seat_id, zone=zone,
                # `.value`, not `str()`. State is a str-Enum and `str()` yields
                # "State.ACTIVE" rather than "ACTIVE", which would put the
                # Python class name into every persisted transition record.
                from_state=getattr(from_state, "value", str(from_state)),
                to_state=getattr(to_state, "value", str(to_state)), reason=reason,
                deviation=round(item.deviation, 4) if item else 0.0,
                area_ratio=round(item.window.area_ratio, 5) if item else 0.0,
                moving_blocks=item.window.moving_blocks if item else 0,
                thresholds={"start_z": cfg.start_z, "stop_z": cfg.stop_z,
                            "start_min_area_ratio": cfg.start_min_area_ratio,
                            "min_duration_ms": cfg.min_duration_ms},
            ))

        def close(reason: str, end_ms: float | None = None) -> None:
            nonlocal builder, counter, state, cooling_since, trough_since
            if builder is None:
                return
            counter += 1
            # Priority follows *sustained activity*, not elapsed wall time.
            # Measuring elapsed time counts the quiet windows that follow a
            # spike, so a dropped pen accumulates duration from stillness.
            sustained = builder.last_active_ms - builder.start_ms
            priority = (PRIORITY_NORMAL if sustained >= cfg.min_duration_ms
                        else PRIORITY_LOW)
            events.append(Event(
                event_id=f"{event_prefix}_{seat_id}_{zone}_{counter:04d}",
                seat_id=seat_id, zone=zone,
                start_pts_ms=builder.start_ms,
                end_pts_ms=end_ms or builder.end_ms,
                peak_pts_ms=builder.peak_ms,
                peak_deviation=round(builder.peak_deviation, 4),
                priority=priority, close_reason=reason,
                windows=builder.windows,
                peak_area_ratio=round(builder.peak_area, 5),
                below_desk_peak=round(builder.below_desk, 5),
            ))
            log(end_ms or builder.end_ms, state, State.CLOSED, reason)
            builder = None
            state = State.QUIET
            cooling_since = trough_since = None

        for item in group:
            window = item.window
            deviation = item.deviation

            # Starting needs motion, not merely an unusual number (PRD 8).
            can_start = (deviation >= cfg.start_z
                         and window.area_ratio >= cfg.start_min_area_ratio
                         and window.moving_blocks >= cfg.start_min_moving_blocks)
            above_stop = deviation >= cfg.stop_z

            # A wall overlapping this window ends whatever is open: the evidence
            # on the far side is about something else.
            #
            # Checked against the window's own span, not the gap between
            # windows. Windows are contiguous, so `crossing(builder.end_ms,
            # window.start_ms)` is an empty interval that never matched
            # anything -- measured: an event ran straight through a decode gap
            # and reported itself as one continuous action.
            wall = boundaries.crossing(window.start_ms, window.end_ms)
            if wall is not None:
                if builder is not None:
                    end = wall[0] if wall[0] > builder.start_ms else builder.end_ms
                    close(f"{CLOSE_BOUNDARY}: {wall[1]}", end_ms=end)
                # Nothing starts inside a wall either.
                continue

            if state is State.QUIET:
                if can_start:
                    builder = _Builder(seat_id, zone, window.start_ms, window.end_ms,
                                       window.start_ms, deviation, window.area_ratio,
                                       window.below_desk, 1, window.end_ms)
                    state = State.CANDIDATE
                    log(window.start_ms, State.QUIET, State.CANDIDATE,
                        "deviation and area cleared the start thresholds", item)
                elif above_stop and item.suppressed:
                    # Recorded, not started. A suppressed window that would
                    # otherwise have qualified is exactly what a reviewer wants
                    # to know was seen and set aside.
                    log(window.start_ms, State.QUIET, State.QUIET,
                        f"above stop threshold but suppressed: {item.suppressed_by}",
                        item)
                continue

            assert builder is not None
            builder.observe(window, deviation)
            if above_stop:
                builder.last_active_ms = window.end_ms

            if state is State.CANDIDATE:
                # Duration is measured to the last ACTIVE window, never to the
                # current one. Measuring to `end_ms` counts the quiet windows
                # that follow a spike, so a dropped pen accumulates 1200 ms of
                # "persistence" out of stillness and promotes to ACTIVE — the
                # exact failure PRD 8 forbids, reappearing in new code. Caught
                # on the first smoke test: a single spike produced a 2000 ms
                # normal-priority event.
                sustained_ms = builder.last_active_ms - builder.start_ms
                if sustained_ms >= cfg.min_duration_ms:
                    state = State.ACTIVE
                    log(window.start_ms, State.CANDIDATE, State.ACTIVE,
                        f"sustained {sustained_ms:.0f} ms of activity, past the "
                        f"{cfg.min_duration_ms:.0f} ms minimum", item)
                elif not above_stop and \
                        window.end_ms - builder.last_active_ms > cfg.candidate_bridge_ms:
                    # Retained as low priority rather than discarded (PRD 8).
                    # It ends where activity ended, not where the bridge expired,
                    # so a retained candidate carries no tail of stillness a
                    # reviewer would have to sit through.
                    close(CLOSE_QUIET, end_ms=builder.last_active_ms)
                continue

            if state is State.ACTIVE:
                if builder.end_ms - builder.start_ms >= cfg.max_duration_ms:
                    close(CLOSE_MAX_DURATION)
                    continue
                if deviation < cfg.trough_z:
                    trough_since = trough_since or window.start_ms
                    if window.end_ms - trough_since >= cfg.trough_ms:
                        close(CLOSE_TROUGH, end_ms=trough_since)
                        continue
                else:
                    trough_since = None
                if not above_stop:
                    state = State.COOLING
                    cooling_since = window.end_ms
                    log(window.start_ms, State.ACTIVE, State.COOLING,
                        f"fell below stop threshold {cfg.stop_z}", item)
                continue

            if state is State.COOLING:
                if above_stop:
                    state = State.ACTIVE
                    cooling_since = trough_since = None
                    log(window.start_ms, State.COOLING, State.ACTIVE,
                        "activity resumed inside the merge gap", item)
                    continue
                # A trough is tracked here as well as in ACTIVE. Tracking it
                # only in ACTIVE meant a deep quiet stretch entered COOLING on
                # its first window and then closed as `activity_ceased`, so a
                # split that happened for a specific measurable reason was
                # reported to the reviewer as an ordinary end.
                if deviation < cfg.trough_z:
                    trough_since = trough_since or window.start_ms
                    if window.end_ms - trough_since >= cfg.trough_ms:
                        close(CLOSE_TROUGH, end_ms=trough_since)
                        continue
                if cooling_since is not None and \
                        window.end_ms - cooling_since > cfg.merge_gap_ms:
                    close(CLOSE_QUIET, end_ms=cooling_since)
                continue

        if builder is not None:
            close(CLOSE_END_OF_STREAM)

    events.sort(key=lambda e: (e.start_pts_ms, e.seat_id, e.zone))
    transitions.sort(key=lambda t: (t.pts_ms, t.seat_id, t.zone))
    return events, transitions


def split_long(event: Event, cfg: SegmentConfig) -> list[Event]:
    """Split an over-long event into children that name their parent.

    Used when an event survives to `max_duration_ms`; the machine closes it, and
    this records the lineage so the original extent stays recoverable rather
    than being silently replaced by a shorter one.
    """
    if event.duration_ms <= cfg.max_duration_ms:
        return [event]
    children: list[Event] = []
    start = event.start_pts_ms
    index = 0
    while start < event.end_pts_ms:
        end = min(start + cfg.max_duration_ms, event.end_pts_ms)
        index += 1
        children.append(Event(
            event_id=f"{event.event_id}_p{index}",
            seat_id=event.seat_id, zone=event.zone,
            start_pts_ms=start, end_pts_ms=end,
            peak_pts_ms=event.peak_pts_ms if start <= event.peak_pts_ms <= end
            else start,
            peak_deviation=event.peak_deviation,
            priority=event.priority, close_reason=CLOSE_MAX_DURATION,
            parent_event_id=event.event_id, windows=0,
            peak_area_ratio=event.peak_area_ratio,
            below_desk_peak=event.below_desk_peak,
        ))
        start = end
    return children


def summarise(events: list[Event], total_ms: float) -> dict:
    normal = [e for e in events if e.priority == PRIORITY_NORMAL]
    low = [e for e in events if e.priority == PRIORITY_LOW]
    flagged = sum(e.duration_ms for e in normal)
    return {
        "events": len(events),
        "normal_priority": len(normal),
        "low_priority_retained": len(low),
        "flagged_ms": round(flagged, 3),
        "candidate_fraction": round(flagged / total_ms, 5) if total_ms else None,
        "by_close_reason": {
            reason: sum(1 for e in events if e.close_reason.startswith(reason))
            for reason in (CLOSE_QUIET, CLOSE_MAX_DURATION, CLOSE_BOUNDARY,
                           CLOSE_TROUGH, CLOSE_END_OF_STREAM)
        },
    }
