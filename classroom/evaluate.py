"""Measured accuracy against human labels.

Every number here is restricted to footage a human reviewed exhaustively. That
restriction is the whole design. Recall computed over the full recording when
only ten minutes were labelled is not a pessimistic estimate, it is a
meaningless one, and false-events-per-hour computed against a clock nobody
watched is worse -- it looks rigorous while measuring nothing.

Three deliberate choices in the scoring, each of which changes the headline
number:

  * **Benign motion is not a false positive.** A candidate stretching produces
    real motion and a motion-first system is right to surface it. Counting that
    as an error would reward a duller gate that also misses someone reaching
    into a bag. Those matches are reported separately as `dismissible` -- they
    cost a reviewer a few seconds each, and that cost is reported as review
    burden instead.

  * **Invisible actions are excluded.** An action labelled `insufficient`
    visibility is not in the footage in any recoverable sense. Scoring it caps
    the achievable ceiling below 100% for reasons that have nothing to do with
    the system. An event that fires on one anyway is not an error either -- it
    found something genuine -- so those land with the dismissible matches rather
    than the false positives.

  * **Matching is one-to-one.** A single system event that spans three separate
    labelled actions has found one of them, not three. Greedy assignment by
    descending overlap, each side consumed once.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from . import db
from .annotate import (
    Annotation, Span, load_annotations, load_spans, merge_spans, within_coverage,
)

# Overlap required to call a system event and a labelled action the same thing.
# 0.1 is low on purpose and reported alongside stricter values rather than
# instead of them: the product claim is "a reviewer is taken to the right
# moment", and an event covering the first third of an action satisfies that
# claim completely. Strict tIoU measures boundary agreement, which is a
# different and lesser question, so both are always printed.
DEFAULT_TIOU = 0.1
REPORTED_TIOU = (0.1, 0.3, 0.5)


@dataclass(frozen=True)
class SystemEvent:
    """One event as the pipeline emitted it."""

    id: int
    source_video_id: int
    t_start: float
    t_end: float
    score: float
    seat_label: str | None = None

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


def load_events(conn: sqlite3.Connection, source_video_id: int | None = None
                ) -> list[SystemEvent]:
    sql = (
        "SELECT e.id, e.source_video_id, e.t_start, e.t_end, e.score, s.seat_label "
        "FROM event e LEFT JOIN seat s ON s.id = e.seat_id"
    )
    params: tuple = ()
    if source_video_id is not None:
        sql += " WHERE e.source_video_id = ?"
        params = (source_video_id,)
    sql += " ORDER BY e.source_video_id, e.t_start"
    return [
        SystemEvent(r["id"], r["source_video_id"], r["t_start"], r["t_end"],
                    r["score"], r["seat_label"])
        for r in db.all_rows(conn, sql, params)
    ]


def temporal_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = min(a_end, b_end) - max(a_start, b_start)
    if inter <= 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def _seats_compatible(a: str | None, b: str | None) -> bool:
    """Whether two seat labels may refer to the same place.

    An unknown seat on either side is not evidence of a mismatch. Annotators
    legitimately record an action without being able to read the placard, and
    refusing those matches would turn an annotation limitation into a system
    error.
    """
    return a is None or b is None or a == b


@dataclass
class Match:
    event: SystemEvent
    annotation: Annotation
    tiou: float


@dataclass
class Result:
    """The measured outcome over one set of videos at one tIoU threshold."""

    tiou_threshold: float
    coverage_s: float
    matches: list[Match] = field(default_factory=list)
    missed: list[Annotation] = field(default_factory=list)
    false_events: list[SystemEvent] = field(default_factory=list)
    dismissible: list[Match] = field(default_factory=list)
    scored_annotations: int = 0
    events_considered: int = 0
    flagged_s: float = 0.0

    @property
    def recall(self) -> float | None:
        if self.scored_annotations == 0:
            return None
        return len(self.matches) / self.scored_annotations

    @property
    def precision(self) -> float | None:
        """Share of events that found a labelled action.

        Dismissible matches sit in neither numerator nor denominator: they are
        neither a hit on something of interest nor an error.
        """
        denominator = len(self.matches) + len(self.false_events)
        if denominator == 0:
            return None
        return len(self.matches) / denominator

    @property
    def false_events_per_hour(self) -> float | None:
        if self.coverage_s <= 0:
            return None
        return len(self.false_events) / (self.coverage_s / 3600.0)

    @property
    def mean_tiou(self) -> float | None:
        if not self.matches:
            return None
        return sum(m.tiou for m in self.matches) / len(self.matches)

    @property
    def review_burden(self) -> float | None:
        """Fraction of reviewed footage a human would have to watch.

        The actual product claim from PRD 3.5. A system with perfect recall that
        flags 90% of the recording has not helped anyone.
        """
        if self.coverage_s <= 0:
            return None
        return self.flagged_s / self.coverage_s


def match_events(
    events: list[SystemEvent],
    annotations: list[Annotation],
    spans: list[Span],
    tiou_threshold: float = DEFAULT_TIOU,
) -> Result:
    """Assign system events to labelled actions, one to one.

    Both sides are first restricted to reviewed footage. Pairs are then taken in
    descending overlap order, each event and each annotation consumed at most
    once, so a long event spanning several actions is credited with exactly one.
    """
    by_video: dict[int, list[tuple[float, float]]] = {}
    for span in spans:
        by_video.setdefault(span.source_video_id, []).append((span.t_start, span.t_end))
    merged = {
        video_id: merge_spans([Span(video_id, s, e, "x") for s, e in intervals])
        for video_id, intervals in by_video.items()
    }
    total_coverage = sum(
        sum(e - s for s, e in intervals) for intervals in merged.values()
    )

    covered_events = [
        e for e in events
        if within_coverage(e.t_start, e.t_end, merged.get(e.source_video_id, []))
    ]
    covered_annotations = [
        a for a in annotations
        if within_coverage(a.t_start, a.t_end, merged.get(a.source_video_id, []))
    ]

    scored = [a for a in covered_annotations if a.scored]
    # Everything labelled that is not in the recall denominator. Two kinds land
    # here and both must, or the system is punished for being right:
    # benign motion, and actions of real interest that the footage cannot
    # actually show. An `object_handling` marked `insufficient` is excluded from
    # recall because the system cannot be expected to find it -- but an event
    # that fires there has found something genuine, and calling that a false
    # positive would penalise the one behaviour the design asks for.
    excused = [a for a in covered_annotations if not a.scored]

    result = Result(
        tiou_threshold=tiou_threshold,
        coverage_s=total_coverage,
        scored_annotations=len(scored),
        events_considered=len(covered_events),
        flagged_s=sum(e.duration for e in covered_events),
    )

    pairs = [
        (temporal_iou(e.t_start, e.t_end, a.t_start, a.t_end), i, j)
        for i, e in enumerate(covered_events)
        for j, a in enumerate(scored)
        if e.source_video_id == a.source_video_id
        and _seats_compatible(e.seat_label, a.seat_label)
    ]
    pairs = [p for p in pairs if p[0] >= tiou_threshold]
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

    used_events: set[int] = set()
    used_annotations: set[int] = set()
    for tiou, i, j in pairs:
        if i in used_events or j in used_annotations:
            continue
        used_events.add(i)
        used_annotations.add(j)
        result.matches.append(Match(covered_events[i], scored[j], tiou))

    result.missed = [a for j, a in enumerate(scored) if j not in used_annotations]

    # Whatever is left over either lands on something labelled but excused --
    # real, correctly surfaced, outside the recall denominator -- or on nothing
    # at all. Only the latter is an error.
    leftover = [e for i, e in enumerate(covered_events) if i not in used_events]
    used_excused: set[int] = set()
    for event in leftover:
        best_tiou, best_index = 0.0, None
        for k, ann in enumerate(excused):
            if k in used_excused or ann.source_video_id != event.source_video_id:
                continue
            if not _seats_compatible(event.seat_label, ann.seat_label):
                continue
            overlap = temporal_iou(event.t_start, event.t_end,
                                   ann.t_start, ann.t_end)
            if overlap > best_tiou:
                best_tiou, best_index = overlap, k
        if best_index is not None and best_tiou >= tiou_threshold:
            used_excused.add(best_index)
            result.dismissible.append(Match(event, excused[best_index], best_tiou))
        else:
            result.false_events.append(event)

    result.matches.sort(key=lambda m: (m.event.source_video_id, m.event.t_start))
    return result


def precision_at_k(
    events: list[SystemEvent],
    annotations: list[Annotation],
    spans: list[Span],
    k: int,
    tiou_threshold: float = DEFAULT_TIOU,
) -> float | None:
    """Share of the top-k highest-scored events that found a labelled action.

    This is the number that describes the actual review experience: an
    investigator works down a ranked list and stops. Recall over the whole
    recording matters less than what the first screenful contains.
    """
    ranked = sorted(events, key=lambda e: -e.score)[:k]
    if not ranked:
        return None
    result = match_events(ranked, annotations, spans, tiou_threshold)
    if result.events_considered == 0:
        return None
    return len(result.matches) / result.events_considered


def by_visibility(result: Result) -> dict[str, tuple[int, int]]:
    """Recall split by how well each action could be seen.

    Reported because a single recall figure hides the only interesting part: a
    system that finds every clear action and no occluded one is behaving
    correctly and should not be averaged into mediocrity.
    """
    found: dict[str, int] = {}
    total: dict[str, int] = {}
    for match in result.matches:
        found[match.annotation.visibility] = found.get(match.annotation.visibility, 0) + 1
    for ann in [m.annotation for m in result.matches] + result.missed:
        total[ann.visibility] = total.get(ann.visibility, 0) + 1
    return {v: (found.get(v, 0), total[v]) for v in sorted(total)}


def by_event_type(result: Result) -> dict[str, tuple[int, int]]:
    found: dict[str, int] = {}
    total: dict[str, int] = {}
    for match in result.matches:
        found[match.annotation.event_type] = found.get(match.annotation.event_type, 0) + 1
    for ann in [m.annotation for m in result.matches] + result.missed:
        total[ann.event_type] = total.get(ann.event_type, 0) + 1
    return {t: (found.get(t, 0), total[t]) for t in sorted(total)}


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:5.1f}%"


def report(
    conn: sqlite3.Connection,
    source_video_id: int | None = None,
    thresholds: tuple[float, ...] = REPORTED_TIOU,
    top_k: tuple[int, ...] = (5, 10, 20),
) -> str:
    """Full measured report, or a clear statement of why there isn't one."""
    events = load_events(conn, source_video_id)
    annotations = load_annotations(conn, source_video_id)
    spans = load_spans(conn, source_video_id)

    coverage = sum(e - s for s, e in merge_spans(spans))
    lines = [
        "Measured accuracy",
        "=" * 64,
        f"reviewed footage   {coverage / 60:.1f} min across {len(spans)} spans",
        f"labelled actions   {len(annotations)}",
        f"system events      {len(events)}",
        "",
    ]

    if coverage <= 0 or not annotations:
        lines += [
            "No exhaustively reviewed footage is available, so nothing here can",
            "be measured. Recall has no denominator and false-events-per-hour",
            "has no clock. Annotate a span with tools/annotate_timeline.py and",
            "run this again.",
            "",
            "Per PRD 15.4, no accuracy claim may be presented until this",
            "section produces numbers.",
        ]
        return "\n".join(lines)

    for threshold in thresholds:
        result = match_events(events, annotations, spans, threshold)
        lines += [
            f"-- temporal IoU >= {threshold:.2f} " + "-" * 40,
            f"  recall            {_pct(result.recall)}"
            f"   ({len(result.matches)}/{result.scored_annotations})",
            f"  precision         {_pct(result.precision)}"
            f"   ({len(result.matches)}/"
            f"{len(result.matches) + len(result.false_events)})",
            f"  false events/hour "
            f"{'n/a' if result.false_events_per_hour is None else f'{result.false_events_per_hour:5.1f}'}",
            f"  dismissible       {len(result.dismissible)}"
            f"   (benign motion, correctly surfaced)",
            f"  mean tIoU         "
            f"{'n/a' if result.mean_tiou is None else f'{result.mean_tiou:.3f}'}",
            f"  review burden     {_pct(result.review_burden)}"
            f"   of reviewed footage flagged",
            "",
        ]

    result = match_events(events, annotations, spans, DEFAULT_TIOU)

    lines.append(f"-- recall by visibility (tIoU >= {DEFAULT_TIOU:.2f}) " + "-" * 22)
    for visibility, (found, total) in by_visibility(result).items():
        lines.append(f"  {visibility:<14} {found:>3}/{total:<3}  {_pct(found / total)}")
    lines.append("")

    lines.append(f"-- recall by action type (tIoU >= {DEFAULT_TIOU:.2f}) " + "-" * 21)
    for event_type, (found, total) in by_event_type(result).items():
        lines.append(f"  {event_type:<22} {found:>3}/{total:<3}  {_pct(found / total)}")
    lines.append("")

    lines.append("-- ranked review " + "-" * 47)
    for k in top_k:
        lines.append(f"  precision@{k:<3} {_pct(precision_at_k(events, annotations, spans, k))}")

    if result.missed:
        lines += ["", "-- missed actions " + "-" * 46]
        for ann in result.missed[:20]:
            seat = ann.seat_label or "?"
            lines.append(
                f"  video {ann.source_video_id}  seat {seat:<4} "
                f"{ann.t_start:8.2f}-{ann.t_end:8.2f}s  "
                f"{ann.event_type:<22} {ann.visibility}"
            )
        if len(result.missed) > 20:
            lines.append(f"  ... and {len(result.missed) - 20} more")

    return "\n".join(lines)
