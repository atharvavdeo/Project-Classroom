"""Ground truth: the labels the system is measured against.

Kept structurally separate from every table the pipeline writes to. System
output and human labels never share a table, because an evaluation that reads
both from one place eventually scores the model against itself.

Three things are recorded, and all three are necessary:

  * **Spans** -- the stretches of footage a human actually watched end to end.
    This is the denominator. Three incidents labelled in a five-hour recording
    tells you nothing about recall unless you also know how much of those five
    hours was reviewed; without spans, an annotator who stopped after ten
    minutes and a system that missed everything afterwards look identical.

  * **Temporal annotations** -- what happened, when, at which seat, and how
    visible it was.

  * **Box annotations** -- object-level labels for the detector fine-tune,
    including explicit negatives.

The event taxonomy below deliberately includes categories the system is
*supposed* to surface and a reviewer is supposed to dismiss. `benign_movement`
and `staff_interaction` are not failures: a motion-first system that did not
fire on someone stretching would also not fire on someone reaching into a bag.
Labelling them separately is what allows precision to be reported honestly --
as "flagged, and a reviewer dismissed it in four seconds" rather than as a bare
false positive.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from . import db

# What a labelled interval can be. Ordered from most to least investigative
# interest; `unclear` is a first-class answer, not a gap in the taxonomy.
EVENT_TYPES = (
    "object_handling",        # hand on a small object that is not workstation equipment
    "neighbour_interaction",  # turning, leaning or passing toward another candidate
    "sustained_lookaway",     # head down or away well beyond the typing baseline
    "leaving_seat",
    "staff_interaction",      # invigilator at the desk: real motion, zero interest
    "benign_movement",        # stretch, drink, adjust clothing, scratch
    "unclear",                # visible motion whose nature cannot be determined
)

# How well the action can be seen. Anything at `insufficient` is excluded from
# recall scoring: a system cannot be marked down for missing what the footage
# does not contain, and including it would silently cap the achievable ceiling.
VISIBILITY = ("clear", "partial", "occluded", "insufficient")

# Box classes. The three confusers are here because they are what the stock
# weights actually got wrong on this footage (PRD 8.1.2), not because a generic
# classroom model would list them.
BOX_CLASSES = (
    "phone_like",
    "paper_like",
    "person",
    "mouse",
    "keyboard",
    "monitor",
    "other",
)

# Classes that exist only to be learned as negatives.
NEGATIVE_CLASSES = ("mouse", "keyboard", "monitor")

# The action types that count toward recall. Excludes the two benign categories:
# the system is not wrong to fire on a stretch, and scoring it as a miss when it
# does not would reward a less sensitive gate.
SCORED_TYPES = tuple(
    t for t in EVENT_TYPES if t not in ("benign_movement", "staff_interaction")
)


@dataclass(frozen=True)
class Span:
    """A stretch of footage reviewed exhaustively."""

    source_video_id: int
    t_start: float
    t_end: float
    annotator: str
    seat_label: str | None = None
    note: str | None = None

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def validate(self) -> None:
        if self.t_end <= self.t_start:
            raise ValueError(f"span ends before it starts: {self.t_start}-{self.t_end}")
        if not self.annotator:
            raise ValueError("a span must record who reviewed it")


@dataclass(frozen=True)
class Annotation:
    """One labelled action."""

    source_video_id: int
    t_start: float
    t_end: float
    event_type: str
    visibility: str
    seat_label: str | None = None
    annotator: str | None = None
    note: str | None = None

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def validate(self) -> None:
        if self.t_end <= self.t_start:
            raise ValueError(
                f"annotation ends before it starts: {self.t_start}-{self.t_end}"
            )
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event type {self.event_type!r}")
        if self.visibility not in VISIBILITY:
            raise ValueError(f"unknown visibility {self.visibility!r}")

    @property
    def scored(self) -> bool:
        """Whether this action counts toward recall."""
        return self.event_type in SCORED_TYPES and self.visibility != "insufficient"


@dataclass(frozen=True)
class Box:
    """One object-level label on an extracted still."""

    frame_path: str
    cls: str
    x1: float
    y1: float
    x2: float
    y2: float
    source_video_id: int | None = None
    t: float | None = None
    seat_label: str | None = None
    visibility: str = "clear"
    annotator: str | None = None

    @property
    def px_area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def negative(self) -> bool:
        return self.cls in NEGATIVE_CLASSES

    def validate(self) -> None:
        if self.cls not in BOX_CLASSES:
            raise ValueError(f"unknown box class {self.cls!r}")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"degenerate box on {self.frame_path}")
        if self.visibility not in VISIBILITY:
            raise ValueError(f"unknown visibility {self.visibility!r}")


# ------------------------------------------------------------------ storage --

def save_span(conn: sqlite3.Connection, span: Span) -> int:
    span.validate()
    row_id = db.insert(
        conn, "annotated_span",
        source_video_id=span.source_video_id, t_start=span.t_start,
        t_end=span.t_end, seat_label=span.seat_label,
        annotator=span.annotator, note=span.note,
    )
    db.audit(conn, "annotate.span", "annotated_span", row_id,
             f"{span.t_start:.2f}-{span.t_end:.2f}s", span.annotator)
    return row_id


def save_annotation(conn: sqlite3.Connection, ann: Annotation) -> int:
    ann.validate()
    row_id = db.insert(
        conn, "annotation",
        source_video_id=ann.source_video_id, seat_label=ann.seat_label,
        t_start=ann.t_start, t_end=ann.t_end, event_type=ann.event_type,
        visibility=ann.visibility, annotator=ann.annotator,
        exhaustive=1, note=ann.note,
    )
    db.audit(conn, "annotate.event", "annotation", row_id,
             f"{ann.event_type} {ann.t_start:.2f}-{ann.t_end:.2f}s",
             ann.annotator or "unknown")
    return row_id


def save_box(conn: sqlite3.Connection, box: Box) -> int:
    box.validate()
    return db.insert(
        conn, "box_annotation",
        source_video_id=box.source_video_id, frame_path=box.frame_path,
        t=box.t, seat_label=box.seat_label, cls=box.cls,
        negative=int(box.negative), x1=box.x1, y1=box.y1, x2=box.x2, y2=box.y2,
        px_area=box.px_area, visibility=box.visibility, annotator=box.annotator,
    )


def mark_frame_reviewed(
    conn: sqlite3.Connection, frame_path: str,
    source_video_id: int | None = None, t: float | None = None,
    annotator: str | None = None,
) -> None:
    """Record that a frame was looked at, whether or not it yielded boxes.

    Idempotent: re-reviewing a frame is normal and must not raise.
    """
    conn.execute(
        "INSERT OR REPLACE INTO reviewed_frame "
        "(frame_path, source_video_id, t, annotator) VALUES (?, ?, ?, ?)",
        (frame_path, source_video_id, t, annotator),
    )


def clear_boxes(conn: sqlite3.Connection, frame_path: str) -> None:
    """Drop existing boxes for a frame before rewriting it.

    Re-annotating a frame replaces its labels rather than accumulating a second
    copy of every box.
    """
    conn.execute("DELETE FROM box_annotation WHERE frame_path = ?", (frame_path,))


def load_spans(conn: sqlite3.Connection, source_video_id: int | None = None) -> list[Span]:
    sql = "SELECT * FROM annotated_span"
    params: tuple = ()
    if source_video_id is not None:
        sql += " WHERE source_video_id = ?"
        params = (source_video_id,)
    sql += " ORDER BY source_video_id, t_start"
    return [
        Span(r["source_video_id"], r["t_start"], r["t_end"], r["annotator"],
             r["seat_label"], r["note"])
        for r in db.all_rows(conn, sql, params)
    ]


def load_annotations(
    conn: sqlite3.Connection, source_video_id: int | None = None
) -> list[Annotation]:
    sql = "SELECT * FROM annotation"
    params: tuple = ()
    if source_video_id is not None:
        sql += " WHERE source_video_id = ?"
        params = (source_video_id,)
    sql += " ORDER BY source_video_id, t_start"
    return [
        Annotation(r["source_video_id"], r["t_start"], r["t_end"], r["event_type"],
                   r["visibility"], r["seat_label"], r["annotator"], r["note"])
        for r in db.all_rows(conn, sql, params)
    ]


def load_boxes(conn: sqlite3.Connection) -> list[Box]:
    return [
        Box(r["frame_path"], r["cls"], r["x1"], r["y1"], r["x2"], r["y2"],
            r["source_video_id"], r["t"], r["seat_label"], r["visibility"],
            r["annotator"])
        for r in db.all_rows(
            conn, "SELECT * FROM box_annotation ORDER BY frame_path, id")
    ]


def reviewed_frames(conn: sqlite3.Connection) -> list[str]:
    return [
        r["frame_path"] for r in db.all_rows(
            conn, "SELECT frame_path FROM reviewed_frame ORDER BY frame_path")
    ]


# ----------------------------------------------------------------- coverage --

def merge_spans(spans: list[Span]) -> list[tuple[float, float]]:
    """Collapse overlapping spans into disjoint intervals.

    Annotators revisit footage and overlapping passes are normal. Summing the
    raw durations would inflate the review clock and deflate every per-hour
    rate computed from it.
    """
    if not spans:
        return []
    ordered = sorted((s.t_start, s.t_end) for s in spans)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def coverage_s(spans: list[Span]) -> float:
    """Total distinct seconds reviewed."""
    return sum(end - start for start, end in merge_spans(spans))


def within_coverage(
    t_start: float, t_end: float, merged: list[tuple[float, float]]
) -> bool:
    """Whether an interval's midpoint falls inside reviewed footage.

    Midpoint rather than full containment: a system event legitimately runs a
    little past the end of a reviewed span, and discarding it there would
    penalise the system for where the annotator happened to stop.
    """
    mid = 0.5 * (t_start + t_end)
    return any(start <= mid <= end for start, end in merged)


# ------------------------------------------------------------- COCO export --

def to_coco(
    boxes: list[Box], reviewed: list[str], image_size: dict[str, tuple[int, int]],
    classes: tuple[str, ...] = BOX_CLASSES,
) -> dict:
    """Build a COCO detection set from box labels plus reviewed-but-empty frames.

    Every reviewed frame becomes an image, including those with no boxes. Those
    empty frames are the point: a background image carrying zero annotations is
    how a detector is taught that a computer mouse is not a phone, and this
    corpus's measured failure mode is false positives rather than misses.
    """
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(classes)]
    cat_id = {name: i + 1 for i, name in enumerate(classes)}

    paths = sorted(set(reviewed) | {b.frame_path for b in boxes})
    images, image_id = [], {}
    for i, path in enumerate(paths, start=1):
        image_id[path] = i
        width, height = image_size.get(path, (0, 0))
        if not width or not height:
            raise ValueError(f"no image size known for {path}")
        images.append({"id": i, "file_name": path, "width": width, "height": height})

    annotations = []
    for i, box in enumerate(boxes, start=1):
        box.validate()
        if box.cls not in cat_id:
            raise ValueError(f"class {box.cls!r} is not in the exported category set")
        annotations.append({
            "id": i,
            "image_id": image_id[box.frame_path],
            "category_id": cat_id[box.cls],
            "bbox": [box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1],
            "area": box.px_area,
            "iscrowd": 0,
        })

    return {"images": images, "annotations": annotations, "categories": categories}


def write_coco(path: str | Path, coco: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coco, indent=1), encoding="utf-8")
    return path


def dataset_summary(boxes: list[Box], reviewed: list[str]) -> str:
    """Human-readable census, printed before any training run starts.

    Stated plainly so that a fine-tune against twelve boxes is visibly a
    fine-tune against twelve boxes.
    """
    per_class: dict[str, int] = {}
    for box in boxes:
        per_class[box.cls] = per_class.get(box.cls, 0) + 1
    labelled = {b.frame_path for b in boxes}
    all_frames = set(reviewed) | labelled
    lines = [
        f"frames reviewed      {len(all_frames)}",
        f"  with boxes         {len(labelled)}",
        f"  background (empty) {len(all_frames - labelled)}",
        f"boxes                {len(boxes)}",
    ]
    for name in BOX_CLASSES:
        if name in per_class:
            tag = "   <- negative" if name in NEGATIVE_CLASSES else ""
            lines.append(f"  {name:<12} {per_class[name]:>5}{tag}")
    return "\n".join(lines)
