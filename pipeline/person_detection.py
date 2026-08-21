"""Candidate-only person detection (PRD 9).

PRD 9 opens with the constraint that shapes this whole phase:

> Heavy pose cannot run on every person/frame. Full coverage comes from motion.

Full coverage is already done -- Phase 3 scanned every decodable window and
produced `candidate_manifest.jsonl`. This module does not re-derive candidates
and cannot widen them; it reads the frozen manifest and decides only *which
frames within an already-chosen interval* to look at. That separation is what
makes the later A/B honest: if a pose model looks better, it cannot be because
it was quietly handed a different set of frames.

Two things here are easy to get wrong and are handled explicitly.

**Sampling is a plan, not a side effect.** The sample plan is built and written
before a single frame is decoded, so the rows a detector saw are reconstructible
from the artifacts alone, and every configuration is given exactly those rows.

**Interpolated boxes are labelled.** PRD 9 permits "labelled tracker
interpolation" between sampled frames. A box that no detector ever produced must
never be indistinguishable from one that a detector did produce, so `origin`
travels with every box and the count of each is reported separately.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Where a box came from. The distinction is load-bearing: `detected` boxes are
# measurements, `interpolated` boxes are inferences between measurements, and a
# metric that mixes them is not measuring the detector.
DETECTED = "detected"
INTERPOLATED = "interpolated"
ORIGINS = (DETECTED, INTERPOLATED)


@dataclass
class SampleConfig:
    # How often to run the detector inside a candidate interval. 2 Hz is the
    # rate at which a seated person's posture changes meaningfully; faster costs
    # GPU time for frames that repeat, slower loses the reach that started the
    # event.
    sample_hz: float = 2.0

    # Context decoded either side of the event. An event's first frame is the
    # moment motion crossed a threshold, which is already partway into the
    # movement; without pre-roll the reviewer sees the follow-through only.
    pre_roll_ms: float = 1500.0
    post_roll_ms: float = 1500.0

    # A cap so one long event cannot consume the frame budget for a whole run.
    # Exceeding it is recorded on the plan rather than silently truncating.
    max_frames_per_event: int = 240

    # Detector score below which a box is not proposed at all.
    min_score: float = 0.30

    def interval_ms(self) -> float:
        return 1000.0 / self.sample_hz


@dataclass
class SampleRow:
    """One frame one detector must look at. Deterministic and frozen."""

    row_id: str
    # The event that first requested this timestamp, and every event it serves.
    # A frame is decoded and detected once no matter how many candidate
    # intervals overlap it; keeping the full list means an observation can still
    # be attributed to all of them without detecting the same frame twice.
    event_id: str
    event_ids: list[str] = field(default_factory=list)
    seat_id: str = ""
    source_video: str = ""
    source_video_sha256: str = ""
    pts_ms: float = 0.0
    calibration_version: str = ""
    camera_epoch: str = ""
    health_state: list[str] = field(default_factory=list)
    gate_state: str = "pass"
    # The seat-context rectangle from the candidate manifest. Detection runs on
    # the full frame; this is carried so a crop-limited configuration can be
    # compared against a full-frame one on identical rows.
    source_crop: dict | None = None

    def to_row(self) -> dict:
        return asdict(self)


@dataclass
class PersonBox:
    """One person observation, with its provenance attached."""

    row_id: str
    pts_ms: float
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    origin: str = DETECTED
    config_id: str = ""
    track_id: int | None = None

    def __post_init__(self) -> None:
        if self.origin not in ORIGINS:
            raise ValueError(
                f"unknown box origin {self.origin!r}; a box is either measured "
                f"({DETECTED}) or inferred ({INTERPOLATED}), never unlabelled")

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)

    def to_row(self) -> dict:
        return {**asdict(self), "area_px": round(self.area, 1)}


def row_id(video_sha: str, pts_ms: float) -> str:
    """A stable ID for one manifest row: one video, one timestamp.

    Derived from content, not from a counter, so two runs over the same video
    produce the same IDs and a comparison across runs is possible. PRD 9
    requires a deterministic ID; a sequence number would silently renumber
    everything the moment an earlier event changed.

    Deliberately **not** keyed on the event. A row is a frame to decode and
    detect, and two candidate intervals overlapping in time want the same frame,
    not two copies of it. Including the event produced exactly that: the same
    person detected twice at one timestamp, colliding pose row IDs downstream,
    and a tracker that saw duplicate boxes per frame and opened a spurious track
    for each -- 16 tracks over 3 events on real footage.
    """
    key = f"{video_sha}|{pts_ms:.3f}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def plan_continuous(duration_ms: float, source_video: str = "",
                    source_video_sha256: str = "",
                    calibration_version: str = "", camera_epoch: str = "",
                    cfg: SampleConfig | None = None
                    ) -> tuple[list[SampleRow], list[dict]]:
    """Sample the **whole recording**, not only the motion candidates.

    `plan()` above builds its rows from the candidate manifest, so every stage
    downstream inherits that gate. Measured on run 1446, the gate is severe:

        01_phone    46 events ->  4 merged spans covering  26.5s of 131.3s (20.2%)
        04_talking  38 events -> 16 merged spans covering  55.8s of 143.2s (39.0%)
        12_paper    29 events ->  8 merged spans covering  33.2s of  88.4s (37.5%)

    So **60-80% of every recording is never decoded for person detection**, and
    video 01 is only looked at between 53.6s and 102.9s. A candidate sitting
    still while using a phone under the desk produces little motion, raises no
    candidate, and is therefore never sampled at all -- no later stage can
    recover them, because the frames were never read.

    This planner ignores candidates entirely and lays a uniform grid across the
    full duration. Every person in every sampled frame gets detected, tracked,
    posed and measured, whether or not anything "interesting" was happening.
    That is what per-person metrics require: a baseline needs the ordinary
    frames far more than the exceptional ones.

    Rows carry `event_id="continuous"` and an empty `event_ids`, which is how a
    consumer tells a continuous row from an event-gated one.
    """
    cfg = cfg or SampleConfig()
    step = cfg.interval_ms()
    rows: list[SampleRow] = []
    pts = 0.0
    while pts <= duration_ms:
        rows.append(SampleRow(
            row_id=row_id(source_video_sha256, pts),
            event_id="continuous",
            event_ids=[],
            seat_id="",
            source_video=source_video,
            source_video_sha256=source_video_sha256,
            pts_ms=round(pts, 3),
            calibration_version=calibration_version,
            camera_epoch=camera_epoch,
            health_state=[],
            gate_state="pass",
            source_crop=None))
        pts += step

    notes = [{
        "event_id": "continuous",
        "planned_rows": len(rows),
        "sample_hz": cfg.sample_hz,
        "duration_ms": round(duration_ms, 1),
        "coverage": 1.0,
        "note": "uniform grid over the entire recording. Unlike the "
                "candidate-gated plan, no interval of the video is excluded, "
                "so a person who never triggers a motion candidate is still "
                "observed.",
    }]
    return rows, notes


def plan(candidates: list[dict], cfg: SampleConfig | None = None
         ) -> tuple[list[SampleRow], list[dict]]:
    """Build the frozen sample plan from the candidate manifest.

    Returns the rows and one note per event describing what was planned and
    whether the frame cap bit. Rows are deduplicated **by timestamp**, so two
    overlapping candidate intervals share one decode and one detection rather
    than producing two identical copies of the same person; each row records
    every event it serves. Rows are sorted by PTS so decoding is one forward
    pass.
    """
    cfg = cfg or SampleConfig()
    step = cfg.interval_ms()
    rows: dict[str, SampleRow] = {}
    notes: list[dict] = []

    for candidate in candidates:
        # A candidate row that carries reference annotations means the isolation
        # boundary has already been breached upstream; refuse rather than run.
        leaked = [k for k in candidate if "reference" in k.lower()]
        if leaked:
            raise ValueError(
                f"candidate row for {candidate.get('event_id')} carries "
                f"reference field(s) {leaked}; the reference manifest is "
                f"readable only at stage 12 (PRD 15)")

        start = float(candidate["start_pts_ms"]) - cfg.pre_roll_ms
        end = float(candidate["end_pts_ms"]) + cfg.post_roll_ms
        start = max(start, 0.0)

        wanted = int((end - start) // step) + 1
        capped = min(wanted, cfg.max_frames_per_event)

        pts = start
        made = 0
        while made < capped:
            rid = row_id(candidate.get("source_video_sha256", ""), pts)
            existing = rows.get(rid)
            if existing is not None:
                # Another event already claimed this frame. Record that this one
                # is served by it too, and move on without a second decode.
                if candidate["event_id"] not in existing.event_ids:
                    existing.event_ids.append(candidate["event_id"])
                pts += step
                made += 1
                continue
            rows[rid] = SampleRow(
                row_id=rid,
                event_id=candidate["event_id"],
                event_ids=[candidate["event_id"]],
                seat_id=candidate.get("seat_id", ""),
                source_video=candidate.get("source_video", ""),
                source_video_sha256=candidate.get("source_video_sha256", ""),
                pts_ms=round(pts, 3),
                calibration_version=candidate.get("calibration_version", ""),
                camera_epoch=candidate.get("camera_epoch", ""),
                health_state=list(candidate.get("health_state", [])),
                gate_state=candidate.get("gate_state", "pass"),
                source_crop=candidate.get("source_crop"),
            )
            pts += step
            made += 1

        notes.append({
            "event_id": candidate["event_id"],
            "seat_id": candidate.get("seat_id", ""),
            "requested_frames": wanted,
            "planned_frames": capped,
            "frame_cap_applied": capped < wanted,
            "sample_hz": cfg.sample_hz,
            "window_start_ms": round(start, 3),
            "window_end_ms": round(start + capped * step, 3),
            "reason": "frame cap reached; the interval is longer than one "
                      "configuration's budget and the tail is not sampled"
            if capped < wanted else "full interval sampled at the configured rate",
        })

    ordered = sorted(rows.values(), key=lambda r: r.pts_ms)
    return ordered, notes


# --------------------------------------------------------------- running --

@dataclass
class DetectionResult:
    """What one detector configuration produced over the frozen rows."""

    config_id: str
    state: str                       # a run_manifest model state
    boxes: list[PersonBox] = field(default_factory=list)
    rows_seen: int = 0
    rows_with_person: int = 0
    proposed: int = 0                # before the score threshold
    returned: int = 0                # after it
    seconds: float = 0.0
    peak_vram_mb: float | None = None
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.state == "available"

    def metrics(self) -> dict:
        per_row = self.returned / self.rows_seen if self.rows_seen else 0.0
        return {
            "config_id": self.config_id,
            "state": self.state,
            "reason": self.reason,
            "rows_seen": self.rows_seen,
            "rows_with_person": self.rows_with_person,
            "row_coverage": round(self.rows_with_person / self.rows_seen, 4)
            if self.rows_seen else 0.0,
            "persons_proposed": self.proposed,
            "persons_returned": self.returned,
            "persons_per_row": round(per_row, 3),
            "detected_boxes": sum(1 for b in self.boxes if b.origin == DETECTED),
            "interpolated_boxes": sum(1 for b in self.boxes
                                      if b.origin == INTERPOLATED),
            "seconds": round(self.seconds, 3),
            "rows_per_second": round(self.rows_seen / self.seconds, 2)
            if self.seconds > 0 else None,
            "peak_vram_mb": self.peak_vram_mb,
        }


def detect(rows: list[SampleRow], runner, config_id: str,
           cfg: SampleConfig | None = None) -> DetectionResult:
    """Run one person-detector configuration over the frozen rows.

    `runner` is any callable taking a `SampleRow` and returning an iterable of
    `(x1, y1, x2, y2, score)`. Keeping it a plain callable means the harness,
    the tests and a real ONNX session are the same code path -- PRD 17.13
    forbids an integration test that passes by skipping the model, and the
    surest way to honour that is for the test to exercise the same function.

    A runner that raises is recorded as `launch_failed` with the exception text.
    The rows already seen are kept: partial evidence is evidence, and discarding
    it would make the failure look like an absence of activity.
    """
    cfg = cfg or SampleConfig()
    result = DetectionResult(config_id=config_id, state="available")
    started = time.perf_counter()

    for row in rows:
        try:
            proposals = list(runner(row))
        except Exception as exc:                       # noqa: BLE001
            result.state = "launch_failed"
            result.reason = f"{type(exc).__name__}: {exc}"
            break

        result.rows_seen += 1
        result.proposed += len(proposals)
        kept = 0
        for x1, y1, x2, y2, score in proposals:
            if score < cfg.min_score:
                continue
            result.boxes.append(PersonBox(
                row_id=row.row_id, pts_ms=row.pts_ms,
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                score=round(float(score), 4), origin=DETECTED,
                config_id=config_id))
            kept += 1
        result.returned += kept
        if kept:
            result.rows_with_person += 1

    result.seconds = time.perf_counter() - started
    return result


def unavailable(config_id: str, state: str, reason: str) -> DetectionResult:
    """A configuration that could not run, kept in the comparison table.

    PRD 5: "A failed configuration stays in the comparison table as
    unavailable." Returning a result object rather than None is what makes that
    structural instead of a thing the report author must remember.
    """
    return DetectionResult(config_id=config_id, state=state, reason=reason)


def onnx_runner(model_path: str | Path, providers: list[str] | None = None,
                frame_reader=None, input_size: int = 640,
                confidence: float = 0.25):
    """Adapter over the measured `classroom.detect.ONNXDetector`.

    Built lazily so importing this module needs neither onnxruntime nor a
    checkpoint; a missing model surfaces as `missing_weights` at preflight, not
    as an ImportError halfway through a run.

    Person detection runs on the **whole frame**, not on the seat crop. A crop
    tight to one seat cuts a leaning neighbour in half, and the tracker then
    sees two people appear and disappear at the crop boundary. The seat crop's
    job is to bound the *object* search later, not the person search now.

    Boxes come back in source-frame coordinates via `CropTransform.to_frame`,
    because a box left in letterboxed input space is wrong by the pad offset and
    every overlay drawn from it still looks plausible.
    """
    from classroom.crops import extract
    from classroom.detect import ONNXDetector

    detector = ONNXDetector(str(model_path), providers=providers,
                            input_size=input_size)

    def run(row: SampleRow):
        if frame_reader is None:
            raise RuntimeError(
                "onnx_runner needs a frame_reader that maps a SampleRow to a "
                "decoded frame; without one the detector has nothing to see")
        frame = frame_reader(row)
        if frame is None:
            return []
        height, width = frame.shape[:2]
        canvas, transform = extract(frame, (0, 0, width, height), input_size)
        out = []
        for detection in detector(canvas, transform, confidence, row.pts_ms):
            if detection.cls != "person":
                continue
            out.append((*detection.box, detection.confidence))
        return out

    return run
