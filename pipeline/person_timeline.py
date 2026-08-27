"""One continuous record per person, over the whole recording.

### Why this exists

Everything before this measured *events*. `person_detection.plan()` builds its
rows from the motion candidate manifest, so pose, head orientation and object
crops all inherit that gate. Measured on run 1446:

    01_phone     26.5s of 131.3s sampled (20.2%)
    04_talking   55.8s of 143.2s sampled (39.0%)
    12_paper     33.2s of  88.4s sampled (37.5%)

**60-80% of every recording was never decoded for person detection.** A
candidate who sits still while using a phone under the desk generates little
motion, raises no candidate, and is therefore never looked at -- and no later
stage can recover them, because the frames were never read. For a three-hour
recording that gap grows.

It also made per-person metrics impossible. A metric like "this candidate turns
right more than they usually do" needs the *ordinary* frames far more than the
exceptional ones, and the ordinary frames were exactly what the gate discarded.

This module builds the other half: a continuous, per-person timeline sampled
uniformly across the entire video, for **every** person in **every** sampled
frame, whether or not anything interesting is happening.

### What it produces

For each track, a `PersonTimeline` holding one `PersonSample` per sighting --
box, pose keypoints, head yaw/pitch/roll, facing state, wrist positions, and
any object detected near that person's hands. Then a `PersonProfile`
aggregating the whole track: how long they were observed, their resting head
orientation and its spread, how far they departed from it and when, how much
time they spent facing away from their own workstation, and whether they left
frame and returned.

### What it deliberately does not do

It does not decide that anyone cheated. A profile is a description of a
person's observed behaviour over the recording, with the ordinary and the
unusual measured on the same scale, so a reviewer can see which is which.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field


@dataclass
class TimelineConfig:
    """Thresholds for describing a person's record. All relative."""

    # Departure from this person's own resting yaw, in normalised units, past
    # which a sample counts as "attention away from their workstation". Matches
    # `HeadPoseConfig.min_departure_from_baseline` so the two stages agree.
    departure: float = 0.30

    # A gap longer than this in a person's sightings is an absence, not a
    # missed frame. Expressed in multiples of the sampling interval so it
    # transfers between sample rates.
    absence_intervals: float = 4.0

    # An absence shorter than this is a tracking dropout, not the person
    # leaving. Distinguishing them matters: one is a data-quality fact about
    # us, the other is an observation about them.
    min_absence_ms: float = 2000.0

    # Minimum sightings before a profile reports a baseline at all. A median
    # over three samples is not a resting posture.
    min_samples_for_baseline: int = 8

    # Object centre within this many torso widths of a wrist counts as "near
    # this person's hands". Same ruler as `flag_handheld`.
    near_hand_torso: float = 2.0

    # Minimum detector score before an attachment counts toward a flag.
    #
    # Left at 0.0 the pipeline reports everything, which is what produced the
    # seat-placard false positives: D-FINE calls the numbered cards on the wall
    # cell phones at 0.30-0.49, against 0.933 for the one phone verified by eye.
    # Raising this to ~0.6 removes that whole class of error without touching
    # the true positive. It is a decision about what to report, not a
    # measurement, so it stays configurable and defaults to reporting
    # everything -- and because confidence is now stored per attachment,
    # changing it is a `tools/reprofile.py` run, not a re-detection.
    min_object_confidence: float = 0.0

    # A class attached in more than this fraction of a person's sightings is
    # treated as a fixed object beside that seat rather than something they
    # picked up. Persistence is not corroboration for a handheld object -- the
    # same argument `flag_recurrent` makes in image space, applied here in a
    # person's own timeline.
    #
    # Measured on the paper video before this rule: 1,032 "phone" attachments,
    # with 18 of 45 tracks carrying one in over half their sightings, and rates
    # of 1.62 and 3.00 per sighting -- arithmetically impossible for a single
    # held phone. Flags jumped from 4 people to 32.
    persistent_object_rate: float = 0.50

    # Below this rate an attachment is too rare to report on its own; a single
    # frame of a mouse read as a phone should not raise anything.
    min_object_rate: float = 0.02

    # --- Dashboard orientation labels ---------------------------------------
    # These are deliberately presentation classifications, not a second
    # verdict path. They are measured directly from pose and retained so the
    # dashboard can distinguish ordinary desk work from the few spans that
    # deserve contextual review.
    looking_up_pitch_max: float = -0.20
    shallow_down_pitch_min: float = 0.20
    deep_down_pitch_min: float = 0.45
    deep_down_min_ms: float = 15000.0

    # `yaw_relative_to_torso` is a shoulder-normalised proxy: it stops pinning
    # at the clamp (0.1% of resolved frames against 28.8% for raw yaw), but on
    # the frames where raw yaw was one of its two constants it varies with
    # shoulder squareness rather than with neck rotation -- see the field
    # comment on PersonSample. So the departure below is measured against the
    # subject's own resting baseline, never read as an angle.
    torso_relative_turn_departure: float = 0.30
    torso_relative_turn_min_ms: float = 4000.0
    orientation_merge_intervals: float = 2.5


@dataclass
class PersonSample:
    """One sighting of one person at one timestamp."""

    track_id: int
    pts_ms: float
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    seat_state: str = ""
    torso_scale_px: float | None = None

    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    facing: str = ""
    head_scale_px: float = 0.0

    # Head yaw measured against the shoulder line rather than against head
    # width. `head_pose.py` has always computed this and stage 14 discarded it,
    # so nothing downstream could use it.
    #
    # It matters because plain `yaw` carries no magnitude on roughly half the
    # frames. Recomputed from the stored joints of runs 1511/01_phone,
    # 1510/12_paper and 1509/04_talking -- 5,112 yaw-resolved samples:
    #
    #   |yaw| == 1.5 (the clamp)              1,471   28.8%
    #   |yaw| == 1.0 exactly (one-ear branch)   969   19.0%
    #   ------------------------------------------------------
    #   no magnitude available                2,440   47.7%
    #
    # The second row is the one that misleads. When a turn occludes an ear,
    # `head_pose.estimate` reports a *constant* +/-1.0 naming the direction,
    # because the geometry that would give a magnitude is the occluded part.
    # A slight glance and a full turn then score identically.
    #
    # What this field fixes, and what it does not:
    #   - it resolves on 4,350 of those 5,112 frames (85.1%) and is at the
    #     clamp on 0.1% of them, so the scale stops pinning; but
    #   - on 43.2% of them the underlying yaw was one of the two constants
    #     above, and `head_pose` computes this as `yaw * shoulder squareness`.
    #     There the number varies with SHOULDER geometry -- whole-body
    #     rotation -- not with how far the neck turned.
    #
    # That is still the right signal for `turned_and_held`, which is about a
    # sustained departure from a subject's own resting posture rather than an
    # angle. It is not a gaze measurement and must never be reported as one.
    #
    # Carried as None when the shoulders were not resolvable, never as 0.0.
    yaw_relative_to_torso: float | None = None

    # Frame-level pitch band. It is descriptive only: the two ordinary bands
    # never enter the review queue, and the sustained/turn labels are produced
    # later from PTS-continuous spans in `profile()`.
    orientation_state: str = "not_assessable"

    left_wrist: tuple[float, float] | None = None
    right_wrist: tuple[float, float] | None = None
    wrist_below_desk: bool = False

    # Objects the detector reported near this person's hands at this instant,
    # as {"cls", "confidence", "box", "source"}.
    #
    # This used to hold bare class names, which made the attachments
    # undrawable and unauditable: the annotated video had nothing to render,
    # and checking why a flag fired meant re-running the detector to recover
    # the boxes. Keeping the geometry and the score means a confidence floor
    # can be applied later by reprofiling instead of by a GPU re-run.
    near_hand_objects: list = field(default_factory=list)

    # Every resolved keypoint, as {name: [x, y, confidence]}. Carried so a
    # renderer can draw the actual skeleton rather than a box and an arrow --
    # a reviewer judging "is that person turned toward their neighbour" needs
    # to see the shoulders the yaw was measured against.
    joints: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


@dataclass
class PersonProfile:
    """Everything observed about one person across the whole recording."""

    track_id: int
    seat_state: str

    samples: int = 0
    first_seen_ms: float = 0.0
    last_seen_ms: float = 0.0
    observed_span_ms: float = 0.0
    # Span divided by the recording's duration. A person seen for 12% of the
    # video has a profile built on 12% of the evidence, and every number below
    # must be read in that light.
    coverage_of_recording: float = 0.0

    resolvable_samples: int = 0
    median_head_scale_px: float = 0.0

    baseline_yaw: float | None = None
    baseline_yaw_spread: float | None = None
    baseline_pitch: float | None = None
    baseline_pitch_spread: float | None = None

    baseline_yaw_relative_to_torso: float | None = None
    baseline_yaw_relative_to_torso_spread: float | None = None

    # Time spent departed from their own resting orientation, by direction.
    samples_turned_left: int = 0
    samples_turned_right: int = 0
    samples_looking_down: int = 0
    fraction_away_from_baseline: float = 0.0

    # Absences: gaps in sightings long enough to be the person leaving rather
    # than the tracker dropping them.
    absences: list = field(default_factory=list)
    longest_absence_ms: float = 0.0

    wrist_below_desk_samples: int = 0
    near_hand_object_counts: dict = field(default_factory=dict)
    # Attachments per sighting, per class. A rate near or above 1.0 cannot be a
    # single held object and marks a fixed feature beside the seat.
    near_hand_object_rates: dict = field(default_factory=dict)
    persistent_object_classes: list = field(default_factory=list)

    facing_counts: dict = field(default_factory=dict)

    # Additive, dashboard-safe labels. These do not alter `flags`, fusion, or
    # a machine state; a client can opt into rendering them without changing
    # the established review queue.
    orientation_labels: list = field(default_factory=list)
    orientation_events: list = field(default_factory=list)

    # Observations worth a reviewer's attention, each with the number that
    # produced it. A flag is never a conclusion that someone cheated -- it is a
    # pointer at a span of video and a reason to look.
    flags: list = field(default_factory=list)
    note: str = ""

    def to_row(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


@dataclass
class PersonTimeline:
    track_id: int
    samples: list = field(default_factory=list)


def build_timelines(samples: list[PersonSample]) -> dict:
    """Group samples into one timeline per track, ordered in time."""
    out: dict[int, PersonTimeline] = {}
    for sample in samples:
        timeline = out.setdefault(sample.track_id,
                                  PersonTimeline(track_id=sample.track_id))
        timeline.samples.append(sample)
    for timeline in out.values():
        timeline.samples.sort(key=lambda s: s.pts_ms)
    return out


def _spread(values: list[float]) -> tuple[float, float]:
    """Median and MAD, the pair every baseline in this project uses."""
    median = statistics.median(values)
    mad = statistics.median([abs(v - median) for v in values]) or 0.0
    return median, mad


def pitch_state(pitch: float | None, cfg: TimelineConfig | None = None) -> str:
    """Return the reviewer-facing pitch band for one resolvable frame.

    `looking_up` and `shallow_down` are explicitly non-escalating. A raw
    `deep_down` frame is not enough for a card either: only a PTS-continuous
    15-second span becomes the `deep_down_sustained` dashboard label.
    """
    cfg = cfg or TimelineConfig()
    if pitch is None:
        return "not_assessable"
    if pitch < cfg.looking_up_pitch_max:
        return "looking_up"
    if cfg.shallow_down_pitch_min <= pitch <= cfg.deep_down_pitch_min:
        return "shallow_down"
    if pitch > cfg.deep_down_pitch_min:
        return "deep_down"
    return "neutral"


def _spans(rows: list[PersonSample], predicate, interval_ms: float,
           merge_intervals: float) -> list[dict]:
    """Make PTS-continuous spans, tolerating a bounded sampling wobble.

    `rows` must be ordered by `pts_ms`; `build_timelines` guarantees that.
    Out-of-order rows would make the gap test compare against the wrong
    predecessor and silently split or merge spans.
    """
    if not rows:
        return []
    merge_gap_ms = max(interval_ms * merge_intervals, interval_ms)
    spans, start, end = [], None, None
    for row in rows:
        if predicate(row):
            if start is None:
                start = row.pts_ms
            end = row.pts_ms
        elif start is not None and row.pts_ms - end > merge_gap_ms:
            spans.append({"from_ms": round(start, 1),
                          "to_ms": round(end + interval_ms, 1),
                          "duration_ms": round(end - start + interval_ms, 1)})
            start = end = None
    if start is not None:
        spans.append({"from_ms": round(start, 1),
                      "to_ms": round(end + interval_ms, 1),
                      "duration_ms": round(end - start + interval_ms, 1)})
    return spans


def _orientation_outputs(rows: list[PersonSample], baseline_torso_yaw: float | None,
                         interval_ms: float, cfg: TimelineConfig) -> tuple[list, list]:
    """Build labels/events for the four approved dashboard classifications.

    The output is intentionally separate from legacy `flags`. It lets a
    dashboard expose the requested states while preserving the model's current
    machine routing and historical profile semantics.
    """
    labels, events = [], []
    counts = {state: sum(pitch_state(r.pitch, cfg) == state for r in rows)
              for state in ("looking_up", "shallow_down", "deep_down")}
    for state in ("looking_up", "shallow_down"):
        if counts[state]:
            labels.append({
                "label": state,
                "classification": "observed_only",
                "samples": counts[state],
                "dashboard_action": "display_only",
                "guardrail": "This band never escalates on duration alone.",
            })

    for span in _spans(rows,
                       lambda r: pitch_state(r.pitch, cfg) == "deep_down",
                       interval_ms, cfg.orientation_merge_intervals):
        if span["duration_ms"] >= cfg.deep_down_min_ms:
            event = {"label": "deep_down_sustained", "classification": "review_context",
                     **span,
                     "dashboard_action": "show_review_context",
                     "guardrail": "Head-down posture is not intent; show this with the clip and object/hand context."}
            events.append(event)
            labels.append({"label": "deep_down_sustained",
                           "classification": "review_context",
                           "events": 1,
                           "dashboard_action": "show_review_context",
                           "guardrail": event["guardrail"]})

    if baseline_torso_yaw is not None:
        def turned(row):
            value = row.yaw_relative_to_torso
            return value is not None and abs(value - baseline_torso_yaw) >= \
                cfg.torso_relative_turn_departure

        for span in _spans(rows, turned, interval_ms, cfg.orientation_merge_intervals):
            if span["duration_ms"] < cfg.torso_relative_turn_min_ms:
                continue
            values = [r.yaw_relative_to_torso for r in rows
                      if span["from_ms"] <= r.pts_ms < span["to_ms"]
                      and r.yaw_relative_to_torso is not None]
            direction = "left" if statistics.median(values) > baseline_torso_yaw \
                else "right"
            event = {"label": "turned_and_held", "classification": "context_requires_second_modality",
                     "direction": direction, **span,
                     "dashboard_action": "show_only_with_second_modality",
                     "guardrail": "Torso-relative turn is not gaze and never routes to review on its own."}
            events.append(event)
            labels.append({"label": "turned_and_held",
                           "classification": "context_requires_second_modality",
                           "events": 1,
                           "dashboard_action": "show_only_with_second_modality",
                           "guardrail": event["guardrail"]})
    return labels, events


def profile(timeline: PersonTimeline, duration_ms: float,
            interval_ms: float, cfg: TimelineConfig | None = None
            ) -> PersonProfile:
    """Describe one person's whole observed record.

    The profile is built for **every** person, not only those who did something
    notable. That is the point: without the ordinary samples there is no
    baseline to be unusual against, and a metric computed only on flagged
    moments has already assumed its own conclusion.
    """
    cfg = cfg or TimelineConfig()
    rows = timeline.samples
    out = PersonProfile(track_id=timeline.track_id,
                        seat_state=rows[0].seat_state if rows else "")
    if not rows:
        out.note = "no sightings"
        return out

    out.samples = len(rows)
    out.first_seen_ms = rows[0].pts_ms
    out.last_seen_ms = rows[-1].pts_ms
    out.observed_span_ms = rows[-1].pts_ms - rows[0].pts_ms
    out.coverage_of_recording = (out.observed_span_ms / duration_ms
                                 if duration_ms else 0.0)

    resolvable = [r for r in rows if r.yaw is not None or r.pitch is not None]
    out.resolvable_samples = len(resolvable)
    scales = [r.head_scale_px for r in rows if r.head_scale_px]
    out.median_head_scale_px = round(statistics.median(scales), 1) if scales else 0.0

    facing = {}
    for row in rows:
        if row.facing:
            facing[row.facing] = facing.get(row.facing, 0) + 1
    out.facing_counts = facing

    yaws = [r.yaw for r in rows if r.yaw is not None]
    pitches = [r.pitch for r in rows if r.pitch is not None]
    torso_yaws = [r.yaw_relative_to_torso for r in rows
                  if r.yaw_relative_to_torso is not None]

    if len(yaws) >= cfg.min_samples_for_baseline:
        out.baseline_yaw, out.baseline_yaw_spread = _spread(yaws)
        away = 0
        for value in yaws:
            if value - out.baseline_yaw >= cfg.departure:
                out.samples_turned_left += 1
                away += 1
            elif out.baseline_yaw - value >= cfg.departure:
                out.samples_turned_right += 1
                away += 1
        out.fraction_away_from_baseline = round(away / len(yaws), 4)
    if len(pitches) >= cfg.min_samples_for_baseline:
        out.baseline_pitch, out.baseline_pitch_spread = _spread(pitches)
        out.samples_looking_down = sum(
            1 for v in pitches if v - out.baseline_pitch >= cfg.departure)
    if len(torso_yaws) >= cfg.min_samples_for_baseline:
        (out.baseline_yaw_relative_to_torso,
         out.baseline_yaw_relative_to_torso_spread) = _spread(torso_yaws)

    out.orientation_labels, out.orientation_events = _orientation_outputs(
        rows, out.baseline_yaw_relative_to_torso, interval_ms, cfg)

    # Absences: a gap larger than a few sample intervals AND long enough in
    # wall-clock terms to be the person rather than the tracker.
    gap_floor = max(cfg.absence_intervals * interval_ms, cfg.min_absence_ms)
    for earlier, later in zip(rows, rows[1:]):
        gap = later.pts_ms - earlier.pts_ms
        if gap >= gap_floor:
            out.absences.append({"from_ms": round(earlier.pts_ms, 1),
                                 "to_ms": round(later.pts_ms, 1),
                                 "duration_ms": round(gap, 1)})
    out.longest_absence_ms = round(
        max((a["duration_ms"] for a in out.absences), default=0.0), 1)

    out.wrist_below_desk_samples = sum(1 for r in rows if r.wrist_below_desk)
    counts: dict[str, int] = {}
    for row in rows:
        for item in row.near_hand_objects:
            # Tolerate the old bare-string form so profiles written before
            # boxes were stored can still be re-read.
            name = item["cls"] if isinstance(item, dict) else item
            score = float(item.get("confidence", 1.0)) \
                if isinstance(item, dict) else 1.0
            if score < cfg.min_object_confidence:
                continue
            counts[name] = counts.get(name, 0) + 1
    out.near_hand_object_counts = counts
    out.near_hand_object_rates = {
        name: round(total / len(rows), 3) for name, total in counts.items()}
    out.persistent_object_classes = sorted(
        name for name, rate in out.near_hand_object_rates.items()
        if rate > cfg.persistent_object_rate)

    if out.baseline_yaw is None:
        out.note = (f"only {len(yaws)} resolvable yaw sample(s); under the "
                    f"{cfg.min_samples_for_baseline} needed for a baseline, so "
                    f"no departure figure is reported for this person")
    elif out.coverage_of_recording < 0.1:
        out.note = (f"observed for {out.coverage_of_recording:.1%} of the "
                    f"recording; every figure here rests on that fraction")
    return out


def flag(profile: PersonProfile, cfg: TimelineConfig | None = None) -> list:
    """Turn one person's measured record into reviewer-facing observations.

    Every flag carries the figure that produced it and the population it was
    measured against, because a reviewer who cannot see why an item is on the
    list cannot disagree with it. Nothing here asserts misconduct; the wording
    is observational by design (PS #1 requires alerts that highlight observed
    behaviour rather than accusing).

    Thresholds are deliberately generous. The pipeline's earlier guardrails were
    strict enough that nothing reached a reviewer at all -- 0 events in the top
    disposition band across 113 ranked events -- and a triage tool that surfaces
    nothing is worse than one that surfaces too much.
    """
    cfg = cfg or TimelineConfig()
    out = []

    if profile.baseline_yaw is None:
        out.append({
            "flag": "no_orientation_baseline",
            "severity": "info",
            "evidence": f"{profile.resolvable_samples} resolvable of "
                        f"{profile.samples} sighting(s)",
            "why": "too few resolvable head samples to establish this person's "
                   "resting orientation, so no departure figure can be "
                   "reported for them. A data-quality fact about us, not an "
                   "observation about them.",
        })
        return out

    if profile.fraction_away_from_baseline >= 0.40:
        direction = ("left" if profile.samples_turned_left >
                     profile.samples_turned_right else "right")
        out.append({
            "flag": "sustained_attention_away",
            "severity": "high" if profile.fraction_away_from_baseline >= 0.60
                        else "medium",
            "evidence": f"{profile.fraction_away_from_baseline:.0%} of "
                        f"{profile.samples} sightings departed from this "
                        f"person's own resting yaw of "
                        f"{profile.baseline_yaw:+.2f} "
                        f"({profile.samples_turned_left} left, "
                        f"{profile.samples_turned_right} right)",
            "why": f"head held away from their resting orientation for much of "
                   f"the time observed, more often to their {direction}.",
        })
    elif profile.fraction_away_from_baseline >= 0.25:
        out.append({
            "flag": "frequent_orientation_change",
            "severity": "low",
            "evidence": f"{profile.fraction_away_from_baseline:.0%} of "
                        f"{profile.samples} sightings away from a resting yaw "
                        f"of {profile.baseline_yaw:+.2f}",
            "why": "moves away from their resting orientation more than most, "
                   "which is ordinary in itself and only matters alongside "
                   "other evidence.",
        })

    if profile.samples and profile.samples_looking_down / profile.samples >= 0.15:
        out.append({
            "flag": "prolonged_downward_orientation",
            "severity": "low",
            "evidence": f"{profile.samples_looking_down} of {profile.samples} "
                        f"sightings pitched below their own resting level",
            "why": "expected during written work; length is what distinguishes "
                   "it from reading a screen.",
        })

    # Judge on RATE, not raw count. A class attached in most of a person's
    # sightings is a fixed object beside their seat; a handheld one is
    # transient by definition, and the transient case is the reportable one.
    # `phone` is deliberately absent. That class is stock COCO `cell phone`
    # from D-FINE, and the output contract (docs/OUTPUT_CONTRACT.md, reason
    # code DFINE_OBJECT_CONTEXT) bars it from routing anyone to review: on this
    # corpus it fires on mice, keyboards and paper edges, and PRD 10 names that
    # cluster as a failure. The phone route runs through a phone-specific
    # detector verified by SAM 3 instead -- SAM3_PHONE_NAMED. D-FINE's phone
    # detections are still carried as workstation context in
    # `near_hand_object_rates`; they simply do not raise a flag.
    target_names = ("secondary_paper_chit", "book_or_notebook_like_object")
    transient, persistent = {}, {}
    for name in target_names:
        rate = profile.near_hand_object_rates.get(name, 0.0)
        if rate <= 0.0:
            continue
        if rate > cfg.persistent_object_rate:
            persistent[name] = rate
        elif rate >= cfg.min_object_rate:
            transient[name] = rate

    if transient:
        out.append({
            "flag": "target_object_near_hands",
            # Uniformly medium. The old rule promoted to `high` on a COCO
            # phone rate, which was the D-FINE phone route this contract
            # removes; nothing left in `target_names` justifies `high` on its
            # own, and severity is decided by fusion, not here.
            "severity": "medium",
            "evidence": ", ".join(
                f"{k} in {v:.0%} of sightings "
                f"({profile.near_hand_object_counts.get(k, 0)} of "
                f"{profile.samples})"
                for k, v in sorted(transient.items(), key=lambda kv: -kv[1])),
            "why": "a prohibited-class object appeared near this person's hands "
                   "for part of the time they were observed, which is the "
                   "pattern of something picked up rather than something that "
                   "sits on the desk. The detector is stock COCO and not "
                   "fine-tuned on this corpus, so the crop must be looked at.",
        })

    if persistent:
        out.append({
            "flag": "persistent_object_beside_seat",
            "severity": "info",
            "evidence": ", ".join(f"{k} in {v:.0%} of sightings"
                                  for k, v in sorted(persistent.items(),
                                                     key=lambda kv: -kv[1])),
            "why": "present near this person's hands for most of the recording, "
                   "so it is a fixed feature of their workstation rather than "
                   "something they picked up. Recorded, not flagged: "
                   "persistence is not corroboration for a handheld object.",
        })

    if profile.wrist_below_desk_samples >= 3:
        out.append({
            "flag": "hand_below_desk_line",
            "severity": "medium",
            "evidence": f"{profile.wrist_below_desk_samples} sighting(s) with a "
                        f"wrist below the desk line",
            "why": "an object held below the desk cannot be seen by any "
                   "detector, so the behaviour is reported instead of the "
                   "object.",
        })

    if profile.absences:
        out.append({
            "flag": "absent_from_frame",
            "severity": "medium",
            "evidence": f"{len(profile.absences)} absence(s), longest "
                        f"{profile.longest_absence_ms / 1000:.1f}s",
            "why": "no sighting for long enough to be the person leaving rather "
                   "than a tracker dropout. Standing up is ordinary; duration "
                   "and what happens meanwhile are what matter.",
        })

    if profile.coverage_of_recording < 0.10:
        out.append({
            "flag": "low_coverage",
            "severity": "info",
            "evidence": f"observed across {profile.coverage_of_recording:.1%} "
                        f"of the recording",
            "why": "every figure for this person rests on that fraction. "
                   "Tracking breaks and there is no re-identification, so one "
                   "physical person may appear as several tracks.",
        })
    return out


def summarise(profiles: list[PersonProfile], duration_ms: float,
              sampled_rows: int, cfg: TimelineConfig | None = None) -> dict:
    """Cross-person view, with the coverage caveat stated in the artifact."""
    cfg = cfg or TimelineConfig()
    if not profiles:
        return {"people": 0,
                "note": "no person was tracked. With no detections this is a "
                        "valid result, not a crash."}

    covered = sorted(p.coverage_of_recording for p in profiles)
    away = sorted(p.fraction_away_from_baseline for p in profiles
                  if p.baseline_yaw is not None)
    with_baseline = sum(1 for p in profiles if p.baseline_yaw is not None)

    return {
        "people": len(profiles),
        "sampled_rows": sampled_rows,
        "duration_ms": round(duration_ms, 1),
        "people_with_a_yaw_baseline": with_baseline,
        "coverage_of_recording": {
            "min": round(covered[0], 4),
            "median": round(covered[len(covered) // 2], 4),
            "max": round(covered[-1], 4),
        },
        "fraction_away_from_baseline": {
            "min": round(away[0], 4) if away else None,
            "median": round(away[len(away) // 2], 4) if away else None,
            "max": round(away[-1], 4) if away else None,
        },
        "people_with_absences": sum(1 for p in profiles if p.absences),
        "config": asdict(cfg),
        "note":
            "Every person observed is profiled, whether or not anything "
            "notable happened to them. A person's departure figures are "
            "measured against their own resting orientation, so they are "
            "comparable within that person over time and not between people.",
        "coverage_caveat":
            "coverage_of_recording is the span between a person's first and "
            "last sighting divided by the recording length. It is NOT a "
            "guarantee they were watched throughout that span -- tracking "
            "breaks and re-identification does not exist, so one physical "
            "person may appear as several tracks.",
    }
