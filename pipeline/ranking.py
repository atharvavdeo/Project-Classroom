"""Transparent ranking (PRD 12).

PRD 12's first sentence sets the terms:

> Priority is transparent ranking, not cheating probability.

The difference is not cosmetic. A probability invites the reader to treat the
number as the likelihood that something happened. A rank says only: of the
things this run surfaced, look at these first. So the output here is an ordered
list with every component and coefficient persisted, and no total that could be
read as a probability. Scores are unbounded above and carry no units.

Every component is listed in PRD 12 and each is computed from evidence that
already exists upstream. Nothing here recomputes motion, re-runs a model, or
consults the reference manifest.

### What Gemma may and may not do

PRD 11: the VLM "cannot create candidates, alter boundaries, alter rank". So
`gemma` appears in this module in exactly one place -- setting `needs_human_review`
on the output row. It contributes no points, positive or negative. An invalid or
absent review changes an event's position by nothing at all.

### Zero high-priority events is a result

PRD 12: with none, "report analysed/failed/affected duration, retained
low-priority/uncertain counts, configs, and reference results -- never 'nothing
happened'." `summarise` produces exactly that shape whether the list is full or
empty, so the empty case cannot fall through to a cheerful default.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from pipeline.object_detection import TARGET_CLASSES

# Coefficients. Persisted with every ranked event so a position can be
# re-derived, and so a change in weighting is visible as a change in the
# artifacts rather than as a silent reordering between runs.
WEIGHTS = {
    # Motion evidence: how far the window departed from the seat's own baseline,
    # and how much of the zone moved. Both already robust and baseline-relative.
    "motion_deviation": 1.0,
    "motion_area": 0.6,
    # Duration and repetition. A sustained or repeated movement is more worth a
    # reviewer's time than an instant one, but only mildly -- the interesting
    # events in this corpus are short.
    "duration": 0.3,
    "repetition": 0.5,
    # Seat association quality. An event nobody can attribute is still worth
    # surfacing, but below one that is cleanly attributed.
    "seat_quality": 0.8,
    # Pose context, and only the sustained kind. PRD 9 forbids a single neck
    # turn or low-confidence wrist from creating a high-priority event, so an
    # unsustained pose signal contributes zero rather than a little.
    "pose_sustained": 1.2,
    # Temporally supported object evidence. The strongest single component,
    # because it is the only one corroborated across frames by a model.
    "object_supported": 2.0,
    "object_single_frame": 0.4,
}

# Penalties are multiplicative, applied after the components sum. A quality
# problem should scale the whole score down rather than subtract a fixed amount
# from it: on a strong event a fixed subtraction is negligible, which is exactly
# where an integrity problem matters most.
PENALTIES = {
    "quality_affected": 0.6,
    "environmental": 0.5,
    "uncertain_seat": 0.7,
    "camera_uncompensated": 0.4,
}

NORMAL = "normal"
LOW = "low"
SUPPRESSED = "suppressed_from_auto_ranking"

# ------------------------------------------------------------ dispositions --
# `priority` says where an event sits in the ordering. `disposition` says what
# a reviewer is being asked to do with it, which is a different question and
# the one an investigator actually has.
#
# The band boundaries cannot be absolute score thresholds. Measured: the top
# event on video 01 scores 85.63 and the top event on video 03 scores 9.92,
# because the dominant component is a motion deviation in that seat's own
# baseline units. A fixed cut would put every video-01 event above every
# video-03 event regardless of what is in them. So the motion-only route into
# the review queue is defined on **rank within the run**, and the corroborated
# route is defined on evidence and ignores rank entirely.
REJECTED = "rejected_not_offered"
RETAINED = "retained_low_priority"
FLAGGED = "flagged_for_review"
HIGH_POTENTIAL = "high_potential_corroborated"
DISPOSITIONS = (HIGH_POTENTIAL, FLAGGED, RETAINED, REJECTED)

# Fraction of a run's gate-passing events that enter the queue on motion
# strength alone. Everything corroborated enters regardless of where it falls.
REVIEW_FRACTION = 0.25


@dataclass
class RankedEvent:
    """One event's position, and every number that produced it."""

    event_id: str
    seat_state: str
    start_pts_ms: float
    end_pts_ms: float
    peak_pts_ms: float
    score: float
    rank: int = 0
    priority: str = NORMAL
    disposition: str = RETAINED
    disposition_reason: str = ""
    corroboration: list = field(default_factory=list)
    components: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    penalties: dict = field(default_factory=dict)
    gate_state: str = "pass"
    health_state: list[str] = field(default_factory=list)
    needs_human_review: bool = True
    review_source: str = ""
    config_ids: dict = field(default_factory=dict)
    artifact_paths: list[str] = field(default_factory=list)
    reason: str = ""

    def to_row(self) -> dict:
        return {**asdict(self), "score": round(self.score, 4)}


def _components(event: dict, pose_evidence=None, objects=None) -> dict:
    """Every ranking component, from evidence that already exists.

    Only **target-class** object evidence scores. Temporal confirmation runs on
    everything the detector emits, and on video 03 that is 334 confirmed
    `unknown_object`, 103 `monitor_or_bezel` and 76 `keyboard` against 36
    `phone`. A confirmed monitor is a confirmed monitor: it is real, it is
    correctly labelled, and it corroborates nothing. Counting it would let the
    furniture in the room carry an event into the top band, which is precisely
    the false-event generation PS #2 names as a deployment risk.
    """
    objects = objects or []
    targets = [o for o in objects if o.cls in TARGET_CLASSES]

    supported = sum(1 for o in targets
                    if o.status == "temporally_supported_object_evidence")
    singles = sum(1 for o in targets
                  if o.status == "single_frame_object_candidate")

    duration_s = max(event.get("end_pts_ms", 0) - event.get("start_pts_ms", 0),
                     0.0) / 1000.0

    return {
        "motion_deviation": float(event.get("peak_deviation", 0.0)),
        "motion_area": float(event.get("peak_area_ratio", 0.0)),
        # Compressed: a 30-second event is not thirty times more interesting
        # than a one-second one, and a raw duration term would make it so.
        "duration": min(duration_s / 10.0, 1.0),
        "repetition": min(float(event.get("repetition", 0)) / 3.0, 1.0),
        "seat_quality": float(event.get("seat_confidence", 0.0)),
        "pose_sustained": 1.0 if (pose_evidence is not None
                                  and getattr(pose_evidence, "sustained", False))
        else 0.0,
        "object_supported": float(supported),
        "object_single_frame": float(singles),
    }


def _penalties(event: dict) -> dict:
    """Which multiplicative penalties apply, and why."""
    applied: dict[str, float] = {}
    health = set(event.get("health_state", []))

    if health & {"blur", "exposure_change", "compression_noise", "whiteout",
                 "blackout", "frozen_video"}:
        applied["quality_affected"] = PENALTIES["quality_affected"]
    if "environmental_motion" in health:
        applied["environmental"] = PENALTIES["environmental"]
    if event.get("seat_state", event.get("seat_id")) in (
            "ambiguous_seat", "unattributed", None):
        applied["uncertain_seat"] = PENALTIES["uncertain_seat"]
    if "camera_motion_uncertain" in health or "camera_motion" in health:
        applied["camera_uncompensated"] = PENALTIES["camera_uncompensated"]
    return applied


def rank(events: list[dict], pose_by_event: dict | None = None,
         objects_by_event: dict | None = None,
         reviews_by_event: dict | None = None,
         gate_state_of=None) -> list[RankedEvent]:
    """Rank events. Gemma sets review status only, never position."""
    pose_by_event = pose_by_event or {}
    objects_by_event = objects_by_event or {}
    reviews_by_event = reviews_by_event or {}

    out: list[RankedEvent] = []
    for event in events:
        event_id = event["event_id"]
        components = _components(event, pose_by_event.get(event_id),
                                 objects_by_event.get(event_id))
        penalties = _penalties(event)

        total = sum(WEIGHTS[name] * value for name, value in components.items())
        multiplier = 1.0
        for value in penalties.values():
            multiplier *= value
        score = total * multiplier

        gate_state = (gate_state_of(event) if gate_state_of
                      else event.get("gate_state", "pass"))

        # A gate that suppresses an event removes it from automatic ordering and
        # nothing else. The evidence stays, searchable, with its score intact so
        # a reviewer who goes looking can still see what it would have scored.
        if gate_state in ("suppress_from_auto_ranking", "uncertain", "failed"):
            priority = SUPPRESSED
            reason = (f"gate state {gate_state!r} excludes this from automatic "
                      f"priority. It remains searchable with its score intact "
                      f"(PRD 8)")
        elif event.get("priority") == "low" or score <= 0.0:
            priority = LOW
            reason = ("retained at low priority; short or subtle candidates are "
                      "kept even when they do not activate (PRD 8)")
        else:
            priority = NORMAL
            reason = (f"{total:.2f} from components"
                      + (f", scaled by {multiplier:.2f} for "
                         f"{sorted(penalties)}" if penalties else ""))

        review = reviews_by_event.get(event_id)
        out.append(RankedEvent(
            event_id=event_id,
            seat_state=event.get("seat_id") or event.get("seat_state", "unattributed"),
            start_pts_ms=float(event.get("start_pts_ms", 0.0)),
            end_pts_ms=float(event.get("end_pts_ms", 0.0)),
            peak_pts_ms=float(event.get("peak_pts_ms", 0.0)),
            score=score, priority=priority, components=components,
            weights=dict(WEIGHTS), penalties=penalties,
            gate_state=gate_state,
            health_state=list(event.get("health_state", [])),
            # The one thing the VLM touches. It contributes no points.
            needs_human_review=review.needs_human_review if review else True,
            review_source=review.config_id if review else "",
            reason=reason))

    # Suppressed events are ordered after everything else regardless of score,
    # then by score. They keep a rank so they are still citable.
    out.sort(key=lambda e: (e.priority == SUPPRESSED, e.priority == LOW, -e.score))
    for index, event in enumerate(out, start=1):
        event.rank = index
    _assign_dispositions(out)
    return out


def _assign_dispositions(ranked: list[RankedEvent],
                         review_fraction: float = REVIEW_FRACTION) -> None:
    """Decide what a reviewer is asked to do with each event.

    Four bands, in the order they are tested:

    **rejected_not_offered** -- a gate refused it. This means *not placed in
    the queue*, never *judged innocent*: the evidence stays on disk with its
    score intact and remains searchable. Calling it "rejected" and leaving it
    retrievable is the only honest pair of properties; a band that meant
    "cleared" would be a finding this pipeline is not entitled to make.

    **high_potential_corroborated** -- a conjunction, not a high score. It
    requires object evidence that survived temporal confirmation *and*
    sustained pose context *and* a resolved seat *and* no active quality or
    geometry penalty. Four independent things have to agree. A single loud
    motion spike cannot reach this band no matter how large it is, which is the
    point: PS #2 names false event generation and alert fatigue as risks, and
    a top band reachable by one modality is how both happen.

    **flagged_for_review** -- the working queue. Either partial corroboration
    (object or pose, not both) or, on motion alone, the top `review_fraction`
    of this run's gate-passing events.

    **retained_low_priority** -- everything else that passed the gate. Kept and
    searchable, at the bottom.
    """
    eligible = [e for e in ranked if e.priority != SUPPRESSED]
    cutoff_rank = max(int(round(len(eligible) * review_fraction)), 1) \
        if eligible else 0
    motion_rank = {event.event_id: index
                   for index, event in enumerate(eligible, start=1)}

    for event in ranked:
        if event.priority == SUPPRESSED:
            event.disposition = REJECTED
            event.disposition_reason = (
                f"gate state {event.gate_state!r} withheld this from the review "
                f"queue. The evidence is retained and searchable with its score "
                f"intact; this is not a finding that nothing happened")
            continue

        components = event.components
        corroboration = []
        if components.get("object_supported", 0.0) >= 1.0:
            corroboration.append("temporally_supported_object_evidence")
        if components.get("pose_sustained", 0.0) >= 1.0:
            corroboration.append("sustained_pose_context")
        if components.get("repetition", 0.0) > 0.0:
            corroboration.append("repeated_movement")
        event.corroboration = corroboration

        seat_resolved = event.seat_state not in (
            "", None, "unattributed", "ambiguous_seat")
        modalities = {"temporally_supported_object_evidence",
                      "sustained_pose_context"} & set(corroboration)

        if len(modalities) == 2 and seat_resolved and not event.penalties:
            event.disposition = HIGH_POTENTIAL
            event.disposition_reason = (
                f"object evidence confirmed across frames and sustained pose "
                f"context agree at seat {event.seat_state}, with no quality or "
                f"geometry penalty applied. Corroboration across independent "
                f"modalities, not score, put this here")
        elif corroboration:
            event.disposition = FLAGGED
            event.disposition_reason = (
                f"partial corroboration ({', '.join(corroboration)}); "
                f"not enough independent agreement for the top band")
        elif motion_rank.get(event.event_id, len(eligible) + 1) <= cutoff_rank:
            event.disposition = FLAGGED
            event.disposition_reason = (
                f"rank {motion_rank[event.event_id]} of {len(eligible)} on "
                f"motion alone, inside the top {review_fraction:.0%} of this "
                f"run. No object or pose evidence corroborates it")
        else:
            event.disposition = RETAINED
            event.disposition_reason = (
                f"motion below this run's top {review_fraction:.0%} and no "
                f"corroborating evidence; retained and searchable")


def summarise(ranked: list[RankedEvent], analysed_ms: float = 0.0,
              failed_ms: float = 0.0, affected_ms: float = 0.0,
              configurations: dict | None = None,
              evaluation_status: str = "") -> dict:
    """The reviewer summary, in the shape PRD 12 requires even when empty."""
    normal = [e for e in ranked if e.priority == NORMAL]
    low = [e for e in ranked if e.priority == LOW]
    suppressed = [e for e in ranked if e.priority == SUPPRESSED]

    report = {
        "events_ranked": len(ranked),
        "normal_priority": len(normal),
        "low_priority_retained": len(low),
        "suppressed_from_auto_ranking": len(suppressed),
        "needs_human_review": sum(1 for e in ranked if e.needs_human_review),
        "analysed_ms": round(analysed_ms, 1),
        "failed_ms": round(failed_ms, 1),
        "quality_affected_ms": round(affected_ms, 1),
        "analysed_fraction": round(
            (analysed_ms - failed_ms) / analysed_ms, 4) if analysed_ms else None,
        "configurations": configurations or {},
        # Named for what it holds -- a sentence about whether evaluation
        # happened -- not "reference_result", which named it after data it
        # deliberately does not contain and tripped the stage-12 isolation
        # check on every run. The check is right to be strict; the field name
        # was wrong.
        "evaluation_status": evaluation_status or
        "no reference manifest was read; object evaluation is incomplete and no "
        "quality claim is made (PRD 3)",
        "weights": dict(WEIGHTS),
        "penalties": dict(PENALTIES),
        "top_events": [e.to_row() for e in normal[:10]],
        # What a reviewer is actually asked to open, and what was withheld.
        # `review_queue` is the number of events a human is being asked to
        # look at; measuring reviewer effort saved needs that number, not the
        # number of events the segmenter produced.
        "dispositions": {name: sum(1 for e in ranked if e.disposition == name)
                         for name in DISPOSITIONS},
        "review_queue": sum(1 for e in ranked if e.disposition
                            in (HIGH_POTENTIAL, FLAGGED)),
        "review_fraction": REVIEW_FRACTION,
        "disposition_note":
            "high_potential_corroborated requires temporally supported object "
            "evidence AND sustained pose context AND a resolved seat AND no "
            "active penalty -- a conjunction across independent modalities, "
            "never a score. rejected_not_offered means withheld from the queue "
            "by a gate, never cleared: those events keep their score and stay "
            "searchable.",
    }

    if not normal:
        # PRD 12 forbids reporting this as "nothing happened".
        report["headline"] = (
            f"No high-priority events. {report['analysed_ms'] / 1000.0:.1f} s "
            f"analysed, {report['failed_ms'] / 1000.0:.1f} s failed, "
            f"{report['quality_affected_ms'] / 1000.0:.1f} s quality-affected; "
            f"{len(low)} low-priority and {len(suppressed)} suppressed "
            f"candidates retained and searchable. This is a statement about "
            f"what this configuration surfaced over the analysed duration, not "
            f"a finding that nothing occurred.")
    else:
        report["headline"] = (
            f"{len(normal)} event(s) at normal priority over "
            f"{report['analysed_ms'] / 1000.0:.1f} s analysed; {len(low)} "
            f"retained at low priority and {len(suppressed)} excluded from "
            f"automatic ordering. Priority is a reading order, not a "
            f"probability that anything occurred.")
    return report
