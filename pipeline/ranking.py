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
    """Every ranking component, from evidence that already exists."""
    objects = objects or []

    supported = sum(1 for o in objects
                    if o.status == "temporally_supported_object_evidence")
    singles = sum(1 for o in objects
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
    return out


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
