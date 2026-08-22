"""Person-to-seat association (PRD 7).

PRD 7 gives the procedure and two prohibitions. The prohibitions are the
interesting part:

> Never assign by nearest-box centre alone. Never force a seat across shared
> desk/aisle boundaries.

Both describe the same failure from different angles: producing an answer
because an answer was wanted, rather than because the geometry supported one.
The whole module is arranged so that refusing is as easy to express as deciding,
and so that a refusal carries the full score vector rather than a null.

Association happens twice, and the order matters.

**Per box**, `pipeline.calibration.attribute_box` scores the box's lower
footprint against each seat polygon. That is the geometric evidence and it is
computed independently for every observation, with no memory.

**Per track**, the per-box answers are aggregated. Temporal continuity may
*resolve* an ambiguity -- a frame whose two leading seats are within the margin
can be settled by the seat the rest of the track sits in. It may never
*overrule* a contradiction: if the track's majority seat is not among a frame's
candidates at all, that frame stays ambiguous. Continuity is evidence about
which of several supported answers is right, never evidence for an answer the
geometry did not support.

Every association records the full candidate score vector, the rule that fired,
and the reason in words, so a reviewer can see not just what was decided but
what the alternative was and by how much it lost.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .calibration import (AMBIGUOUS_SEAT, UNATTRIBUTED, AttributionConfig,
                          CalibrationProfile, attribute_box)
from .person_detection import PersonBox
from .tracking import SEAT_ONLY, Track, TrackingResult

# Which rule produced the association. Recorded so the mix can be audited: a run
# where most seats came from continuity rather than geometry is a run whose
# calibration or detector needs attention, not a run with good attribution.
BY_GEOMETRY = "footprint_overlap"
BY_CONTINUITY = "track_continuity"
BY_REFUSAL = "refused"


@dataclass
class AssociationConfig:
    attribution: AttributionConfig = field(default_factory=AttributionConfig)

    # Share of a track's resolved boxes that must agree before continuity is
    # allowed to settle an ambiguous frame. Below this the track itself is
    # unstable and has no opinion worth applying.
    min_track_majority: float = 0.60

    # A track must have at least this many geometrically resolved boxes before
    # its majority means anything. Two boxes agreeing is not a majority; it is
    # a coincidence with a denominator.
    min_resolved_boxes: int = 3

    # Health conditions under which attribution is not attempted at all. Under
    # uncompensated camera motion the calibration polygons no longer describe
    # where the seats are, so every score computed against them is wrong in an
    # unknown direction.
    blocking_health: tuple[str, ...] = (
        "camera_repositioned", "camera_motion_uncertain", "decode_gap",
        "blackout")


@dataclass
class SeatAssociation:
    """One box's seat, or an explicit refusal to name one."""

    row_id: str
    pts_ms: float
    track_id: int | None
    seat_id: str | None
    state: str                       # a seat id, AMBIGUOUS_SEAT, or UNATTRIBUTED
    rule: str
    reason: str
    scores: dict[str, float] = field(default_factory=dict)
    margin: float | None = None
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    health_state: list[str] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.seat_id is not None

    def to_row(self) -> dict:
        return {**asdict(self), "resolved": self.resolved}


def _margin(scores: dict[str, float]) -> float | None:
    if len(scores) < 2:
        return None
    ranked = sorted(scores.values(), reverse=True)
    return round(ranked[0] - ranked[1], 4)


def associate_box(box: PersonBox, profile: CalibrationProfile,
                  cfg: AssociationConfig | None = None,
                  health_state: list[str] | None = None) -> SeatAssociation:
    """Geometric association for one box, with no temporal memory."""
    cfg = cfg or AssociationConfig()
    health = list(health_state or [])

    blocking = [h for h in health if h in cfg.blocking_health]
    if blocking:
        return SeatAssociation(
            row_id=box.row_id, pts_ms=box.pts_ms, track_id=box.track_id,
            seat_id=None, state=UNATTRIBUTED, rule=BY_REFUSAL,
            reason=f"health state {blocking} makes the calibration polygons "
                   f"unreliable at this timestamp; attribution is not attempted",
            box=box.box, health_state=health)

    attribution = attribute_box(box.box, profile, cfg.attribution)
    rule = BY_GEOMETRY if attribution.resolved else BY_REFUSAL
    return SeatAssociation(
        row_id=box.row_id, pts_ms=box.pts_ms, track_id=box.track_id,
        seat_id=attribution.seat_id, state=attribution.state, rule=rule,
        reason=attribution.reason, scores=attribution.scores,
        margin=_margin(attribution.scores), box=box.box, health_state=health)


def _majority(associations: list[SeatAssociation], cfg: AssociationConfig
              ) -> tuple[str | None, float, int]:
    """The seat a track sits in, if one commands a majority of resolved boxes."""
    resolved = [a for a in associations if a.resolved]
    if len(resolved) < cfg.min_resolved_boxes:
        return None, 0.0, len(resolved)
    votes: dict[str, int] = {}
    for association in resolved:
        votes[association.seat_id] = votes.get(association.seat_id, 0) + 1
    seat, count = max(votes.items(), key=lambda kv: kv[1])
    share = count / len(resolved)
    if share < cfg.min_track_majority:
        return None, share, len(resolved)
    return seat, share, len(resolved)


def associate_track(candidate: Track, profile: CalibrationProfile,
                    cfg: AssociationConfig | None = None,
                    health_of=None) -> list[SeatAssociation]:
    """Associate every box in one track, then let continuity settle ties.

    `health_of` is an optional callable from PTS to the health conditions in
    force there.
    """
    cfg = cfg or AssociationConfig()
    associations = [
        associate_box(box, profile, cfg,
                      health_of(box.pts_ms) if health_of else None)
        for box in candidate.boxes]

    seat, share, resolved = _majority(associations, cfg)
    if seat is None:
        return associations

    for association in associations:
        if association.resolved or association.state != AMBIGUOUS_SEAT:
            continue
        # Continuity resolves a tie only between seats the geometry already put
        # on the table. If the track's seat is not a candidate for this frame,
        # the two sources of evidence disagree, and a disagreement is not
        # resolved by preferring the one we happen to have more of.
        if seat not in association.scores:
            association.reason += (
                f"; track {candidate.track_id} sits in {seat} for "
                f"{share:.0%} of its resolved frames, but {seat} scores nothing "
                f"here -- continuity does not overrule a contradiction")
            continue
        association.seat_id = seat
        association.state = seat
        association.rule = BY_CONTINUITY
        association.reason = (
            f"geometry was ambiguous ({association.reason}); track "
            f"{candidate.track_id} sits in {seat} for {share:.0%} of its "
            f"{resolved} resolved frames, and {seat} is among this frame's "
            f"candidates at {association.scores[seat]:.3f}")
    return associations


def associate(result: TrackingResult, profile: CalibrationProfile,
              cfg: AssociationConfig | None = None,
              health_of=None) -> list[SeatAssociation]:
    """Associate a whole tracking result.

    The no-track baseline is handled by its own path rather than by a track of
    one box each: it must produce purely geometric associations with no
    continuity rule available, which is the entire point of having it.
    """
    cfg = cfg or AssociationConfig()
    if result.config_id == SEAT_ONLY or not result.tracks:
        return [associate_box(box, profile, cfg,
                              health_of(box.pts_ms) if health_of else None)
                for box in result.boxes]

    out: list[SeatAssociation] = []
    for candidate in result.tracks:
        out += associate_track(candidate, profile, cfg, health_of)
    out.sort(key=lambda a: (a.pts_ms, a.track_id or 0))
    return out


def summarise(associations: list[SeatAssociation]) -> dict:
    """Association quality, reported so that refusals are visible."""
    total = len(associations)
    if not total:
        return {"associations": 0,
                "note": "no person boxes were associated; there is nothing to "
                        "report and this is not a perfect-attribution result."}

    resolved = [a for a in associations if a.resolved]
    ambiguous = [a for a in associations if a.state == AMBIGUOUS_SEAT]
    unattributed = [a for a in associations if a.state == UNATTRIBUTED]
    by_rule: dict[str, int] = {}
    for association in associations:
        by_rule[association.rule] = by_rule.get(association.rule, 0) + 1

    per_seat: dict[str, int] = {}
    for association in resolved:
        per_seat[association.seat_id] = per_seat.get(association.seat_id, 0) + 1

    margins = [a.margin for a in resolved if a.margin is not None]
    return {
        "associations": total,
        "resolved": len(resolved),
        "resolved_fraction": round(len(resolved) / total, 4),
        "ambiguous_seat": len(ambiguous),
        "unattributed": len(unattributed),
        "by_rule": by_rule,
        "continuity_share": round(by_rule.get(BY_CONTINUITY, 0) / total, 4),
        "per_seat": dict(sorted(per_seat.items())),
        "median_margin": round(sorted(margins)[len(margins) // 2], 4)
        if margins else None,
        "note": "a high continuity share means most seats came from a track's "
                "majority rather than from this frame's geometry, which is a "
                "signal to review calibration or detector quality.",
    }
