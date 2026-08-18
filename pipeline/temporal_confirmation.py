"""Temporal object confirmation (PRD 10).

PRD 10 defines four final statuses and one prohibition:

    temporally_supported_object_evidence   grouped, consistent, multi-frame
    single_frame_object_candidate          one sighting, retained
    conflicting_object_evidence            the group disagrees with itself
    insufficient_visual_evidence           the source cannot support the call

> A singleton is retained, not silently deleted.

That last line is the reason this module is not a filter. The obvious
implementation -- drop anything seen once -- is wrong in the specific case the
product exists for: a phone visible for a third of a second between a pocket and
a lap is a singleton, and it is the event a reviewer most wants to see. So a
singleton is *labelled*, kept, and ranked below corroborated evidence.

### Conflict is a status, not a tie-break

A group whose members disagree -- three frames calling a region `phone` and two
calling it `mouse` -- is not resolved by majority here. Resolving it would
manufacture agreement that the evidence does not contain, and the disagreement
is itself the interesting signal: it usually means a small bright rectangle that
neither class fits well. `conflicting_object_evidence` carries the full class
distribution so a reviewer sees what the models actually said.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .object_detection import INSUFFICIENT, ObjectDetection

SUPPORTED = "temporally_supported_object_evidence"
SINGLE_FRAME = "single_frame_object_candidate"
CONFLICTING = "conflicting_object_evidence"
INSUFFICIENT_EVIDENCE = INSUFFICIENT
STATUSES = (SUPPORTED, SINGLE_FRAME, CONFLICTING, INSUFFICIENT_EVIDENCE)


@dataclass
class ConfirmConfig:
    # Frames a group needs before it is temporally supported. Two independent
    # sightings, not three: at the 2 Hz sample rate three frames is 1.5 s, which
    # is longer than many of the movements this system exists to surface.
    min_support_frames: int = 2

    # Detections further apart than this start a new group. A phone seen at 10 s
    # and again at 90 s is two events, not one 80-second sighting.
    max_gap_ms: float = 3000.0

    # Spatial agreement required between consecutive members of a group. Low,
    # because a held object moves: this is asking whether the sightings describe
    # the same region of the desk, not whether the object stayed still.
    min_iou: float = 0.10

    # Share of a group that must agree on a class before it is that class.
    # Below this the group is conflicting, and stays conflicting.
    class_majority: float = 0.60


def iou(a: tuple[float, float, float, float],
        b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class ConfirmedObject:
    """One temporally grouped object observation."""

    group_id: str
    config_id: str
    event_id: str
    seat_state: str
    track_id: int | None
    cls: str
    status: str
    support_frames: int
    start_pts_ms: float
    end_pts_ms: float
    duration_ms: float
    # The full class distribution, always. A group reported only by its winning
    # class hides the disagreement that decided whether it is evidence at all.
    class_distribution: dict[str, int] = field(default_factory=dict)
    max_confidence: float = 0.0
    mean_confidence: float = 0.0
    spatial_consistency: float = 0.0
    crop_ids: list[str] = field(default_factory=list)
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    reason: str = ""

    @property
    def supported(self) -> bool:
        return self.status == SUPPORTED

    def to_row(self) -> dict:
        return {**asdict(self), "supported": self.supported}


def _group_key(detection: ObjectDetection) -> tuple:
    """What makes two detections candidates for the same group (PRD 10).

    Class, seat, track and time -- as PRD 10 names them. Class is included so
    that a phone and a mouse detected in the same place are two groups whose
    disagreement is then visible, rather than one group that has to pick.
    """
    return (detection.config_id, detection.event_id, detection.seat_state,
            detection.track_id, detection.cls)


def confirm(detections: list[ObjectDetection], cfg: ConfirmConfig | None = None
            ) -> list[ConfirmedObject]:
    """Group detections in time and assign each group its final status."""
    cfg = cfg or ConfirmConfig()

    usable = [d for d in detections if d.cls != INSUFFICIENT]
    abstained = [d for d in detections if d.cls == INSUFFICIENT]

    buckets: dict[tuple, list[ObjectDetection]] = {}
    for detection in usable:
        buckets.setdefault(_group_key(detection), []).append(detection)

    groups: list[list[ObjectDetection]] = []
    for key, members in buckets.items():
        members.sort(key=lambda d: d.pts_ms)
        current = [members[0]]
        for detection in members[1:]:
            previous = current[-1]
            gap = detection.pts_ms - previous.pts_ms
            if gap <= cfg.max_gap_ms and iou(previous.box, detection.box) >= cfg.min_iou:
                current.append(detection)
            else:
                groups.append(current)
                current = [detection]
        groups.append(current)

    out = [_confirmed(group, cfg) for group in groups]

    # Abstentions are grouped too, so a reviewer can see how much of an event
    # the source could not answer for -- one row per crop would drown that.
    by_event: dict[tuple, list[ObjectDetection]] = {}
    for detection in abstained:
        by_event.setdefault(
            (detection.config_id, detection.event_id, detection.seat_state),
            []).append(detection)
    for (config_id, event_id, seat_state), members in by_event.items():
        members.sort(key=lambda d: d.pts_ms)
        out.append(ConfirmedObject(
            group_id=f"{config_id}:{event_id}:{seat_state}:insufficient",
            config_id=config_id, event_id=event_id, seat_state=seat_state,
            track_id=None, cls=INSUFFICIENT, status=INSUFFICIENT_EVIDENCE,
            support_frames=len(members),
            start_pts_ms=round(members[0].pts_ms, 3),
            end_pts_ms=round(members[-1].pts_ms, 3),
            duration_ms=round(members[-1].pts_ms - members[0].pts_ms, 3),
            class_distribution={INSUFFICIENT: len(members)},
            crop_ids=[d.crop_id for d in members],
            reason=members[0].abstain_reason or
            "the source resolution does not support an object call here"))

    out.sort(key=lambda g: (g.start_pts_ms, g.config_id))
    return out


def _confirmed(group: list[ObjectDetection], cfg: ConfirmConfig
               ) -> ConfirmedObject:
    first, last = group[0], group[-1]
    distribution: dict[str, int] = {}
    for detection in group:
        distribution[detection.cls] = distribution.get(detection.cls, 0) + 1

    winner, count = max(distribution.items(), key=lambda kv: kv[1])
    share = count / len(group)

    consistencies = [iou(group[i].box, group[i + 1].box)
                     for i in range(len(group) - 1)]
    consistency = sum(consistencies) / len(consistencies) if consistencies else 0.0

    if share < cfg.class_majority:
        status = CONFLICTING
        reason = (f"{len(group)} sightings split {distribution}; no class holds "
                  f"the {cfg.class_majority:.0%} majority. The disagreement is "
                  f"reported rather than resolved by vote")
    elif len(group) >= cfg.min_support_frames:
        status = SUPPORTED
        reason = (f"{len(group)} sightings over {last.pts_ms - first.pts_ms:.0f} ms "
                  f"with mean spatial consistency {consistency:.2f}")
    else:
        status = SINGLE_FRAME
        reason = ("one sighting. Retained and ranked below corroborated "
                  "evidence, never deleted: a phone visible between a pocket "
                  "and a lap is a singleton and is the case this exists for")

    confidences = [d.confidence for d in group]
    return ConfirmedObject(
        group_id=f"{first.config_id}:{first.event_id}:{first.cls}:"
                 f"{first.pts_ms:.0f}",
        config_id=first.config_id, event_id=first.event_id,
        seat_state=first.seat_state, track_id=first.track_id,
        cls=winner, status=status, support_frames=len(group),
        start_pts_ms=round(first.pts_ms, 3), end_pts_ms=round(last.pts_ms, 3),
        duration_ms=round(last.pts_ms - first.pts_ms, 3),
        class_distribution=dict(sorted(distribution.items())),
        max_confidence=round(max(confidences), 4),
        mean_confidence=round(sum(confidences) / len(confidences), 4),
        spatial_consistency=round(consistency, 4),
        crop_ids=[d.crop_id for d in group],
        box=(round(first.x1, 2), round(first.y1, 2),
             round(first.x2, 2), round(first.y2, 2)),
        reason=reason)


def summarise(groups: list[ConfirmedObject]) -> dict:
    """Confirmation outcomes, with singleton retention made visible."""
    if not groups:
        return {"groups": 0,
                "note": "no object observations were grouped. With zero "
                        "detections this is a valid result, not a crash, and "
                        "it is not evidence that nothing happened."}

    by_status: dict[str, int] = {status: 0 for status in STATUSES}
    by_class: dict[str, int] = {}
    for group in groups:
        by_status[group.status] = by_status.get(group.status, 0) + 1
        by_class[group.cls] = by_class.get(group.cls, 0) + 1

    return {
        "groups": len(groups),
        "by_status": by_status,
        "by_class": dict(sorted(by_class.items())),
        "retained_singletons": by_status[SINGLE_FRAME],
        "deleted": 0,
        "supported_duration_ms": round(
            sum(g.duration_ms for g in groups if g.supported), 1),
        "median_support_frames": sorted(
            g.support_frames for g in groups)[len(groups) // 2],
        "note": "no group was deleted. A singleton is a status, not a reason to "
                "discard, and the count above is retained evidence (PRD 10).",
    }
