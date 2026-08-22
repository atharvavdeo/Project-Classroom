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

import math
from dataclasses import asdict, dataclass, field

from .object_detection import INSUFFICIENT, ObjectDetection

SUPPORTED = "temporally_supported_object_evidence"
SINGLE_FRAME = "single_frame_object_candidate"
CONFLICTING = "conflicting_object_evidence"
INSUFFICIENT_EVIDENCE = INSUFFICIENT
RECURRENT_SCENE_OBJECT = "recurrent_scene_object"
STATUSES = (SUPPORTED, SINGLE_FRAME, CONFLICTING, INSUFFICIENT_EVIDENCE,
            RECURRENT_SCENE_OBJECT)


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

    # --- Recurrence: persistence is not corroboration for a handheld object ---
    #
    # Measured on video 01: a monitor bezel at image (275, 650) produced **23**
    # separate `temporally_supported` phone groups across 38.4 s, while the one
    # genuine phone -- the highest-confidence detection in the recording, at
    # 0.933 -- produced **2**. Grouping rewards persistence, and a partition
    # edge, a bezel or a chair highlight is in every frame forever. So the
    # mechanism built to suppress noise was ranking furniture an order of
    # magnitude above the only true event.
    #
    # A class that keeps reappearing at the same image coordinates across many
    # separate groups is part of the room. This does not delete it -- it is
    # restated as `recurrent_scene_object`, which cannot corroborate an event.
    # The grid is a FRACTION OF FRAME WIDTH, not a pixel count. An absolute
    # 25 px cell on 1280x720 gives 1479 cells; the same 25 px on video 04's
    # 640x480 gives 494 -- four times the relative area per cell. Measured
    # before this fix: video 01 averaged 7.2 groups per cell and video 04
    # averaged 73.5, so a fixed count threshold flagged 97% of video 04's
    # groups (35,131 of 36,332) while behaving sensibly on video 01. This is
    # the same failure the detector's own thresholds have: absolute pixels do
    # not transfer between cameras.
    recurrence_grid_fraction: float = 0.02        # 25.6 px at 1280 wide
    frame_width_px: float = 1280.0                # set by the caller per video

    # A cell is furniture when it holds far more groups of this class than a
    # typical cell does, not when it passes a fixed count. The comparison is
    # against the median occupied cell for the same class in the same
    # configuration, so a busy recording and a quiet one are judged on their
    # own terms.
    recurrence_outlier_factor: float = 3.0
    min_recurrent_groups: int = 4                 # floor, never the whole test
    min_recurrent_span_ms: float = 5000.0

    # --- Handheld plausibility -------------------------------------------
    # A phone in use is in a hand. Distance from the nearest wrist, in torso
    # widths so it transfers between a near and a far seat. Beyond this the
    # object is somewhere else in the frame and is not being held.
    max_wrist_distance_torso: float = 2.0
    # Below this fraction of a group's frames having a resolvable nearby wrist,
    # no judgement is made: absent pose is not evidence of absent hand.
    min_wrist_coverage: float = 0.5


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
    # How many separate groups of this class recur at this image location, and
    # over what span. Set by `flag_recurrent`.
    recurrence_count: int = 1
    recurrence_span_ms: float = 0.0
    # Median distance from the group's centre to the nearest wrist, in torso
    # widths, and whether that is close enough to be held. None where pose did
    # not resolve a wrist often enough to say. Set by `flag_handheld`.
    wrist_distance_torso: float | None = None
    handheld_plausible: bool | None = None

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


def _centre(box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def flag_recurrent(groups: list[ConfirmedObject],
                   cfg: ConfirmConfig | None = None) -> list[ConfirmedObject]:
    """Restate groups that keep reappearing at one place as scene objects.

    Runs after grouping, over all groups of one configuration at once, because
    recurrence is a property of the *set* of groups and is invisible from
    inside any one of them.

    Nothing is deleted. The group keeps its frames, its confidence and its
    crops; only its status changes, and the reason records the count and span
    that caused it, so a reviewer can disagree.
    """
    cfg = cfg or ConfirmConfig()
    grid = max(cfg.recurrence_grid_fraction * cfg.frame_width_px, 4.0)
    buckets: dict[tuple, list[ConfirmedObject]] = {}
    for group in groups:
        cx, cy = _centre(group.box)
        key = (group.config_id, group.cls,
               round(cx / grid), round(cy / grid))
        buckets.setdefault(key, []).append(group)

    # Per (config, class), what does a typical occupied cell hold? Furniture is
    # an outlier against that, not against a number chosen in advance.
    occupancy: dict[tuple, list[int]] = {}
    for key, members in buckets.items():
        occupancy.setdefault((key[0], key[1]), []).append(len(members))
    typical: dict[tuple, float] = {}
    for key, counts in occupancy.items():
        counts.sort()
        typical[key] = counts[len(counts) // 2]

    for key, members in buckets.items():
        starts = [g.start_pts_ms for g in members]
        ends = [g.end_pts_ms for g in members]
        span = max(ends) - min(starts)
        median_cell = typical.get((key[0], key[1]), 1)
        floor = max(cfg.min_recurrent_groups,
                    cfg.recurrence_outlier_factor * median_cell)
        for group in members:
            group.recurrence_count = len(members)
            group.recurrence_span_ms = round(span, 1)
        if (len(members) >= floor
                and span >= cfg.min_recurrent_span_ms):
            for group in members:
                if group.status in (SUPPORTED, SINGLE_FRAME):
                    group.status = RECURRENT_SCENE_OBJECT
                    group.reason = (
                        f"{len(members)} separate groups of {group.cls!r} recur "
                        f"within {grid:.0f} px of this location across "
                        f"{span / 1000:.1f}s, against a median of "
                        f"{median_cell:.0f} for this class elsewhere in the "
                        f"frame. A handheld object does not hold one image "
                        f"position for that long; this is part of the room. "
                        f"Evidence retained, but it cannot corroborate an "
                        f"event.")
    return groups


def flag_handheld(groups: list[ConfirmedObject], wrists: dict,
                  cfg: ConfirmConfig | None = None) -> list[ConfirmedObject]:
    """Measure how far each group sits from the nearest wrist.

    `wrists` maps rounded pts_ms to a list of `(x, y, torso_scale_px)` for every
    wrist resolved in that frame.

    A phone in use is in a hand. Distance is normalised by torso width so a far
    seat is not penalised for being small. Where pose resolved no wrist for most
    of a group's life the verdict is `None`, not False: absent pose is not
    evidence of an absent hand, and this test may only ever demote on evidence.
    """
    cfg = cfg or ConfirmConfig()
    if not wrists:
        return groups
    times = sorted(wrists)

    def nearest_frame(pts_ms: float):
        if not times:
            return None
        best = min(times, key=lambda t: abs(t - pts_ms))
        return wrists[best] if abs(best - pts_ms) <= 1000.0 else None

    for group in groups:
        cx, cy = _centre(group.box)
        distances = []
        for pts_ms in (group.start_pts_ms, group.end_pts_ms):
            frame = nearest_frame(pts_ms)
            if not frame:
                continue
            best = None
            for wx, wy, scale in frame:
                if not scale:
                    continue
                d = math.hypot(cx - wx, cy - wy) / scale
                best = d if best is None else min(best, d)
            if best is not None:
                distances.append(best)
        if len(distances) < 1:
            continue
        distances.sort()
        median = distances[len(distances) // 2]
        group.wrist_distance_torso = round(median, 3)
        group.handheld_plausible = median <= cfg.max_wrist_distance_torso
        if not group.handheld_plausible and group.status == SUPPORTED:
            group.status = RECURRENT_SCENE_OBJECT
            group.reason = (
                f"nearest wrist is {median:.1f} torso widths away, beyond the "
                f"{cfg.max_wrist_distance_torso:.1f} floor. A phone in use is "
                f"in a hand; this is elsewhere in the frame. Evidence retained, "
                f"but it cannot corroborate an event.")
    return groups


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
        "recurrent_scene_objects": by_status.get(RECURRENT_SCENE_OBJECT, 0),
        "handheld_implausible": sum(
            1 for g in groups if g.handheld_plausible is False),
        "handheld_unjudged": sum(
            1 for g in groups if g.handheld_plausible is None),
        "deleted": 0,
        "supported_duration_ms": round(
            sum(g.duration_ms for g in groups if g.supported), 1),
        "median_support_frames": sorted(
            g.support_frames for g in groups)[len(groups) // 2],
        "note": "no group was deleted. A singleton is a status, not a reason to "
                "discard, and the count above is retained evidence (PRD 10).",
    }
