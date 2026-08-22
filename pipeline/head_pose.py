"""Head and neck orientation from the keypoints AlphaPose already returns.

### Why this exists

`pose.py` decoded `nose`, `left_eye`, `right_eye`, `left_ear` and `right_ear`
on every observation, stored all five, and then used exactly one of them:

    observation.head_pitch = (nose_y - shoulder_y) / scale

That is a vertical offset, not an angle, and there was **no yaw at all** -- so
"turned toward a neighbour", the primary copying signal, was not measured
anywhere in the pipeline. This module uses the other four keypoints.

### Why not a face-landmark head-pose model

Every published exam-proctoring head-pose stack (MediaPipe FaceMesh, dlib-68,
RetinaFace, all feeding `cv2.solvePnP`) is built for webcam proctoring: one
frontal face filling the frame. Measured on this corpus, the interocular
distance is **9.3 px on video 01, 10.1 px on video 03 and 4.7 px on the paper
video**. FaceMesh wants ~64 px of face and dlib-68 wants ~80 px. Those methods
cannot run here, and eye-gaze tracking is not merely hard but impossible: at
this scale an eye is two or three pixels and there is no iris to track.

What *is* available is the position of five facial keypoints at 0.84-0.93
median confidence, on 73-96% of observations. Position survives at 10 px even
though detail does not, and orientation is a question about position.

### These are projective proxies, not Euler angles

A single uncalibrated camera cannot recover true head rotation from five
coplanar-ish points without a face model and intrinsics. Everything here is a
**normalised projective proxy**: monotonic in the real angle, comparable within
one seat over time, and not comparable to a protractor. They are reported in
normalised units and, where a degree figure is given, it is labelled
`approx_deg` and derived from an explicit spherical-head assumption.

That is why every threshold downstream is **relative to the seat's own
baseline** (PRD 17.6). The camera looks down at the front row and across at the
back, so a fixed "yaw beyond 30 degrees" rule would flag one row permanently
and never flag another.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field

# Facing states, in the pipeline's existing four-state observability idiom.
FRONTAL = "frontal"
TURNED_LEFT = "turned_left"
TURNED_RIGHT = "turned_right"
DOWNWARD = "downward"
NOT_RESOLVABLE = "not_resolvable_at_source_scale"
FACING_STATES = (FRONTAL, TURNED_LEFT, TURNED_RIGHT, DOWNWARD, NOT_RESOLVABLE)

VISIBLE = "visible"


@dataclass
class HeadPoseConfig:
    """Thresholds, and the reason each one has its value."""

    # A keypoint below this contributes nothing. AlphaPose's median facial
    # confidence on this corpus is 0.84-0.93, so 0.35 admits the weak tail
    # without admitting noise.
    min_keypoint_confidence: float = 0.35

    # Head scale, in pixels, below which no orientation claim is made. Ear
    # separation under this leaves the yaw denominator dominated by keypoint
    # jitter rather than by geometry.
    min_head_scale_px: float = 6.0

    # Yaw magnitude, in normalised units, past which the head is called turned
    # rather than frontal. 0.35 of a half-head-width is roughly 20 degrees
    # under the spherical assumption in `approx_yaw_deg`.
    turned_threshold: float = 0.35

    # How far past the subject's own resting orientation a value must sit
    # before it counts as a departure, on top of clearing `turned_threshold`.
    #
    # Both conditions are required, and each fixes the other's failure. Using
    # the absolute threshold alone flags normal posture: candidates face their
    # monitors while the camera sits at an angle, so a permanently turned head
    # is the resting state. Measured on video 04 before this rule: 45 of 53
    # excursions were `yaw_right`, 9 had a baseline exactly equal to their peak
    # -- tracks turned for their entire visible life -- and 18 peaked at
    # saturation. Using the baseline alone fails the other way: yaw saturates
    # at +/-1.0, so a bimodal distribution gives a huge MAD and a threshold
    # beyond the achievable range, which produced zero events across 42
    # subjects on video 01.
    min_departure_from_baseline: float = 0.30

    # Median absolute deviations above the seat's own median before a frame
    # counts toward a sustained excursion. Matches `PoseConfig.head_dwell_mad`.
    excursion_mad: float = 3.0

    # A dwell or excursion must last at least this long to be reported. A
    # glance is not an event; PS #1 names "look around due to distractions" and
    # "raise their heads while thinking" as normal behaviour, and both are
    # short.
    min_event_ms: float = 1500.0

    # Two excursions closer than this are one event with a wobble in it, not
    # two events. Left as None it is derived per track from the observed
    # sampling interval, which is the only correct default: the pose stage
    # samples at 1 Hz by default, and a fixed 800 ms gap is *shorter* than one
    # sample period, so every run was split into singletons and no event could
    # ever reach `min_event_ms`. Measured before the fix: 1 event across 379
    # observations of 29 tracks.
    merge_gap_ms: float | None = None
    merge_gap_intervals: float = 2.5

    # Repetition: excursions in the same direction within this window are
    # counted as a repeated pattern, which is the discriminator PS #1 asks for
    # between a stretch and sustained attention elsewhere.
    repetition_window_ms: float = 60000.0


@dataclass
class HeadPose:
    """One observation's head orientation, with everything it was derived from."""

    pose_row_id: str
    config_id: str
    pts_ms: float
    seat_state: str
    track_id: int | None = None

    yaw: float | None = None            # signed, normalised; +ve = subject's left
    pitch: float | None = None          # signed, normalised; +ve = downward
    roll: float | None = None           # signed, normalised
    yaw_relative_to_torso: float | None = None
    approx_yaw_deg: float | None = None

    head_scale_px: float = 0.0
    facing: str = NOT_RESOLVABLE
    evidence: str = ""
    keypoints_used: list = field(default_factory=list)

    def to_row(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def _point(joints: dict, name: str, cfg: HeadPoseConfig):
    joint = joints.get(name)
    if not joint:
        return None
    if joint.get("observability") != VISIBLE:
        return None
    if float(joint.get("confidence", 0.0)) < cfg.min_keypoint_confidence:
        return None
    return (float(joint["x"]), float(joint["y"]))


def estimate(joints: dict, pose_row_id: str = "", config_id: str = "",
             pts_ms: float = 0.0, seat_state: str = "",
             track_id: int | None = None,
             cfg: HeadPoseConfig | None = None) -> HeadPose:
    """Head orientation from one observation's facial keypoints.

    `joints` maps joint name to the stored row (`x`, `y`, `confidence`,
    `observability`).

    **Yaw.** The ears sit near the head's vertical rotation axis and the nose
    sits forward of it, so the nose's horizontal displacement from the ear
    midpoint, divided by half the ear separation, is `sin(yaw)` for a spherical
    head. When only one ear is resolved the head is turned far enough to
    occlude the other, which is itself the measurement -- direction is taken
    from which ear survived and the magnitude is reported as saturated.

    **Pitch.** The eyes sit forward of the ear axis, so pitching down carries
    the eye midpoint downward relative to the ear midpoint. Normalised by head
    scale, that is a projective proxy for pitch that does not depend on the
    shoulder line, which a slouching subject moves independently.

    **Roll.** The eye line's tilt, measured against the shoulder line rather
    than against the image horizontal, so a camera mounted off-level does not
    read as every subject having a tilted head.
    """
    cfg = cfg or HeadPoseConfig()
    out = HeadPose(pose_row_id=pose_row_id, config_id=config_id, pts_ms=pts_ms,
                   seat_state=seat_state, track_id=track_id)

    nose = _point(joints, "nose", cfg)
    left_eye = _point(joints, "left_eye", cfg)
    right_eye = _point(joints, "right_eye", cfg)
    left_ear = _point(joints, "left_ear", cfg)
    right_ear = _point(joints, "right_ear", cfg)
    left_shoulder = _point(joints, "left_shoulder", cfg)
    right_shoulder = _point(joints, "right_shoulder", cfg)

    used = [n for n, p in (("nose", nose), ("left_eye", left_eye),
                           ("right_eye", right_eye), ("left_ear", left_ear),
                           ("right_ear", right_ear)) if p]
    out.keypoints_used = used

    # --------------------------------------------------------------- scale --
    # Ear separation is the natural head width. Interocular is the fallback and
    # is roughly 0.5 of it, so it is scaled to stay on one ruler.
    if left_ear and right_ear:
        head_scale = math.hypot(left_ear[0] - right_ear[0],
                                left_ear[1] - right_ear[1])
    elif left_eye and right_eye:
        head_scale = 2.0 * math.hypot(left_eye[0] - right_eye[0],
                                      left_eye[1] - right_eye[1])
    else:
        head_scale = 0.0
    out.head_scale_px = round(head_scale, 2)

    if head_scale < cfg.min_head_scale_px:
        out.facing = NOT_RESOLVABLE
        out.evidence = (
            f"head scale {head_scale:.1f} px is under the "
            f"{cfg.min_head_scale_px:.0f} px floor; at this size the yaw "
            f"denominator is keypoint jitter, not geometry")
        return out

    # ----------------------------------------------------------------- yaw --
    if left_ear and right_ear and nose:
        ear_mid_x = (left_ear[0] + right_ear[0]) / 2.0
        # Sign convention: +ve when the nose lies toward the subject's left ear,
        # i.e. the subject has turned to their own left.
        direction = 1.0 if left_ear[0] >= right_ear[0] else -1.0
        out.yaw = max(-1.5, min(1.5,
                                direction * (nose[0] - ear_mid_x)
                                / (head_scale / 2.0)))
        out.approx_yaw_deg = round(
            math.degrees(math.asin(max(-1.0, min(1.0, out.yaw)))), 1)
        out.evidence = (f"nose offset from ear midpoint over half ear "
                        f"separation ({head_scale:.1f} px)")
    elif nose and (left_ear or right_ear):
        # One ear occluded: the head is turned far enough to hide it. The
        # surviving ear names the direction. Reported saturated rather than
        # numerically precise, because the geometry that would give a magnitude
        # is exactly the part that is occluded.
        out.yaw = 1.0 if left_ear else -1.0
        out.approx_yaw_deg = 90.0 if left_ear else -90.0
        out.evidence = (f"only the {'left' if left_ear else 'right'} ear is "
                        f"resolved; the other is occluded by the turn, so yaw "
                        f"is reported saturated in that direction")
    else:
        out.evidence = "neither ear pair nor a nose-plus-ear was resolved"

    # --------------------------------------------------------------- pitch --
    if left_eye and right_eye and left_ear and right_ear:
        eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0
        ear_mid_y = (left_ear[1] + right_ear[1]) / 2.0
        out.pitch = (eye_mid_y - ear_mid_y) / head_scale
    elif nose and left_eye and right_eye:
        eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0
        out.pitch = (nose[1] - eye_mid_y) / head_scale

    # ---------------------------------------------------------------- roll --
    if left_eye and right_eye:
        eye_angle = math.degrees(math.atan2(left_eye[1] - right_eye[1],
                                            left_eye[0] - right_eye[0]))
        if left_shoulder and right_shoulder:
            shoulder_angle = math.degrees(
                math.atan2(left_shoulder[1] - right_shoulder[1],
                           left_shoulder[0] - right_shoulder[0]))
            out.roll = ((eye_angle - shoulder_angle + 180.0) % 360.0) - 180.0
        else:
            out.roll = ((eye_angle + 180.0) % 360.0) - 180.0

    # ------------------------------------------- yaw relative to the torso --
    # A subject who rotates their whole body toward a neighbour and one who
    # keeps their shoulders square and turns only their neck are different
    # behaviours. Shoulder foreshortening carries the body's own rotation, so
    # subtracting it isolates the neck.
    if out.yaw is not None and left_shoulder and right_shoulder:
        shoulder_span = abs(left_shoulder[0] - right_shoulder[0])
        shoulder_len = math.hypot(left_shoulder[0] - right_shoulder[0],
                                  left_shoulder[1] - right_shoulder[1])
        if shoulder_len > 1e-6:
            # 1.0 when the shoulders face the camera squarely, falling toward 0
            # as the torso rotates away.
            squareness = max(0.0, min(1.0, shoulder_span / shoulder_len))
            out.yaw_relative_to_torso = round(out.yaw * squareness, 4)

    # -------------------------------------------------------------- facing --
    if out.yaw is None and out.pitch is None:
        out.facing = NOT_RESOLVABLE
    elif out.yaw is not None and abs(out.yaw) >= cfg.turned_threshold:
        out.facing = TURNED_LEFT if out.yaw > 0 else TURNED_RIGHT
    elif out.pitch is not None and out.pitch > 0.0:
        out.facing = DOWNWARD if out.pitch > 0.18 else FRONTAL
    else:
        out.facing = FRONTAL

    for name in ("yaw", "pitch", "roll"):
        value = getattr(out, name)
        if value is not None:
            setattr(out, name, round(value, 4))
    return out


# ------------------------------------------------------------- temporal --

@dataclass
class Excursion:
    """A sustained departure from this seat's own resting orientation."""

    seat_state: str
    track_id: int | None
    kind: str                      # "yaw_left" | "yaw_right" | "look_down"
    start_pts_ms: float
    end_pts_ms: float
    duration_ms: float
    peak_value: float
    frames: int
    baseline: float
    threshold: float

    def to_row(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def _robust_baseline(values: list[float]) -> tuple[float, float]:
    """Median and MAD -- the seat's own resting orientation and its spread."""
    if not values:
        return 0.0, 0.0
    median = statistics.median(values)
    mad = statistics.median([abs(v - median) for v in values]) or 0.0
    return median, mad


def _sample_interval(samples: list[tuple[float, float]]) -> float:
    """Median spacing between consecutive observations of one track."""
    gaps = [b - a for (a, _x), (b, _y) in zip(samples, samples[1:]) if b > a]
    return statistics.median(gaps) if gaps else 1000.0


def _runs(samples: list[tuple[float, float]], threshold: float, above: bool,
          cfg: HeadPoseConfig, gap_ms: float | None = None
          ) -> list[list[tuple[float, float]]]:
    """Contiguous stretches past a threshold, with short gaps bridged."""
    gap = gap_ms if gap_ms is not None else (
        cfg.merge_gap_ms if cfg.merge_gap_ms is not None else 2500.0)
    runs, current, last_ms = [], [], None
    for pts_ms, value in samples:
        hit = (value >= threshold) if above else (value <= threshold)
        if hit:
            if current and last_ms is not None and pts_ms - last_ms > gap:
                runs.append(current)
                current = []
            current.append((pts_ms, value))
            last_ms = pts_ms
        elif current and last_ms is not None and pts_ms - last_ms > gap:
            runs.append(current)
            current, last_ms = [], None
    if current:
        runs.append(current)
    return runs


def excursions(poses: list[HeadPose], cfg: HeadPoseConfig | None = None
               ) -> list[Excursion]:
    """Sustained head events, measured against each seat's own baseline.

    Three kinds, because they mean different things: a sustained turn to the
    left, a sustained turn to the right, and a prolonged downward dwell. The
    direction is kept because "toward the neighbour on one side, repeatedly" is
    the signal, and an unsigned magnitude discards it.
    """
    cfg = cfg or HeadPoseConfig()
    by_seat: dict[tuple, list[HeadPose]] = {}
    for pose in poses:
        by_seat.setdefault((pose.seat_state, pose.track_id), []).append(pose)

    out: list[Excursion] = []
    for (seat, track), group in by_seat.items():
        group.sort(key=lambda p: p.pts_ms)

        for kind, attribute, above in (("yaw_left", "yaw", True),
                                       ("yaw_right", "yaw", False),
                                       ("look_down", "pitch", True)):
            samples = [(p.pts_ms, getattr(p, attribute)) for p in group
                       if getattr(p, attribute) is not None]
            if len(samples) < 3:
                continue
            interval = _sample_interval(samples)
            gap = (cfg.merge_gap_ms if cfg.merge_gap_ms is not None
                   else interval * cfg.merge_gap_intervals)
            baseline, mad = _robust_baseline([v for _t, v in samples])
            # A seat whose MAD is zero has not moved; a threshold at the median
            # would then fire on every frame. Fall back to a fixed margin.
            spread = mad if mad > 1e-6 else 0.10
            threshold = (baseline + cfg.excursion_mad * spread if above
                         else baseline - cfg.excursion_mad * spread)
            # Cap the adaptive threshold at the absolute definition of the
            # behaviour, so it can never demand more than the measure can give.
            #
            # Yaw saturates at +/-1.0: when a turn occludes the far ear the
            # magnitude is reported saturated, because the geometry that would
            # give a finer number is exactly what is hidden. That makes the
            # distribution bimodal -- frontal near 0, turns at +/-1.0 -- and a
            # median-absolute-deviation over a bimodal signal is huge. Measured
            # on video 01 track 12: median -0.294, MAD 0.4621, so
            # median + 3*MAD = +1.093, ABOVE the saturation point. The
            # threshold was unreachable and zero events formed across 42
            # subjects and 1076 observations, while the same rows classified
            # 360 frames turned_left and 236 turned_right.
            if attribute == "yaw":
                limit = cfg.turned_threshold
                threshold = (min(threshold, limit) if above
                             else max(threshold, -limit))

            # A frame must clear the absolute threshold *and* sit at least
            # `min_departure_from_baseline` beyond this subject's own resting
            # orientation. Filtering the samples before run-finding keeps the
            # gap-bridging logic working on the qualifying frames only.
            departure = cfg.min_departure_from_baseline
            qualifying = [
                (t, v) for t, v in samples
                if (v - baseline >= departure if above
                    else baseline - v >= departure)]
            if len(qualifying) < 2:
                continue

            for run in _runs(qualifying, threshold, above, cfg, gap):
                # A run of N samples at this interval covers (N-1)*interval of
                # wall clock between its first and last sighting, plus the
                # interval it continues to hold after the last one. Using only
                # last-minus-first would report a two-sample event as
                # instantaneous at high sample rates and as long at low ones.
                duration = run[-1][0] - run[0][0] + interval
                if duration < cfg.min_event_ms or len(run) < 2:
                    continue
                peak = max(v for _t, v in run) if above \
                    else min(v for _t, v in run)
                out.append(Excursion(
                    seat_state=seat, track_id=track, kind=kind,
                    start_pts_ms=run[0][0], end_pts_ms=run[-1][0],
                    duration_ms=duration, peak_value=peak, frames=len(run),
                    baseline=baseline, threshold=threshold))

    out.sort(key=lambda e: (e.start_pts_ms, e.seat_state))
    return out


def neighbour_bias(events: list[Excursion], seat_positions: dict,
                   cfg: HeadPoseConfig | None = None) -> dict:
    """Do a seat's turns point at an occupied neighbour, and repeatedly?

    `seat_positions` maps seat id to an (x, y) centre from the calibration
    profile. A turn is neighbour-directed when a seat exists on the side the
    subject turned toward. Turning toward an aisle or a wall is the same head
    movement and is not the same observation.
    """
    cfg = cfg or HeadPoseConfig()
    report: dict[str, dict] = {}

    for event in events:
        if event.kind not in ("yaw_left", "yaw_right"):
            continue
        seat = event.seat_state
        block = report.setdefault(seat, {
            "yaw_left_events": 0, "yaw_right_events": 0,
            "toward_occupied_neighbour": 0, "toward_no_seat": 0,
            "total_turn_ms": 0.0, "repeated_same_direction": 0,
            "neighbours": {}})
        block[f"{event.kind}_events"] += 1
        block["total_turn_ms"] += event.duration_ms

        here = seat_positions.get(seat)
        if not here:
            block["toward_no_seat"] += 1
            continue
        # +yaw is the subject's own left. With a camera facing the seats, the
        # subject's left is toward smaller x in the image.
        wanted = -1 if event.kind == "yaw_left" else 1
        found = None
        for other, position in seat_positions.items():
            if other == seat:
                continue
            dx = position[0] - here[0]
            if dx == 0 or (dx > 0) != (wanted > 0):
                continue
            if abs(position[1] - here[1]) > abs(dx) * 1.5:
                continue                      # a different row, not a neighbour
            if found is None or abs(dx) < abs(found[1]):
                found = (other, dx)
        if found:
            block["toward_occupied_neighbour"] += 1
            block["neighbours"][found[0]] = \
                block["neighbours"].get(found[0], 0) + 1
        else:
            block["toward_no_seat"] += 1

    # Repetition, per direction, within the configured window.
    for seat, block in report.items():
        same = [e for e in events if e.seat_state == seat
                and e.kind in ("yaw_left", "yaw_right")]
        for kind in ("yaw_left", "yaw_right"):
            times = sorted(e.start_pts_ms for e in same if e.kind == kind)
            block["repeated_same_direction"] += sum(
                1 for a, b in zip(times, times[1:])
                if b - a <= cfg.repetition_window_ms)
        block["total_turn_ms"] = round(block["total_turn_ms"], 1)
    return report


def summarise(poses: list[HeadPose], events: list[Excursion],
              bias: dict | None = None,
              cfg: HeadPoseConfig | None = None) -> dict:
    """Per-seat head-behaviour metrics, with the resolvable fraction stated."""
    cfg = cfg or HeadPoseConfig()
    # Key by track as well as seat. Five of six cameras carry DRAFT
    # calibrations, so `seat_state` is "unattributed" for every observation and
    # keying on it alone pools every person in the hall into one baseline --
    # which is exactly the thing a per-seat baseline exists to avoid.
    def key_of(pose):
        if pose.seat_state and pose.seat_state != "unattributed":
            return pose.seat_state
        return (f"{pose.seat_state or 'unattributed'}/track_{pose.track_id}"
                if pose.track_id is not None else "unattributed/no_track")

    by_seat: dict[str, list[HeadPose]] = {}
    for pose in poses:
        by_seat.setdefault(key_of(pose), []).append(pose)

    def spread(values):
        values = sorted(v for v in values if v is not None)
        if not values:
            return None
        return {"n": len(values),
                "median": round(statistics.median(values), 3),
                "p10": round(values[int(0.10 * (len(values) - 1))], 3),
                "p90": round(values[int(0.90 * (len(values) - 1))], 3)}

    seats = {}
    for seat, group in sorted(by_seat.items()):
        resolvable = [p for p in group if p.facing != NOT_RESOLVABLE]
        facing = {state: sum(1 for p in group if p.facing == state)
                  for state in FACING_STATES}
        tracks = {p.track_id for p in group}
        mine = [e for e in events
                if e.seat_state == group[0].seat_state and e.track_id in tracks]
        downs = [e for e in mine if e.kind == "look_down"]
        turns = [e for e in mine if e.kind.startswith("yaw")]
        seats[seat] = {
            "observations": len(group),
            "resolvable": len(resolvable),
            "resolvable_fraction": round(len(resolvable) / len(group), 3)
            if group else 0.0,
            "median_head_scale_px": round(
                statistics.median([p.head_scale_px for p in group]), 1)
            if group else 0.0,
            "facing": facing,
            "yaw": spread([p.yaw for p in group]),
            "yaw_relative_to_torso": spread(
                [p.yaw_relative_to_torso for p in group]),
            "pitch": spread([p.pitch for p in group]),
            "roll": spread([p.roll for p in group]),
            "look_down_events": len(downs),
            "longest_look_down_ms": round(max((e.duration_ms for e in downs),
                                              default=0.0), 1),
            "total_look_down_ms": round(sum(e.duration_ms for e in downs), 1),
            "turn_events": len(turns),
            "longest_turn_ms": round(max((e.duration_ms for e in turns),
                                         default=0.0), 1),
            "neighbour_bias": (bias or {}).get(seat),
        }

    return {
        "observations": len(poses),
        "seats": len(seats),
        "resolvable_fraction": round(
            sum(1 for p in poses if p.facing != NOT_RESOLVABLE) / len(poses), 3)
        if poses else 0.0,
        "events": len(events),
        "events_by_kind": {kind: sum(1 for e in events if e.kind == kind)
                           for kind in ("yaw_left", "yaw_right", "look_down")},
        "config": asdict(cfg),
        "per_seat": seats,
        "note":
            "yaw, pitch and roll are normalised projective proxies, monotonic "
            "in the true angle and comparable within one seat over time. They "
            "are not protractor angles: recovering those needs camera "
            "intrinsics and a face model, and at 4.7-10.1 px interocular "
            "distance this corpus supports neither. Every threshold is set "
            "against the seat's own median and MAD (PRD 17.6), never a "
            "universal angle, because the camera looks down at the front row "
            "and across at the back.",
        "not_measurable":
            "Eye gaze and iris direction are not measured and cannot be at "
            "this source scale. Face identity is not computed anywhere.",
    }
