"""The frozen pose manifest and the pose A/B harness (PRD 9).

PRD 4 makes this stage's shape mandatory rather than advisory:

> Run tracking and RTMPose/RTMO/AlphaPose on the identical pose manifest. Do not
> select a pose model yet. [...] This is mandatory: otherwise an input/crop/
> threshold change can be incorrectly credited to a model.

So the manifest is frozen first, hashed, and handed unchanged to every
configuration. `manifest_digest` exists so that claim is checkable rather than
asserted: two configurations that report different digests were not compared,
whatever the report says, and `tests/test_pose_manifest_equivalence.py` fails
the run if they diverge.

### The four observability states

PRD 9 requires output that distinguishes visible, occluded, not detected, and
not resolvable at source scale. Collapsing any two of them loses the finding
this corpus is most likely to produce. A wrist that is *occluded by the desk* is
the interesting case -- it is what a hand in a lap looks like. A wrist that is
*not resolvable at source scale* is a statement about the camera, not the
person, and it is the difference between "we looked and saw nothing" and "this
footage cannot answer that question". A pipeline that reports both as a missing
keypoint has thrown away the distinction that decides whether more analysis or a
better camera is the answer.

### What pose may say

Only observable temporal context: baseline-relative head dwell, a wrist entering
desk/lap/bag context, repeated reach, torso movement, cross-seat ambiguity. And
one hard rule from PRD 9, enforced in `observe` rather than left to the ranker:

> A single neck turn or low-confidence wrist must never create a high-priority
> event.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .calibration import (AMBIGUOUS_SEAT, DESK_WORK_SURFACE, LAP_BAG,
                          UNATTRIBUTED, UPPER_BODY, CalibrationProfile)
from .seat_association import SeatAssociation

# COCO-17, matching `classroom.pose` so the two agree on what a wrist is.
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12

JOINT_NAMES = {
    NOSE: "nose", L_EYE: "left_eye", R_EYE: "right_eye",
    L_EAR: "left_ear", R_EAR: "right_ear",
    L_SHOULDER: "left_shoulder", R_SHOULDER: "right_shoulder",
    L_ELBOW: "left_elbow", R_ELBOW: "right_elbow",
    L_WRIST: "left_wrist", R_WRIST: "right_wrist",
    L_HIP: "left_hip", R_HIP: "right_hip",
}

# The four observability states PRD 9 requires be distinguishable.
VISIBLE = "visible"
OCCLUDED = "occluded"
NOT_DETECTED = "not_detected"
NOT_RESOLVABLE = "not_resolvable_at_source_scale"
OBSERVABILITY = (VISIBLE, OCCLUDED, NOT_DETECTED, NOT_RESOLVABLE)

# Named pose configurations (PRD 5).
RTMPOSE = "rtmpose_baseline"
RTMO = "rtmo_baseline"
ALPHAPOSE = "alphapose_crowd_baseline"
ALPHAPOSE_WHOLEBODY = "alphapose_wholebody"
POSE_CONFIGS = (RTMPOSE, RTMO, ALPHAPOSE)


@dataclass
class PoseConfig:
    # Below this a keypoint is not a measurement. Matches
    # `classroom.pose.KEYPOINT_CONF_MIN` so the legacy and new paths agree.
    keypoint_conf_min: float = 0.30

    # A person box shorter than this cannot support a wrist keypoint at all: at
    # 720p across an examination hall, a 60 px person is roughly 4 px of hand.
    # Anything reported there is the estimator's prior, not the image, and is
    # recorded as NOT_RESOLVABLE rather than believed or dropped.
    min_person_px: float = 60.0

    # A wrist must be this many px below the desk line before it counts as
    # having entered the lap context, so that a hand resting on the desk edge
    # does not read as a hand in a bag.
    lap_entry_margin_px: float = 8.0

    # PRD 9's hard floor: a single frame's observation never carries weight.
    # Two independent observations at different timestamps are the minimum for
    # a pose signal to be reported as sustained.
    min_support_frames: int = 2

    # Head dwell is measured against the seat's own baseline, never a universal
    # angle (PRD 17.6). This is how far past the baseline counts as a dwell.
    head_dwell_mad: float = 3.0


# ------------------------------------------------------------- manifest --

@dataclass
class PoseRow:
    """One person crop every pose configuration must process, identically."""

    pose_row_id: str
    row_id: str                      # the person-detection sample row
    event_id: str
    source_video_sha256: str
    pts_ms: float
    frame_index: int | None
    # The person source, so a configuration cannot quietly pick a different crop
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    box_origin: str = "detected"
    track_id: int | None = None
    # Seat candidates, not a seat. A pose row over an ambiguous boundary carries
    # both candidates and their scores; forcing one here would push a decision
    # the geometry refused to make into every downstream artifact.
    seat_state: str = UNATTRIBUTED
    seat_candidates: dict[str, float] = field(default_factory=dict)
    health_state: list[str] = field(default_factory=list)
    calibration_version: str = ""
    camera_epoch: str = ""

    def to_row(self) -> dict:
        return asdict(self)

    def canonical(self) -> str:
        """The bytes that define this row's identity, for the digest."""
        return json.dumps({
            "row_id": self.row_id,
            "sha256": self.source_video_sha256,
            "pts_ms": round(self.pts_ms, 3),
            "box": [round(v, 3) for v in self.box],
            "track_id": self.track_id,
            "seat_state": self.seat_state,
        }, sort_keys=True)


def pose_row_id(video_sha: str, pts_ms: float,
                box: tuple[float, float, float, float]) -> str:
    """Deterministic per PRD 9, and derived from the crop rather than an index.

    Keyed on the box as well as the timestamp because two people in one frame
    are two rows, and a per-frame counter would renumber them the moment the
    detector's output order changed.
    """
    key = f"{video_sha}|{pts_ms:.3f}|{box[0]:.2f},{box[1]:.2f},{box[2]:.2f},{box[3]:.2f}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def freeze(associations: list[SeatAssociation], rows_by_id: dict,
           frame_index_of=None) -> list[PoseRow]:
    """Freeze the pose manifest from associated person boxes.

    `rows_by_id` maps a person-detection `row_id` to its `SampleRow`, so the
    video hash, event and calibration version travel with the pose row rather
    than being re-derived later from something that might have changed.
    """
    out: list[PoseRow] = []
    for association in associations:
        sample = rows_by_id.get(association.row_id)
        if sample is None:
            # An interpolated box has no manifest row behind it. It is a
            # continuity aid, not something to run a pose model on.
            continue

        # Checked against the instance's real attributes, not just its declared
        # fields. A leak that arrives by assignment rather than by schema is
        # still a leak, and it is the easier of the two to introduce.
        carried = set(sample.to_row()) | set(getattr(sample, "__dict__", {}))
        leaked = [k for k in sorted(carried) if "reference" in k.lower()]
        if leaked:
            raise ValueError(
                f"sample row {association.row_id} carries reference field(s) "
                f"{leaked}; PRD 15 makes the reference manifest readable only "
                f"at stage 12")

        out.append(PoseRow(
            pose_row_id=pose_row_id(sample.source_video_sha256,
                                    association.pts_ms, association.box),
            row_id=association.row_id,
            event_id=sample.event_id,
            source_video_sha256=sample.source_video_sha256,
            pts_ms=round(association.pts_ms, 3),
            frame_index=frame_index_of(association.pts_ms)
            if frame_index_of else None,
            box=tuple(round(float(v), 3) for v in association.box),
            box_origin="detected",
            track_id=association.track_id,
            seat_state=association.state,
            seat_candidates=dict(association.scores),
            health_state=list(association.health_state),
            calibration_version=sample.calibration_version,
            camera_epoch=sample.camera_epoch,
        ))

    out.sort(key=lambda r: (r.pts_ms, r.pose_row_id))
    return out


def manifest_digest(rows: list[PoseRow]) -> str:
    """One hash over the whole manifest.

    The proof that PRD 4's "identical rows" requirement held. Every
    configuration records this digest; a report comparing two configurations
    with different digests is comparing two experiments, not two models.
    """
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda r: r.pose_row_id):
        digest.update(row.canonical().encode())
        digest.update(b"\n")
    return digest.hexdigest()


# ---------------------------------------------------------- observation --

@dataclass
class JointState:
    name: str
    x: float
    y: float
    confidence: float
    observability: str

    def to_row(self) -> dict:
        return asdict(self)


@dataclass
class PoseObservation:
    """One configuration's reading of one manifest row."""

    pose_row_id: str
    config_id: str
    pts_ms: float
    seat_state: str
    joints: list[JointState] = field(default_factory=list)
    head_pitch: float | None = None
    wrist_in_desk_context: bool = False
    wrist_in_lap_context: bool = False
    torso_scale_px: float | None = None
    person_height_px: float = 0.0
    # A left/right pair whose horizontal order is reversed relative to the
    # shoulders. Reported for review, never auto-corrected: silently swapping
    # them would make a wrong estimate look right.
    suspected_joint_swaps: list[str] = field(default_factory=list)
    note: str = ""

    def state(self, index: int) -> str:
        for joint in self.joints:
            if joint.name == JOINT_NAMES.get(index):
                return joint.observability
        return NOT_DETECTED

    @property
    def visible_joints(self) -> int:
        return sum(1 for j in self.joints if j.observability == VISIBLE)

    def to_row(self) -> dict:
        return {
            "pose_row_id": self.pose_row_id,
            "config_id": self.config_id,
            "pts_ms": round(self.pts_ms, 3),
            "seat_state": self.seat_state,
            "joints": [j.to_row() for j in self.joints],
            "head_pitch": round(self.head_pitch, 4)
            if self.head_pitch is not None else None,
            "wrist_in_desk_context": self.wrist_in_desk_context,
            "wrist_in_lap_context": self.wrist_in_lap_context,
            "torso_scale_px": round(self.torso_scale_px, 2)
            if self.torso_scale_px is not None else None,
            "person_height_px": round(self.person_height_px, 1),
            "visible_joints": self.visible_joints,
            "suspected_joint_swaps": self.suspected_joint_swaps,
            "note": self.note,
        }

    @classmethod
    def from_row(cls, row: dict) -> "PoseObservation":
        """Rebuild an observation from its persisted row.

        The inverse of `to_row`, so a later stage can weigh pose evidence
        without re-running a pose model. `visible_joints` is deliberately not
        restored: it is a derived property, and accepting a stored value would
        let a hand-edited artifact claim visibility the joints do not show.
        """
        return cls(
            pose_row_id=row["pose_row_id"],
            config_id=row.get("config_id", ""),
            pts_ms=float(row.get("pts_ms", 0.0)),
            seat_state=row.get("seat_state", ""),
            joints=[JointState(**joint) for joint in row.get("joints", [])],
            head_pitch=row.get("head_pitch"),
            wrist_in_desk_context=bool(row.get("wrist_in_desk_context", False)),
            wrist_in_lap_context=bool(row.get("wrist_in_lap_context", False)),
            torso_scale_px=row.get("torso_scale_px"),
            person_height_px=float(row.get("person_height_px", 0.0)),
            suspected_joint_swaps=list(row.get("suspected_joint_swaps", [])),
            note=row.get("note", ""),
        )


def _observability(keypoints: np.ndarray, index: int, row: PoseRow,
                   desk_line_y: float | None, cfg: PoseConfig) -> str:
    """Which of PRD 9's four states this joint is in.

    Order matters. Source scale is checked first because it is a property of the
    footage and applies whatever the estimator returned; a confident keypoint on
    a 40 px person is still not resolvable, and believing it is exactly the
    error this state exists to prevent.
    """
    height = row.box[3] - row.box[1]
    if height < cfg.min_person_px:
        return NOT_RESOLVABLE

    confidence = float(keypoints[index, 2])
    if confidence >= cfg.keypoint_conf_min:
        return VISIBLE

    # A low-confidence wrist below the desk line is the desk occluding it, which
    # is a different fact from the estimator having failed. Distinguishing them
    # is what turns "no keypoint" into evidence rather than a gap.
    if index in (L_WRIST, R_WRIST) and desk_line_y is not None \
            and float(keypoints[index, 1]) > desk_line_y:
        return OCCLUDED
    if confidence > 0.0:
        return OCCLUDED
    return NOT_DETECTED


def observe(keypoints: np.ndarray, row: PoseRow, config_id: str,
            profile: CalibrationProfile | None = None,
            cfg: PoseConfig | None = None) -> PoseObservation:
    """Turn raw keypoints into an observation, with every absence named."""
    cfg = cfg or PoseConfig()
    keypoints = np.asarray(keypoints, dtype=float).reshape(-1, 3)

    desk_line_y = None
    if profile is not None and row.seat_state not in (AMBIGUOUS_SEAT, UNATTRIBUTED):
        seat = profile.seat(row.seat_state)
        if seat is not None:
            desk_line_y = seat.desk_line_y

    height = row.box[3] - row.box[1]
    observation = PoseObservation(
        pose_row_id=row.pose_row_id, config_id=config_id, pts_ms=row.pts_ms,
        seat_state=row.seat_state, person_height_px=height)

    for index, name in JOINT_NAMES.items():
        if index >= len(keypoints):
            observation.joints.append(
                JointState(name, 0.0, 0.0, 0.0, NOT_DETECTED))
            continue
        observation.joints.append(JointState(
            name=name,
            x=round(float(keypoints[index, 0]), 2),
            y=round(float(keypoints[index, 1]), 2),
            confidence=round(float(keypoints[index, 2]), 4),
            observability=_observability(keypoints, index, row, desk_line_y, cfg)))

    if height < cfg.min_person_px:
        observation.note = (
            f"person box is {height:.0f} px tall, under the {cfg.min_person_px:.0f} "
            f"px this source can resolve a hand at; every joint is reported "
            f"{NOT_RESOLVABLE} and no pose context is derived")
        return observation

    ls, rs = keypoints[L_SHOULDER], keypoints[R_SHOULDER]
    if min(ls[2], rs[2]) >= cfg.keypoint_conf_min:
        scale = float(np.hypot(ls[0] - rs[0], ls[1] - rs[1]))
        observation.torso_scale_px = scale
        nose = keypoints[NOSE]
        if nose[2] >= cfg.keypoint_conf_min and scale > 1e-3:
            shoulder_y = (ls[1] + rs[1]) / 2.0
            observation.head_pitch = float((nose[1] - shoulder_y) / scale)

    for index in (L_WRIST, R_WRIST):
        if observation.state(index) != VISIBLE:
            continue
        y = float(keypoints[index, 1])
        if desk_line_y is None:
            continue
        if y > desk_line_y + cfg.lap_entry_margin_px:
            observation.wrist_in_lap_context = True
        else:
            observation.wrist_in_desk_context = True

    # Left/right ordering check. A seated person facing the camera has their
    # left shoulder on the image right; a reversal means the estimator has
    # likely swapped the pair. Flagged, never fixed.
    for left, right, label in ((L_SHOULDER, R_SHOULDER, "shoulders"),
                               (L_WRIST, R_WRIST, "wrists"),
                               (L_HIP, R_HIP, "hips")):
        if (observation.state(left) == VISIBLE
                and observation.state(right) == VISIBLE
                and observation.state(L_SHOULDER) == VISIBLE
                and observation.state(R_SHOULDER) == VISIBLE):
            shoulder_sign = np.sign(ls[0] - rs[0])
            pair_sign = np.sign(keypoints[left, 0] - keypoints[right, 0])
            if shoulder_sign != 0 and pair_sign != 0 \
                    and shoulder_sign != pair_sign and label != "shoulders":
                observation.suspected_joint_swaps.append(label)

    return observation


# ------------------------------------------------------------- A/B runner --

@dataclass
class PoseResult:
    config_id: str
    state: str = "available"
    manifest_digest: str = ""
    observations: list[PoseObservation] = field(default_factory=list)
    rows_seen: int = 0
    persons_proposed: int = 0
    persons_returned: int = 0
    seconds: float = 0.0
    peak_vram_mb: float | None = None
    checkpoint_sha256: str = ""
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.state == "available"

    def metrics(self) -> dict:
        if not self.available:
            return {"config_id": self.config_id, "state": self.state,
                    "reason": self.reason, "manifest_digest": self.manifest_digest,
                    "note": "this configuration did not run; its absence is "
                            "recorded rather than filled in from another model"}

        counts = {state: 0 for state in OBSERVABILITY}
        wrist_counts = {state: 0 for state in OBSERVABILITY}
        for observation in self.observations:
            for joint in observation.joints:
                counts[joint.observability] += 1
            for index in (L_WRIST, R_WRIST):
                wrist_counts[observation.state(index)] += 1

        total_joints = sum(counts.values()) or 1
        pitched = [o.head_pitch for o in self.observations
                   if o.head_pitch is not None]
        return {
            "config_id": self.config_id,
            "state": self.state,
            "manifest_digest": self.manifest_digest,
            "checkpoint_sha256": self.checkpoint_sha256,
            "rows_seen": self.rows_seen,
            "persons_proposed": self.persons_proposed,
            "persons_returned": self.persons_returned,
            "row_coverage": round(len(self.observations) / self.rows_seen, 4)
            if self.rows_seen else 0.0,
            "joint_observability": counts,
            "joint_visible_fraction": round(counts[VISIBLE] / total_joints, 4),
            "wrist_observability": wrist_counts,
            "head_pitch_resolved": len(pitched),
            "suspected_joint_swaps": sum(len(o.suspected_joint_swaps)
                                         for o in self.observations),
            "wrist_in_lap_context": sum(1 for o in self.observations
                                        if o.wrist_in_lap_context),
            "seconds": round(self.seconds, 3),
            "rows_per_second": round(self.rows_seen / self.seconds, 2)
            if self.seconds > 0 else None,
            "peak_vram_mb": self.peak_vram_mb,
        }


def run(rows: list[PoseRow], runner, config_id: str,
        profile: CalibrationProfile | None = None,
        cfg: PoseConfig | None = None,
        checkpoint_sha256: str = "") -> PoseResult:
    """Run one pose configuration over the frozen manifest.

    `runner` maps a `PoseRow` to an iterable of (17, 3) keypoint arrays. Every
    configuration receives this identical list, and the digest it records is
    computed from the rows it was actually given, not from the rows it was
    supposed to be given.
    """
    cfg = cfg or PoseConfig()
    result = PoseResult(config_id=config_id,
                        manifest_digest=manifest_digest(rows),
                        checkpoint_sha256=checkpoint_sha256)
    started = time.perf_counter()

    for row in rows:
        try:
            people = list(runner(row))
        except Exception as exc:                        # noqa: BLE001
            result.state = "launch_failed"
            result.reason = f"{type(exc).__name__}: {exc}"
            break

        result.rows_seen += 1
        result.persons_proposed += len(people)
        for keypoints in people:
            result.observations.append(
                observe(keypoints, row, config_id, profile, cfg))
            result.persons_returned += 1

    result.seconds = time.perf_counter() - started
    return result


def unavailable(config_id: str, state: str, reason: str,
                rows: list[PoseRow] | None = None) -> PoseResult:
    """A pose configuration that could not run, retained in the comparison."""
    return PoseResult(config_id=config_id, state=state, reason=reason,
                      manifest_digest=manifest_digest(rows or []))


# ------------------------------------------------------- pose evidence --

@dataclass
class PoseEvidence:
    """What pose observed over one event. Observational vocabulary only."""

    event_id: str
    seat_state: str
    config_id: str
    frames: int
    observations_used: int
    head_dwell_frames: int = 0
    wrist_desk_frames: int = 0
    wrist_lap_frames: int = 0
    repeated_reach: int = 0
    sustained: bool = False
    cross_seat_ambiguity: bool = False
    reason: str = ""

    def to_row(self) -> dict:
        return asdict(self)


def evidence(observations: list[PoseObservation], event_id: str,
             baseline_pitch: float | None = None,
             baseline_mad: float | None = None,
             cfg: PoseConfig | None = None) -> PoseEvidence:
    """Summarise pose over one event under PRD 9's constraints.

    Head dwell is measured against the seat's own baseline, never a universal
    angle (PRD 17.6): the camera looks down at some rows and across at others,
    so a fixed "head below 20 degrees" rule would flag one row of the hall
    permanently and never flag another.

    `sustained` is the flag downstream ranking is permitted to weight, and it
    requires at least `min_support_frames` independent observations. A single
    neck turn cannot set it, which is PRD 9's rule made structural.
    """
    cfg = cfg or PoseConfig()
    used = [o for o in observations if o.person_height_px >= cfg.min_person_px]
    seats = {o.seat_state for o in observations}

    out = PoseEvidence(
        event_id=event_id,
        seat_state=next(iter(seats)) if len(seats) == 1 else AMBIGUOUS_SEAT,
        config_id=observations[0].config_id if observations else "",
        frames=len(observations),
        observations_used=len(used),
        cross_seat_ambiguity=len(seats) > 1,
    )

    if not used:
        out.reason = ("no observation met the source-scale floor; pose cannot "
                      "say anything about this event and does not")
        return out

    if baseline_pitch is not None and baseline_mad:
        threshold = baseline_pitch + cfg.head_dwell_mad * baseline_mad
        out.head_dwell_frames = sum(
            1 for o in used if o.head_pitch is not None and o.head_pitch > threshold)

    out.wrist_desk_frames = sum(1 for o in used if o.wrist_in_desk_context)
    out.wrist_lap_frames = sum(1 for o in used if o.wrist_in_lap_context)

    # A reach is a lap entry preceded by a frame with no lap entry: the
    # transition, not the state. Counting states would report one long rest as
    # thirty reaches.
    previous = False
    for observation in used:
        if observation.wrist_in_lap_context and not previous:
            out.repeated_reach += 1
        previous = observation.wrist_in_lap_context

    strongest = max(out.head_dwell_frames, out.wrist_lap_frames)
    out.sustained = strongest >= cfg.min_support_frames
    out.reason = (
        f"{strongest} supporting observation(s) against a floor of "
        f"{cfg.min_support_frames}; "
        + ("sustained pose context" if out.sustained else
           "single-frame or absent pose context, which PRD 9 forbids from "
           "creating a high-priority event"))
    return out


# --------------------------------------------------------- comparison --

def compare(results: list[PoseResult], rows: list[PoseRow]) -> dict:
    """The matched pose comparison report (PRD 9).

    Reports every configuration including the ones that did not run, and
    verifies that all of them saw the same manifest before comparing anything.
    """
    expected = manifest_digest(rows)
    mismatched = [r.config_id for r in results
                  if r.manifest_digest and r.manifest_digest != expected]

    crowding: dict[str, int] = {}
    for row in rows:
        bucket = "ambiguous" if row.seat_state == AMBIGUOUS_SEAT else (
            "unattributed" if row.seat_state == UNATTRIBUTED else "seated")
        crowding[bucket] = crowding.get(bucket, 0) + 1

    health: dict[str, int] = {}
    for row in rows:
        for condition in row.health_state or ["none"]:
            health[condition] = health.get(condition, 0) + 1

    return {
        "manifest_digest": expected,
        "manifest_rows": len(rows),
        "manifest_equivalence": "verified" if not mismatched else "VIOLATED",
        "mismatched_configs": mismatched,
        "equivalence_note":
            "every configuration below was given byte-identical manifest rows; "
            "a difference in their metrics is therefore attributable to the "
            "model (PRD 4)" if not mismatched else
            "configurations disagreed on the manifest they processed. Their "
            "metrics are NOT comparable and no selection may be made from them.",
        "rows_by_seat_state": crowding,
        "rows_by_health_state": health,
        "configurations": [r.metrics() for r in results],
        "available": [r.config_id for r in results if r.available],
        "unavailable": [{"config_id": r.config_id, "state": r.state,
                         "reason": r.reason} for r in results if not r.available],
        "selection": "none. PRD 5 forbids naming a configuration best until "
                     "every runnable mandatory configuration has a matched "
                     "report; selection happens in stage 12 against references.",
    }


# ------------------------------------------------------------- runners --

# Which rtmlib model each named configuration means. Kept here rather than
# inferred from the config_id string so that renaming a configuration cannot
# silently change which model runs under it.
RTMLIB_MODEL = {
    RTMPOSE: "topdown",       # YOLOX-m person detector into RTMPose-m
    RTMO: "rtmo-m",           # one-stage, cost independent of crowd size
    "rtmo_large": "rtmo-l",
}


def rtmlib_runner(config_id: str, frame_reader=None,
                  backend: str = "onnxruntime", device: str = "cpu"):
    """Adapter over the measured `classroom.pose.PoseEstimator` (rtmlib).

    Built lazily. A configuration whose weights or runtime are absent must
    surface as an unavailable record from preflight, not as an import error part
    way through a run.
    """
    from classroom.pose import PoseEstimator

    model = RTMLIB_MODEL.get(config_id)
    if model is None:
        raise ValueError(
            f"{config_id!r} has no declared rtmlib model. Add it to "
            f"RTMLIB_MODEL rather than guessing from the name.")
    estimator = PoseEstimator(model, backend=backend, device=device)

    # Each model runs on the **full frame**, in its native mode, and its output
    # is then matched to the manifest row's person box.
    #
    # The alternative -- cropping to the row's box and estimating inside it --
    # was tried first and produced a result that looked like a finding and was
    # not one: RTMPose returned 70 poses over 74 rows, RTMO returned 1. That is
    # not a difference between the models. A top-down model runs its own person
    # detector and finds the single person filling a crop easily; a one-stage
    # model is trained on whole scenes and an upscaled single-person crop is out
    # of distribution for it. The crop was handicapping one competitor, which is
    # precisely the "input change credited to a model" PRD 4 exists to prevent.
    #
    # Identical rows are still guaranteed: both models are asked about the same
    # person at the same timestamp, and both are free to answer natively.
    cache: dict[float, list] = {}

    def run_row(row: PoseRow):
        if frame_reader is None:
            raise RuntimeError(
                f"{config_id} needs a frame_reader mapping a PoseRow to a "
                f"decoded frame; without one there is nothing to estimate on")

        # Co-temporal rows -- several people in one frame -- share one
        # inference. Keyed on PTS, and holding one entry, because the reader
        # walks the video forward and never returns to an earlier timestamp.
        if row.pts_ms not in cache:
            frame = frame_reader(row)
            cache.clear()
            cache[row.pts_ms] = [] if frame is None else estimator(frame)
        people = cache[row.pts_ms]

        matched, best = [], 0.0
        for keypoints in people:
            array = np.asarray(keypoints, dtype=float)
            visible = array[array[:, 2] >= 0.30]
            if len(visible) < 2:
                continue
            person = (float(visible[:, 0].min()), float(visible[:, 1].min()),
                      float(visible[:, 0].max()), float(visible[:, 1].max()))
            score = _overlap(person, row.box)
            if score > best:
                best, matched = score, [array]

        # Below this the returned person is somebody else in the frame. An
        # unmatched row is a row this configuration did not answer, which is a
        # measurement, not a gap to be filled from the nearest pose.
        return matched if best >= 0.30 else []

    return run_row


def alphapose_runner(config_id: str, model_config: str, checkpoint: str,
                     frame_reader=None, device: str = "cpu"):
    """Adapter over AlphaPose, built from official MVIG-SJTU source.

    Top-down by construction: AlphaPose estimates keypoints inside a person box
    that something else supplied. Here that box is the manifest row's, which is
    exactly what PRD 4 wants -- every configuration answering about the same
    person at the same timestamp.

    Unlike the rtmlib adapter this one is *given* the box rather than finding
    it, so there is no full-frame pass and no IoU match back. That is a real
    difference between the configurations and it is recorded in the report
    rather than hidden: a top-down model handed a box cannot miss the person,
    but it also cannot recover from a bad box.

    The affine crop below is AlphaPose's own `get_affine_transform` convention
    (centre/scale to a 256x192 input). Using a plain resize instead would shift
    every keypoint by the aspect-ratio difference, which looks plausible and is
    wrong everywhere.
    """
    import numpy as _np
    import torch
    from alphapose.models import builder
    from alphapose.utils.config import update_config
    from alphapose.utils.transforms import get_affine_transform, get_func_heatmap_to_coord

    import cv2

    cfg = update_config(model_config)
    model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu",
                                     weights_only=True), strict=True)

    # Asking for CUDA on a CPU-only torch raises inside `.to()`, which would
    # fail the whole pose stage rather than run it slowly. Degrade instead, and
    # say so: a run that silently used the CPU when the operator asked for the
    # GPU is a run whose timings mean something other than they appear to.
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"    AlphaPose: {device!r} requested but torch reports no CUDA "
              f"device (torch {torch.__version__}); running on CPU")
        device = "cpu"
    if device.startswith("cuda"):
        print(f"    AlphaPose on {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 2**20:.0f} MiB)")
    model.to(device).eval()
    heatmap_to_coord = get_func_heatmap_to_coord(cfg)
    input_h, input_w = cfg.DATA_PRESET.IMAGE_SIZE
    aspect_ratio = float(input_w) / input_h

    def run_row(row: PoseRow):
        if frame_reader is None:
            raise RuntimeError(
                f"{config_id} needs a frame_reader mapping a PoseRow to a "
                f"decoded frame; without one there is nothing to estimate on")
        frame = frame_reader(row)
        if frame is None:
            return []

        x1, y1, x2, y2 = row.box
        centre = _np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=_np.float32)
        width, height = x2 - x1, y2 - y1
        # Pad the box to the model's aspect ratio, as AlphaPose does, so the
        # person is not squashed into the input.
        if width > aspect_ratio * height:
            height = width / aspect_ratio
        else:
            width = height * aspect_ratio
        scale = _np.array([width, height], dtype=_np.float32) * 1.25

        trans = get_affine_transform(centre, scale, 0, [input_w, input_h])
        crop = cv2.warpAffine(frame, trans, (int(input_w), int(input_h)),
                              flags=cv2.INTER_LINEAR)
        blob = crop[:, :, ::-1].astype(_np.float32) / 255.0
        blob = (blob - _np.array([0.406, 0.457, 0.480], dtype=_np.float32))
        tensor = torch.from_numpy(blob.transpose(2, 0, 1)[None]).to(device)

        with torch.no_grad():
            heatmap = model(tensor).cpu().numpy()

        # Project heatmap peaks back to source coordinates. A keypoint left in
        # heatmap space is wrong by the affine transform and still draws a
        # plausible overlay.
        bbox = [float(centre[0] - scale[0] / 2), float(centre[1] - scale[1] / 2),
                float(centre[0] + scale[0] / 2), float(centre[1] + scale[1] / 2)]
        coords, scores = heatmap_to_coord(
            heatmap[0], bbox, hm_shape=cfg.DATA_PRESET.HEATMAP_SIZE,
            norm_type=None)
        out = _np.zeros((17, 3), dtype=float)
        out[:, :2] = _np.asarray(coords)[:17]
        out[:, 2] = _np.asarray(scores).reshape(-1)[:17]
        return [out] if float(out[:, 2].max()) >= 0.05 else []

    return run_row


def _overlap(a: tuple[float, float, float, float],
             b: tuple[float, float, float, float]) -> float:
    """IoU, used to decide whether a returned pose is the row's person."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
