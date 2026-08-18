"""The object crop manifest and its four named crop policies (PRD 10).

PRD 10 opens with the sentence that orders this whole stage:

> Hand-first is refinement, not primary detection. A hand may be hidden or
> mislocalised.

So the crop policy is a *fallback chain*, not a choice. `pose_hand_context` is
used when the wrist is reliable; when it is not, the crop falls back through
`lap_bag_context` and `seat_desk_context` to `native_source_context_fallback`,
and the reason it fell back is recorded on the crop. A pipeline that produced a
hand crop regardless would be inventing a hand location from a low-confidence
keypoint, and every detection inside it would inherit that invention while
looking like ordinary evidence.

### On resolution

PRD 10 is explicit that upscaling cannot create detail:

> Upscaling/tile/VLM cannot invent detail; output `insufficient_visual_evidence`
> where resolution does not support the object.

Every crop therefore records its **native** pixel size -- the source-frame
footprint before any resize -- alongside the resized input. Downstream
abstention is decided on the native size. A 20x14 px region magnified to 640x640
is still 20x14 px of evidence, and the only honest thing a detector can say
about a phone-sized object in it is that the source does not support the call.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

# The four named crop configurations (PRD 5).
POSE_HAND = "pose_hand_context"
SEAT_DESK = "seat_desk_context"
LAP_BAG = "lap_bag_context"
NATIVE_FALLBACK = "native_source_context_fallback"
CROP_CONFIGS = (POSE_HAND, SEAT_DESK, LAP_BAG, NATIVE_FALLBACK)

# Why a crop is not the one that was first asked for.
FALLBACK_NO_POSE = "no_pose_observation"
FALLBACK_LOW_WRIST = "wrist_confidence_below_floor"
FALLBACK_UNRESOLVABLE = "wrist_not_resolvable_at_source_scale"
FALLBACK_NO_SEAT = "seat_unattributed"
FALLBACK_OUT_OF_FRAME = "crop_left_the_frame"


@dataclass
class CropConfig:
    # Context around a wrist, in multiples of the person's shoulder width. A
    # hand crop must contain the hand, the forearm direction and the held-object
    # area (PRD 10.4); one shoulder width either side covers a held phone and
    # the desk surface under it without swallowing the neighbour.
    hand_context_torso: float = 1.0

    # Floor below which a wrist is not a location. Above this the keypoint
    # positions the crop; below it the crop falls back and says so.
    wrist_confidence_min: float = 0.40

    # Smallest crop, in source pixels, that is worth cutting at all. Below this
    # the crop is emitted with an abstention flag rather than dropped: PRD 10
    # wants an explicit `insufficient_visual_evidence`, not an absence.
    min_native_px: int = 24

    # Native footprint below which no object call may be made. Measured on this
    # corpus a phone occupies roughly 30x20 px; a region under this cannot hold
    # one plus enough surround to identify it.
    abstain_below_native_px: int = 32

    # The square input every detector configuration resizes into.
    input_size: int = 640


@dataclass
class ObjectCrop:
    """One crop every detector configuration must process, identically."""

    crop_id: str
    pose_row_id: str
    event_id: str
    source_video_sha256: str
    pts_ms: float
    frame_index: int | None = None

    crop_config: str = NATIVE_FALLBACK
    requested_config: str = NATIVE_FALLBACK
    fallback_reason: str = ""

    # Source-frame rectangle. Every later projection is defined against this.
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0

    resize_method: str = "letterbox_aspect_preserved"
    input_size: int = 640

    seat_state: str = "unattributed"
    seat_candidates: dict[str, float] = field(default_factory=dict)
    track_id: int | None = None
    pose_source: str = ""            # which pose configuration positioned it
    health_state: list[str] = field(default_factory=list)
    calibration_version: str = ""

    @property
    def native_w(self) -> float:
        return self.x2 - self.x1

    @property
    def native_h(self) -> float:
        return self.y2 - self.y1

    @property
    def native_px(self) -> float:
        """The source footprint. The only size an abstention may be judged on."""
        return max(self.native_w, 0.0) * max(self.native_h, 0.0)

    @property
    def magnification(self) -> float:
        """How much this crop is enlarged to reach the detector input.

        Recorded so a reader can see at a glance that a confident detection came
        from a region magnified 20x, which is a statement about interpolation
        rather than about the object.
        """
        longest = max(self.native_w, self.native_h)
        return self.input_size / longest if longest > 0 else 0.0

    def insufficient(self, cfg: "CropConfig") -> bool:
        return min(self.native_w, self.native_h) < cfg.abstain_below_native_px

    def to_row(self) -> dict:
        return {
            **asdict(self),
            "native_w": round(self.native_w, 2),
            "native_h": round(self.native_h, 2),
            "native_px": round(self.native_px, 1),
            "magnification": round(self.magnification, 2),
        }


def crop_id(video_sha: str, pts_ms: float, config: str,
            box: tuple[float, float, float, float]) -> str:
    """Deterministic, and keyed on the policy as well as the geometry.

    Two policies may produce the same rectangle on a given frame; they are still
    two crops, because the *reason* differs and a comparison of crop policies
    needs both rows.
    """
    key = (f"{video_sha}|{pts_ms:.3f}|{config}|"
           f"{box[0]:.2f},{box[1]:.2f},{box[2]:.2f},{box[3]:.2f}")
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _clamp(box: tuple[float, float, float, float], w: int, h: int
           ) -> tuple[tuple[float, float, float, float], bool]:
    x1, y1, x2, y2 = box
    cx1, cy1 = max(x1, 0.0), max(y1, 0.0)
    cx2, cy2 = min(x2, float(w)), min(y2, float(h))
    clipped = (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2)
    return (cx1, cy1, cx2, cy2), clipped


def build(pose_row, observation, profile, cfg: CropConfig | None = None,
          frame_w: int | None = None, frame_h: int | None = None
          ) -> ObjectCrop:
    """Build one crop for one pose row, walking PRD 10's fallback chain.

    `observation` may be None -- no pose configuration answered this row -- in
    which case the chain starts already fallen back, with the reason recorded.
    """
    cfg = cfg or CropConfig()
    frame_w = frame_w or (profile.frame_w if profile else 1920)
    frame_h = frame_h or (profile.frame_h if profile else 1080)

    seat = None
    if profile is not None and pose_row.seat_state not in (
            "ambiguous_seat", "unattributed"):
        seat = profile.seat(pose_row.seat_state)

    requested = POSE_HAND
    chosen, reason, box = None, "", None

    # ---- 1. pose_hand_context, when a wrist is actually a location ----------
    if observation is None:
        reason = FALLBACK_NO_POSE
    else:
        from .pose import L_WRIST, R_WRIST, NOT_RESOLVABLE, VISIBLE

        best = None
        for index in (L_WRIST, R_WRIST):
            state = observation.state(index)
            if state == NOT_RESOLVABLE:
                reason = reason or FALLBACK_UNRESOLVABLE
                continue
            if state != VISIBLE:
                continue
            joint = next(j for j in observation.joints
                         if j.name == ("left_wrist" if index == L_WRIST
                                       else "right_wrist"))
            if joint.confidence < cfg.wrist_confidence_min:
                reason = reason or FALLBACK_LOW_WRIST
                continue
            if best is None or joint.confidence > best.confidence:
                best = joint

        if best is not None:
            # The context radius scales with the person, so a candidate at the
            # back of the hall gets a proportionally sized crop rather than a
            # fixed pixel box that would be their whole torso.
            scale = observation.torso_scale_px or (
                (pose_row.box[3] - pose_row.box[1]) * 0.30)
            radius = max(scale * cfg.hand_context_torso, cfg.min_native_px)
            box = (best.x - radius, best.y - radius,
                   best.x + radius, best.y + radius)
            chosen = POSE_HAND
        elif not reason:
            reason = FALLBACK_LOW_WRIST

    # ---- 2. lap_bag_context, then seat_desk_context -------------------------
    if chosen is None and seat is not None:
        for kind, name in (("lap_bag", LAP_BAG),
                           ("desk_work_surface", SEAT_DESK)):
            polygon = seat.zones.get(kind)
            if not polygon:
                continue
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            box = (min(xs), min(ys), max(xs), max(ys))
            chosen = name
            break

    # ---- 3. native_source_context_fallback ---------------------------------
    if chosen is None:
        box = pose_row.box
        chosen = NATIVE_FALLBACK
        reason = reason or FALLBACK_NO_SEAT

    box, clipped = _clamp(box, frame_w, frame_h)
    if clipped and not reason:
        reason = FALLBACK_OUT_OF_FRAME

    return ObjectCrop(
        crop_id=crop_id(pose_row.source_video_sha256, pose_row.pts_ms,
                        chosen, box),
        pose_row_id=pose_row.pose_row_id,
        event_id=pose_row.event_id,
        source_video_sha256=pose_row.source_video_sha256,
        pts_ms=pose_row.pts_ms,
        frame_index=pose_row.frame_index,
        crop_config=chosen,
        requested_config=requested,
        fallback_reason="" if chosen == requested else reason,
        x1=round(box[0], 2), y1=round(box[1], 2),
        x2=round(box[2], 2), y2=round(box[3], 2),
        input_size=cfg.input_size,
        seat_state=pose_row.seat_state,
        seat_candidates=dict(pose_row.seat_candidates),
        track_id=pose_row.track_id,
        pose_source=observation.config_id if observation else "",
        health_state=list(pose_row.health_state),
        calibration_version=pose_row.calibration_version,
    )


def freeze(pose_rows, observations_by_row: dict, profile,
           cfg: CropConfig | None = None) -> list[ObjectCrop]:
    """Freeze `object_crop_manifest.jsonl` from the pose manifest (PRD 4.4).

    Every detector configuration is handed this list unchanged. As with the pose
    manifest, sameness is checkable rather than asserted -- see `digest`.
    """
    cfg = cfg or CropConfig()
    out = []
    for row in pose_rows:
        leaked = [k for k in sorted(set(row.to_row())
                                    | set(getattr(row, "__dict__", {})))
                  if "reference" in k.lower()]
        if leaked:
            raise ValueError(
                f"pose row {row.pose_row_id} carries reference field(s) "
                f"{leaked}; PRD 15 makes references readable only at stage 12")
        out.append(build(row, observations_by_row.get(row.pose_row_id),
                         profile, cfg))
    out.sort(key=lambda c: (c.pts_ms, c.crop_id))
    return out


def digest(crops: list[ObjectCrop]) -> str:
    """One hash over the crop manifest, for the same reason the pose one exists."""
    sha = hashlib.sha256()
    for crop in sorted(crops, key=lambda c: c.crop_id):
        sha.update(f"{crop.crop_id}|{crop.x1:.2f},{crop.y1:.2f},"
                   f"{crop.x2:.2f},{crop.y2:.2f}|{crop.crop_config}".encode())
        sha.update(b"\n")
    return sha.hexdigest()


def extract(frame, crop: ObjectCrop, cfg: CropConfig | None = None):
    """Cut and letterbox one crop, returning the array and its transform.

    Delegates to the measured `classroom.crops.extract` so the projection
    arithmetic that Phase 6 already exercised is not rewritten here.
    """
    from classroom.crops import extract as _extract

    cfg = cfg or CropConfig()
    bounds = (int(crop.x1), int(crop.y1), int(crop.x2), int(crop.y2))
    return _extract(frame, bounds, cfg.input_size)


def summarise(crops: list[ObjectCrop], cfg: CropConfig | None = None) -> dict:
    """Crop-policy mix, which is itself a finding.

    A run where most crops fell back to `native_source_context_fallback` is a run
    where pose did not position anything, and the detector A/B below it is
    measuring seat crops whatever the configuration names claim.
    """
    cfg = cfg or CropConfig()
    if not crops:
        return {"crops": 0, "note": "no crops were built; there is nothing to "
                                    "detect on and this is not a clean result."}

    by_config: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    for crop in crops:
        by_config[crop.crop_config] = by_config.get(crop.crop_config, 0) + 1
        if crop.fallback_reason:
            by_reason[crop.fallback_reason] = by_reason.get(crop.fallback_reason, 0) + 1

    insufficient = [c for c in crops if c.insufficient(cfg)]
    magnifications = sorted(c.magnification for c in crops)
    return {
        "crops": len(crops),
        "digest": digest(crops),
        "by_crop_config": dict(sorted(by_config.items())),
        "fallback_reasons": dict(sorted(by_reason.items())),
        "hand_context_share": round(by_config.get(POSE_HAND, 0) / len(crops), 4),
        "insufficient_native_resolution": len(insufficient),
        "insufficient_fraction": round(len(insufficient) / len(crops), 4),
        "median_magnification": round(magnifications[len(magnifications) // 2], 2),
        "max_magnification": round(magnifications[-1], 2),
        "note": "magnification is interpolation, not detail. A crop at 20x is "
                "640 px of upscaled evidence over a 32 px source footprint, and "
                "the abstention rule is judged on the footprint (PRD 10).",
    }
