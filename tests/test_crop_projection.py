"""Phase 5 gate: crop geometry, provenance and the fallback chain (PRD 10).

Two properties, both of which fail silently when they fail.

**Projection.** A box detected in a letterboxed 640x640 input and left there is
wrong by the pad offset and the scale factor. It still draws a plausible
rectangle on an overlay, still lands inside the frame, and still looks like
evidence. Every crop therefore round-trips: source rectangle to input space and
back, and the arithmetic must return what it was given.

**The fallback chain.** PRD 10 requires a `pose_hand_context` crop only when the
wrist is reliable, and a recorded fallback otherwise:

> Hand-first is refinement, not primary detection. A hand may be hidden or
> mislocalised.

The failure to prevent is a hand crop built from a 0.1-confidence wrist. It is
indistinguishable in the artifacts from a real one unless the reason travels
with the crop, so the reason is what this tests.

Also covered: PRD 10's rule that upscaling cannot invent detail. Magnification
is recorded, and abstention is judged on the **native** footprint, so a 20x14 px
region blown up to 640x640 abstains no matter how confident the output looks.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.calibration import (  # noqa: E402
    APPROVED, STATIC_BACKGROUND, CalibrationProfile, MaskRegion, Seat,
)
from pipeline.crops import (  # noqa: E402
    FALLBACK_LOW_WRIST, FALLBACK_NO_POSE, FALLBACK_UNRESOLVABLE, LAP_BAG,
    NATIVE_FALLBACK, POSE_HAND, SEAT_DESK, CropConfig, build, crop_id, digest,
    freeze, summarise,
)
from pipeline.pose import (  # noqa: E402
    JointState, NOT_RESOLVABLE, OCCLUDED, PoseObservation, PoseRow, VISIBLE,
)


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def rect(x1, y1, x2, y2):
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def make_profile() -> CalibrationProfile:
    return CalibrationProfile(
        camera_id="cam-t", epoch_id="e1", version=1, frame_w=1280, frame_h=720,
        operator="test", reviewer="test-reviewer", approval_state=APPROVED,
        seats=[Seat(seat_id="S-A",
                    zones={"seat_context": rect(100, 300, 400, 600),
                           "desk_work_surface": rect(100, 500, 400, 600),
                           "upper_body": rect(100, 300, 400, 500),
                           "lap_bag": rect(120, 560, 380, 600)},
                    desk_line_y=520.0)],
        masks=[MaskRegion(STATIC_BACKGROUND, rect(0, 0, 1280, 200))])


def make_row(seat_state="S-A", box=(150.0, 300.0, 350.0, 600.0),
             pose_row_id="p1") -> PoseRow:
    return PoseRow(pose_row_id=pose_row_id, row_id="r1", event_id="EV-1",
                   source_video_sha256="abc123", pts_ms=1000.0,
                   frame_index=25, box=box, seat_state=seat_state,
                   seat_candidates={"S-A": 0.9},
                   calibration_version="cam-t/e1/v1")


def make_observation(wrist_conf=0.9, wrist_state=VISIBLE, torso=60.0,
                     wrist_xy=(300.0, 480.0)) -> PoseObservation:
    observation = PoseObservation(
        pose_row_id="p1", config_id="rtmpose_baseline", pts_ms=1000.0,
        seat_state="S-A", torso_scale_px=torso, person_height_px=300.0)
    for name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear",
                 "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                 "left_hip", "right_hip"):
        observation.joints.append(JointState(name, 240.0, 360.0, 0.9, VISIBLE))
    observation.joints.append(
        JointState("left_wrist", wrist_xy[0], wrist_xy[1], wrist_conf,
                   wrist_state))
    observation.joints.append(
        JointState("right_wrist", 190.0, 480.0, 0.05, OCCLUDED))
    return observation


def main() -> int:
    ok = True
    profile = make_profile()
    cfg = CropConfig()

    print("a reliable wrist produces a pose_hand_context crop")
    hand = build(make_row(), make_observation(), profile, cfg)
    ok &= check("policy is pose_hand_context", hand.crop_config == POSE_HAND,
                hand.crop_config)
    ok &= check("no fallback reason is recorded", not hand.fallback_reason)
    ok &= check("the crop is centred on the wrist",
                abs((hand.x1 + hand.x2) / 2 - 300.0) < 1.0 and
                abs((hand.y1 + hand.y2) / 2 - 480.0) < 1.0,
                f"({(hand.x1 + hand.x2) / 2:.0f}, {(hand.y1 + hand.y2) / 2:.0f})")
    ok &= check("context scales with the person, not a fixed pixel box",
                abs(hand.native_w - 2 * 60.0) < 1.0,
                f"{hand.native_w:.0f} px across a 60 px torso")
    ok &= check("the pose configuration that positioned it is recorded",
                hand.pose_source == "rtmpose_baseline", hand.pose_source)

    print("\na larger person gets a proportionally larger hand crop")
    far = build(make_row(), make_observation(torso=25.0), profile, cfg)
    near = build(make_row(), make_observation(torso=120.0), profile, cfg)
    ok &= check("crop size tracks torso scale", near.native_w > far.native_w,
                f"{far.native_w:.0f} px vs {near.native_w:.0f} px")

    print("\na low-confidence wrist falls back, and says so (PRD 10)")
    weak = build(make_row(), make_observation(wrist_conf=0.10), profile, cfg)
    ok &= check("policy is not pose_hand_context",
                weak.crop_config != POSE_HAND, weak.crop_config)
    ok &= check("it fell back to a calibrated zone",
                weak.crop_config in (LAP_BAG, SEAT_DESK), weak.crop_config)
    ok &= check("the reason names the wrist confidence",
                weak.fallback_reason == FALLBACK_LOW_WRIST, weak.fallback_reason)
    ok &= check("the requested policy is still recorded",
                weak.requested_config == POSE_HAND, weak.requested_config)

    print("\na wrist below the source-scale floor never positions a crop")
    unresolvable = build(make_row(), make_observation(wrist_state=NOT_RESOLVABLE),
                         profile, cfg)
    ok &= check("policy fell back", unresolvable.crop_config != POSE_HAND,
                unresolvable.crop_config)
    ok &= check("the reason distinguishes source scale from low confidence",
                unresolvable.fallback_reason == FALLBACK_UNRESOLVABLE,
                unresolvable.fallback_reason)

    print("\nno pose observation at all falls back with its own reason")
    blind = build(make_row(), None, profile, cfg)
    ok &= check("policy fell back", blind.crop_config != POSE_HAND,
                blind.crop_config)
    ok &= check("the reason is no_pose_observation",
                blind.fallback_reason == FALLBACK_NO_POSE, blind.fallback_reason)
    ok &= check("no pose source is claimed", blind.pose_source == "")

    print("\nan unattributed seat with no pose reaches the native fallback")
    orphan = build(make_row(seat_state="unattributed"), None, profile, cfg)
    ok &= check("policy is native_source_context_fallback",
                orphan.crop_config == NATIVE_FALLBACK, orphan.crop_config)
    ok &= check("it falls back to the person box itself",
                (orphan.x1, orphan.y1, orphan.x2, orphan.y2) ==
                (150.0, 300.0, 350.0, 600.0),
                f"({orphan.x1}, {orphan.y1}, {orphan.x2}, {orphan.y2})")

    print("\ncrops are clamped to the frame, never allowed outside it")
    edge = build(make_row(), make_observation(wrist_xy=(20.0, 30.0), torso=120.0),
                 profile, cfg)
    ok &= check("x1 and y1 are inside the frame",
                edge.x1 >= 0.0 and edge.y1 >= 0.0, f"({edge.x1}, {edge.y1})")
    ok &= check("x2 and y2 are inside the frame",
                edge.x2 <= profile.frame_w and edge.y2 <= profile.frame_h,
                f"({edge.x2}, {edge.y2})")

    print("\nsource-to-input projection round-trips (the silent-failure case)")
    from classroom.crops import extract as raw_extract
    import numpy as np
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas, transform = raw_extract(
        frame, (int(hand.x1), int(hand.y1), int(hand.x2), int(hand.y2)),
        cfg.input_size)
    ok &= check("the input is the configured square",
                canvas.shape[:2] == (cfg.input_size, cfg.input_size),
                str(canvas.shape))
    source_box = (hand.x1 + 10, hand.y1 + 12, hand.x1 + 40, hand.y1 + 34)
    round_tripped = transform.to_frame(transform.to_input(source_box))
    ok &= check("source -> input -> source returns the same box",
                all(abs(a - b) < 0.51 for a, b in zip(source_box, round_tripped)),
                f"{[round(v, 2) for v in round_tripped]}")
    # The failure this catches: a box left in input space.
    naive = transform.to_input(source_box)
    ok &= check("an unprojected box is materially wrong, not close",
                max(abs(a - b) for a, b in zip(source_box, naive)) > 5.0,
                f"off by {max(abs(a - b) for a, b in zip(source_box, naive)):.0f} px")

    print("\nmagnification is recorded, and abstention is judged on native px")
    tiny_row = make_row(box=(200.0, 400.0, 220.0, 414.0), pose_row_id="p-tiny")
    tiny = build(tiny_row, None, profile, cfg)
    ok &= check("a tiny crop is still emitted, not dropped", tiny is not None)
    big_magnification = build(
        make_row(seat_state="unattributed", box=(200.0, 400.0, 222.0, 416.0)),
        None, profile, cfg)
    ok &= check("magnification above 20x is recorded",
                big_magnification.magnification > 20.0,
                f"{big_magnification.magnification:.1f}x")
    ok &= check("and it is judged insufficient on the native footprint",
                big_magnification.insufficient(cfg),
                f"{big_magnification.native_w:.0f}x"
                f"{big_magnification.native_h:.0f} px native")
    ok &= check("a full-size crop is not insufficient",
                not hand.insufficient(cfg),
                f"{hand.native_w:.0f}x{hand.native_h:.0f} px native")

    print("\ncrop identity is deterministic and policy-aware")
    a = crop_id("abc", 1000.0, POSE_HAND, (10.0, 20.0, 30.0, 40.0))
    b = crop_id("abc", 1000.0, POSE_HAND, (10.0, 20.0, 30.0, 40.0))
    c = crop_id("abc", 1000.0, SEAT_DESK, (10.0, 20.0, 30.0, 40.0))
    ok &= check("the same crop hashes the same", a == b, a)
    ok &= check("the same rectangle under a different policy is a different crop",
                a != c, "two policies producing one rectangle are still two rows")

    print("\nthe manifest freezes, hashes, and refuses reference leakage")
    rows = [make_row(pose_row_id=f"p{i}") for i in range(4)]
    manifest = freeze(rows, {"p0": make_observation()}, profile, cfg)
    ok &= check("one crop per pose row", len(manifest) == 4, str(len(manifest)))
    ok &= check("the digest is stable", digest(manifest) == digest(manifest))
    ok &= check("order does not change the digest",
                digest(list(reversed(manifest))) == digest(manifest))

    leaky = make_row(pose_row_id="p-leak")
    object.__setattr__(leaky, "reference_class", "phone")
    try:
        freeze([leaky], {}, profile, cfg)
        ok &= check("freeze refuses a leaked reference field", False)
    except ValueError as exc:
        ok &= check("freeze refuses a leaked reference field", True, str(exc)[:60])

    print("\nthe summary exposes the crop-policy mix as a finding")
    report = summarise(manifest, cfg)
    for field in ("by_crop_config", "fallback_reasons", "hand_context_share",
                  "insufficient_native_resolution", "median_magnification"):
        ok &= check(f"summary reports {field}", field in report)
    ok &= check("an empty manifest is not a clean result",
                "not a clean result" in summarise([], cfg)["note"])

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
